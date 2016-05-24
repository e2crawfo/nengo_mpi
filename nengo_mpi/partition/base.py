from collections import defaultdict
from functools import partial
import warnings

import networkx as nx
import numpy as np

from nengo import Direct, Node, Ensemble, Connection, Probe
from nengo.base import ObjView
from nengo.ensemble import Neurons
from nengo.utils.builder import find_all_io
from nengo.utils.compat import is_iterable, itervalues

from nengo_mpi import PartitionError
from nengo_mpi.spaun_mpi import SpaunStimulus
from nengo_mpi.partition.work_balanced import work_balanced_partitioner
from nengo_mpi.partition.metis import metis_available, metis_partitioner
from nengo_mpi.partition.random import random_partitioner


import logging
logger = logging.getLogger(__name__)

_partitioners = [random_partitioner, work_balanced_partitioner]
if metis_available():
    _partitioners.append(metis_partitioner)


def partitioners():
    return _partitioners[:]


def get_probed_connections(network):
    """ Return all connections that have probes in ``network``. """
    probed_connections = []
    for probe in network.all_probes:
        if isinstance(probe.target, Connection):
            probed_connections.append(probe.target)

    return probed_connections


def _can_cross_boundary(conn, probed_connections, max_size=np.inf):
    """ Return whether ``conn`` is allowed to cross component boundaries. """

    can_cross = conn.synapse is not None
    can_cross &= conn.size_mid <= max_size
    can_cross &= conn not in probed_connections
    can_cross &= not conn.learning_rule_type

    return can_cross


def make_boundary_predicate(network, max_size=np.inf):
    probed_connections = set(get_probed_connections(network))
    predicate = partial(
        _can_cross_boundary,
        probed_connections=probed_connections,
        max_size=max_size)

    return predicate


def verify_assignments(network, assignments):
    """ Propagate the assignments given in ``assignments``.

    This is used when an assignment of objects to components is supplied
    directly to ``nengo_mpi.Simulator`` in lieu of a ``Partitioner`` object.
    It does not scale well and is mainly useful for demonstration purposes.

    Parameters
    ----------
    network: nengo.Network
        The network whose assignments are to be verified.
    assignments: dict
        A mapping from each nengo object in the network to an integer
        specifying which component of the partition the object is assigned
        to. Component 0 is simulated by the master process.

    """
    n_components = max(assignments.values()) + 1

    can_cross_boundary = make_boundary_predicate(network)
    propogate_assignments(network, assignments, can_cross_boundary)

    if n_components > 1:
        component0, cluster_graph = network_to_cluster_graph(
            network, can_cross_boundary)

        evaluate_partition(
            network, n_components, assignments, cluster_graph,
            can_cross_boundary)

    return n_components, assignments


class Partitioner(object):
    """ Divide a nengo network into components that can be simulated independently.

    Parameters
    ----------
    n_components: int (optional)
        The number of components to divide the nengo network into.

    func: function (optional)
        A function to partition the nengo graph, assigning nengo objects
        to component indices. Ignored if ``n_components == 1``.

        Arguments for func:
            cluster_graph
            n_components

    use_weights: boolean (optional)
        Whether to use the size_mid attribute of connections to weight the
        edges in the graph that we partition. If False, then all edges have
        the same weight.

    straddle_conn_max_size: int/float (optional)
        Connections of this size or greater are not permitted to straddle
        component boundaries. Two nengo objects that are connected by a
        Connection that is bigger than this size are forced to be in the
        same component.

    args: Extra positional args passed to func.

    kwargs: Extra keyword args passed to func.

    """
    def __init__(
            self, n_components=1, func=None, use_weights=True,
            straddle_conn_max_size=np.inf, *args, **kwargs):

        self.n_components = n_components

        if func is None:
            func = self.default_partition_func
        self.func = func

        self.straddle_conn_max_size = straddle_conn_max_size
        self.args = args
        self.kwargs = kwargs

    @property
    def default_partition_func(self):
        if metis_available():
            print ("Defaulting to metis partitioner")
            return metis_partitioner
        else:
            print ("Defaulting to work-balanced partitioner")
            return work_balanced_partitioner

    def partition(self, network):
        """
        Partition ``network`` using the partitioning function ``self.func``.

        If ``self.n_components == 1`` or the number of independently
        simulatable chunks of the network is less than ``self.n_components``,
        the partitioning function is not used. In the former case, all objects
        go on component 0, and in the latter case, we arbitrarily assign each
        independently simulatable chunk to its own component.

        Steps:
            1. Construct a cluster graph from the nengo network.
            2. Partition the cluster graph using ``self.func``.

        Parameters
        ----------
        network: nengo.Network
            The network to partition.

        Returns
        -------
        n_components: int
            The number of components the nengo network is split into. May
            be different than the value supplied to Partitioner.__init__ in
            cases where it is deemed impossible to split the network into the
            desired number of components.
        assignments: dict
            A mapping from each nengo object in the network to an index
            specifying which component of the partition the object is assigned
            to. Component 0 is simulated by the master process.

        """
        # A mapping from each nengo object to its component index
        object_assignments = {}

        duplicate_spaun_stim(network)

        can_cross_boundary = make_boundary_predicate(
            network, self.straddle_conn_max_size)

        if self.n_components > 1:
            # component0 is also in the cluster graph
            component0, cluster_graph = network_to_cluster_graph(
                network, can_cross_boundary)

            n_clusters = len(cluster_graph)

            # ``cluster_assignments`` maps each cluster to its component idx
            if n_clusters <= self.n_components:
                self.n_components = n_clusters
                cluster_assignments = {
                    cluster: i for i, cluster in enumerate(cluster_graph)}
            else:
                cluster_assignments = self.func(
                    cluster_graph, self.n_components,
                    *self.args, **self.kwargs)

            if component0:
                # Assign the objects in ``component0`` to component 0
                on_zero = filter(
                    lambda n: cluster_assignments[n] == 0, cluster_assignments)
                c = cluster_assignments[component0]
                on_c = filter(
                    lambda n: cluster_assignments[n] == c, cluster_assignments)

                cluster_assignments.update({cluster: c for cluster in on_zero})
                cluster_assignments.update({cluster: 0 for cluster in on_c})

            for cluster, component in cluster_assignments.iteritems():
                cluster.assign_to_component(object_assignments, component)

        propogate_assignments(network, object_assignments, can_cross_boundary)

        if self.n_components > 1:
            evaluate_partition(
                network, self.n_components, object_assignments,
                cluster_graph, can_cross_boundary)

        return self.n_components, object_assignments


class NengoObjectCluster(object):
    """ A collection of nengo objects that must be simulated together.

    Two nengo objects must be simulated together iff there exists a
    path between them in the *undirected* graph of nengo objects which
    contains a Connection that is *not* permitted to cross component
    boundaries (as defined by the function ``can_cross_boundary``).

    NengoObjectClusters are used as the nodes in the cluster graph
    that is created as part of the partitioning process.

    Parameters
    ----------
    obj: nengo object
        The initial nengo object stored in the node.

    """
    def __init__(self, obj):
        self.objects = set()
        self.inputs = set()
        self.outputs = set()
        self._n_neurons = 0
        self.connections = []

        self.add_object(obj)

    @property
    def n_neurons(self):
        return self._n_neurons

    def add_object(self, obj):
        self.objects.add(obj)

        if hasattr(obj, 'n_neurons'):
            self._n_neurons += obj.n_neurons

    def add_input(self, i):
        self.inputs.add(i)

    def add_output(self, o):
        self.outputs.add(o)

    def empty(self):
        return len(self.objects) == 0

    def __str__(self):
        s = (
            "<NengoObjectCluster: "
            + ", ".join(str(o) for o in self.objects) + ">")
        return s

    def __repr__(self):
        return str(self)

    def assign_to_component(self, assignments, component):
        """
        Assign all nengo objects in ``self`` to the ``component``.

        Alters the provided ``assignments`` dictionary.

        Parameters
        ----------
        assignments: dict
            A dict mapping each nengo object to its component.
        component: intent
            The component to assign the nengo objects to.

        """
        for obj in self.objects:
            assignments[obj] = component

    def merge(self, other):
        """ Merge ``self`` with ``other``. Return True if a merging occurs. """

        if self == other:
            return False

        for obj in other.objects:
            if hasattr(obj, 'n_neurons'):
                self._n_neurons += obj.n_neurons

        self.objects = self.objects.union(other.objects)

        self.inputs = set([
            i for i in self.inputs.union(other.inputs)
            if not (i.pre_obj in self.objects and i.post_obj in self.objects)])

        self.outputs = set([
            o for o in self.outputs.union(other.outputs)
            if not (o.pre_obj in self.objects and o.post_obj in self.objects)])

        self.connections.extend(other.connections)

        return True


def network_to_cluster_graph(
        network, can_cross_boundary=None,
        use_weights=True, merge_nengo_nodes=True):
    """ Create a cluster graph from a nengo Network.

    A cluster graph is a graph wherein the nodes are maximally large
    NengoObjectClusters, and edges are nengo Connections that are permitted to
    cross component boundaries.

    Parameters
    ----------
    network: nengo.Network
        The network whose cluster graph we want to construct.

    can_cross_boundary: function
        A function which accepts a Connection, and returns a boolean specifying
        whether the Connection is allowed to cross component boundaries.

    use_weights: boolean
        Whether edges in the cluster graph should be weighted by the
        ``size_mid`` attribute of the connection. If not, then all connections
        are weighted equally.

    merge_nengo_nodes: boolean
        If True, then clusters which would consist entirely of nengo Nodes are
        merged with a neighboring cluster. This is done because it is typically
        not useful to have a processor simulating only Nodes, as it will only
        add extra communication without easing the computational burden.

    Returns
    -------
    component0: NengoObjectCluster
        A NengoObjectCluster containing all nengo objects which must be
        simulated on the master process in the nengo_mpi simulator. If there
        are no such objects, then this has value None.

    cluster_graph: networkx.Graph
        A graph wherein the nodes are instances of NengoObjectCluster.
        Importantly, if ``component0`` is not None, then it is included
        in ``cluster_graph``.

    """
    def merge_clusters(cluster_map, a, b, conn=None):
        if a.merge(b):
            for obj in b.objects:
                cluster_map[obj] = a

            del b

        if conn is not None:
            a.connections.append(conn)

        return a

    if can_cross_boundary is None:
        can_cross_boundary = make_boundary_predicate(network)

    # A mapping from each nengo object to the cluster it is a member of
    cluster_map = {
        obj: NengoObjectCluster(obj) for obj
        in network.all_nodes + network.all_ensembles}

    for conn in network.all_connections:
        pre_obj = neurons2ensemble(conn.pre_obj)
        pre_cluster = cluster_map[pre_obj]

        post_obj = neurons2ensemble(conn.post_obj)
        post_cluster = cluster_map[post_obj]

        if can_cross_boundary(conn):
            pre_cluster.add_output(conn)
            post_cluster.add_input(conn)
        else:
            merge_clusters(cluster_map, pre_cluster, post_cluster, conn)

    all_clusters = list(set(cluster_map.values()))

    _, outputs = find_all_io(network.all_connections)

    # merge together all clusters that have to go on component 0
    component0 = filter(
        lambda x: for_component0(x, outputs), all_clusters)

    if component0:
        all_clusters = filter(lambda x: x not in component0[1:], all_clusters)
        component0 = reduce(
            lambda u, v: merge_clusters(cluster_map, u, v), component0)
    else:
        component0 = None

    any_neurons = any(cluster.n_neurons > 0 for cluster in all_clusters)
    if merge_nengo_nodes and any_neurons:
        # For each cluster which does not contain any neurons, merge the
        # cluster with another cluster which *does* contain neurons,
        # and which the original cluster communicates strongly with.

        without_neurons = filter(lambda c: c.n_neurons == 0, all_clusters)

        for cluster in without_neurons:
            # figure out which cluster would be most beneficial to merge with.
            counts = defaultdict(int)

            for i in cluster.inputs:
                pre_obj = neurons2ensemble(i.pre_obj)
                pre_cluster = cluster_map[pre_obj]
                if pre_cluster.n_neurons > 0:
                    counts[pre_cluster] += i.size_mid

            for o in cluster.outputs:
                post_obj = neurons2ensemble(o.post_obj)
                post_cluster = cluster_map[post_obj]
                if post_cluster.n_neurons > 0:
                    counts[post_cluster] += o.size_mid

            if counts:
                best_cluster = max(counts, key=counts.__getitem__)
            else:
                best_cluster = (
                    n for n in all_clusters if n.n_neurons > 0).next()

            merge_clusters(cluster_map, best_cluster, cluster)
            all_clusters.remove(cluster)

    G = nx.Graph()
    G.add_nodes_from(all_clusters)

    boundary_connections = filter(can_cross_boundary, network.all_connections)

    for conn in boundary_connections:
        pre_cluster = cluster_map[neurons2ensemble(conn.pre_obj)]
        post_cluster = cluster_map[neurons2ensemble(conn.post_obj)]

        if pre_cluster != post_cluster:
            weight = conn.size_mid if use_weights else 1.0

            if G.has_edge(pre_cluster, post_cluster):
                G[pre_cluster][post_cluster]['weight'] += weight
                G[pre_cluster][post_cluster]['connections'].append(conn)
            else:
                G.add_edge(
                    pre_cluster, post_cluster,
                    weight=weight, connections=[conn])

    return component0, G


def neurons2ensemble(e):
    return e.ensemble if isinstance(e, Neurons) else e


def for_component0(cluster, outputs):
    """ Returns whether the cluster must be simulated on process 0. """

    for obj in cluster.objects:
        if isinstance(obj, Node) and callable(obj.output):
            return True

        if isinstance(obj, Node):
            if any([conn.function is not None for conn in outputs[obj]]):
                return True

        if isinstance(obj, Ensemble) and isinstance(obj.neuron_type, Direct):
            return True

    return False


def duplicate_spaun_stim(network):
    """ Make copies of SpaunStimulus nodes to limit communication. """

    with network:
        for node in network.all_nodes:
            if isinstance(node, SpaunStimulus):
                conns = [
                    c for c in network.all_connections
                    if c.pre_obj is node]

                if len(conns) <= 1:
                    # Don't bother if there's only one connection
                    continue

                removed_probes = []
                for probe in network.all_probes:
                    if probe.target is node:
                        removed_probes.append(probe)
                        assert remove_from_network(probe)

                for conn in conns:
                    if conn.pre_obj is node:
                        stim = SpaunStimulus(
                            dimension=node.dimension,
                            stimulus_sequence=node.stimulus_sequence,
                            present_interval=node.present_interval,
                            present_blanks=node.present_blanks,
                            identifier=node.identifier)

                        if isinstance(conn.pre, ObjView):
                            stim = ObjView(stim, conn.pre.slice)

                        Connection(
                            stim, conn.post,
                            transform=conn.transform,
                            function=conn.function,
                            synapse=conn.synapse)

                        assert remove_from_network(network, conn)

                        for probe in removed_probes:
                            Probe(
                                node,
                                attr=probe.attr,
                                sample_every=probe.sample_every,
                                synapse=probe.synapse,
                                solver=probe.solver,
                                seed=probe.seed,
                                label=probe.label)

                assert remove_from_network(network, node)


def propogate_assignments(network, assignments, can_cross_boundary):
    """ Assign every object in ``network`` to a component.

    Propogates the component assignments stored in the dict ``assignments``
    (which only needs to contain assignments for top level Networks, Nodes and
    Ensembles) down to objects that are contained in those top-level objects.
    If assignments is empty, then all objects will be assigned to the 1st
    component, which has index 0. The intent is to have some partitioning
    algorithm determine some of the assignments before this function is called,
    and then have this function propogate those assignments.

    Also does validation, making sure that connections that cross component
    boundaries have certain properties (see ``can_cross_boundary``) and making
    sure that certain types of objects are assigned to component 0.

    Objects that must be simulated on component 0 are:
        1. Nodes with callable outputs.
        2. Ensembles of Direct neurons.
        3. Any Node that is the source for a Connection that has a function.

    Parameters
    ----------
    network: nengo.Network
        The network we are partitioning.

    assignments: dict
        A dictionary mapping from nengo objects to component indices.
        This dictionary will be altered to contain assignments for all objects
        in the network. If a network appears in assignments, then all objects
        in that network which do not also appear in assignments will be given
        the same assignment as the network.

    can_cross_boundary: function
        A function which accepts a Connection, and returns a boolean specifying
        whether the Connection is allowed to cross component boundaries.

    Returns
    -------
    Nothing, but ``assignments`` is modified.

    """
    def helper(network, assignments, outputs):
        for node in network.nodes:
            if callable(node.output):
                if node in assignments and assignments[node] != 0:
                    warnings.warn(
                        "Found Node with callable output that was assigned to "
                        "a component other than component 0. Overriding "
                        "previous assignment.")

                assignments[node] = 0

            else:
                if any([conn.function is not None for conn in outputs[node]]):
                    if node in assignments and assignments[node] != 0:
                        warnings.warn(
                            "Found Node with an output connection whose "
                            "function is not None, which is assigned to a "
                            "component other than component 0. Overriding "
                            "previous assignment.")

                    assignments[node] = 0

                elif node not in assignments:
                    assignments[node] = assignments[network]

        for ensemble in network.ensembles:
            if isinstance(ensemble.neuron_type, Direct):
                if ensemble in assignments and assignments[ensemble] != 0:
                    warnings.warn(
                        "Found Direct-mode ensemble that was assigned to a "
                        "component other than component 0. Overriding "
                        "previous assignment.")

                assignments[ensemble] = 0

            elif ensemble not in assignments:
                assignments[ensemble] = assignments[network]

            assignments[ensemble.neurons] = assignments[ensemble]

        for n in network.networks:
            if n not in assignments:
                assignments[n] = assignments[network]

            helper(n, assignments, outputs)

    assignments[network] = 0

    _, outputs = find_all_io(network.all_connections)

    try:
        helper(network, assignments, outputs)

        # Assign learning rules
        for conn in network.all_connections:
            if conn.learning_rule is not None:
                rule = conn.learning_rule
                if is_iterable(rule):
                    rule = itervalues(rule) if isinstance(rule, dict) else rule
                    for r in rule:
                        assignments[r] = assignments[conn.pre_obj]
                elif rule is not None:
                    assignments[rule] = assignments[conn.pre_obj]

        # Check for connections erroneously crossing component boundaries
        non_crossing = [
            conn for conn in network.all_connections
            if not can_cross_boundary(conn)]

        for conn in non_crossing:
            pre_component = assignments[conn.pre_obj]
            post_component = assignments[conn.post_obj]

            if pre_component != post_component:
                raise PartitionError(
                    "Connection %s crosses a component "
                    "boundary, but it is not permitted to. "
                    "Pre-object assigned to %d, post-object "
                    "assigned to %d." % (conn, pre_component, post_component))

        # Assign probes
        for probe in network.all_probes:
            target = (
                probe.target.obj
                if isinstance(probe.target, ObjView)
                else probe.target)

            if isinstance(target, Connection):
                target = target.pre_obj

            assignments[probe] = assignments[target]

    except KeyError as e:
        # Nengo tests require a value error to be raised in these cases.
        msg = "Invalid Partition. KeyError: %s" % e.message,
        raise ValueError(msg)

    nodes = network.all_nodes
    nodes_in = all([node in assignments for node in nodes])
    assert nodes_in, "Assignments incomplete, missing some nodes."

    ensembles = network.all_ensembles
    ensembles_in = all([ensemble in assignments for ensemble in ensembles])
    assert ensembles_in, "Assignments incomplete, missing some ensembles."


def total_neurons(network):
    n_neurons = 0

    for e in network.ensembles:
        n_neurons += e.n_neurons

    for n in network.networks:
        n_neurons += total_neurons(n)

    return n_neurons


def evaluate_partition(
        network, n_components, assignments, cluster_graph, can_cross_boundary):
    """ Print a summary of the quality of a partition. """

    print ( "*" * 80)

    key = lambda n: sum(
        e.n_neurons for e in n.objects if hasattr(e, 'n_neurons'))

    cluster_n_neurons = [key(n) for n in cluster_graph.nodes()]
    cluster_n_items = [len(n.objects) for n in cluster_graph.nodes()]

    only_nodes = [
        all(isinstance(o, Node) for o in n.objects)
        for n in cluster_graph.nodes()]

    print ("Cluster graph statistics:")
    print ("Number of clusters: ", cluster_graph.number_of_nodes())
    print ("Number of edges: ", cluster_graph.number_of_edges())
    print ("Number of clusters containing only nengo Nodes: ", sum(only_nodes))

    print ("Mean neurons per cluster: ", np.mean(cluster_n_neurons))
    print ("Std of neurons per cluster", np.std(cluster_n_neurons))
    print ("Min number of neurons", np.min(cluster_n_neurons))
    print ("Max number of neurons", np.max(cluster_n_neurons))

    print ("Mean nengo objects per cluster: ", np.mean(cluster_n_items))
    print ("Std of nengo objects per cluster", np.std(cluster_n_items))
    print ("Min number of nengo objects", np.min(cluster_n_items))
    print ("Max number of nengo objects", np.max(cluster_n_items))

    component_neuron_counts = [0] * n_components
    component_item_counts = [0] * n_components

    for ens in network.all_ensembles:
        if ens in assignments:
            component_neuron_counts[assignments[ens]] += ens.n_neurons
            component_item_counts[assignments[ens]] += 1

    for node in network.all_nodes:
        if node in assignments:
            component_item_counts[assignments[ens]] += 1

    mean_neuron_count = np.mean(component_neuron_counts)
    neuron_count_std = np.std(component_neuron_counts)

    print ("*" * 20)
    print ("Evaluating partition of network")

    print ("Total number of neurons: ", total_neurons(network))
    print ("Mean neurons per component: ", mean_neuron_count)
    print ("Standard deviation of neurons per component", neuron_count_std)
    print ("Min number of neurons", np.min(component_neuron_counts))
    print ("Max number of neurons", np.max(component_neuron_counts))
    print (
        "Number of empty partitions: "
        "%d" % (n_components - np.count_nonzero(component_neuron_counts)))

    mean_item_count = np.mean(component_item_counts)
    item_count_std = np.std(component_item_counts)

    print ("*" * 10)

    print (
        "Total number of nengo objects (Nodes and Ensembles): "
        "%d" % len(network.all_nodes + network.all_ensembles))
    print ("Mean nengo objects per component: ", mean_item_count)
    print ("Standard deviation of nengo objects per component", item_count_std)
    print ("Min number of nengo objects", np.min(component_item_counts))
    print ("Max number of nengo objects", np.max(component_item_counts))

    communication_weight = 0
    total_weight = 0

    for conn in network.all_connections:
        if can_cross_boundary(conn):
            pre_obj = neurons2ensemble(conn.pre_obj)
            post_obj = neurons2ensemble(conn.post_obj)

            if assignments[pre_obj] != assignments[post_obj]:
                communication_weight += conn.size_mid

            total_weight += conn.size_mid

    print ("*" * 10)
    print ("Number of dimensions that are communicated: ", communication_weight)
    print ("Total number of communicable dimensions: ", total_weight)
    print (
        "Percentage of communicable dimensions that *are* "
        "communicated: %f" % (float(communication_weight) / total_weight))

    send_partners = [set() for i in range(n_components)]
    recv_partners = [set() for i in range(n_components)]
    for conn in network.all_connections:
        if can_cross_boundary(conn):
            pre_obj = neurons2ensemble(conn.pre_obj)
            post_obj = neurons2ensemble(conn.post_obj)

            pre_component = assignments[pre_obj]
            post_component = assignments[post_obj]

            if pre_component != post_component:
                send_partners[pre_component].add(post_component)
                recv_partners[post_component].add(pre_component)

    n_send_partners = [len(s) for s in send_partners]
    n_recv_partners = [len(s) for s in recv_partners]

    print ("*" * 10)
    print ("Mean number of send partners: ", np.mean(n_send_partners))
    print ("Standard dev of send partners: ", np.std(n_send_partners))
    print ("Max number of send partners: ", np.max(n_send_partners))
    print ("Min number of send partners: ", np.min(n_send_partners))

    print ("*" * 10)
    print ("Mean number of recv partners: ", np.mean(n_recv_partners))
    print ("Standard dev of recv partners: ", np.std(n_recv_partners))
    print ("Max number of recv partners: ", np.max(n_recv_partners))
    print ("Min number of recv partners: ", np.min(n_recv_partners))


def remove_from_network(network, obj):
    """ Remove ``obj`` from network.

    Returns True if ``obj`` was successfully found and removed.

    """
    key = None
    for t in network.objects.keys():
        if isinstance(obj, t):
            key = t
            break

    if key:
        if obj in network.objects[key]:
            network.objects[key].remove(obj)
            return True

    for sub_net in network.networks:
        removed = remove_from_network(sub_net, obj)

        if removed:
            return True

    return False
