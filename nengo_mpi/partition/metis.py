import subprocess
import tempfile
import os

_metis_available = True

try:
    devnull = open(os.devnull, 'w')
    subprocess.call(["gpmetis"], stdout=devnull, stderr=subprocess.STDOUT)
except OSError as e:
    if e.errno == os.errno.ENOENT:
        _metis_available = False
    else:
        raise


def metis_available():
    return _metis_available


def write_metis_input_file(cluster_graph):
    """
    Writes a cluster graph (created via network_to_cluster_graph) to file in
    the format required by metis (a graph-partitioning utility).

    Parameters
    ----------
    cluster_graph: networkx Graph
        A graph created from a network using network_to_cluster_graph.

    Returns
    -------
    filename: string
        The name of the file storing the graph.
    """
    f = tempfile.NamedTemporaryFile(mode='w', prefix='metis', delete=False)

    print ("Writing metis file: %s" % f.name)

    with f:
        m = cluster_graph.number_of_edges()
        n = cluster_graph.number_of_nodes()

        f.write("%d %d 011" % (n, m))

        indices = {node: i+1 for i, node in enumerate(cluster_graph.nodes())}

        for u in cluster_graph.nodes():
            f.write('\n')

            vertex_weight = 0

            for obj in u.objects:
                if hasattr(obj, 'n_neurons'):
                    vertex_weight += obj.n_neurons

            f.write("%d" % vertex_weight)

            for v, weight_dict in cluster_graph[u].iteritems():
                f.write(" %d %d" % (indices[v], weight_dict['weight']))

    return f.name


def read_metis_output_file(filename):
    """
    Read the given file, assuming it is the output from a run of gpmetis.
    The format of the file is: n lines (n being number of nodes), the i-th line
    contains a single int giving the component that the i-th node is assigned
    to. The first component has index 0.

    Returns
    -------
    A list L where the i-th element is the component index of the i-th node.

    """
    print ("Reading metis output file: %s" % filename)

    node_assignments = []

    with open(filename, 'r') as f:
        node_assignments = [
            int(line) for line in iter(f.readline, '')]

    return node_assignments


def metis_output_filename(filename, n_components):
    return '%s.part.%d' % (filename, n_components)


def metis_partitioner(cluster_graph, n_components, delete_file=True):
    """
    Partitions a cluster graph using the metis partitioning package.

    Parameters
    ----------
    cluster_graph: networkx Graph
        A graph created from a network using network_to_cluster_graph.

    n_components: int
        Desired number of components in the partition.

    Returns
    -------
    assignments: dict
        A mapping from nodes in the cluster graph to components.

    """
    if not metis_available():
        raise Exception(
            "Cannot use metis_partitioner. "
            "gpmetis is not present on the system.")

    assert n_components > 1

    filename = write_metis_input_file(cluster_graph)

    print ("Running metis...")
    subprocess.check_call(['gpmetis', filename, str(n_components)])

    output_filename = metis_output_filename(filename, n_components)
    node_assignments = read_metis_output_file(output_filename)

    if delete_file:
        os.remove(filename)

    return {
        node: component
        for node, component
        in zip(cluster_graph.nodes(), node_assignments)}
