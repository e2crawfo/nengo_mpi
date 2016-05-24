import argparse
import time
from collections import defaultdict
import os

import numpy as np
import pandas as pd
import networkx as nx

import nengo
from nengo.solvers import Solver
import nengo_mpi
from nengo_mpi.partition import metis_partitioner, work_balanced_partitioner
from nengo_mpi.partition import random_partitioner, EnsembleArraySplitter

from utils import write_to_csv


class FakeSolver(Solver):
    def __init__(self, weights=False, rcond=0.01):
        self.weights = weights

    def __call__(self, A, Y, rng=None, E=None):
        X = np.zeros((A.shape[1], Y.shape[1]))
        info = {}
        return self.mul_encoders(X, E), info


def default_generator(n_nodes, q, rng=None):
    assert n_nodes > 0
    assert q >= 0 and q <= 1

    n_cur_edges = 0
    n_possible_edges = n_nodes ** 2
    n_edges = q * n_possible_edges
    n_cur_edges = 0
    adj_list = defaultdict(list)
    edges = []

    rng = np.random.RandomState(seed)

    while n_cur_edges < n_edges:
        A = rng.choice(n_nodes)
        B = rng.choice(n_nodes)

        if B not in adj_list[A]:
            adj_list[A].append(B)
            edges.append((A, B, {}))
            n_cur_edges += 1

            if n_cur_edges % 100 == 0:
                print ( "Done %d edges." % n_cur_edges )

    return edges


def nengo_network_from_graph(
        label, n_nodes, edges, use_ea=True, fake=False,
        dim=1, npd=30, pct_probed=0.0, seed=None):

    assert pct_probed >= 0 and pct_probed <= 1

    model = nengo.Network(label=label, seed=seed)
    rng = np.random.RandomState(seed)

    with model:
        model.config[nengo.Ensemble].neuron_type = nengo.LIF()
        if fake:
            model.config[nengo.Connection].solver = FakeSolver()

        input_node = nengo.Node(output=[0.25] * dim)
        input_p = nengo.Probe(input_node, synapse=0.01)

        probes = [input_p]
        ensembles = []

        for i in range(n_nodes):
            if use_ea:
                ensemble = nengo.networks.EnsembleArray(
                    npd, dim, label="ensemble array %d" % i)
                outp = ensemble.output
                ensembles.append(ensemble)
            else:
                ensemble = nengo.Ensemble(
                    npd * dim, dimensions=dim, label="ensemble %d" % i)
                outp = ensemble

                ensembles.append(ensemble)

            if rng.rand() < pct_probed:
                probe = nengo.Probe(outp)
                probes.append(probe)

            if i % 100 == 0:
                print ( "Done %d ensembles." % i )

        if use_ea:
            nengo.Connection(input_node, ensembles[0].input)
        else:
            nengo.Connection(input_node, ensembles[0])

        for e in edges:
            A = ensembles[e[0]]
            B = ensembles[e[1]]

            if use_ea:
                nengo.Connection(A.output, B.input)
            else:
                nengo.Connection(A, B)

    return model, probes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmarking script for nengo_mpi. "
                    "Network takes the form of a randomly generated graph, "
                    "where each node is a nengo ensemble.")

    parser.add_argument(
        '-n', type=int, default=10,
        help='Number of ensembles in the network.')

    parser.add_argument(
        '-q', type=float, default=0.1,
        help='Probability of each directed edge in the graph existing.')

    parser.add_argument(
        '-d', type=int, default=1,
        help='Number of dimensions in each neural ensemble.')

    parser.add_argument(
        '--npd', type=int, default=50,
        help='Number of neurons per dimension in each neural ensemble.')

    parser.add_argument(
        '--mpi', type=int, default=1, help='Whether to use MPI.')

    parser.add_argument(
        '-p', type=int, default=1,
        help='If using MPI, the number of processors to use.')

    parser.add_argument(
        '--probes', type=float, default=0.1,
        help='Percentage of ensembles that are probed.')

    parser.add_argument(
        '-t', type=float, default=1.0,
        help='Length of the simulation in seconds.')

    parser.add_argument(
        '--noprog', action='store_true',
        help='Supply to omit the progress bar.')

    parser.add_argument(
        '--fake', action='store_true',
        help='Supply to solve for fake decoders. Connections '
             'will not compute the correct function, but creating the network '
             'becomes much faster.')

    parser.add_argument(
        '--gen', type=str, default='default',
        help='Specify the generative model to use to construct the graph.')

    parser.add_argument(
        '--pfunc', type=str, default='',
        help='Specify the algorithm to use for partitioning. '
             'Possible values are: default, metis, random, work.')

    parser.add_argument(
        '--save', nargs='?', type=str, default='', const='random_graph',
        help="Supply a filename to write the network to (so it can be "
             "later be used by the stand-alone version of nengo_mpi). "
             "In this case, the network will not be simulated.")

    parser.add_argument(
        '--mpi-log', nargs='?', type=str,
        default='', const='random_graph', dest='mpi_log',
        help="Supply a filename to write the results of the simulation "
             "to, if an MPI simulation is performed.")

    parser.add_argument(
        '--ea', action='store_true',
        help="Supply to use ensemble arrays instead of ensembles. Each "
             "ensemble within each ensemble array will have dimension 1, "
             "and npd neurons.")

    parser.add_argument(
        '--split-ea', type=int, default=1,
        help="Number of parts to split each ensemble array up into. "
             "Obviously, only has an effect if --ea is also supplied.")

    parser.add_argument(
        '--seed', type=int, default=None,
        help="Seed for random number generation.")

    args = parser.parse_args()
    print ( "Parameters are: ", args )

    name = 'MpiRandomGraphBenchmark'
    npd = args.npd
    dim = args.d
    seed = args.seed
    rng = np.random.RandomState(args.seed)

    n_nodes = args.n
    q = args.q
    n_processors = args.p
    pct_probed = args.probes
    use_ea = args.ea
    split_ea = args.split_ea

    gen_model = args.gen

    use_mpi = args.mpi

    bench_home = os.getenv("NENGO_MPI_BENCH_HOME")
    build_times = os.path.join(bench_home, 'random/buildtimes.db')
    run_times = os.path.join(bench_home, 'random/runtimes.db')

    save_file = args.save
    if save_file == 'random_graph':
        save_file = (
            'random_graph_p{0}_n{1}_q{2}_npd{3}'.format(
                args.p, n_nodes, q, npd))
        save_file = save_file.replace('.', '_')
        save_file += '.net'

    mpi_log = args.mpi_log
    if mpi_log == 'random_graph':
        mpi_log = (
            'random_graph_p{0}_n{1}_q{2}_npd{3}'.format(
                args.p, n_nodes, q, npd))
        mpi_log = mpi_log.replace('.', '_')
        mpi_log += '.h5'

    if mpi_log:
        print ( "Logging simulation results to", mpi_log )

    sim_time = args.t

    progress_bar = not args.noprog

    partitioner = args.pfunc

    fake = args.fake

    assert n_processors >= 1
    assert sim_time > 0

    if gen_model == "default":
        edges = default_generator(n_nodes, q, rng.randint(2000))
    elif gen_model == "ba":
        from networkx.generators import barabasi_albert_graph
        m = int(q * n_nodes)
        g1 = barabasi_albert_graph(n_nodes, m, rng.randint(2000))
        edges1 = g1.edges()

        g2 = barabasi_albert_graph(n_nodes, m)
        edges2 = g2.edges()

        edges2 = [(B, A) for A, B in edges2]
        edges = list(set(edges1 + edges2))

        dg = nx.DiGraph(edges)
        out_degree = pd.Series([o[1] for o in dg.out_degree_iter()])
        print ("Out-degree: ")
        print (out_degree.describe())

        in_degree = pd.Series([i[1] for i in dg.in_degree_iter()])
        print ("In-degree: ")
        print (in_degree.describe())
    else:
        raise NotImplemented()

    model, probes = nengo_network_from_graph(
        name, n_nodes, edges, use_ea, fake, dim,
        npd, pct_probed, rng.randint(2000))

    if use_ea and split_ea > 1:
        splitter = EnsembleArraySplitter()
        max_neurons = np.ceil(float(dim) / split_ea) * npd
        splitter.split(model, max_neurons)

    t0 = time.time()
    if use_mpi:
        fmap = {
            'default': None, '': None,
            'metis': metis_partitioner, 'random': random_partitioner,
            'work': work_balanced_partitioner}

        partitioner = nengo_mpi.Partitioner(
            n_processors, func=fmap[partitioner])
        sim = nengo_mpi.Simulator(
            model, dt=0.001, partitioner=partitioner, save_file=save_file)

        if save_file:
            print ("Saved network to", save_file)
    else:
        sim = nengo.Simulator(model, dt=0.001)

    t1 = time.time()

    n_neurons = npd * dim * n_nodes

    vals = vars(args).copy()
    vals['buildtime'] = t1 - t0
    vals['n_neurons'] = n_neurons
    write_to_csv(build_times, vals)

    print ("BUILD TIME: %f" % (t1 - t0))

    if not save_file:
        t0 = time.time()

        if use_mpi:
            sim.run(sim_time, progress_bar, mpi_log)
        else:
            sim.run(sim_time, progress_bar)

        t1 = time.time()

        if not mpi_log:
            for i, p in enumerate(probes):
                print ("Result from %s: " % p)
                print (sim.data[p][-10:])

        print ("Total simulation time: %g seconds" % (t1 - t0))
        print ("Parameters were: ", args)
        print ("Number of neurons in network: ", n_neurons)
        print ("Number of nodes in network: ", n_nodes)

        vals = vars(args).copy()
        vals['runtime'] = t1 - t0
        vals['n_neurons'] = n_neurons
        write_to_csv(run_times, vals)
