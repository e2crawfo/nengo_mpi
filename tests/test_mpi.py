import os
import pytest

import numpy as np

import nengo
from nengo.neurons import LIF, LIFRate, RectifiedLinear, Sigmoid
from nengo.neurons import AdaptiveLIF, AdaptiveLIFRate  # , Izhikevich

import nengo_mpi
from nengo_mpi import partition
from nengo_mpi.tests.utils import run_standalone_mpi, run_python_mpi

nengo_mpi_dir = os.path.dirname(nengo_mpi.__file__)
mpi_test_script_dir = os.path.join(nengo_mpi_dir, "tests/mpi_tests")
mpi_test_scripts = [
    os.path.join(mpi_test_script_dir, s)
    for s in os.listdir(mpi_test_script_dir)
    if s.endswith(".py")]


@pytest.mark.parametrize("script_name", mpi_test_scripts)
@pytest.mark.parametrize("n_processors", [1, 4])
def test_mpi_script(n_processors, script_name):
    """ A test that consists of running a script in the nengo_mpi context.

    For each *.py file in nengo_mpi/tests/mpi_tests/, create a test
    which simply runs the script using the function run_python_mpi.
    Consider the test passed if the script returns an exit code of 0.

    """
    output, exit_code = run_python_mpi(n_processors, script_name, [])
    print (output)
    assert not exit_code, "Script exited with non-zero exit status."


all_neurons = [
    LIF, LIFRate, RectifiedLinear, Sigmoid,
    AdaptiveLIF, AdaptiveLIFRate]  # Izhikevich]


@pytest.mark.parametrize("neuron_type", all_neurons)
@pytest.mark.parametrize("synapse", [None, 0.0, 0.02, 0.05])
def test_basic_mpi(neuron_type, synapse):
    n_neurons = 40

    m = nengo.Network(seed=1)
    with m:
        A = nengo.Ensemble(
            n_neurons, dimensions=3, neuron_type=neuron_type())

        B = nengo.Ensemble(
            n_neurons, dimensions=3, neuron_type=neuron_type())

        nengo.Connection(A, B, synapse=synapse)

        A_p = nengo.Probe(A)
        B_p = nengo.Probe(B)

        input = nengo.Node([0.1, 0.2, -0.3])
        nengo.Connection(input, A, synapse=0.05)

    sim_time = 1

    refimpl_sim = nengo.Simulator(m)
    refimpl_sim.run(sim_time)

    network_file = "test_nengo_mpi.net"
    log_file = "test_nengo_mpi.h5"
    n_processors = 2

    try:
        nengo_mpi.Simulator(
            m, partitioner=nengo_mpi.Partitioner(n_processors),
            save_file=network_file)

        results = run_standalone_mpi(
            network_file, log_file, n_processors, sim_time)

        assert np.allclose(
            refimpl_sim.data[A_p], results[str(id(A_p))],
            atol=0.00001, rtol=0.00)
        assert np.allclose(
            refimpl_sim.data[B_p], results[str(id(B_p))],
            atol=0.00001, rtol=0.00)
    finally:
        try:
            os.remove(network_file)
        except:
            pass


def random_graph(
        neuron_type, n_nodes, pct_connections,
        pct_probed, pct_self_loops, npd, D):

    seed = 10

    assert n_nodes > 0
    assert pct_connections >= 0 and pct_connections <= 1
    assert pct_self_loops >= 0 and pct_self_loops <= 1
    assert pct_probed >= 0 and pct_probed <= 1

    ensembles = []

    rng = np.random.RandomState(seed)

    name = "RandomGraph"
    m = nengo.Network(label=name, seed=seed)
    with m:
        m.config[nengo.Ensemble].neuron_type = nengo.LIF()
        input_node = nengo.Node(output=[0.25] * D)
        nengo.Probe(input_node, synapse=0.01)

        probes = []

        for i in range(n_nodes):
            ensemble = nengo.Ensemble(
                npd * D, dimensions=D, label="ensemble %d" % i,
                neuron_type=neuron_type())

            if rng.rand() < pct_self_loops:
                nengo.Connection(ensemble, ensemble)

            if rng.rand() < pct_probed:
                probe = nengo.Probe(ensemble)
                probes.append(probe)

            ensembles.append(ensemble)

            for j in range(i):
                if rng.rand() < pct_connections:
                    nengo.Connection(ensemble, ensembles[j])

                if rng.rand() < pct_connections:
                    nengo.Connection(ensembles[j], ensemble)

        nengo.Connection(input_node, ensembles[0])

    return m


@pytest.fixture(params=all_neurons)
def refimpl_results(request):
    """ Returns results of running random graph model.

    Returns the model itself, the nengo.Simulator object used to execute
    the simulation, and the simulation time.
    """

    neuron_type = request.param

    n_nodes = 10
    pct_connections = 0.2
    pct_probed = 0.2
    pct_self_loops = 0.1
    npd = 40
    D = 3

    m = random_graph(
        neuron_type, n_nodes, pct_connections,
        pct_probed, pct_self_loops, npd, D)

    sim_time = 1.0

    refimpl_sim = nengo.Simulator(m)
    refimpl_sim.run(sim_time)

    return (m, refimpl_sim, sim_time)


@pytest.mark.parametrize("partitioner", partition.partitioners())
def test_random_graph(partitioner, refimpl_results):
    m, refimpl_sim, sim_time = refimpl_results

    n_processors = 8
    partitioner = nengo_mpi.Partitioner(n_processors, func=partitioner)

    network_file = "test_nengo_mpi.net"
    log_file = "test_nengo_mpi.h5"
    log_file_1p = "test_nengo_mpi_1p.h5"

    try:
        nengo_mpi.Simulator(
            m, partitioner=partitioner, save_file=network_file)

        results = run_standalone_mpi(
            network_file, log_file, n_processors, sim_time)
        results_1p = run_standalone_mpi(
            network_file, log_file_1p, 1, sim_time)

        for p in m.probes:
            assert np.allclose(
                refimpl_sim.data[p], results[str(id(p))],
                atol=0.00001, rtol=0.00)

        for p in m.probes:
            assert np.allclose(
                refimpl_sim.data[p], results_1p[str(id(p))],
                atol=0.00001, rtol=0.00)
    finally:
        try:
            os.remove(network_file)
        except:
            pass
