from nengo.simulator import ProbeDict
from nengo.cache import get_default_decoder_cache
import nengo.utils.numpy as npext

from nengo_mpi.model import MpiBuilder, MpiModel
from nengo_mpi.partition import Partitioner, verify_assignments

import numpy as np
import atexit
from functools import partial
import logging
logger = logging.getLogger(__name__)


class Simulator(object):
    """MPI simulator for nengo 2.0."""

    # Only one instance of nengo_mpi.Simulator can be unclosed at any time
    __unclosed_simulators = []

    def __init__(
            self, network, dt=0.001, seed=None, model=None,
            partitioner=None, assignments=None, save_file=""):
        """
        Creates a Simulator for a nengo network than can be executed
        in parallel using MPI.

        Parameters
        ----------
        network : nengo.Network
            A network object to be built and then simulated.

        dt : float
            The length of a simulator timestep, in seconds.

        seed : int
            A seed for all stochastic operators used in this simulator.
            Note that there are not stochastic operators implemented
            currently, so this parameters does nothing.

        model : nengo.builder.Model
            A model object that contains build artifacts to be simulated.
            Usually the simulator will build this model for you; however,
            if you want to build the network manually, or to inject some
            build artifacts in the Model before building the network,
            then you can pass in an instance of ``MpiModel'' instance
            or a ``nengo.builder.Model`` instance. If the latter, it
            will be converted into an ``MpiModel''.

        partitioner: Partitioner
            Specifies how to assign nengo objects to MPI processes.
            ``partitioner'' and ``assignment'' cannot both be supplied.

        assignments: dict
            Dictionary mapping from nengo objects to indices of
            partitions components. ``partitioner'' and ``assignment''
            cannot both be supplied.

        save_file: string
            Name of file that will store all data added to the simulator.
            The simulator can later be reconstructed from this file. If
            equal to the empty string, then no file is created.
        """

        self.runnable = not save_file

        if self.runnable and self.__unclosed_simulators:
            raise RuntimeError(
                "Attempting to create active instance of nengo_mpi.Simulator "
                "while another instance exists that has not been "
                "closed. Call `close` on existing instances before "
                "creating new ones.")

        self.n_steps = 0
        self.dt = dt

        if partitioner is not None and assignments is not None:
            raise ValueError(
                "Cannot supply both ``assignments'' and ``partitioner'' to "
                "Simulator.__init__.")

        if assignments is not None:
            p = verify_assignments(network, assignments)
        else:
            if partitioner is None:
                partitioner = Partitioner()

            print ("Partitioning network...")
            p = partitioner.partition(network)

        self.n_components, self.assignments = p

        print ("Building MPI model...")
        self.model = MpiModel(
            self.n_components, self.assignments, dt=dt,
            label="%s, dt=%f" % (network, dt),
            decoder_cache=get_default_decoder_cache(),
            save_file=save_file)

        MpiBuilder.build(self.model, network)

        print ("Finalizing MPI model...")
        self.model.finalize_build()

        # probe -> list
        self._probe_outputs = self.model.params

        self.data = ProbeDict(self._probe_outputs)

        print ("MPI model ready.")

        if self.runnable:
            seed = np.random.randint(npext.maxint) if seed is None else seed
            self.reset(seed=seed)

            self.__unclosed_simulators.append(self)

    @property
    def mpi_sim(self):
        if not self.model.runnable:
            raise Exception(
                "Cannot access C++ simulator of MpiModel, MpiModel is "
                "not in a runnable state. Either in save-file mode, "
                "or the MpiModel instance has not been finalized.")

        return self.model.mpi_sim

    def __str__(self):
        return self.mpi_sim.to_string()

    def run_steps(self, steps, progress_bar, log_filename):
        """ Simulate for the given number of `dt` steps. """

        self.mpi_sim.run_n_steps(steps, progress_bar, log_filename)

        if not log_filename:
            for probe, probe_key in self.model.probe_keys.items():
                data = self.mpi_sim.get_probe_data(probe_key, np.empty)

                # The C++ code doesn't always exactly preserve the shape
                true_shape = self.model.sig[probe]['in'].shape
                if data[0].shape != true_shape:
                    data = map(
                        partial(np.reshape, newshape=true_shape), data)

                if probe not in self._probe_outputs:
                    self._probe_outputs[probe] = data
                else:
                    self._probe_outputs[probe].extend(data)

        self.n_steps += steps

        print ("MPI Simulation complete.")

    def step(self):
        """ Advance the simulator by `self.dt` seconds. """
        self.run_steps(1)

    def run(self, time_in_seconds, progress_bar=True, log_filename=""):
        """ Simulate for the given length of time. """

        steps = int(np.round(float(time_in_seconds) / self.dt))
        self.run_steps(steps, progress_bar, log_filename)

    def trange(self, dt=None):
        dt = self.dt if dt is None else dt
        n_steps = int(self.n_steps * (self.dt / dt))
        return dt * np.arange(1, n_steps + 1)

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed

        self.n_steps = 0

        if self.runnable:
            self.mpi_sim.reset(self.seed)
        else:
            raise RuntimeError(
                "Attempting to reset a non-runnable instance of "
                "nengo_mpi.Simulator.")

        for pk in self.model.probe_keys:
            self._probe_outputs[pk] = []

    def close(self):
        if self.runnable:
            try:
                self.mpi_sim.close()
                self.__unclosed_simulators.remove(self)
            except ValueError:
                raise RuntimeError(
                    "Attempting to close a runnable instance of "
                    "nengo_mpi.Simulator that has already been closed.")

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    @staticmethod
    @atexit.register
    def close_simulators():
        unclosed_simulators = Simulator.__unclosed_simulators
        if len(unclosed_simulators) > 1:
            raise RuntimeError(
                "Multiple instances of nengo_mpi.Simulator "
                "open simulatenously.")
        if unclosed_simulators:
            unclosed_simulators[0].close()

    @staticmethod
    def all_closed():
        return Simulator.__unclosed_simulators == []
