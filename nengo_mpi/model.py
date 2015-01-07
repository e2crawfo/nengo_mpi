"""MPIModel"""

from nengo import builder
from nengo.neurons import LIF, LIFRate, RectifiedLinear, Sigmoid
from nengo.synapses import LinearFilter, Lowpass, Alpha
from nengo.utils.filter_design import cont2discrete
from nengo.utils.graphs import toposort
from nengo.utils.simulator import operator_depencency_graph
from nengo.cache import NoDecoderCache

from nengo.connection import Connection
from nengo.ensemble import Ensemble
from nengo.node import Node
from nengo.probe import Probe

import mpi_sim

import numpy as np
from collections import defaultdict
import warnings

import logging
logger = logging.getLogger(__name__)


def make_builder(base):
    """
    Create a version of an existing builder function whose only difference
    is that it assumes the model is an instance of MpiModel, and uses that
    model to record which ops are built as part of building which high-level
    objects.

    Parameters
    ----------
    base: The existing builder function that we want to augment.

    """

    def build_object(model, obj):
        try:
            model.push_object(obj)
        except AttributeError:
            raise ValueError(
                "Must use an instance of MpiModel.")

        base(model, obj)
        model.pop_object()

    build_object.__doc__ = (
        "Builder function augmented to make use "
        "of MpiModels.\n\n" + str(base.__doc__))

    return build_object


# Overwrite some of the default builder functions with functions that keep
# track of the operators that are created as part of building the objects, and
# store that information in a dictionary in the MpiModel. This information
# is used to assign operators to the correct components.
with warnings.catch_warnings():

    # Ignore the warning generated by overwriting the builder functions.
    warnings.simplefilter('ignore')

    builder.Builder.register(Ensemble)(
        make_builder(builder.build_ensemble))

    builder.Builder.register(Node)(
        make_builder(builder.build_node))

    builder.Builder.register(Connection)(
        make_builder(builder.build_connection))

    builder.Builder.register(Probe)(
        make_builder(builder.build_probe))


class DummyNdarray():
    def __init__(self, value):
        self.dtype = value.dtype
        self.shape = value.shape
        self.size = value.size
        self.strides = value.strides


def adjust_linear_filter(op, synapse, num, den, dt, method='zoh'):
    num, den, _ = cont2discrete(
        (num, den), dt, method=method)
    num = num.flatten()
    num = num[1:] if num[0] == 0 else num
    den = den[1:]  # drop first element (equal to 1)

    return num, den


def pyfunc_checks(val):
    """
    If the output can possibly be treated as a scalar, convert it
    to a python float. Otherwise, convert it to a numpy ndarray.
    """

    if isinstance(val, list):
        val = np.array(val, dtype=np.float64)

    elif isinstance(val, int):
        val = float(val)

    elif isinstance(val, float):
        if isinstance(val, np.float64):
            val = float(val)

    elif not isinstance(val, np.ndarray):
        raise ValueError(
            "python function returning unexpected value, %s" % str(val))

    if isinstance(val, np.ndarray):
        val = np.squeeze(val)

        if val.size == 1:
            val = float(val)
        elif getattr(val, 'dtype', None) != np.float64:
            val = np.asarray(val, dtype=np.float64)

    return val


def make_checked_func(func, t_in, takes_input):
    def f():
        return pyfunc_checks(func())

    def ft(t):
        return pyfunc_checks(func(t))

    def fit(t, i):
        return pyfunc_checks(func(t, i))

    if t_in and takes_input:
        return fit
    elif t_in or takes_input:
        return ft
    else:
        return f


class MpiSend(builder.operator.Operator):
    """
    MpiSend placeholder operator. Stores the signal that the operator will
    send and the partition that it will be sent to. No makestep is defined,
    as it will never be called.
    """
    def __init__(self, dst, signal):
        self.sets = []
        self.incs = []
        self.reads = []
        self.updates = []
        self.dst = dst
        self.signal = signal


class MpiRecv(builder.operator.Operator):
    """
    MpiRecv placeholder operator. Stores the signal that the operator will
    receive and the partition that it will be received from. No makestep is
    defined, as it will never be called.
    """
    def __init__(self, src, signal):
        self.sets = []
        self.incs = []
        self.reads = []
        self.updates = []
        self.src = src
        self.signal = signal


class MpiWait(builder.operator.Operator):
    def __init__(self, signal):
        """Sets the signal so that the signal has a set. Otherwise an assertion
        is violated in the order finding algorithm. Also puts the MpiWait in
        the right place, i.e. before any operators that read from the signal.
        """

        self.sets = [signal]
        self.incs = []
        self.reads = []
        self.updates = []
        self.signal = signal

    def make_step():
        """Dummy function, so this op gets included in the ordering"""
        pass


def split_connection(conn_ops, signal):
    """
    Split the operators belonging to a connection into a
    ``pre'' group and a ``post'' group. The connection is assumed
    to contain exactly 1 operation performing an update, which
    is assigned to the pre group. All ops that write to signals
    which are read by this updating op are assumed to belong to
    the pre group (as are all ops that write to signals which
    *those* ops read from, etc.). The remaining ops are assigned
    to the post group.

    Parameters
    ----------
    conn_ops: A list containing the operators implementing a nengo connection.

    signal: The signal where the connection will be split. Must be updated by
        one of the operators in ``conn_ops''.

    Returns
    -------
    pre_ops: A list of the ops that come before the updated signal.
    post_ops: A list of the ops that come after the updated signal.

    """

    pre_ops = []

    for op in conn_ops:
        if signal in op.updates:
            pre_ops.append(op)

    assert len(pre_ops) == 1

    reads = pre_ops[0].reads

    post_ops = filter(
        lambda op: op not in pre_ops, conn_ops)

    changed = True
    while changed:
        changed = []

        for op in post_ops:
            writes = set(op.incs) | set(op.sets)

            if writes & set(reads):
                pre_ops.append(op)
                reads.extend(op.reads)
                changed.append(op)

        post_ops = filter(
            lambda op: op not in changed, post_ops)

    return pre_ops, post_ops


def make_key(obj):
    """
    Create a key for an object. Must be unique, and reproducable (i.e. produce
    the same key if called with the same object multiple times).
    """
    if isinstance(obj, builder.signal.SignalView):
        return id(obj.base)
    else:
        return id(obj)


class MpiModel(builder.Model):
    """
    Output of the Builder, used by the Simulator.

    Differs from the Model in the reference implementation in that
    as the model is built, it keeps track of the object currently being
    built. This permits it to track which operators are added as part
    of which high-level objects, so that those operators can later be
    added to the correct MPI component (required since MPI components are
    specified in terms of the high-level nengo objects like nodes,
    networks and ensembles).
    """

    # implemented_ops = [Reset, Copy, ]

    def __init__(
            self, num_components, assignments, dt=0.001, label=None,
            decoder_cache=NoDecoderCache()):

        self.num_components = num_components
        self.assignments = assignments

        self.mpi_sim = mpi_sim.PythonMpiSimulator(num_components, dt)

        # TODO: not sure if we still need this
        # C++ key (int) -> ndarray
        self.sig_dict = {}

        # for each component, stores the keys of the signals that have
        # to be sent and received, respectively
        self.send_signals = defaultdict(list)
        self.recv_signals = defaultdict(list)

        # for each component, stores the keys of the signals that have
        # already been added to that component.
        self.added_signals = defaultdict(list)

        # operators for each component
        self.component_ops = defaultdict(list)

        # probe -> C++ key (int)
        # Used to query the C++ simulator for probe data
        self.probe_keys = {}

        self._object_context = [None]
        self.object_ops = defaultdict(list)

        super(MpiModel, self).__init__(dt, label, decoder_cache)

    def __str__(self):
        return "MpiModel: %s" % self.label

    def push_object(self, object):
        self._object_context.append(object)

    def pop_object(self):

        obj = self._object_context.pop()

        if not isinstance(obj, Connection):
            component = self.assignments[obj]

            self.add_ops(component, self.object_ops[obj])

        else:
            conn = obj
            pre_component = self.assignments[conn.pre_obj]
            post_component = self.assignments[conn.post_obj]

            if pre_component == post_component:
                self.add_ops(pre_component, self.object_ops[conn])

            else:
                # conn crosses component boundaries
                if conn.modulatory:
                    raise Exception(
                        "Connections crossing component boundaries "
                        "must not be modulatory.")

                if conn.learning_rule_type:
                    raise Exception(
                        "Connections crossing component boundaries "
                        "must not have learning rules.")

                if 'synapse_out' in self.sig[conn]:
                    signal = self.sig[conn]['synapse_out']
                else:
                    raise Exception(
                        "Connections crossing component boundaries "
                        "must be filtered so that there is an update.")

                self.send_signals[pre_component].append(
                    (signal, post_component))
                self.recv_signals[post_component].append(
                    (signal, pre_component))

                pre_ops, post_ops = split_connection(
                    self.object_ops[conn], signal)

                # Have to add the signal to both components, so can't delete it
                # the first time.
                self.add_signal(pre_component, signal, delete=False)
                self.add_signal(post_component, signal, delete=True)

                self.add_ops(pre_component, pre_ops)
                self.add_ops(post_component, post_ops)

    def add_ops(self, component, ops):
        for op in ops:
            for signal in op.all_signals:
                self.add_signal(component, signal)

        self.component_ops[component].extend(ops)

    def add_signal(self, component, signal, delete=True):
        key = make_key(signal)

        if key not in self.added_signals[component]:
            # TODO: Should copy the logic used to add signals
            # to the SignalDict in the refimpl builder
            logger.debug(
                "Adding signal %s with key: %s", signal, make_key(signal))

            self.added_signals[component].append(key)

            label = str(signal)

            A = signal.base._value

            if A.ndim == 0:
                A = np.reshape(A, (1, 1))

            self.mpi_sim.add_signal(component, key, label, A)

            if delete:
                # Replace the data stored in the signal by a dummy array,
                # which has # no contents but has the same shape, size, etc
                # as the original. This should allow the memory to be
                # reclaimed.
                signal.base._value = DummyNdarray(signal.base._value)

    def add_op(self, op):
        """
        Records that the operator was added as part of building
        the object that is on the top of _object_context stack,
        and then uses refimpl add_op to finish adding the op.
        """
        self.object_ops[self._object_context[-1]].append(op)

    def finalize(self):
        logger.debug("ADDING OPS TO MPI:")
        for component in range(self.num_components):
            self.add_ops_to_mpi(component)

        logger.debug("ADDING PROBES TO MPI:")
        logger.debug(self.probes)

        for probe in self.probes:
            self.add_probe(
                probe, make_key(self.sig[probe]['in']),
                sample_every=probe.sample_every)

        self.mpi_sim.finalize()

    def from_refimpl_model(self, model):
        """Create an MpiModel from an instance of a refimpl Model."""

        if not isinstance(model, builder.Model):
            raise TypeError(
                "Model must be an instance of "
                "%s." % builder.model.__name__)

        self.dt = model.dt
        self.label = model.label
        self.decoder_cache = model.decoder_cache

        self.toplevel = model.toplevel
        self.config = model.config

        self.operators = model.operators
        self.params = model.params
        self.seeds = model.seeds
        self.probes = model.probes
        self.sig = model.sig

    def add_ops_to_mpi(self, component):
        logger.debug("IN ADD OPS TO MPI!")
        # logger.debug("MODEL: %s", self)
        # logger.debug("SEND SIGNALS: %s", self.send_signals)
        # logger.debug("RECV SIGNALS: %s", self.recv_signals)

        send_signals = self.send_signals[component]
        recv_signals = self.recv_signals[component]

        for signal, dst in send_signals:
            mpi_wait = MpiWait(signal)
            self.component_ops[component].append(mpi_wait)

        for signal, src in recv_signals:
            mpi_wait = MpiWait(signal)
            self.component_ops[component].append(mpi_wait)

        dg = operator_depencency_graph(self.component_ops[component])
        step_order = [node for node in toposort(dg)
                      if hasattr(node, 'make_step')]

        for signal, dst in send_signals:
            # find the op that updates the signal
            updates_signal = map(
                lambda x: signal in x.updates, step_order)

            update_index = updates_signal.index(True)

            mpi_send = MpiSend(dst, signal)

            step_order.insert(update_index+1, mpi_send)

        for signal, src in recv_signals:
            # find the first op that reads from the signal
            reads = map(
                lambda x: signal in x.reads, step_order)

            read_index_last = len(reads) - reads[::-1].index(True) - 1

            mpi_recv = MpiRecv(src, signal)

            # Put the recv after the last read,
            # and the wait before the first read
            step_order.insert(read_index_last+1, mpi_recv)

        for op in step_order:
            op_type = type(op)

            if op_type == builder.node.SimPyFunc:
                t_in = op.t_in
                fn = op.fn
                x = op.x

                output_id = (make_key(op.output)
                             if op.output is not None
                             else -1)

                if x is None:
                    logger.debug(
                        "Creating PyFunc, output:%d", make_key(op.output))

                    if op.output is None:
                        self.mpi_sim.create_PyFunc(fn, t_in)
                    else:
                        self.mpi_sim.create_PyFuncO(
                            output_id, make_checked_func(fn, t_in, False),
                            t_in)

                else:
                    logger.debug(
                        "Creating PyFunc with input, output:%d",
                        make_key(op.output))

                    if isinstance(x.value, DummyNdarray):
                        input_array = np.zeros(x.shape)
                    else:
                        input_array = x.value

                    if op.output is None:

                        self.mpi_sim.create_PyFuncI(
                            fn, t_in, make_key(x), input_array)

                    else:
                        self.mpi_sim.create_PyFuncIO(
                            output_id, make_checked_func(fn, t_in, True),
                            t_in, make_key(x), input_array)
            else:
                op_string = self.op_to_string(op)

                if op_string:
                    self.mpi_sim.add_op(component, op_string)

    def op_to_string(self, op):
        """
        Convert an operator into a string. The string will be passed into
        the C++ simulator, where it will be communicated using MPI to the
        correct MPI process. That process will then use build
        an operator using the parameters specified in the string.
        """

        op_type = type(op)

        if op_type == builder.operator.Reset:
            op_args = ["Reset", make_key(op.dst), op.value]

        elif op_type == builder.operator.Copy:
            op_args = ["Copy", make_key(op.dst), make_key(op.src)]

        elif op_type == builder.operator.DotInc:
            op_args = [
                "DotInc", make_key(op.A), make_key(op.X), make_key(op.Y)]

        elif op_type == builder.operator.ElementwiseInc:
            op_args = [
                "ElementwiseInc", make_key(op.A),
                make_key(op.X), make_key(op.Y)]

        elif op_type == builder.neurons.SimNeurons:
            num_neurons = op.J.size
            neuron_type = type(op.neurons)

            if neuron_type is LIF:
                tau_ref = op.neurons.tau_ref
                tau_rc = op.neurons.tau_rc
                op_args = [
                    "LIF", num_neurons, tau_rc, tau_ref, self.dt,
                    make_key(op.J), make_key(op.output)]

            elif neuron_type is LIFRate:
                tau_ref = op.neurons.tau_ref
                tau_rc = op.neurons.tau_rc
                op_args = [
                    "LIFRate", num_neurons, tau_rc, tau_ref,
                    make_key(op.J), make_key(op.output)]

            elif neuron_type is RectifiedLinear:
                op_args = [
                    "RectifiedLinear", num_neurons, make_key(op.J),
                    make_key(op.output)]

            elif neuron_type is Sigmoid:
                op_args = [
                    "Sigmoid", num_neurons, op.neurons.tau_ref,
                    make_key(op.J), make_key(op.output)]
            else:
                raise NotImplementedError(
                    'nengo_mpi cannot handle neurons of type ' +
                    str(neuron_type))

        elif op_type == builder.synapses.SimSynapse:

            synapse = op.synapse

            if isinstance(synapse, LinearFilter):

                do_adjust = not((isinstance(synapse, Alpha) or
                                 isinstance(synapse, Lowpass)) and
                                synapse.tau <= .03 * self.dt)

                if do_adjust:
                    num, den = adjust_linear_filter(
                        op, synapse, synapse.num, synapse.den, self.dt)
                else:
                    num, den = synapse.num, synapse.den

                op_args = [
                    "LinearFilter", make_key(op.input),
                    make_key(op.output), str(num), str(den)]

            else:
                raise NotImplementedError(
                    'nengo_mpi cannot handle synapses of '
                    'type %s' % str(type(synapse)))

        elif op_type == builder.operator.PreserveValue:
            logger.debug(
                "Skipping PreserveValue, signal: %d, signal_key: %d",
                str(op.dst), make_key(op.dst))

            op_args = []

        elif op_type == MpiSend:
            signal_key = make_key(op.signal)
            op_args = ["MpiSend", op.dst, signal_key, signal_key]

        elif op_type == MpiRecv:
            signal_key = make_key(op.signal)
            op_args = ["MpiRecv", op.src, signal_key, signal_key]

        elif op_type == MpiWait:
            signal_key = make_key(op.signal)
            op_args = ["MpiWait", signal_key]

        else:
            raise NotImplementedError(
                "nengo_mpi cannot handle operator of "
                "type %s" % str(op_type))

        delim = ';'
        op_string = delim.join(map(str, op_args))
        return op_string

    def add_probe(self, probe, signal_key, probe_key=None, sample_every=None):

        period = 1 if sample_every is None else sample_every / self.dt

        self.probe_keys[probe] = (make_key(probe)
                                  if probe_key is None
                                  else probe_key)

        self.mpi_sim.add_probe(
            self.assignments[probe], self.probe_keys[probe],
            signal_key, period)
