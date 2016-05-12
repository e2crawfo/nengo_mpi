import nengo_mpi
import nengo

model = nengo.Network()
with model:
    A = nengo.Ensemble(n_neurons=50, dimensions=1)
    B = nengo.Ensemble(n_neurons=50, dimensions=1)
    nengo.Connection(A, B)
    
sim = nengo_mpi.Simulator(model)
sim.run(1.0)

