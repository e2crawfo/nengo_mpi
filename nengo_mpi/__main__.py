""" Setup mpi context.

The worker processes enter the C++ code and wait for signals from the master
process. The master process executes the script given as the first argument.

Example usage: mpirun -np <np> python -m nengo_mpi <script>

This code will is only executed if nengo_mpi is run as the main script with -m.
"""


# cargo cult to get around errors. See
# https://xrunhprof.wordpress.com/2014/11/04/an-openmpi-python-and-dlopen-issue/
import ctypes
ctypes.CDLL("libmpi.so", mode=ctypes.RTLD_GLOBAL)

import mpi_sim
import sys
import os
import atexit

mpi_sim.mpi_init()

rank = mpi_sim.get_mpi_rank()
n_procs = mpi_sim.get_mpi_n_procs()

if rank == 0 and (not sys.argv[1:] or sys.argv[1] in ("--help", "-h")):
    print ("usage: mpirun -np <np> python -m nengo_mpi scriptfile [arg] ...")
    sys.exit(2)

if rank > 0:
    mpi_sim.worker_start()
else:
    # Note: Largely copied from /usr/lib/python2.7/pdb.py
    mainpyfile = sys.argv[1]     # Get script filename
    if not os.path.exists(mainpyfile):
        print ('Error:', mainpyfile, 'does not exist')
        sys.exit(1)

    del sys.argv[0]         # Hide "pdb.py" from argument list

    # Replace pdb's dir with script's dir in front of module search path.
    sys.path[0] = os.path.dirname(mainpyfile)

    try:
        # import __main__
        # __main__.__dict__.clear()
        # __main__.__dict__.update(
        #     {"__name__"    : "__main__",
        #      "__file__"    : mainpyfile,
        #      "__builtins__": __builtins__})

        globals = {
            "__name__": "__main__",
            "__file__": mainpyfile,
            "__builtins__": __builtins__}
        statement = 'execfile(%r)\n' % mainpyfile
        locals = globals
        statement += '\n'

        exec statement in globals, locals

    except SystemExit:
        print ("The program exited via sys.exit(). Exit status: ",)
        print ( sys.exc_info()[1] )

    # Closes all nengo_mpi Simulator instances, since
    # nengo_mpi.Simulator.close_simulators is an atexit exitfunc.
    atexit._run_exitfuncs()

    mpi_sim.kill_workers()

mpi_sim.mpi_finalize()
