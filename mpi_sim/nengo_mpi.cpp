#include <iostream>
#include <string>
#include <memory>

#include <mpi.h>
#include <hdf5.h>

#include "optionparser.h"

#include "mpi_simulator.hpp"


using namespace std;

enum serialOptionIndex {UNKNOWN, HELP, NO_PROG, TIMING, LOG, SEED, MERGED};

const option::Descriptor serial_usage[] =
{
 {UNKNOWN, 0, "" , "",          option::Arg::None, "USAGE: nengo_mpi [options] <network file> <simulation time> \n\n"
                                                   "where <network file> is a file specifying a network to simulate,\n"
                                                   "and <simulation time> is the amount of time to simulate the network\n"
                                                   "for, in seconds. Simulation results are logged to an HDF5 file.\n"
                                                   "Note that the options must come before the network file.\n"
                                                   "Options:" },
 {HELP,     0, "" , "help",     option::Arg::None, "  --help  \tPrint usage and exit." },
 {NO_PROG,  0, "",  "noprog",   option::Arg::None, "  --noprog  \tSupply to omit the progress bar." },
 {TIMING,   0, "",  "timing",   option::Arg::None, "  --timing  \tSupply to collect timing info." },
 {LOG,      0, "",  "log",      option::Arg::NonEmpty, "  --log  \tName of file to log results to using HDF5. "
                                                               "If not specified, the log filename is the same as the "
                                                               "name of the network file, but with the .h5 extension."},
 {SEED,     0, "",  "seed",     option::Arg::Numeric, "  --seed  \tSeed for stochastic processes in the network."},
 {MERGED,   0, "",  "merged",   option::Arg::None, "  --merged  \tSupply to use merged communication mode."},
 {UNKNOWN,  0, "" , ""   ,      option::Arg::None, "\nExamples:\n"
                                                   "  nengo_mpi --noprog basal_ganglia.net 1.0\n"
                                                   "  nengo_mpi --log ~/spaun_results.h5 spaun.net 7.5\n" },
 {0,0,0,0,0,0}
};

int mpi_master_start(int argc, char **argv){

    argc -= (argc > 0); argv += (argc > 0); // skip program name argv[0] if present
    option::Stats  stats(serial_usage, argc, argv);
    option::Option options[stats.options_max], buffer[stats.buffer_max];
    option::Parser parse(serial_usage, argc, argv, options, buffer);

    if (parse.error()){
        return 0;
    }

    if (options[HELP] || argc == 0) {
        option::printUsage(std::cout, serial_usage);
        return 0;
    }

    string net_filename;
    float sim_length;

    // Handle mandatory options.
    if(parse.nonOptionsCount() != 2){
        cout << "Please specify a network to simulate and a simulation time." << endl << endl;
        option::printUsage(std::cout, serial_usage);
        return 0;
    }else{
        net_filename = parse.nonOptions()[0];

        string s_sim_length(parse.nonOptions()[1]);

        try{
            sim_length = boost::lexical_cast<float>(s_sim_length);
        }catch(const boost::bad_lexical_cast& e){
            stringstream msg;
            msg << "Specified simulation time, " << s_sim_length
                << ", could not be interpreted as a float." << endl;
            throw runtime_error(msg.str());
        }
    }

    cout << "Will load network from file: " << net_filename << "." << endl;
    cout << "Will run simulation for " << sim_length << " second(s)." << endl;

    bool show_progress = !bool(options[NO_PROG]);
    cout << "Show progress bar: " << show_progress << endl;

    bool collect_timings = bool(options[TIMING]);
    cout << "Collect timing info: " << collect_timings << endl;

    bool mpi_merged = bool(options[MERGED]);
    cout << "Merged communication mode: " << mpi_merged << endl;

    string log_filename;
    if(options[LOG]){
        log_filename = options[LOG].arg;
    }else{
        log_filename = net_filename.substr(0, net_filename.find_last_of(".")) + ".h5";
    }
    cout << "Will write simulation results to: " << log_filename << endl;

    int seed = 1;
    if(options[SEED]){
        seed = boost::lexical_cast<unsigned>(options[SEED].arg);
    }
    cout << "Will simulate with seed: " << seed << endl;
    cout << endl;

    cout << "Building network..." << endl;
    auto sim = unique_ptr<MpiSimulator>(new MpiSimulator(mpi_merged, collect_timings));
    sim->from_file(net_filename);
    sim->finalize_build();

    cout << "Done building network." << endl;
    cout << endl;

    sim->reset(seed);
    cout << endl;

    int n_steps = int(round(sim_length / sim->dt()));
    cout << "Running simulation for " << n_steps << " steps with dt = " << sim->dt() << "." << endl;

    sim->run_n_steps(n_steps, show_progress, log_filename);
    sim->close();

    mpi_kill_workers();

    return 0;
}

int main(int argc, char **argv){

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank == 0){
        mpi_master_start(argc, argv);
    }else{
        mpi_worker_start(MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}