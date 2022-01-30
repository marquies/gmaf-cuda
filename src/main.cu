#include <iostream>
#include <unistd.h>
#include <fstream>

#include <dirent.h>

#include<string.h>
#include <thread>

#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include "graphcode.h"
#include "gcloadunit.cuh"
#include "queryhandler.cuh"
#include "helper.h"
#include "cpualgorithms.h"



//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>



void runSequential(std::vector<json> &arr, gmaf::GraphCode gc);


void printUsageAndExit(char *const *argv);

bool mainLoop = true;


enum Algorithms {
    Algo_Invalid,
    Algo_pc_cuda,
    Algo_pc_cpu_seq,
    Algo_pc_cpu_par,
    Algo_pm_cuda,
    Algo_pmr_cuda
    //others...
};

Algorithms resolveAlgorithm(std::string input);


void ctrl_c(int sig) {
    fprintf(stderr, "Ctrl-C caught - Please press enter\n");
    mainLoop = false;
    signal(sig, ctrl_c); /* re-installs handler */

}

Algorithms resolveAlgorithm(std::string input) {
    if (input == "pc_cuda") return Algo_pc_cuda;
    if (input == "pc_cpu_seq") return Algo_pc_cpu_seq;
    if (input == "pc_cpu_par") return Algo_pc_cpu_par;
    if (input == "pm_cuda") return Algo_pm_cuda;
    if (input == "pmr_cuda") return Algo_pmr_cuda;
    return Algo_Invalid;
}

void printUsageAndExit(char *const *argv) {
    fprintf(stderr, "Usage: %s [-v] -a ALGORITHM -d dir -c limit_files\n", argv[0]);
    std::cout << "    -v verbose in terms of debug" << std::endl;
    std::cout << "    -c limits the maximum number of GCs" << std::endl;
    std::cout << "Algorithms available" << std::endl;
    std::cout << "    pc" << std::endl;
    std::cout << "    pc_cpu_seq" << std::endl;
    std::cout << "    pc_cpu_par" << std::endl;
    std::cout << "    pm_cuda" << std::endl;


    exit(EXIT_FAILURE);
}


/**
 * Main function of the program.
 * @param argc
 * @param argv
 * @return
 */
int main_init(int argc, char *argv[]) {

    // Handling the command line arguments
    int opt;
    char *cvalue = NULL;
    char *algorithm = NULL;
    int limit = 100;
    bool simulation = false;


    while ((opt = getopt(argc, argv, "vd:c:sba:")) != -1) {
        switch (opt) {
            case 's':
                simulation = true;
                break;
            case 'b':
                G_BENCHMARK = true;
                break;
            case 'v':
                G_DEBUG = true;
                break;
            case 'd':
                cvalue = optarg;
                break;
            case 'c':
                limit = atoi(optarg);
                break;
            case 'a':
                algorithm = optarg;
                break;
            default:
                printUsageAndExit(argv);
        }
    }

    // Setup based on parameters

    QueryHandler qh;
    GcLoadUnit *loadUnit = NULL;


    if (cvalue == NULL && !simulation) {
        printUsageAndExit(argv);
    }

    if (algorithm == NULL) {
        printUsageAndExit(argv);
    }

    switch (resolveAlgorithm(algorithm)) {
        case Algo_pc_cuda:

            qh.setStrategy(std::unique_ptr<Strategy>(new CudaTask1OnGpuMemory));
            //qh.setStrategy(std::unique_ptr<Strategy>(new CudaTask1MemCopy));
            loadUnit = new GcLoadUnit(GcLoadUnit::Modes::MODE_MEMORY_MAP);
            break;
        case Algo_pc_cpu_seq:
            qh.setStrategy(std::unique_ptr<Strategy>(new CpuSequentialTask1));
            loadUnit = new GcLoadUnit(GcLoadUnit::Modes::MODE_VECTOR_MAP);
            break;
        case Algo_pc_cpu_par:
            qh.setStrategy(std::unique_ptr<Strategy>(new CpuParallelTask1));
            loadUnit = new GcLoadUnit(GcLoadUnit::Modes::MODE_VECTOR_MAP);
            break;
        case Algo_pm_cuda:
            qh.setStrategy(std::unique_ptr<Strategy>(new CudaTask2a));
            loadUnit = new GcLoadUnit(GcLoadUnit::Modes::MODE_VECTOR_MAP);
            break;
        case Algo_pmr_cuda:
            qh.setStrategy(std::unique_ptr<Strategy>(new CudaTask2ab));
            loadUnit = new GcLoadUnit(GcLoadUnit::Modes::MODE_VECTOR_MAP);
            break;

        case Algo_Invalid:
        default:
            std::cout << "Unknown algorithm: " << algorithm << std::endl;
            printUsageAndExit(argv);
            break;

    }


    if (simulation) {
        loadUnit->loadArtificialGcs(limit, 100);
    } else {
        loadUnit->loadGraphCodes(cvalue, limit);
    }

    std::string queryString;

    char buf[256];
    void (*old)(int);

    old = signal(SIGINT, ctrl_c); /* installs handler */

    // Enable to work without query
    //qh.processQuery("Query by Example: 1.gc", *loadUnit);
    do {
        std::cout << "Enter Query: ";
        if (fgets(buf, sizeof(buf), stdin) != NULL && mainLoop) {
            printf("Got : %s", buf);
            std::string str(buf);
            str.erase(std::remove(str.begin(), str.end(), '\n'), str.end());
            if (str.compare("quit") == 0) {
                mainLoop = false;
            } else {
                if (qh.validate(str)) {
                    qh.processQuery(str, *loadUnit);
                } else {
                    std::cout << "Query invalid" << std::endl;
                }
            }
        }
    } while (mainLoop);
    signal(SIGINT, old); /* restore initial handler */

    loadUnit->freeAll();

    return 0;

}


