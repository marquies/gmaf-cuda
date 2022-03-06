#include <iostream>
#include <unistd.h>
#include <fstream>

#include <dirent.h>

#include<string.h>
#include <thread>

#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

#include <stdio.h>
#include <signal.h>

#include <cstdlib>
#include <memory>
#include <string>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "graphcode.h"
#include "gcloadunit.cuh"
#include "queryhandler.cuh"
#include "helper.h"
#include "algorithmstrategies.cuh"



//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>



void runSequential(std::vector<json> &arr, gmaf::GraphCode gc);


void printUsageAndExit(char *const *argv);

bool mainLoop = true;


Algorithms resolveAlgorithm(std::string input);


void handleConsoleInput(QueryHandler *qh, GcLoadUnit *loadUnit);

void handleNetworkInput(QueryHandler *qh, GcLoadUnit *loadUnit);

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
    if (input == "pcs_cuda") return Algo_pcs_cuda;

    return Algo_Invalid;
}

void printUsageAndExit(char *const *argv) {
    fprintf(stderr, "Usage: %s [-v] -a ALGORITHM -d dir -c limit_files\n", argv[0]);
    std::cout << "    -v verbose in terms of debug" << std::endl;
    std::cout << "    -c limits the maximum number of GCs (default is 100)" << std::endl;
    std::cout << "    -n starts in network mode, binding to port 4711 (default is console)" << std::endl;

    std::cout << "Algorithms available" << std::endl;
    std::cout << "    pc_cuda" << std::endl;
    std::cout << "    pc_cpu_seq" << std::endl;
    std::cout << "    pc_cpu_par" << std::endl;
    std::cout << "    pm_cuda" << std::endl;
    std::cout << "    pmr_cuda" << std::endl;
    std::cout << "    pcs_cuda" << std::endl;


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
    bool network = false;


    while ((opt = getopt(argc, argv, "vd:c:sba:n")) != -1) {
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
            case 'n':
                network = true;
                break;
            default:
                printUsageAndExit(argv);
        }
    }

    // Setup based on parameters

    QueryHandler *qh = new QueryHandler();
    GcLoadUnit *loadUnit;


    if (cvalue == NULL && !simulation) {
        printUsageAndExit(argv);
    }

    if (algorithm == NULL) {
        printUsageAndExit(argv);
    }

    StrategyFactory sf;
    try {
        loadUnit = sf.setupStrategy(resolveAlgorithm(algorithm), qh);
    } catch (std::runtime_error e) {
        std::cout << "Unknown algorithm: " << algorithm << std::endl;
        printUsageAndExit(argv);
    }


    if (simulation) {
        loadUnit->loadArtificialGcs(limit, 100);
    } else {
        loadUnit->loadGraphCodes(cvalue, limit);
    }

    void (*old)(int);

    old = signal(SIGINT, ctrl_c); /* installs handler */
    if (network) {
        handleNetworkInput(qh, loadUnit);

    } else {
        handleConsoleInput(qh, loadUnit);
    }
    signal(SIGINT, old); /* restore initial handler */

    loadUnit->freeAll();

    return 0;

}

void handleNetworkInput(QueryHandler *qh, GcLoadUnit *loadUnit) {
    int server_fd, new_socket;
    ssize_t valread;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    unsigned short port = 4711;

    const char *hello = "Enter Query: ";

    // Creating socket file descriptor
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    // Forcefully attaching socket to the port 8080
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT,
                   &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    // Forcefully attaching socket to the port
    if (bind(server_fd, (struct sockaddr *) &address,
             sizeof(address)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }
    if (listen(server_fd, 3) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }
    std::cout << "Started network daemon IP '" << inet_ntoa(address.sin_addr) << "' Port: '" << port << "'"
              << std::endl;
    if ((new_socket = accept(server_fd, (struct sockaddr *) &address,
                             (socklen_t *) &addrlen)) < 0) {
        perror("accept");
        exit(EXIT_FAILURE);
    }

    const int bufferSize = 1024;

    do {
        char buffer[bufferSize] = {0};
        send(new_socket, hello, strlen(hello), 0);
        valread = read(new_socket, buffer, ssize_t(bufferSize));
        if (valread > 0 && mainLoop) {
            printf("Got : %s\n", buffer);
            std::string str(buffer);
            str.erase(std::remove(str.begin(), str.end(), '\r'), str.end());
            str.erase(std::remove(str.begin(), str.end(), '\n'), str.end());
            if (str.compare("quit") == 0) {
                mainLoop = false;
            } else {
                if (qh->validate(str)) {
                    qh->processQuery(str, loadUnit);
                } else {
                    const char *msg = "Query invalid";
                    send(new_socket, msg, strlen(msg), 0);
                    std::cout << msg << std::endl;
                }
            }
        }
    } while (mainLoop);
}

void handleConsoleInput(QueryHandler *qh, GcLoadUnit *loadUnit) {
    std::string queryString;

    char buf[256];


    do {
        std::cout << "Enter Query: " << std::endl;
        if (fgets(buf, sizeof(buf), stdin) != NULL && mainLoop) {
            printf("Got : %s", buf);
            std::string str(buf);
            str.erase(std::remove(str.begin(), str.end(), '\n'), str.end());
            if (str.compare("quit") == 0) {
                mainLoop = false;
            } else {
                if (qh->validate(str)) {
                    qh->processQuery(str, loadUnit);
                } else {
                    std::cout << "Query invalid" << std::endl;
                }
            }
        }
    } while (mainLoop);

}


