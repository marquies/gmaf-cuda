#include <iostream>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <signal.h>
#include <cstdlib>
#include <memory>
#include <string>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include<pthread.h>
#include <c++/9/fstream>

#include "graphcode.h"
#include "gcloadunit.cuh"
#include "queryhandler.cuh"
#include "helper.h"
#include "algorithmstrategies.cuh"


bool mainLoop = true;

/**
 * Print the usage text information to standard out.
 * @param argv argument values handed over to the program
 */
void printUsageAndExit(char *const *argv);

/**
 * Evaluates the algorithm based on command line argument.
 * @param input value of a command line argument
 * @return the algorithm or Algorithms::Algo_Invalid if not resolved.
 */
Algorithms resolveAlgorithm(std::string input);

/**
 * Handling input in console mode.
 * @param qh initialized QueryHandler
 * @param loadUnit initialized load unit
 */
void handleConsoleInput(QueryHandler *qh, GcLoadUnit *loadUnit);

/**
 * Handling input in network mode.
 * @param qh initialized QueryHandler
 * @param loadUnit initialized load unit
 */
void handleNetworkInput(QueryHandler *qh, GcLoadUnit *loadUnit);

/**
 * Handler for CTRL+C input to stop the program.
 * @param sig
 */
void ctrl_c(int sig) {
    fprintf(stderr, "Ctrl-C caught - Stopping program\n");
    mainLoop = false;
    exit(EXIT_SUCCESS);
    signal(sig, ctrl_c); /* re-installs handler */

}

/**
 * Maps String into Algorithms enum value.
 * @param input string from command line or config
 * @return the mappes algorithm or Algo_Invalid if not existing.
 */
Algorithms resolveAlgorithm(std::string input) {
    if (input == "pc_cuda") return Algo_pc_cuda;
    if (input == "pc_cpu_seq") return Algo_pc_cpu_seq;
    if (input == "pc_cpu_par") return Algo_pc_cpu_par;
    if (input == "pm_cuda") return Algo_pm_cuda;
    if (input == "pmr_cuda") return Algo_pmr_cuda;
    if (input == "pcs_cuda") return Algo_pcs_cuda;

    return Algo_Invalid;
}

/**
 * Prints a usage text for this program.
 * @param argv command line arguments.
 */
void printUsageAndExit(char *const *argv) {
    fprintf(stderr, "Usage: %s [-vdsbn] -a ALGORITHM -d dir -c limit_data -l gc_size \n", argv[0]);
    std::cout << "    -v verbose in terms of debug" << std::endl;
    std::cout << "    -s simulation mode (artificial data is used)" << std::endl;
    std::cout << "    -c limits the maximum number of GCs (default is 100)" << std::endl;
    std::cout << "    -l set the length of elements of the artificial graph codes in simulation mode (default is 100)"
              << std::endl;
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

std::string trim(std::string const &source, char const *delims = " \t\r\n") {
    std::string result(source);
    std::string::size_type index = result.find_last_not_of(delims);
    if (index != std::string::npos)
        result.erase(++index);

    index = result.find_first_not_of(delims);
    if (index != std::string::npos)
        result.erase(0, index);
    else
        result.erase();
    return result;
}

/**
 * Main function of the program.
 * @param argc command line arguments
 * @param argv command line argument values
 * @return 0 if successful
 */
int main_init(int argc, char *argv[]) {

    // Handling the command line arguments
    int opt;
    char *cvalue = NULL;
    char *algorithm = NULL;
    int limit = 100;
    int dimension = 100;
    bool simulation = false;
    bool network = false;
    char *iniFile = NULL;


    while ((opt = getopt(argc, argv, "vd:c:sba:nl:i:")) != -1) {
        switch (opt) {
            case 'i':
                iniFile = optarg;
                break;
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
            case 'l':
                dimension = atoi(optarg);
                break;
            default:
                printUsageAndExit(argv);
        }
    }

    // Evaluating properties if provided
    if (iniFile != NULL) {
        std::ifstream file(iniFile);

        std::string line;
        std::string name;
        std::string value;
        std::string inSection;
        int posEqual;
        while (std::getline(file, line)) {
            if (!line.length()) continue;
            if (line[0] == '#') continue;
            if (line[0] == ';') continue;

            posEqual = line.find('=');
            name = trim(line.substr(0, posEqual));
            value = trim(line.substr(posEqual + 1));

            if (name == "s") {
                if (value == "true") {
                    simulation = true;
                } else if (value == "false") {
                    simulation = false;
                } else {
                    std::cout << "Invalid value '" << value << "' for parameter '" << name << "'" << std::endl;
                    exit(EXIT_FAILURE);
                }
            }
            if (name == "d") {
                if (value.length() <= 0) {
                    std::cout << "Invalid value '" << value << "' for parameter '" << name << "'" << std::endl;
                    exit(EXIT_FAILURE);
                }
                const std::string::size_type size = value.size();
                cvalue = new char[size + 1];   //we need extra char for NUL
                memcpy(cvalue, value.c_str(), size + 1);
            }
            if (name == "a") {
                if (value.length() <= 0) {
                    std::cout << "Invalid value '" << value << "' for parameter '" << name << "'" << std::endl;
                    exit(EXIT_FAILURE);
                }
                const std::string::size_type size = value.size();
                algorithm = new char[size + 1];   //we need extra char for NUL
                memcpy(algorithm, value.c_str(), size + 1);
            }
            if (name == "c") {
                if (value.length() <= 0) {
                    std::cout << "Invalid value '" << value << "' for parameter '" << name << "'" << std::endl;
                    exit(EXIT_FAILURE);
                }
                limit = atoi(value.c_str());
                if (limit <= 0) {
                    std::cout << "Invalid value '" << value << "' for parameter '" << name << "'" << std::endl;
                    exit(EXIT_FAILURE);
                }
            }
            if (name == "l") {
                if (value.length() <= 0) {
                    std::cout << "Invalid value '" << value << "' for parameter '" << name << "'" << std::endl;
                    exit(EXIT_FAILURE);
                }
                dimension = atoi(value.c_str());
                if (dimension <= 0) {
                    std::cout << "Invalid value '" << value << "' for parameter '" << name << "'" << std::endl;
                    exit(EXIT_FAILURE);
                }
            }

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

        loadUnit->loadArtificialGcs(limit, dimension);
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

void *connection_handler(void *arguments);

struct arg_struct {
    int socket;
    QueryHandler *qh;
    GcLoadUnit *loadUnit;
};

void handleNetworkInput(QueryHandler *qh, GcLoadUnit *loadUnit) {
    int server_fd, new_socket, *new_sock;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    unsigned short port = 4711;


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
    while ((new_socket = accept(server_fd, (struct sockaddr *) &address,
                                (socklen_t *) &addrlen))) {
//        perror("accept");
//        exit(EXIT_FAILURE);
        pthread_t sniffer_thread;
        new_sock = static_cast<int *>(malloc(1));
        *new_sock = new_socket;
        struct arg_struct args;
        args.socket = *new_sock;
        args.qh = qh;
        args.loadUnit = loadUnit;

        if (pthread_create(&sniffer_thread, NULL, connection_handler, (void *) &args) < 0) {
            perror("could not create thread");
            return;
        }

        //Now join the thread , so that we dont terminate before the thread
        //pthread_join( sniffer_thread , NULL);
        puts("Handler assigned");
    }
    if (new_socket < 0) {
        perror("accept failed");
        return;
    }
    return;
}


void *connection_handler(void *arguments) {
    const int bufferSize = 1024;
    const char *hello = "Enter Query: \r\n";
    ssize_t valread;
    struct arg_struct *args = (struct arg_struct *) arguments;

    int sock = args->socket;
    QueryHandler *qh = args->qh;
    GcLoadUnit *loadUnit = args->loadUnit;

    send(sock, hello, strlen(hello), 0);
    do {
        char buffer[bufferSize] = {0};
        valread = read(sock, buffer, ssize_t(bufferSize));


        if (valread == 0) {
            puts("Client disconnected");
            fflush(stdout);
            return 0;
        } else if (valread == -1) {
            perror("recv failed");
            return 0;
        }

        if (valread > 0 && mainLoop) {
            printf("Got : %s\n", buffer);
            std::string str(buffer);
            str.erase(std::remove(str.begin(), str.end(), '\r'), str.end());
            str.erase(std::remove(str.begin(), str.end(), '\n'), str.end());
            if (str.compare("quit") == 0) {
                mainLoop = false;
            } else {
                if (qh->validate(str)) {
                    Metrics *metrics = qh->processQuery(str, loadUnit);
                    if (metrics != NULL) {


                        auto result = nlohmann::json::array();
                        for (int i = 0; i < loadUnit->getNumberOfGc(); i++) {
                            nlohmann::json metric;
                            metric["gc_filename"] = loadUnit->getGcNameOnPosition(metrics[i].idx);
                            metric["inferencing"] = metrics[i].inferencing;
                            metric["similarity"] = metrics[i].similarity;
                            metric["recommendation"] = metrics[i].recommendation;
                            result.push_back(metric);
                        }
                        std::string s = result.dump();
                        const char *res_msg = s.c_str();
                        send(sock, res_msg, strlen(res_msg), 0);

                        free(metrics);
                    } else {
                        const char *msg = "Item not found";
                        send(sock, msg, strlen(msg), 0);
                        std::cout << msg << std::endl;
                    }

                } else {
                    const char *msg = "Query invalid";
                    send(sock, msg, strlen(msg), 0);
                    std::cout << msg << std::endl;
                }
            }
            char buf[] = "\r\n";
            send(sock, buf, sizeof(buf), 0);
            send(sock, hello, strlen(hello), 0);

        }
    } while (mainLoop);
    return 0;
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
                    Metrics *result = qh->processQuery(str, loadUnit);
                    free(result);
                } else {
                    std::cout << "Query invalid" << std::endl;
                }
            }
        }
    } while (mainLoop);

}


