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



//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>



void runThreaded(std::vector<json> &arr, const gmaf::GraphCode &gc, int s);

void runSequential(std::vector<json> &arr, gmaf::GraphCode gc);


bool mainLoop = true;

void ctrl_c(int sig) {
    fprintf(stderr, "Ctrl-C caught - Please press enter\n");
    mainLoop = false;
    signal(sig, ctrl_c); /* re-installs handler */
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
    int limit = 100;
    bool simulation = false;

    while ((opt = getopt(argc, argv, "d:c:s")) != -1) {
        switch (opt) {
            case 's':
                simulation = true;
            case 'd':
                cvalue = optarg;
                break;
            case 'c':
                limit = atoi(optarg);
                break;

            default:
                fprintf(stderr, "Usage: %s -d dir -c limit_files\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    if (cvalue == NULL) {
        fprintf(stderr, "Usage: %s -d dir -c limit_files\n", argv[0]);
        exit(EXIT_FAILURE);
    }


    GcLoadUnit loadUnit;
    if (simulation) {
        loadUnit.loadArtificialGcs(limit, 100);
    } else {

        int n = 0;


        for (const auto &entry: fs::directory_iterator(cvalue)) {
//            files.push_back(entry.path().string());
try {
            loadUnit.addGcFromFile(entry.path().string());
        } catch (json::exception &e) {
            std::cerr << e.what() << '\n';
        }
            n++;
            if (n > limit) {
                break;
            }
        }

    }

    std::string queryString;

    char buf[256];
    void (*old)(int);

    old = signal(SIGINT, ctrl_c); /* installs handler */

    do {
        std::cout << "Enter Query";
        if (fgets(buf, sizeof(buf), stdin) != NULL && mainLoop) {
            printf("Got : %s", buf);
            std::string str(buf);
            str.erase(std::remove(str.begin(), str.end(), '\n'), str.end());
            if (str.compare("quit") == 0) {
                mainLoop = false;
            } else {
                if (QueryHandler::validate(str)) {
                    QueryHandler::processQuery(str, loadUnit);
                } else {
                    std::cout << "Query invalid" << std::endl;
                }
            }
        }
    } while (mainLoop);
    signal(SIGINT, old); /* restore initial handler */
    return 0;

//    auto start = std::chrono::system_clock::now();
//
//    std::vector<json> arr;
//
//
//    // Loading the graph codes
//    gmaf::GraphCode gc;
//    gc.loadGraphCodes(cvalue, limit, &arr);
//    auto loaded = std::chrono::system_clock::now();
//    std::chrono::duration<double> elapsed_seconds = loaded - start;
//
//    std::cout << "loaded " << arr.size() << " graph code files. (" << "elapsed time: " << elapsed_seconds.count() << ")"
//              << std::endl;
//
//    // Run the code
//
//    //runThreaded(arr, gc, 4);
//    runSequential(arr, gc);
//
//    // Time evaluation
//    auto end = std::chrono::system_clock::now();
//
//    elapsed_seconds = end - start;
//    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
//
//    std::cout << "finished computation at " << std::ctime(&end_time)
//              << "elapsed time: " << elapsed_seconds.count() << "s\n";
    return 0;
}

void runSequential(std::vector<json> &arr, gmaf::GraphCode gc) {
    gc.calculateSimilarityV(0, &arr.at(0), &arr, 1, arr.size());
}


void runThreaded(std::vector<json> &arr, const gmaf::GraphCode &gc, int s) {
    int x = arr.size() / s;
    std::vector<std::thread> threads;

    for (int i = 0; i < s; i++) {
        int end = i == s - 1 ? arr.size() : (i + 1) * x;
        threads.push_back(std::thread(&gmaf::GraphCode::calculateSimilarityV, gc, i, &arr.at(0), &arr, i * x + 1, end));
    }

    for (auto &th: threads) {
        th.join();
    }
}

