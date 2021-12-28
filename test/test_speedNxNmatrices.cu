//
// Created by breucking on 27.12.21.
//

#include <iostream>
#include <graphcode.h>
#include <chrono>
#include <ctime>

#include "../src/helper.h"
#include "../src/cuda_algorithms.cuh"

#include "testhelper.cpp"

#define N 1000
#define L 100

void testCpuSeqCalc();
void testCpuThreadCalc();
int calculateSimilaritySequentialOrdered(json gc1, json gc2, float *results);

void testCudaCalc();

/**
 * MAIN
 */
int main(int, char**)
{

    testCpuSeqCalc();
//    testCpuThreadCalc();
    testCudaCalc();

}

void testCudaCalc() {
    // 1. Create n gc matrices of size l
    // 2. Calc metrix for each GC



    const json &gc_sample = generateTestData(L);

    std::vector<json> *others = new std::vector<json>() ;

    for(int i = 0; i < N; i++) {
        others->push_back(gc_sample);
    }

    gmaf::GraphCode gc;

    auto start = std::chrono::system_clock::now();

    float results[3];

    // Do a plain simple version of the calc
    for (const auto agc: *others) {
//        calculateSimilaritySequentialOrdered(gc_sample, agc, results);
        testCudaLinearMatrixMemory(gc_sample, agc);
    }

    // Do a plain simple version of the calc



    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "finished CUDA computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n";

}

void testCpuSeqCalc() {

    // 1. Create n gc matrices of size l
    // 2. Calc metrix for each GC



    const json gc_sample = generateTestData(L);
//    std::cout << "tc1" << std::endl;

    std::vector<json> *others = new std::vector<json>(N) ;
//    std::cout << "tc2" << std::endl;

    for(int i = 0; i < N; i++) {
        others->push_back(gc_sample);
    }
//    std::cout << "tc3" << std::endl;

    gmaf::GraphCode gc;

    auto start = std::chrono::system_clock::now();

    float results[3];

    // Do a plain simple version of the calc
    for (const auto agc: *others) {
//        std::cout << "tc4" << std::endl;
        calculateSimilaritySequentialOrdered(gc_sample, agc, results);
    }

    // Do a plain simple version of the calc



    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "finished CPU seq computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n";

}

void testCpuThreadCalc() {

    // 1. Create n gc matrices of size l
    // 2. Calc metrix for each GC

    nlohmann::json gcq;
    gcq["dictionary"] = { "head", "body", "foot"};
    gcq["matrix"] = {{1,1,0}, {0,1,0}, {0,1,1}};

    nlohmann::json gce;
    gce["dictionary"] = { "head", "body", "torso"};
    gce["matrix"] = {{1,2,0}, {0,1,0}, {0,0,1}};

    std::vector<json> others;



    for(int i = 0; i < N; i++) {
        others.push_back(gce);
    }

    gmaf::GraphCode gc;

    auto start = std::chrono::system_clock::now();


    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "finished computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n";

}



int calculateSimilaritySequentialOrdered(json gc1, json gc2, float *results) {
    json gc1Dictionary = gc1["dictionary"];
    json gc2Dictionary = gc2["dictionary"];

    std::string gc1Dict[gc1Dictionary.size()];

    int n = 0;

    int sim = 0;


    int matrix1[gc1Dictionary.size()][gc1Dictionary.size()];
    convertDict2Matrix(gc1Dictionary.size(), (int *) matrix1, gc1["matrix"]);

    int matrix2[gc2Dictionary.size()][gc2Dictionary.size()];
    convertDict2Matrix(gc2Dictionary.size(), (int *) matrix2, gc2["matrix"]);



    std::vector<std::string> dict2;
    for (const auto &item2: gc2Dictionary.items()) {
        dict2.push_back(item2.value().get<std::string>());
    }


    for (const auto &item: gc1Dictionary.items()) {
        //std::cout << item.value() << "\n";

        std::string str = item.value().get<std::string>();
        gc1Dict[n++] = str;


        for (const auto &item2: gc2Dictionary.items()) {
            if (str == item2.value()) {
                //std::cout << "Match" << std::endl;
                sim++;
            }
        }

    }
    int num_of_non_zero_edges = 0;
    int edge_metric_count = 0;
    int edge_type = 0;

    for (int i = 0; i < gc1Dictionary.size(); i++) {
        for (int j = 0; j < gc1Dictionary.size(); j++) {

            if (i != j && matrix1[i][j] != 0) {
                num_of_non_zero_edges++;

                int position1 = i;
                int position2 = j;
                //std::cout << "Pos " << position1 << " " << position2 << std::endl;
                if (position1 == -1 || position2 == -1) {
                    continue;
                }

                int edge = matrix2[position1][position2];
                if (edge != 0) {
                    edge_metric_count++;
                }
                if (edge == matrix1[i][j]) {
                    edge_type++;
                }

            }
        }
    }

    float node_metric = (float) sim / (float) gc1Dictionary.size();
    float edge_metric = 0.0;
    if (num_of_non_zero_edges > 0)
        edge_metric = (float) edge_metric_count / (float) num_of_non_zero_edges;
    float edge_type_metric = 0.0;
    if (edge_metric_count > 0)
        edge_type_metric = (float) edge_type / (float) edge_metric_count;

    results[0] = node_metric;
    results[1] = edge_metric;
    results[2] = edge_type_metric;
    return 0;

}
