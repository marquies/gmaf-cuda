//
// Created by breucking on 27.12.21.
//

#include <iostream>
#include <graphcode.h>
#include <chrono>
#include <ctime>

#include "../src/cuda_algorithms.cuh"

#include "testhelper.cpp"

#define N 100
#define L 2040

void testCpuSeqCalc();

void testCpuThreadCalc();

void testCpuSeqCalcPlain();

void testCudaCalc();

void testCudaCalcPlain();

/**
 * MAIN
 */
int main(int, char **) {

//    testCpuSeqCalc();
//    testCpuThreadCalc();
//    testCudaCalc();
   // testCpuSeqCalcPlain();
    testCudaCalcPlain();
}

void testCudaCalc() {
    // 1. Create n gc matrices of size l
    // 2. Calc metrix for each GC



    const json &gc_sample = generateTestData(L);

    std::vector<json> *others = new std::vector<json>();

    for (int i = 0; i < N; i++) {
        others->push_back(gc_sample);
    }

    auto start = std::chrono::system_clock::now();


    // Do a plain simple version of the calc
    for (const auto agc: *others) {
        testCudaLinearMatrixMemory(gc_sample, agc);
    }

    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "finished CUDA computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n";

}

void testCpuSeqCalcPlain() {

    // 1. Create n gc matrices of size l
    // 2. Calc metrix for each GC
    const GraphCode gc_sample = generateTestDataGc(L);

    std::vector<GraphCode> *others = new std::vector<GraphCode>();

    for (int i = 0; i < N; i++) {
        others->push_back(gc_sample);
    }

    auto start = std::chrono::system_clock::now();

    // Do a plain simple version of the calc
    for (const auto agc: *others) {
        calculateSimilaritySequentialOrdered(gc_sample, agc);
    }

    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "finished CPU seq computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n";

}

void testCudaCalcPlain() {

    // 1. Create n gc matrices of size l
    // 2. Calc metrix for each GC
    const GraphCode gc_sample = generateTestDataGc(L);

    std::vector<GraphCode> *others = new std::vector<GraphCode>();

    for (int i = 0; i < N; i++) {
        others->push_back(gc_sample);
    }

    auto start = std::chrono::system_clock::now();

    // Do a plain simple version of the calc
    for (const auto agc: *others) {
        testCudaLinearMatrixMemory(gc_sample, agc);
    }

    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "finished CUDA seq computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n";

}

void testCpuSeqCalc() {

    // 1. Create n gc matrices of size l
    // 2. Calc metrix for each GC



    const json gc_sample = generateTestData(L);

    std::vector<json> *others = new std::vector<json>();

    for (int i = 0; i < N; i++) {
        others->push_back(gc_sample);
    }

    auto start = std::chrono::system_clock::now();

    // Do a plain simple version of the calc
    for (const auto agc: *others) {
        calculateSimilaritySequentialOrdered(gc_sample, agc);
    }

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
    gcq["dictionary"] = {"head", "body", "foot"};
    gcq["matrix"] = {{1, 1, 0},
                     {0, 1, 0},
                     {0, 1, 1}};

    nlohmann::json gce;
    gce["dictionary"] = {"head", "body", "torso"};
    gce["matrix"] = {{1, 2, 0},
                     {0, 1, 0},
                     {0, 0, 1}};

    std::vector<json> others;


    for (int i = 0; i < N; i++) {
        others.push_back(gce);
    }

    auto start = std::chrono::system_clock::now();

    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "finished computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n";

}



