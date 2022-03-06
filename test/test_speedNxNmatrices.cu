//
// Created by breucking on 27.12.21.
//

#include <iostream>
#include <graphcode.h>
#include <chrono>
#include <ctime>

#include "../src/cudaalgorithms.cuh"

#include "testhelper.h"
#include "cudahelper.cuh"
#include "reduce.cuh"
#include "cpualgorithms.h"

#define N 10
#define L 12500

void testCpuSeqCalc();

void testCpuThreadCalc();

void testCpuSeqCalcPlain();

void testCudaCalc();

void testCudaCalcPlain();


void testReducer();

void generate_input(unsigned int* input,  long input_len)
{
    for ( long i = 0; i < input_len; ++i)
    {
        input[i] = i;
    }
}

 int cpu_simple_sum( int* h_in,  int h_in_len)
{
     int total_sum = 0;

    for ( int i = 0; i < h_in_len; ++i)
    {
        total_sum = total_sum + h_in[i];
    }

    return total_sum;
}



/**
 * MAIN
 */
int main(int, char **) {
    testReducer();

//    testCpuSeqCalc();
//    testCpuThreadCalc();
//    testCudaCalc();
    testCpuSeqCalcPlain();
    testCudaCalcPlain();
}

void testReducer() {
    // Set up clock for timing comparisons
    std::clock_t start;
    double duration;
     long long h_in_len = (long long) L* (long long)L;
    //unsigned int h_in_len = 2048;
    std::cout << "h_in_len: " << h_in_len << std::endl;
    unsigned  int* h_in = new  unsigned  int[h_in_len];
    generate_input(h_in, h_in_len);
    //for (unsigned int i = 0; i < input_len; ++i)
    //{
    //	std::cout << input[i] << " ";
    //}
    //std::cout << std::endl;

    // Set up device-side memory for input
    unsigned int* d_in;
    HANDLE_ERROR(cudaMalloc(&d_in, sizeof( unsigned  int) * h_in_len));
    HANDLE_ERROR(cudaMemcpy(d_in, h_in, sizeof( unsigned  int) * h_in_len, cudaMemcpyHostToDevice));
    start = std::clock();
    unsigned int gpu_total_sum = gpu_sum_reduce(d_in, h_in_len);
    duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
    std::cout << gpu_total_sum << std::endl;
    std::cout << "GPU time: " << duration << " s" << std::endl;


    HANDLE_ERROR(cudaFree(d_in));
    delete[] h_in;

}

void testCudaCalc() {
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
        demoCudaLinearMatrixMemory(gc_sample, agc);
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
        demoCalculateSimilaritySequentialOrdered(gc_sample, agc);
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
        demoCudaLinearMatrixMemoryCudaReduceSum(gc_sample, agc);
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
        demoCalculateSimilaritySequentialOrdered(gc_sample, agc);
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



