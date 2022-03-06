// Ttt

#include <stdio.h>
#include <cuda_runtime.h>
#include <helper.h>


#include "../src/graphcode.h"
#include "../src/cudaalgorithms.cuh"

#include "testhelper.h"

void testFindLargestDivisor();

void testConvertGc4Cuda();

void testDemoCudaLinearMatrixMemoryCudaReduceSum();

void testCalcKernelLaunchConfig();

void testCudaLinearMatrixMemoryRealTest() {
    // Generate test data

    Metrics m;

    json gcq = generateTestData(9);
    m = demoCudaLinearMatrixMemory(gcq, gcq);

    assert(m.similarity == 1);
    assert(m.recommendation == 1);

    json gcq2 = generateTestData(2040);
    m = demoCudaLinearMatrixMemory(gcq2, gcq2);

    assert(m.similarity == 1);
    assert(m.recommendation == 1);


    nlohmann::json gcq3;
    gcq3["dictionary"] = {"head", "body", "foot"};
    gcq3["matrix"] = {{1, 1, 0},
                      {0, 1, 0},
                      {0, 1, 1}};

    nlohmann::json gcq4;
    gcq4["dictionary"] = {"head", "body", "foot"};
    gcq4["matrix"] = {{1, 2, 0},
                      {0, 1, 0},
                      {0, 0, 1}};

    Metrics m2 = demoCudaLinearMatrixMemory(gcq3, gcq4);

    assert(AreSame(m2.similarity, (float) 3. / 3.));
    assert(m2.recommendation == .5);
    assert(m2.inferencing == 0);


}


/**
 * MAIN
 */
int main(int, char **) {
    testFindLargestDivisor();
    testConvertGc4Cuda();
    testCudaLinearMatrixMemoryRealTest();

    testDemoCudaLinearMatrixMemoryCudaReduceSum();

    testCalcKernelLaunchConfig();

}

void testCalcKernelLaunchConfig() {
    dim3 grid, block;

    calcKernelLaunchConfig(1, block, grid);
    assert(block.x == 1);
    assert(block.y == 1);
    assert(grid.x == 1);
    assert(grid.y == 1);

    calcKernelLaunchConfig(2, block, grid);
    assert(block.x == 2);
    assert(block.y == 2);
    assert(grid.x == 1);
    assert(grid.y == 1);

    calcKernelLaunchConfig(32, block, grid);
    assert(block.x == 32);
    assert(block.y == 32);
    assert(grid.x == 1);
    assert(grid.y == 1);

    calcKernelLaunchConfig(33, block, grid);
    assert(block.x == 32);
    assert(block.y == 32);
    assert(grid.x == 2);
    assert(grid.y == 2);

}

void testDemoCudaLinearMatrixMemoryCudaReduceSum() {

    // Generate test data

    Metrics m;

    const GraphCode &gcq = generateTestDataGc(9);
    m = demoCudaLinearMatrixMemoryCudaReduceSum(gcq, gcq);

    assert(m.similarity == 1);
    assert(m.recommendation == 1);

    const GraphCode &gcq2 = generateTestDataGc(2040);
    m = demoCudaLinearMatrixMemoryCudaReduceSum(gcq2, gcq2);

    assert(m.similarity == 1);
    assert(m.recommendation == 1);

    GraphCode gcq3;
    std::vector<std::string> *vect = new std::vector<std::string>{"head", "body", "foot"};
    unsigned short mat1[] = {1, 1, 0, 0, 1, 0, 0, 1, 1};
    unsigned short* mat = mat1;
    gcq3.dict = vect;
    gcq3.matrix = mat;

    GraphCode gcq4;
    std::vector<std::string> *vect2 = new std::vector<std::string>{"head", "body", "foot"};
    unsigned short mat2[] = {1, 2, 0, 0, 1, 0, 0, 0, 1};
    unsigned short* mata = mat2;
    gcq4.dict = vect2;
    gcq4.matrix = mata;

    Metrics m2 = demoCudaLinearMatrixMemoryCudaReduceSum(gcq3, gcq4);

    assert(AreSame(m2.similarity, (float) 3. / 3.));
    assert(m2.recommendation == .5);
    assert(m2.inferencing == 0);


}

void testConvertGc4Cuda() {

    nlohmann::json gcq3;
    gcq3["dictionary"] = {"head", "body", "foot"};
    gcq3["matrix"] = {{1, 1, 0},
                      {0, 1, 0},
                      {0, 1, 1}};

    json dict;
    unsigned long  numberOfElements1;
    unsigned long items1;
    unsigned short *inputMatrix1;
    convertJsonGc2GcDataStructure(gcq3, dict, numberOfElements1, items1, inputMatrix1);

    assert(inputMatrix1[0] == 1);
    assert(inputMatrix1[1] == 1);
    assert(inputMatrix1[2] == 0);
    assert(inputMatrix1[3] == 0);
    assert(inputMatrix1[4] == 1);
    assert(inputMatrix1[5] == 0);
    assert(inputMatrix1[6] == 0);
    assert(inputMatrix1[7] == 1);
    assert(inputMatrix1[8] == 1);

}

void testFindLargestDivisor() {
    // Note that this loop runs till square root
    int d = findLargestDivisor(513);
    assert(d == 171);
    d = findLargestDivisor(73);
    assert(d == 1);
    d = findLargestDivisor(4000000);
    assert(d == 2000000);

}

