//
// Created by breucking on 28.12.21.
//

#include "../src/helper.h"
#include "../src/graphcode.h"
#include "../src/cudaalgorithms.cuh"

#include "testhelper.h"
#include "cpualgorithms.h"

void testCudaLinearMatrixMemoryRealTest();

int main() {
    testCudaLinearMatrixMemoryRealTest();
}

void testCudaLinearMatrixMemoryRealTest() {
    // Generate test data

    Metrics m;

    json gcq = generateTestData(9);
    m = demoCalculateSimilaritySequentialOrdered(gcq, gcq);

    assert(m.similarity == 1);
    assert(m.recommendation == 1);

    json gcq2 = generateTestData(2040);
    m = demoCalculateSimilaritySequentialOrdered(gcq2, gcq2);

    assert(m.similarity == 1);
    assert(m.recommendation == 1);


    nlohmann::json gcq3;
    gcq3["dictionary"] = { "head", "body", "foot"};
    gcq3["matrix"] = {{1,1,0}, {0,1,0}, {0,1,1}};

    nlohmann::json gcq4;
    gcq4["dictionary"] = { "head", "body", "foot"};
    gcq4["matrix"] = {{1,2,0}, {0,1,0}, {0,0,1}};

    Metrics m2 = demoCalculateSimilaritySequentialOrdered(gcq3, gcq4);

    assert(AreSame(m2.similarity,(float) 3./3.));
    assert(m2.recommendation == .5);
    assert(m2.inferencing == 0);




}