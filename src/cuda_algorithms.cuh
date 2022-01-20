//
// Created by breucking on 28.12.21.
//


#ifndef GCSIM_ALGORITHMS_CUH
#define GCSIM_ALGORITHMS_CUH

#include "graphcode.h"
#include <uuid/uuid.h>
#include <cuda_runtime.h>

typedef struct GraphCode {
    std::vector<std::string> *dict;
    unsigned short *matrix;
} GraphCode;

typedef struct GraphCode2 {
    uuid_t dict [100];
    unsigned short matrix[100*100];
} GraphCode2;

void convertGc2Cuda(const json &gcq, json &gc1Dictionary, int &numberOfElements, long &items, unsigned short int *&inputMatrix);
void calcKernelLaunchConfig(int width, dim3 &block, dim3 &grid);


Metrics demoCudaLinearMatrixMemory(json json1, json json2);

Metrics demoCudaLinearMatrixMemory(GraphCode json1, GraphCode json2);

Metrics demoCudaLinearMatrixMemoryCudaReduceSum(GraphCode json1, GraphCode json2);

Metrics demoCalculateSimilaritySequentialOrdered(json gc1, json gc2);

Metrics demoCalculateSimilaritySequentialOrdered(GraphCode gc1, GraphCode gc2);


#endif //GCSIM_ALGORITHMS_CUH
