//
// Created by breucking on 28.12.21.
//


#ifndef GCSIM_ALGORITHMS_CUH
#define GCSIM_ALGORITHMS_CUH

#include "graphcode.h"

typedef struct GraphCode {
    std::vector<std::string> *dict;
    unsigned short *matrix;
} GraphCode;

void convertGc2Cuda(const json &gcq, json &gc1Dictionary, int &numberOfElements, long &items, unsigned short int *&inputMatrix);
void calcKernelLaunchConfig(int width, dim3 &block, dim3 &grid);


Metrics demoCudaLinearMatrixMemory(json json1, json json2);

Metrics demoCudaLinearMatrixMemory(GraphCode json1, GraphCode json2);

Metrics demoCudaLinearMatrixMemoryCudaReduceSum(GraphCode json1, GraphCode json2);

Metrics demoCalculateSimilaritySequentialOrdered(json gc1, json gc2);

Metrics demoCalculateSimilaritySequentialOrdered(GraphCode gc1, GraphCode gc2);


#endif //GCSIM_ALGORITHMS_CUH
