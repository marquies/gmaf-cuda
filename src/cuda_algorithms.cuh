//
// Created by breucking on 28.12.21.
//


#ifndef GCSIM_ALGORITHMS_CUH
#define GCSIM_ALGORITHMS_CUH

#include "graphcode.h"

typedef struct GraphCode {
    std::vector<std::string> *dict;
    int *matrix;
} GraphCode;

void convertGc2Cuda(const json &gcq, json &gc1Dictionary, int &numberOfElements, int &items, int *&inputMatrix);


Metrics testCudaLinearMatrixMemory(json json1, json json2);

Metrics testCudaLinearMatrixMemory(GraphCode json1, GraphCode json2);

Metrics calculateSimilaritySequentialOrdered(json gc1, json gc2);

Metrics calculateSimilaritySequentialOrdered(GraphCode gc1, GraphCode gc2);


#endif //GCSIM_ALGORITHMS_CUH
