//
// Created by breucking on 28.12.21.
//


#ifndef GCSIM_ALGORITHMS_CUH
#define GCSIM_ALGORITHMS_CUH

#include "graphcode.h"

void convertGc2Cuda(const json &gcq, json &gc1Dictionary, int &numberOfElements, int &items, int *&inputMatrix);


Metrics testCudaLinearMatrixMemory(json json1, json json2);
Metrics calculateSimilaritySequentialOrdered(json gc1, json gc2);


#endif //GCSIM_ALGORITHMS_CUH
