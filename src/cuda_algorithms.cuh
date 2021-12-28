//
// Created by breucking on 28.12.21.
//

#include "graphcode.h"

#ifndef GCSIM_CUDA_ALGORITHMS_CUH
#define GCSIM_CUDA_ALGORITHMS_CUH

void convertGc2Cuda(const json &gcq, json &gc1Dictionary, int &numberOfElements, int &items, int *&inputMatrix);


Metrics testCudaLinearMatrixMemory(json json1, json json2);

#endif //GCSIM_CUDA_ALGORITHMS_CUH
