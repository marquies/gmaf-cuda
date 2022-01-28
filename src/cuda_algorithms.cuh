//
// Created by breucking on 28.12.21.
//


#ifndef GCSIM_ALGORITHMS_CUH
#define GCSIM_ALGORITHMS_CUH

#include "graphcode.h"
#include <uuid/uuid.h>
#include <cuda_runtime.h>


typedef struct GraphCode2 {
    uuid_t dict[100];
    unsigned short matrix[100 * 100];
} GraphCode2;

void convertGc2Cuda(const json &gcq, json &gc1Dictionary, int &numberOfElements, long &items,
                    unsigned short int *&inputMatrix);

void calcKernelLaunchConfig(int width, dim3 &block, dim3 &grid);


Metrics demoCudaLinearMatrixMemory(json json1, json json2);

Metrics demoCudaLinearMatrixMemory(GraphCode json1, GraphCode json2);

Metrics demoCudaLinearMatrixMemoryCudaReduceSum(GraphCode json1, GraphCode json2);

Metrics * demoCalculateGCsOnCudaWithCopy(int NUMBER_OF_GCS, unsigned int dictCounter,
                                         const unsigned short *gcMatrixData,
                                         const unsigned int *gcDictData, const unsigned int *gcMatrixOffsets,
                                         const unsigned int *gcDictOffsets, const unsigned int *gcMatrixSizes,
                                         int gcQueryPosition = 0);

Metrics *demoCalculateGCsOnCuda(int NUMBER_OF_GCS,
                                unsigned int dictCounter,
                                unsigned short *d_gcMatrixData,
                                unsigned int *d_gcDictData,
                                unsigned int *d_gcMatrixOffsets,
                                unsigned int *d_gcDictOffsets,
                                unsigned int *d_gcMatrixSizes,
                                int gcQueryPosition = 0);

__global__ void compare2(unsigned short *gcMatrixData, unsigned int *gcDictData, unsigned int *gcMatrixOffsets,
                         unsigned int *gcMatrixSizes, unsigned int *gcDictOffsets, int gcToCompare,
                         Metrics *metrics);

#endif //GCSIM_ALGORITHMS_CUH
