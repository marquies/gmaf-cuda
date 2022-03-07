//
// Created by breucking on 28.12.21.
//


#ifndef GCSIM_ALGORITHMS_CUH
#define GCSIM_ALGORITHMS_CUH

#include "graphcode.h"
#include <uuid/uuid.h>
#include <cuda_runtime.h>


typedef struct UUIDGraphCode {
    uuid_t dict[100];
    unsigned short matrix[100 * 100];
} GraphCode2;


/**
 * Calculates a launch config.
 * @param width
 * @param block
 * @param grid
 */
void calcKernelLaunchConfig(int width, dim3 &block, dim3 &grid);


Metrics demoCudaLinearMatrixMemory(json json1, json json2);

Metrics demoCudaLinearMatrixMemoryWithCopy(GraphCode gc1, GraphCode gc2);

Metrics demoCudaLinearMatrixMemoryCudaReduceSum(GraphCode json1, GraphCode json2);

/**
 * Calculates metrics for Graph Codes on CUDA. The host data will be loaded into the device memory.
 * @param numberOfGcs number of Graph Codes in the gcMatrixData
 * @param dictCounter number of Graph Codes in the gcDictData
 * @param gcMatrixData pointer to the Graph Codes matrix data as a coalesced array.
 * @param gcDictData pointer to the Graph Codes dictionary data as a coalesced array of mapped integers.
 * @param gcMatrixOffsets pointer to an index array with the offsets of gcMatricData.
 * @param gcDictOffsets pointer to an index array with the offsets of gcDictData.
 * @param gcMatrixSizes pointer to an index array with the sizes of gcMatricData.
 * @param gcQueryPosition pointer to the query Graph Code
 * @return  pointer to a return array for the calculated metrics
 */
Metrics *demoCalculateGCsOnCudaWithCopy(int numberOfGcs,
                                        unsigned int dictCounter,
                                        const unsigned short *gcMatrixData,
                                        const unsigned int *gcDictData,
                                        const unsigned int *gcMatrixOffsets,
                                        const unsigned int *gcDictOffsets,
                                        const unsigned int *gcMatrixSizes,
                                        int gcQueryPosition = 0);

/**
 * Calculates metrics for Graph Codes on CUDA. The data need to be loaded into the device memory first.
 * @param numberOfGcs number of Graph Codes in the d_gcMatrixData
 * @param dictCounter number of Graph Codes in the d_gcDictData
 * @param d_gcMatrixData device pointer to the Graph Codes matrix data as a coalesced array.
 * @param d_gcDictData device pointer to the Graph Codes dictionary data as a coalesced array of mapped integers.
 * @param d_gcMatrixOffsets device pointer to an index array with the offsets of gcMatricData.
 * @param d_gcDictOffsets device pointer to an index array with the offsets of gcDictData.
 * @param d_gcMatrixSizes device pointer to an index array with the sizes of gcMatricData.
 * @param gcQueryPosition device pointer to the query Graph Code
 * @return device pointer to a return array for the calculated metrics
 */
Metrics *demoCalculateGCsOnCuda(int numberOfGcs,
                                unsigned int dictCounter,
                                unsigned short *d_gcMatrixData,
                                unsigned int *d_gcDictData,
                                unsigned int *d_gcMatrixOffsets,
                                unsigned int *d_gcDictOffsets,
                                unsigned int *d_gcMatrixSizes,
                                int gcQueryPosition = 0);

/**
 * Calculates metrics for Graph Codes on CUDA.
 * @param gcMatrixData device pointer to the Graph Codes matrix data as a coalesced array.
 * @param gcDictData device pointer to the Graph Codes dictionary data as a coalesced array of mapped integers.
 * @param gcMatrixOffsets device pointer to an index array with the offsets of gcMatricData.
 * @param gcMatrixSizes  device pointer to an index array with the sizes of gcMatricData.
 * @param gcDictOffsets  device pointer to an index array with the offsets of gcDictData.
 * @param gcQuery position of the query Graph Code within the gcMatrixData
 * @param numberOfGcs the number of GCs
 * @param metrics device pointer to a return array for the calculated metrics
 */
__global__ void cudaGcCompute(unsigned short *gcMatrixData,
                              unsigned int *gcDictData,
                              unsigned int *gcMatrixOffsets,
                              unsigned int *gcMatrixSizes,
                              unsigned int *gcDictOffsets,
                              int gcQuery,
                              int numberOfGcs,
                              Metrics *metrics);

/**
 * Calc Metrices is a simple example to compareUUID two NxN matrices
 * @param data pinter to vectorized matrix
 * @param comparedata pointer to vectorized matrix
 * @param matrixSize dimension of the NxN matrix
 * @param numOfNonZeroEdges pointer to array to store the values for the non zero edges comparison
 * @param edgeMetricCount pointer to array to store the values for the edge metric comparison
 * @param edgeType pointer to array to store the values for the edge type metric comparison
 */
__global__ void calcMetrices(unsigned short int *data,
                             unsigned short int *comparedata,
                             unsigned long noItems,
                             unsigned int *numOfNonZeroEdges,
                             unsigned int *edgeMetricCount,
                             unsigned int *edgeType);


/**
 * Calculates metrics for Graph Codes on CUDA. The data need to be loaded into the device memory first.
 *
 * @param numberOfGcs number of Graph Codes in the d_gcMatrixData
 * @param dictCounter number of Graph Codes in the d_gcDictData
 * @param d_gcMatrixData device pointer to the Graph Codes matrix data as a coalesced array.
 * @param d_gcDictData device pointer to the Graph Codes dictionary data as a coalesced array of mapped integers.
 * @param d_gcMatrixOffsets device pointer to an index array with the offsets of gcMatricData.
 * @param d_gcDictOffsets device pointer to an index array with the offsets of gcDictData.
 * @param d_gcMatrixSizes device pointer to an index array with the sizes of gcMatricData.
 * @param gcQueryPosition device pointer to the query Graph Code
 * @return device pointer to a return array for the calculated metrics
 */
Metrics *demoCalculateGCsOnCudaAndKeepMetricsInMem(int numberOfGcs,
                                                   unsigned int dictCounter,
                                                   unsigned short *d_gcMatrixData,
                                                   unsigned int *d_gcDictData,
                                                   unsigned int *d_gcMatrixOffsets,
                                                   unsigned int *d_gcDictOffsets,
                                                   unsigned int *d_gcMatrixSizes,
                                                   int gcQueryPosition = 0);

/**
 * Sorts the metrics in the CUDA memory and returns a sorted array in host memory. Sort is descending.
 *
 * @param devicePtr pointer to the metrics array in the CUDA memory.
 * @param numberOfGcs the number of metrics.
 * @return a pointer to the sorted array in host memory.
 */
Metrics *demoSortAndRetrieveMetrics(Metrics *devicePtr, int numberOfGcs);

/**
 * CUDA device function for quicksort sorting algorithm for metrics data.
 * @param data device pointer to the metrics data
 * @param left start position
 * @param right end position
 * @param depth pstep counter arameter on how many recursions.
 */
__global__ void cdp_simple_quicksort(Metrics *data, int left, int right, int depth);

/**
 * Starts a quick sort operation on a CUDA device.
 * @param data device pointer to an array with the Metrics data to sort.
 * @param nitems number of items in the data array.
 */
void run_qsort(Metrics *data, unsigned int nitems);


#endif //GCSIM_ALGORITHMS_CUH
