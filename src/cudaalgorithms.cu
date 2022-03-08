//
// Created by breucking on 28.12.21.
//

#include <time.h>
#include <c++/9/chrono>
#include <string>
#include "cudaalgorithms.cuh"

#include <cstdlib>

#include <cuda_runtime.h>

#include <iostream>

#include <math.h>

#include "graphcode.h"
#include "cudahelper.cuh"
#include "helper.h"

#include "reduce.cuh"

#include <cuda_profiler_api.h>
#include "cuda-quicksort/CUDA-Quicksort.h"

#define MAX_DEPTH       24
#define INSERTION_SORT  32


void checkNxNConstraint(const GraphCode &gc1, const GraphCode &gc2);

void demoCalculateGCsOnCuda(int NUMBER_OF_GCS, unsigned int dictCounter, const unsigned short *gcMatrixData,
                            const unsigned int *gcDictData, const unsigned int *gcMatrixOffsets,
                            const unsigned int *gcDictOffsets, const unsigned int *gcMatrixSizes);


__global__ void
calcMetrices(unsigned short int *data, unsigned short int *comparedata, unsigned long noItems,
             unsigned int *numOfNonZeroEdges, unsigned int *edgeMetricCount, unsigned int *edgeType) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int /*offset*/ tid = x + y * blockDim.x * gridDim.x;

    numOfNonZeroEdges[tid] = 0;
    edgeMetricCount[tid] = 0;
    edgeType[tid] = 0;

    if (tid > noItems) {
        return;
    }

    if (x != y && data[tid] != 0) {
        numOfNonZeroEdges[tid] = 1;
        if (comparedata[tid] != 0) {
            edgeMetricCount[tid] = 1;
            if (data[tid] == comparedata[tid]) {
                edgeType[tid] = 1;
            }
        }
    }

}


void print_d_array(unsigned int *d_array, int len) {
    int *h_array = new int[len];
    HANDLE_ERROR(cudaMemcpy(h_array, d_array, sizeof(int) * len, cudaMemcpyDeviceToHost));
    for (int i = 0; i < len; ++i) {
        std::cout << h_array[i] << " ";
    }
    std::cout << std::endl;

    delete[] h_array;
}

Metrics demoCudaLinearMatrixMemoryCudaReduceSum(GraphCode json1, GraphCode json2) {
    checkNxNConstraint(json1, json2);
    cudaProfilerStart();
    int items1 = pow(json1.dict->size(), 2);

    // Prep for cuda


    unsigned short int *gpu_inputMatrix1;
    unsigned short int *gpu_inputMatrix2;
    unsigned int *darr_edge_metric_count;
    unsigned int *darr_num_of_non_zero_edges;
    unsigned int *darr_edge_type;

    auto start = std::chrono::system_clock::now();

    HANDLE_ERROR(cudaMalloc((void **) &gpu_inputMatrix1, sizeof(unsigned short int) * items1));
    HANDLE_ERROR(cudaMalloc((void **) &gpu_inputMatrix2, sizeof(unsigned short int) * items1));
    HANDLE_ERROR(cudaMalloc((void **) &darr_num_of_non_zero_edges, sizeof(unsigned int) * items1));
    HANDLE_ERROR(cudaMalloc((void **) &darr_edge_metric_count, sizeof(unsigned int) * items1));
    HANDLE_ERROR(cudaMalloc((void **) &darr_edge_type, sizeof(unsigned int) * items1));

    // Transfer data from host to device memory
    HANDLE_ERROR(
            cudaMemcpy(gpu_inputMatrix1, json1.matrix, sizeof(unsigned short int) * items1, cudaMemcpyHostToDevice));
    HANDLE_ERROR(
            cudaMemcpy(gpu_inputMatrix2, json2.matrix, sizeof(unsigned short int) * items1, cudaMemcpyHostToDevice));

    dim3 block;
    dim3 grid;

    int width = json1.dict->size();
    calcKernelLaunchConfig(width, block, grid);

    //HANDLE_ERROR(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calcMetrices, 0, 0));

    // calculation
    auto loaded = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = loaded - start;

    if (G_DEBUG)
        std::cout << "elapsed time: " << elapsed_seconds.count()
                  << std::endl;

    //int q = sqrt((float) items1);
    calcMetrices<<<grid, block>>>(gpu_inputMatrix1, gpu_inputMatrix2, items1,
                                  darr_num_of_non_zero_edges,
                                  darr_edge_metric_count,
                                  darr_edge_type
    );



    //printf("CUDA error %s\n",cudaGetErrorString(cudaPeekAtLastError()));
    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    auto end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    if (G_DEBUG) {
        std::cout << "finished computation at " << std::ctime(&end_time)
                  << "elapsed time: " << elapsed_seconds.count() << "s\n";
        elapsed_seconds = end - loaded;
        std::cout << "Computation time: " << elapsed_seconds.count() << "s\n";
    }
    auto mem_start = std::chrono::system_clock::now();

    unsigned int gts_edge_metric_count = gpu_sum_reduce(darr_edge_metric_count, items1);
    unsigned int gts_edge_type = gpu_sum_reduce(darr_edge_type, items1);
    unsigned int gts_num_of_non_zero_edges = gpu_sum_reduce(darr_num_of_non_zero_edges, items1);

    if (G_DEBUG) {
        std::cout << gts_num_of_non_zero_edges << std::endl;
    }

    auto mem_end = std::chrono::system_clock::now();
    if (G_DEBUG) {
        elapsed_seconds = mem_end - mem_start;
        std::cout << "Sum Reduce time: " << elapsed_seconds.count() << "s\n";
    }


    HANDLE_ERROR(cudaFree(gpu_inputMatrix1));
    HANDLE_ERROR(cudaFree(gpu_inputMatrix2));
    HANDLE_ERROR(cudaFree(darr_edge_metric_count));
    HANDLE_ERROR(cudaFree(darr_num_of_non_zero_edges));
    HANDLE_ERROR(cudaFree(darr_edge_type));

    // Result reduction
    int num_of_non_zero_edges = gts_num_of_non_zero_edges;
    int edge_metric_count = gts_edge_metric_count;
    int edgeTypeCount = gts_edge_type;

    std::string gc1Dict[json1.dict->size()];

    int sim = 0;
    int n = 0;
    for (const auto &item: *json1.dict) {
        std::string str = item;
        gc1Dict[n++] = str;


        for (const auto &item2: *json2.dict) {
            if (str == item2) {
                //std::cout << "Match" << std::endl;
                sim++;
            }
        }
    }

    // Calculate metrices
    float node_metric = (float) sim / (float) json1.dict->size();


    float edge_metric = 0.0;
    if (num_of_non_zero_edges > 0)
        edge_metric = (float) edge_metric_count / (float) num_of_non_zero_edges;

    float edge_type_metric = 0.0;
    if (edge_metric_count > 0)
        edge_type_metric = (float) edgeTypeCount / (float) edge_metric_count;

    if (G_DEBUG)
        std::cout << "Similarity: " << " value: " << node_metric << std::endl;
    if (G_DEBUG)
        std::cout << "Recommendation: " << " value: " << edge_metric << std::endl;


    if (G_DEBUG) {
        auto metrics_end = std::chrono::system_clock::now();
        elapsed_seconds = metrics_end - mem_end;
        std::cout << "Metrics Management time: " << elapsed_seconds.count() << "s\n";
    }

    Metrics m;
    m.similarity = node_metric;
    m.recommendation = edge_metric;
    m.inferencing = edge_type_metric;

    cudaProfilerStop();
    return m;

}

void checkNxNConstraint(const GraphCode &gc1, const GraphCode &gc2) {

    if (gc1.dict->size() != gc2.dict->size()) {
        std::cout << "Graph Codes need to have same size" << std::endl;
        exit(71);
    }
    bool result = std::equal(gc1.dict->begin(), gc1.dict->end(), gc2.dict->begin());

    if (!result) {
        std::cout << "Graph Codes need to have same dict elements" << std::endl;
        exit(71);
    }
}


Metrics demoCudaLinearMatrixMemoryWithCopy(GraphCode gc1, GraphCode gc2) {
    checkNxNConstraint(gc1, gc2);

    int items1 = pow(gc1.dict->size(), 2);

    // Prep for cuda


    unsigned short int *gpu_inputMatrix1;
    unsigned short int *gpu_inputMatrix2;
    unsigned int *darr_edge_metric_count;
    unsigned int *darr_num_of_non_zero_edges;
    unsigned int *darr_edge_type;

    auto start = std::chrono::system_clock::now();

    HANDLE_ERROR(cudaMalloc((void **) &gpu_inputMatrix1, sizeof(unsigned short int) * items1));
    HANDLE_ERROR(cudaMalloc((void **) &gpu_inputMatrix2, sizeof(unsigned short int) * items1));
    HANDLE_ERROR(cudaMalloc((void **) &darr_num_of_non_zero_edges, sizeof(unsigned int) * items1));
    HANDLE_ERROR(cudaMalloc((void **) &darr_edge_metric_count, sizeof(unsigned int) * items1));
    HANDLE_ERROR(cudaMalloc((void **) &darr_edge_type, sizeof(unsigned int) * items1));

    // Transfer data from host to device memory
    HANDLE_ERROR(
            cudaMemcpy(gpu_inputMatrix1, gc1.matrix, sizeof(unsigned short int) * items1, cudaMemcpyHostToDevice));
    HANDLE_ERROR(
            cudaMemcpy(gpu_inputMatrix2, gc2.matrix, sizeof(unsigned short int) * items1, cudaMemcpyHostToDevice));

    dim3 block;
    dim3 grid;

    calcKernelLaunchConfig(gc1.dict->size(), block, grid);

    // calculation
    auto loaded = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = loaded - start;

    if (G_DEBUG)
        std::cout << "elapsed time: " << elapsed_seconds.count()
                  << std::endl;

    calcMetrices<<<grid, block>>>(gpu_inputMatrix1,
                                  gpu_inputMatrix2,
                                  items1,
                                  darr_num_of_non_zero_edges,
                                  darr_edge_metric_count,
                                  darr_edge_type
    );

    HANDLE_ERROR(cudaPeekAtLastError());

    auto end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    if (G_DEBUG) {
        std::cout << "finished computation at " << std::ctime(&end_time)
                  << "elapsed time: " << elapsed_seconds.count() << "s\n";
        elapsed_seconds = end - loaded;
        std::cout << "Computation time: " << elapsed_seconds.count() << "s\n";
    }


    int *arrEdgeTypeMetricCount;
    HANDLE_ERROR(cudaMallocHost((void **) &arrEdgeTypeMetricCount, sizeof(int) * items1));

    int *arr_edge_metric_count;
    HANDLE_ERROR(cudaMallocHost((void **) &arr_edge_metric_count, sizeof(int) * items1));

    int *arr_num_of_non_zero_edges;
    HANDLE_ERROR(cudaMallocHost((void **) &arr_num_of_non_zero_edges, sizeof(int) * items1));


    HANDLE_ERROR(cudaMemcpy(arr_num_of_non_zero_edges, darr_num_of_non_zero_edges, sizeof(int) * items1,
                            cudaMemcpyDeviceToHost));

    HANDLE_ERROR(
            cudaMemcpy(arr_edge_metric_count, darr_edge_metric_count, sizeof(int) * items1, cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaMemcpy(arrEdgeTypeMetricCount, darr_edge_type, sizeof(int) * items1, cudaMemcpyDeviceToHost));


    auto mem_end = std::chrono::system_clock::now();
    if (G_DEBUG) {
        elapsed_seconds = mem_end - end;
        std::cout << "Memory Management time: " << elapsed_seconds.count() << "s\n";
    }


    HANDLE_ERROR(cudaFree(gpu_inputMatrix1));
    HANDLE_ERROR(cudaFree(gpu_inputMatrix2));
    HANDLE_ERROR(cudaFree(darr_edge_metric_count));
    HANDLE_ERROR(cudaFree(darr_num_of_non_zero_edges));
    HANDLE_ERROR(cudaFree(darr_edge_type));

    // Result reduction
    int num_of_non_zero_edges = 0;
    int edge_metric_count = 0;
    int edgeTypeCount = 0;
    for (int i = 0; i < items1; i++) {
        if (arr_edge_metric_count[i] == 1) {
            edge_metric_count++;
        }
        if (arr_num_of_non_zero_edges[i] == 1) {
            num_of_non_zero_edges++;
        }
        if (arrEdgeTypeMetricCount[i] == 1) {
            edgeTypeCount++;
        }
    }

    std::string gc1Dict[gc1.dict->size()];

    int sim = 0;
    int n = 0;
    for (const auto &item: *gc1.dict) {
        //std::cout << item.value() << "\n";
        std::string str = item;
        gc1Dict[n++] = str;


        for (const auto &item2: *gc2.dict) {
            if (str == item2) {
                //std::cout << "Match" << std::endl;
                sim++;
            }
        }
    }

    // Calculate metrices
    float node_metric = (float) sim / (float) gc1.dict->size();


    float edge_metric = 0.0;
    if (num_of_non_zero_edges > 0)
        edge_metric = (float) edge_metric_count / (float) num_of_non_zero_edges;

    float edge_type_metric = 0.0;
    if (edge_metric_count > 0)
        edge_type_metric = (float) edgeTypeCount / (float) edge_metric_count;

    if (G_DEBUG)
        std::cout << "Similarity: " << " value: " << node_metric << std::endl;
    if (G_DEBUG)
        std::cout << "Recommendation: " << " value: " << edge_metric << std::endl;


    if (G_DEBUG) {
        auto metrics_end = std::chrono::system_clock::now();
        elapsed_seconds = metrics_end - mem_end;
        std::cout << "Metrics Management time: " << elapsed_seconds.count() << "s\n";
    }

    Metrics m;
    m.similarity = node_metric;
    m.recommendation = edge_metric;
    m.inferencing = edge_type_metric;


    cudaFreeHost(arrEdgeTypeMetricCount);
    cudaFreeHost(arr_num_of_non_zero_edges);
    cudaFreeHost(arr_edge_metric_count);

    return m;

}


Metrics demoCudaLinearMatrixMemory(json json1, json json2) {
    //checkNxNConstraint(json1, json2);

    json gc1Dictionary;
    unsigned long numberOfElements1;
    unsigned long items1;
    unsigned short int *inputMatrix1;

    convertJsonGc2GcDataStructure(json1, gc1Dictionary, numberOfElements1, items1, inputMatrix1);


    json gc2Dictionary;
    unsigned long numberOfElements2;
    unsigned long items2;
    unsigned short int *inputMatrix2;
    convertJsonGc2GcDataStructure(json2, gc2Dictionary, numberOfElements2, items2, inputMatrix2);

    // Prep for cuda


    unsigned short int *gpu_inputMatrix1;
    unsigned short int *gpu_inputMatrix2;
    unsigned int *darr_edge_metric_count;
    unsigned int *darr_num_of_non_zero_edges;
    unsigned int *darr_edge_type;

    auto start = std::chrono::system_clock::now();

    HANDLE_ERROR(cudaMalloc((void **) &gpu_inputMatrix1, sizeof(unsigned short int) * items1));
    HANDLE_ERROR(cudaMalloc((void **) &gpu_inputMatrix2, sizeof(unsigned short int) * items1));
    HANDLE_ERROR(cudaMalloc((void **) &darr_num_of_non_zero_edges, sizeof(unsigned int) * items1));
    HANDLE_ERROR(cudaMalloc((void **) &darr_edge_metric_count, sizeof(unsigned int) * items1));
    HANDLE_ERROR(cudaMalloc((void **) &darr_edge_type, sizeof(unsigned int) * items1));

    // Transfer data from host to device memory
    HANDLE_ERROR(
            cudaMemcpy(gpu_inputMatrix1, inputMatrix1, sizeof(unsigned short int) * items1, cudaMemcpyHostToDevice));
    HANDLE_ERROR(
            cudaMemcpy(gpu_inputMatrix2, inputMatrix2, sizeof(unsigned short int) * items1, cudaMemcpyHostToDevice));


    dim3 block;
    dim3 grid;

    int width = numberOfElements1;

    calcKernelLaunchConfig(width, block, grid);

    // calculation
    auto loaded = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = loaded - start;

    if (G_DEBUG)
        std::cout << "elapsed time: " << elapsed_seconds.count()
                  << std::endl;

    calcMetrices<<<grid, block>>>(gpu_inputMatrix1, gpu_inputMatrix2, items1,
                                  darr_num_of_non_zero_edges,
                                  darr_edge_metric_count,
                                  darr_edge_type
    );


    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
    auto end = std::chrono::system_clock::now();

    elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    if (G_DEBUG)
        std::cout << "finished computation at " << std::ctime(&end_time)
                  << "elapsed time: " << elapsed_seconds.count() << "s\n";
    elapsed_seconds = end - loaded;
    if (G_DEBUG)
        std::cout << "Computation time: " << elapsed_seconds.count() << "s\n";

    // Retrieve results
    int *arrEdgeTypeMetricCount = (int *) malloc(sizeof(int) * items1);
    int *arr_edge_metric_count = (int *) malloc(sizeof(int) * items1);
    int *arr_num_of_non_zero_edges = (int *) malloc(sizeof(int) * items1);

    HANDLE_ERROR(cudaMemcpy(arr_num_of_non_zero_edges, darr_num_of_non_zero_edges, sizeof(int) * items1,
                            cudaMemcpyDeviceToHost));
    HANDLE_ERROR(
            cudaMemcpy(arr_edge_metric_count, darr_edge_metric_count, sizeof(int) * items1, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(arrEdgeTypeMetricCount, darr_edge_type, sizeof(int) * items1, cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(gpu_inputMatrix1));
    HANDLE_ERROR(cudaFree(gpu_inputMatrix2));
    HANDLE_ERROR(cudaFree(darr_edge_metric_count));
    HANDLE_ERROR(cudaFree(darr_num_of_non_zero_edges));
    HANDLE_ERROR(cudaFree(darr_edge_type));


    free(inputMatrix1);
    free(inputMatrix2);

    // Result reduction
    int num_of_non_zero_edges = 0;
    int edge_metric_count = 0;
    int edgeTypeCount = 0;
    for (int i = 0; i < items1; i++) {
        if (arr_edge_metric_count[i] == 1) {
            edge_metric_count++;
        }
        if (arr_num_of_non_zero_edges[i] == 1) {
            num_of_non_zero_edges++;
        }
        if (arrEdgeTypeMetricCount[i] == 1) {
            edgeTypeCount++;
        }
    }

    std::string gc1Dict[gc1Dictionary.size()];

    int sim = 0;
    int n = 0;
    for (const auto &item: gc1Dictionary.items()) {
        std::string str = item.value().get<std::string>();
        gc1Dict[n++] = str;


        for (const auto &item2: gc2Dictionary.items()) {
            if (str == item2.value()) {
                sim++;
            }
        }
    }

    // Calculate metrices
    float node_metric = (float) sim / (float) gc1Dictionary.size();


    float edge_metric = 0.0;
    if (num_of_non_zero_edges > 0)
        edge_metric = (float) edge_metric_count / (float) num_of_non_zero_edges;

    float edge_type_metric = 0.0;
    if (edge_metric_count > 0)
        edge_type_metric = (float) edgeTypeCount / (float) edge_metric_count;

    if (G_DEBUG)
        std::cout << "Similarity: " << " value: " << node_metric << std::endl;
    if (G_DEBUG)
        std::cout << "Recommendation: " << " value: " << edge_metric << std::endl;

    Metrics m;
    m.similarity = node_metric;
    m.recommendation = edge_metric;
    m.inferencing = edge_type_metric;

    free(arrEdgeTypeMetricCount);
    free(arr_num_of_non_zero_edges);
    free(arr_edge_metric_count);

    return m;

}

void calcKernelLaunchConfig(int width, dim3 &block, dim3 &grid) {

    if (width > 32) {
        int gridSize = ceil(width / 32.0);

        block = dim3(32, 32, 1);
        grid = dim3(gridSize, gridSize, 1);

    } else {

        block = dim3(width, width);
        grid = (1);
    }
}


__global__ void cudaGcCompute(unsigned short *gcMatrixData, unsigned int *gcDictData, unsigned int *gcMatrixOffsets,
                              unsigned int *gcMatrixSizes, unsigned int *gcDictOffsets, int gcQuery, int numberOfGcs,
                              Metrics *metrics) {
//    int index = blockIdx.x;
    //int /*offset*/ tid = x + y * blockDim.x * gridDim.x;
    //int x = threadIdx.x + blockIdx.x * blockDim.x
    unsigned int index =  threadIdx.x + blockIdx.x * blockDim.x ;
    if (index >= numberOfGcs)
        return;
//    int /*offset*/ tid = x ;

    unsigned int gc1 = gcQuery;
    unsigned int gc2 = index;

    int sim = 0;
    int elementsGc1 = sqrtf((float) gcMatrixSizes[gc1]);
    int elementsGc2 = sqrtf((float) gcMatrixSizes[gc2]);

    unsigned int off1 = gcDictOffsets[gc1];
    unsigned int off2 = gcDictOffsets[gc2];
    for (int i = 0; i < elementsGc1; i++) {
        for (int j = 0; j < elementsGc2; j++) {

            unsigned int d1 = gcDictData[off1 + i];
            unsigned int d2 = gcDictData[off2 + j];
            if (d1 == d2) {
                sim++;
//                break;
            }
        }
    }

    int num_of_non_zero_edges = 0;
    int edge_metric_count = 0;
    int edge_type = 0;


    for (int i = 0; i < elementsGc1; i++) {
        for (int j = 0; j < elementsGc1; j++) {

            if (i != j && gcMatrixData[gcMatrixOffsets[gc1] + i * elementsGc1 + j] != 0) {
                num_of_non_zero_edges++;

                int position1 = i;
                int position2 = j;
                if (position1 == -1 || position2 == -1) {
                    continue;
                }

//                std::basic_string<char> &v1 = gcQuery.dict->at(i);
                unsigned int vocVal1 = gcDictData[gcDictOffsets[gc1] + i];

                
//                std::basic_string<char> &v2 = gcQuery.dict->at(j);
                unsigned int vocVal2 = gcDictData[gcDictOffsets[gc1] + j];

                long transX = -1;
                long transY = -1;
                unsigned endposition = gcDictOffsets[gc2] + elementsGc2;
                for (int k = gcDictOffsets[gc2]; k <= endposition; k++){
                    if (gcDictData[k] == vocVal1) {
                        transX = k - gcDictOffsets[gc2];
                    }
                    if (gcDictData[k] == vocVal2) {
                        transY = k - gcDictOffsets[gc2];
                    }
                    if (transX != -1 && transY != -1) {
                        break;
                    }
                }

                if (transX == -1 || transY == -1) {
                    continue;
                }

                unsigned short edge = gcMatrixData[gcMatrixOffsets[gc2] + (transX) * elementsGc2 +
                                                   (transY)];


//                int edge = gcMatrixData[gcMatrixOffsets[gc2] + position1 * elements +
//                                        position2];
                if (edge != 0) {
                    edge_metric_count++;
                }
                if (edge == gcMatrixData[gcMatrixOffsets[gc1] + i * elementsGc1 + j]) {
                    edge_type++;
                }

            }
        }
    }
    metrics[index].similarity = 0.0;
    metrics[index].recommendation = 0.0;
    metrics[index].inferencing = 0.0;
    metrics[index].similarity = (float) sim / (float) elementsGc1;
    metrics[index].idx = index;
    if (num_of_non_zero_edges > 0) {
        /*edge_metric*/ metrics[index].recommendation = (float) edge_metric_count / (float) num_of_non_zero_edges;
    }
    if (edge_metric_count > 0) {
        /*edge_type_metric*/ metrics[index].inferencing = (float) edge_type / (float) edge_metric_count;
    }
    metrics[index].compareValue = metrics[index].similarity * 100000.0f + metrics[index].recommendation * 100.0f +
            metrics[index].inferencing;
}

Metrics *demoCalculateGCsOnCuda(int numberOfGcs,
                                unsigned int dictCounter,
                                unsigned short *d_gcMatrixData,
                                unsigned int *d_gcDictData,
                                unsigned int *d_gcMatrixOffsets,
                                unsigned int *d_gcDictOffsets,
                                unsigned int *d_gcMatrixSizes,
                                int gcQueryPosition) {
    Metrics *d_result;

    HANDLE_ERROR(cudaMalloc((void **) &d_result, numberOfGcs * sizeof(Metrics)));


    int gridDim = ceil((float)numberOfGcs/1024.0);

    int block = (numberOfGcs < 1024) ? numberOfGcs : 1024;

    cudaGcCompute<<<gridDim, block>>>(d_gcMatrixData,
//    compare2<<<numberOfGcs, 1>>>(d_gcMatrixData,
                                      d_gcDictData,
                                      d_gcMatrixOffsets,
                                      d_gcMatrixSizes,
                                      d_gcDictOffsets,
                                      gcQueryPosition,
                                      numberOfGcs,
                                      d_result);

    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    Metrics *result = (Metrics *) malloc(numberOfGcs * sizeof(Metrics));
    HANDLE_ERROR(cudaMemcpy(result, d_result, numberOfGcs * sizeof(Metrics), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(d_result));

    return result;
}

Metrics *demoCalculateGCsOnCudaAndKeepMetricsInMem(int numberOfGcs,
                                unsigned int dictCounter,
                                unsigned short *d_gcMatrixData,
                                unsigned int *d_gcDictData,
                                unsigned int *d_gcMatrixOffsets,
                                unsigned int *d_gcDictOffsets,
                                unsigned int *d_gcMatrixSizes,
                                int gcQueryPosition ) {
    Metrics *d_result;

    HANDLE_ERROR(cudaMalloc((void **) &d_result, numberOfGcs * sizeof(Metrics)));


    int gridDim = ceil((float)numberOfGcs/1024.0);

    cudaGcCompute<<<gridDim, 1024>>>(d_gcMatrixData,
//    compare2<<<numberOfGcs, 1>>>(d_gcMatrixData,
                                     d_gcDictData,
                                     d_gcMatrixOffsets,
                                     d_gcMatrixSizes,
                                     d_gcDictOffsets,
                                     gcQueryPosition,
                                     numberOfGcs,
                                     d_result);

    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    return d_result;
}

Metrics *demoSortAndRetrieveMetrics(Metrics* devicePtr, int numberOfGcs) {
    Metrics *result = (Metrics *) malloc(numberOfGcs * sizeof(Metrics));

    int Device = 0;

    sortOnMem(devicePtr, result, numberOfGcs, 128, Device);

    Metrics *pointer1 = result,
    *pointer2 = result + numberOfGcs - 1;

    //Reverse sort
    while (pointer1 < pointer2) {
        swap(pointer1, pointer2);
        pointer1++;
        pointer2--;
    }

    return result;
}


Metrics *demoCalculateGCsOnCudaWithCopy(int numberOfGcs,
                                        unsigned int dictCounter,
                                        const unsigned short *gcMatrixData,
                                        const unsigned int *gcDictData,
                                        const unsigned int *gcMatrixOffsets,
                                        const unsigned int *gcDictOffsets,
                                        const unsigned int *gcMatrixSizes,
                                        int gcQueryPosition) {
    //------------
    // CUDA prep
    //------------


    unsigned short *d_gcMatrixData;
    unsigned int *d_gcDictData;
    unsigned int *d_gcMatrixOffsets;
    unsigned int *d_gcMatrixSizes;
    unsigned int *d_gcDictOffsets;
    Metrics *d_result;

    long md_size = 0;
    for (int i = 0; i < numberOfGcs; i++) {
        md_size += gcMatrixSizes[i];
    }// ;



    HANDLE_ERROR(cudaMalloc((void **) &d_gcMatrixData, md_size * sizeof(unsigned short)));
    HANDLE_ERROR(cudaMalloc((void **) &d_gcDictData, dictCounter * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void **) &d_gcMatrixOffsets, numberOfGcs * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void **) &d_gcMatrixSizes, numberOfGcs * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void **) &d_gcDictOffsets, numberOfGcs * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void **) &d_result, numberOfGcs * sizeof(Metrics)));


    HANDLE_ERROR(
            cudaMemcpy(d_gcMatrixData, gcMatrixData, md_size * sizeof(unsigned short), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_gcDictData, gcDictData, dictCounter * sizeof(unsigned int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_gcMatrixOffsets, gcMatrixOffsets, numberOfGcs * sizeof(unsigned int),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(
            cudaMemcpy(d_gcMatrixSizes, gcMatrixSizes, numberOfGcs * sizeof(unsigned int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(
            cudaMemcpy(d_gcDictOffsets, gcDictOffsets, numberOfGcs * sizeof(unsigned int), cudaMemcpyHostToDevice));

    auto start = std::chrono::system_clock::now();

    cudaGcCompute<<<numberOfGcs, 1>>>(d_gcMatrixData,
                                      d_gcDictData,
                                      d_gcMatrixOffsets,
                                      d_gcMatrixSizes,
                                      d_gcDictOffsets,
                                      gcQueryPosition,
                                      numberOfGcs,
                                      d_result);

    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    time_t end_time = std::chrono::system_clock::to_time_t(end);

    if (G_DEBUG) {
        std::cout << "finished CUDA computation at " << ctime(&end_time)
                  << "elapsed time: " << elapsed_seconds.count() << "s\n";
    }


    Metrics *result = (Metrics *) malloc(numberOfGcs * sizeof(Metrics));
    HANDLE_ERROR(cudaMemcpy(result, d_result, numberOfGcs * sizeof(Metrics), cudaMemcpyDeviceToHost));


//    for (int i = 0; i < numberOfGcs; i++) {


//        std::cout << "Result (" << i << ") "
//                  << "Similarity " << result[i].similarity << "; "
//                  << "Recommendation " << result[i].recommendation << "; "
//                  << "Inference " << result[i].inferencing << "; "
//                  << std::endl;

//        assert(result[i].similarity == 1);
//        assert(result[i].recommendation == 0.5);
//        assert(result[i].inferencing == 0);
//    }
    HANDLE_ERROR(cudaFree(d_gcMatrixData));
    HANDLE_ERROR(cudaFree(d_gcDictData));
    HANDLE_ERROR(cudaFree(d_gcMatrixOffsets));
    HANDLE_ERROR(cudaFree(d_gcMatrixSizes));
    HANDLE_ERROR(cudaFree(d_result));
    return result;
}




__device__ float dcompare(Metrics positionA, Metrics positionB) {
    float a = positionA.similarity * 100000.0f + positionA.recommendation * 100.0f +
              positionA.inferencing;


    float b = positionB.similarity * 100000.0f + positionB.recommendation * 100.0f +
              positionB.inferencing;

    return a-b;
}

////////////////////////////////////////////////////////////////////////////////
// Selection sort used when depth gets too big or the number of elements drops
// below a threshold.
////////////////////////////////////////////////////////////////////////////////
__device__ void selection_sort(Metrics *data, int left, int right) {
    for (int i = left; i <= right; ++i) {
        Metrics min_val = data[i];
        int min_idx = i;

        // Find the smallest value in the range [left, right].
        for (int j = i + 1; j <= right; ++j) {
            Metrics val_j = data[j];

            if (dcompare(val_j, min_val) > 0) {
                min_idx = j;
                min_val = val_j;
            }
        }

        // Swap the values.
        if (i != min_idx) {
            data[min_idx] = data[i];
            data[i] = min_val;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Very basic quicksort algorithm, recursively launching the next level.
////////////////////////////////////////////////////////////////////////////////
__global__ void cdp_simple_quicksort(Metrics *data, int left, int right, int depth) {
    // If we're too deep or there are few elements left, we use an insertion sort...
    if (depth >= MAX_DEPTH || right - left <= INSERTION_SORT) {
        selection_sort(data, left, right);
        return;
    }

    Metrics *lptr = data + left;
    Metrics *rptr = data + right;
    Metrics pivot = data[(left + right) / 2];

    // Do the partitioning.
    while (lptr <= rptr) {
        // Find the next left- and right-hand values to swap
        Metrics lval = *lptr;
        Metrics rval = *rptr;

        // Move the left pointer as long as the pointed element is smaller than the pivot.
        while (dcompare(lval, pivot) > 0) {
            lptr++;
            lval = *lptr;
        }

        // Move the right pointer as long as the pointed element is larger than the pivot.
        while (dcompare(rval, pivot) < 0) {
            rptr--;
            rval = *rptr;
        }

        // If the swap points are valid, do the swap!
        if (lptr <= rptr) {
            *lptr++ = rval;
            *rptr-- = lval;
        }
    }

    // Now the recursive part
    int nright = rptr - data;
    int nleft = lptr - data;

    // Launch a new block to sort the left part.
    if (left < (rptr - data)) {
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        cdp_simple_quicksort<<< 1, 1, 0, s >>>(data, left, nright, depth + 1);
        cudaStreamDestroy(s);
    }

    // Launch a new block to sort the right part.
    if ((lptr - data) < right) {
        cudaStream_t s1;
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        cdp_simple_quicksort<<< 1, 1, 0, s1 >>>(data, nleft, right, depth + 1);
        cudaStreamDestroy(s1);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Call the quicksort kernel from the host.
////////////////////////////////////////////////////////////////////////////////
void run_qsort(Metrics *data, unsigned int nitems) {
    // Prepare CDP for the max depth 'MAX_DEPTH'.
    HANDLE_ERROR(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH));

    // Launch on device
    int left = 0;
    int right = nitems - 1;
    std::cout << "Launching kernel on the GPU" << std::endl;
    cdp_simple_quicksort<<< 1, 1 >>>(data, left, right, 0);
    HANDLE_ERROR(cudaDeviceSynchronize());
}

