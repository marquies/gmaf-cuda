//
// Created by breucking on 28.12.21.
//

#include "cuda_algorithms.cuh"

#include <cstdlib>

#include <cuda_runtime.h>

#include <iostream>

#include <math.h>
#include <chrono>
#include <ctime>

#include "graphcode.h"
#include "cudahelper.cuh"
#include "helper.h"



/**
 * Calc Metrices is a simple example to compare two NxN matrices
 * @param data pinter to vectorized matrix
 * @param comparedata pointer to vectorized matrix
 * @param matrixSize dimension of the NxN matrix
 * @param numOfNonZeroEdges pointer to array to store the values for the non zero edges comparison
 * @param edgeMetricCount pointer to array to store the values for the edge metric comparison
 * @param edgeType pointer to array to store the values for the edge type metric comparison
 */

__global__ void
calcMetrices(int *data, int *comparedata, unsigned long matrixSize,
             int *numOfNonZeroEdges, int *edgeMetricCount, int *edgeType) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int /*offset*/ tid = x + y * blockDim.x * gridDim.x;

    int q = sqrt((float) matrixSize);

    numOfNonZeroEdges[tid] = 0;
    edgeMetricCount[tid] = 0;
    edgeType[tid] = 0;

    for (int i = 0; i < q; i++) {
        if (tid == i * q + i) {
            //Can be used to debug
            //edgeMetricCount[tid] = -1;
            return;
        }
    }

    if (data[tid] != 0) {
        numOfNonZeroEdges[tid] = 1;
        if (comparedata[tid] != 0) {
            edgeMetricCount[tid] = 1;
            if (data[tid] == comparedata[tid]) {
                edgeType[tid] = 1;
            }

        }
    }

}

Metrics testCudaLinearMatrixMemory(GraphCode json1, GraphCode json2) {

    int items1 = pow(json1.dict->size(), 2);

    // Prep for cuda


    int *gpu_inputMatrix1;
    int *gpu_inputMatrix2;
    int *darr_edge_metric_count;
    int *darr_num_of_non_zero_edges;
    int *darr_edge_type;

    auto start = std::chrono::system_clock::now();

    HANDLE_ERROR(cudaMalloc((void **) &gpu_inputMatrix1, sizeof(int) * items1));
    HANDLE_ERROR(cudaMalloc((void **) &gpu_inputMatrix2, sizeof(int) * items1));
    HANDLE_ERROR(cudaMalloc((void **) &darr_num_of_non_zero_edges, sizeof(int) * items1));
    HANDLE_ERROR(cudaMalloc((void **) &darr_edge_metric_count, sizeof(int) * items1));
    HANDLE_ERROR(cudaMalloc((void **) &darr_edge_type, sizeof(int) * items1));
    /*
    cudaMemcpy2DToArray (dst,
                         0,
                         0,
                         matrix1,
                         sizeof(int),
                         gc1Dictionary.size() * sizeof(int),
                         gc1Dictionary.size(),
                         cudaMemcpyHostToDevice );

    */

    // Transfer data from host to device memory
    HANDLE_ERROR(cudaMemcpy(gpu_inputMatrix1, json1.matrix, sizeof(int) * items1, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gpu_inputMatrix2, json2.matrix, sizeof(int) * items1, cudaMemcpyHostToDevice));

    int gridSize;
    int blockSize;
    dim3 block;
    dim3 grid;
    if (items1 > 1024) {
        gridSize = ceil(items1 / 1024.0);
        blockSize = findLargestDivisor(1024);
        block = 1024;//(blockSize, ceil(1024 / (float) blockSize));
        grid = (gridSize);

    } else {
        gridSize = findLargestDivisor(items1);
        blockSize = findLargestDivisor(gridSize);
        if (blockSize == 0) {
            if (isPrime(gridSize)) {
                //blockSize= findLargestDivisor(gridSize+1);
                gridSize += 1;
                blockSize = findLargestDivisor(gridSize);
            }
        }
        block = (items1);
        //grid = (ceil(items1/(float) gridSize));
        grid = (1);
    }






    //HANDLE_ERROR(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calcMetrices, 0, 0));


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



    //printf("CUDA error %s\n",cudaGetErrorString(cudaPeekAtLastError()));
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
    //int arr_edge_metric_count[items1];
    int *arrEdgeTypeMetricCount = (int *) malloc(sizeof(int) * items1);

    int *arr_edge_metric_count = (int *) malloc(sizeof(int) * items1);

    //int arr_num_of_non_zero_edges[items1];
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


    // Result reduction
    int num_of_non_zero_edges = 0;
    int edge_metric_count = 0;
    int edgeTypeCount = 0;
    for (int i = 0; i < items1; i++) {
        //  std::cout << "pos: " << i
        //    << " arr_edge_metric_count: " << arr_edge_metric_count[i]
        //    << " arr_num_of_non_zero_edges: " << arr_num_of_non_zero_edges[i]
        //    << std::endl;
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

    std::string gc1Dict[json1.dict->size()];

    int sim = 0;
    int n = 0;
    for (const auto &item: *json1.dict) {
        //std::cout << item.value() << "\n";
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
    //float node_metric = (float) numberOfElements1 / (float) gc1Dictionary.size();
    float node_metric = (float) sim / (float) json1.dict->size();


    float edge_metric = 0.0;
    if (num_of_non_zero_edges > 0)
        edge_metric = (float) edge_metric_count / (float) num_of_non_zero_edges;

    float edge_type_metric = 0.0;
    if (edge_metric_count > 0)
        edge_type_metric = (float) edgeTypeCount / (float) edge_metric_count;

    if(G_DEBUG)
        std::cout << "Similarity: " << " value: " << node_metric << std::endl;
    if(G_DEBUG)
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


Metrics testCudaLinearMatrixMemory(json json1, json json2) {

    json gc1Dictionary ;
    int numberOfElements1;
    int items1;
    int *inputMatrix1;

    convertGc2Cuda(json1, gc1Dictionary, numberOfElements1, items1, inputMatrix1);


    json gc2Dictionary;
    int numberOfElements2;
    int items2;
    int *inputMatrix2;
    convertGc2Cuda(json2, gc2Dictionary, numberOfElements2, items2, inputMatrix2);

    // Prep for cuda


    int *gpu_inputMatrix1;
    int *gpu_inputMatrix2;
    int *darr_edge_metric_count;
    int *darr_num_of_non_zero_edges;
    int *darr_edge_type;
    // Allocate device memory for inputMatrix1
    //cudaMalloc((void**)&gpu_inputMatrix1, sizeof(int) );

    auto start = std::chrono::system_clock::now();

    HANDLE_ERROR(cudaMalloc((void **) &gpu_inputMatrix1, sizeof(int) * items1));
    HANDLE_ERROR(cudaMalloc((void **) &gpu_inputMatrix2, sizeof(int) * items1));
    HANDLE_ERROR(cudaMalloc((void **) &darr_num_of_non_zero_edges, sizeof(int) * items1));
    HANDLE_ERROR(cudaMalloc((void **) &darr_edge_metric_count, sizeof(int) * items1));
    HANDLE_ERROR(cudaMalloc((void **) &darr_edge_type, sizeof(int) * items1));
    /*
    cudaMemcpy2DToArray (dst,
                         0,
                         0,
                         matrix1,
                         sizeof(int),
                         gc1Dictionary.size() * sizeof(int),
                         gc1Dictionary.size(),
                         cudaMemcpyHostToDevice );

    */

    // Transfer data from host to device memory
    HANDLE_ERROR(cudaMemcpy(gpu_inputMatrix1, inputMatrix1, sizeof(int) * items1, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gpu_inputMatrix2, inputMatrix2, sizeof(int) * items1, cudaMemcpyHostToDevice));

    int gridSize;
    int blockSize;
    dim3 block;
    dim3 grid;
    if (items1 > 1024) {
        gridSize = ceil(items1 / 1024.0);
        blockSize = findLargestDivisor(1024);
        block = 1024;//(blockSize, ceil(1024 / (float) blockSize));
        grid = (gridSize);

    } else {
        gridSize = findLargestDivisor(items1);
        blockSize = findLargestDivisor(gridSize);
        if (blockSize == 0) {
            if (isPrime(gridSize)) {
                //blockSize= findLargestDivisor(gridSize+1);
                gridSize += 1;
                blockSize = findLargestDivisor(gridSize);
            }
        }
        block = (items1);
        //grid = (ceil(items1/(float) gridSize));
        grid = (1);
    }






    //HANDLE_ERROR(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calcMetrices, 0, 0));


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



    //printf("CUDA error %s\n",cudaGetErrorString(cudaPeekAtLastError()));
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
    //int arr_edge_metric_count[items1];
    int *arrEdgeTypeMetricCount = (int *) malloc(sizeof(int) * items1);

    int *arr_edge_metric_count = (int *) malloc(sizeof(int) * items1);

    //int arr_num_of_non_zero_edges[items1];
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
        //  std::cout << "pos: " << i
        //    << " arr_edge_metric_count: " << arr_edge_metric_count[i]
        //    << " arr_num_of_non_zero_edges: " << arr_num_of_non_zero_edges[i]
        //    << std::endl;
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
        //std::cout << item.value() << "\n";
        std::string str = item.value().get<std::string>();
        gc1Dict[n++] = str;


        for (const auto &item2: gc2Dictionary.items()) {
            if (str == item2.value()) {
                //std::cout << "Match" << std::endl;
                sim++;
            }
        }
    }

    // Calculate metrices
    //float node_metric = (float) numberOfElements1 / (float) gc1Dictionary.size();
    float node_metric = (float) sim / (float) gc1Dictionary.size();


    float edge_metric = 0.0;
    if (num_of_non_zero_edges > 0)
        edge_metric = (float) edge_metric_count / (float) num_of_non_zero_edges;

    float edge_type_metric = 0.0;
    if (edge_metric_count > 0)
        edge_type_metric = (float) edgeTypeCount / (float) edge_metric_count;

    if(G_DEBUG)
        std::cout << "Similarity: " << " value: " << node_metric << std::endl;
    if(G_DEBUG)
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


void convertGc2Cuda(const json &gcq, json &gc1Dictionary, int &numberOfElements, int &items, int *&inputMatrix) {
    gc1Dictionary = gcq["dictionary"];
    numberOfElements = gc1Dictionary.size();
    items = numberOfElements * numberOfElements;// Transform to data structures for calculations
//int matrix1[gc1Dictionary.size()][gc1Dictionary.size()];
    int *matrix1;
    matrix1 = (int *) malloc(sizeof(int) * numberOfElements * numberOfElements);

    convertDict2Matrix(numberOfElements, matrix1, gcq["matrix"]);

    //int inputMatrix[items];
//int count = 0;
//for (int i = 0; i < numberOfElements; i++)
//    for (int j = 0; j < numberOfElements; j++) {
//        inputMatrix[count++] = matrix1[j*numberOfElements + i]; //matrix1[i][j];
//    }
    inputMatrix = (int *) malloc(sizeof(int) * numberOfElements * numberOfElements);

    int count = 0;
    for (int i = 0; i < numberOfElements; i++)
        for (int j = 0; j < numberOfElements; j++) {
            inputMatrix[count++] = matrix1[i * numberOfElements + j]; //matrix1[i][j];
        }
    free(matrix1);
}
Metrics calculateSimilaritySequentialOrdered(GraphCode gc1, GraphCode gc2)
{

    int sim = 0;

    int *matrix1 = gc1.matrix;
    int *matrix2 = gc2.matrix;

    for (const auto &item: *gc1.dict) {
        //gc1Dict[n++] = str;


        for (const auto &item2: *gc2.dict) {
            if (item == item2) {
                //std::cout << "Match" << std::endl;
                sim++;
            }
        }

    }
    int num_of_non_zero_edges = 0;
    int edge_metric_count = 0;
    int edge_type = 0;

    for (int i = 0; i < gc1.dict->size(); i++) {
        for (int j = 0; j < gc1.dict->size(); j++) {

            if (i != j && matrix1[i*gc1.dict->size()+j] != 0) {
                num_of_non_zero_edges++;

                int position1 = i;
                int position2 = j;
                if (position1 == -1 || position2 == -1) {
                    continue;
                }

                int edge = matrix2[position1*gc1.dict->size()+ position2];//matrix2[position1][position2];
                if (edge != 0) {
                    edge_metric_count++;
                }
                if (edge == matrix1[i*gc1.dict->size()+j]) {
                    edge_type++;
                }

            }
        }
    }

    float node_metric = (float) sim / (float) gc1.dict->size();
    float edge_metric = 0.0;
    if (num_of_non_zero_edges > 0)
        edge_metric = (float) edge_metric_count / (float) num_of_non_zero_edges;
    float edge_type_metric = 0.0;
    if (edge_metric_count > 0)
        edge_type_metric = (float) edge_type / (float) edge_metric_count;

    Metrics metrics;
    metrics.similarity = node_metric;
    metrics.recommendation = edge_metric;
    metrics.inferencing = edge_type_metric;
    return metrics;

}


Metrics calculateSimilaritySequentialOrdered(json gc1, json gc2) {
    int sim = 0;

    json gc1Dictionary ;
    int numberOfElements1;
    int items1;
    int *matrix1;

    convertGc2Cuda(gc1, gc1Dictionary, numberOfElements1, items1, matrix1);

    json gc2Dictionary ;
    int numberOfElements2;
    int items2;
    int *matrix2;
    convertGc2Cuda(gc2, gc2Dictionary, numberOfElements2, items2, matrix2);

    std::vector<std::string> dict2;
    for (const auto &item2: gc2Dictionary.items()) {
        dict2.push_back(item2.value().get<std::string>());
    }


    for (const auto &item: gc1Dictionary.items()) {

        std::string str = item.value().get<std::string>();
        //gc1Dict[n++] = str;


        for (const auto &item2: gc2Dictionary.items()) {
            if (str == item2.value()) {
                //std::cout << "Match" << std::endl;
                sim++;
            }
        }

    }
    int num_of_non_zero_edges = 0;
    int edge_metric_count = 0;
    int edge_type = 0;

    for (int i = 0; i < gc1Dictionary.size(); i++) {
        for (int j = 0; j < gc1Dictionary.size(); j++) {

//            if (i != j && matrix1[i][j] != 0) {
            if (i != j && matrix1[i*gc1Dictionary.size()+j] != 0) {
                num_of_non_zero_edges++;

                int position1 = i;
                int position2 = j;
                //std::cout << "Pos " << position1 << " " << position2 << std::endl;
                if (position1 == -1 || position2 == -1) {
                    continue;
                }

                int edge = matrix2[position1*gc1Dictionary.size()+ position2];//matrix2[position1][position2];
                if (edge != 0) {
                    edge_metric_count++;
                }
                if (edge == matrix1[i*gc1Dictionary.size()+j]) {
                    edge_type++;
                }

            }
        }
    }

    float node_metric = (float) sim / (float) gc1Dictionary.size();
    float edge_metric = 0.0;
    if (num_of_non_zero_edges > 0)
        edge_metric = (float) edge_metric_count / (float) num_of_non_zero_edges;
    float edge_type_metric = 0.0;
    if (edge_metric_count > 0)
        edge_type_metric = (float) edge_type / (float) edge_metric_count;

    Metrics metrics;
    metrics.similarity = node_metric;
    metrics.recommendation = edge_metric;
    metrics.inferencing = edge_type_metric;
    return metrics;

}