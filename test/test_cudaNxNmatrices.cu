// Ttt
#include <cstdlib>

#include <stdio.h>
#include <cuda_runtime.h>

#include <iostream>

#include <math.h>
#include <chrono>
#include <ctime>

#include "../src/graphcode.h"
#include "../src/cudahelper.cuh"
#include "../src/cuda_algorithms.cu"
#include "../src/helper.h"

#include "testhelper.cpp"


#define BLOCKSIZE_x 16
#define BLOCKSIZE_y 16

#define Nrows 4
#define Ncols 4

double EPSILON = 0.000001;


void testFindLargestDivisor();



void testConvertGc4Cuda();
/**
 * iDivUp FUNCTION
 */
int iDivUp(int hostPtr, int b){ return ((hostPtr % b) != 0) ? (hostPtr / b + 1) : (hostPtr / b); }

//
///**
// * Calc Metrices is a simple example to compare two NxN matrices
// * @param data pinter to vectorized matrix
// * @param comparedata pointer to vectorized matrix
// * @param matrixSize dimension of the NxN matrix
// * @param numOfNonZeroEdges pointer to array to store the values for the non zero edges comparison
// * @param edgeMetricCount pointer to array to store the values for the edge metric comparison
// * @param edgeType pointer to array to store the values for the edge type metric comparison
// */
//__global__ void
//calcMetrices(int *data, int *comparedata, unsigned long matrixSize,
//             int *numOfNonZeroEdges, int *edgeMetricCount, int *edgeType) {
//    int x = threadIdx.x + blockIdx.x * blockDim.x;
//    int y = threadIdx.y + blockIdx.y * blockDim.y;
//    int /*offset*/ tid = x + y * blockDim.x * gridDim.x;
//
//     int q = sqrt((float)matrixSize);
//
//    numOfNonZeroEdges[tid] = 0;
//    edgeMetricCount[tid] = 0;
//    edgeType[tid] = 0;
//
//     for (int i = 0; i < q; i++) {
//        if (tid == i*q+i) {
//            //Can be used to debug
//            //edgeMetricCount[tid] = -1;
//            return;
//        }
//     }
//
//    if (data[tid] != 0 ) {
//        numOfNonZeroEdges[tid] = 1;
//        if (comparedata[tid] != 0) {
//            edgeMetricCount[tid] = 1;
//            if (data[tid] == comparedata[tid]) {
//                edgeType[tid] = 1;
//            }
//
//        }
//    }
//
//}



/******************/
/* TEST KERNEL 2D */
/******************/

__global__ void test_kernel_2D(float *devPtr, size_t pitch)
{
    int    tidx = blockIdx.x*blockDim.x + threadIdx.x;
    int    tidy = blockIdx.y*blockDim.y + threadIdx.y;

    if ((tidx < Ncols) && (tidy < Nrows))
    {
        float *row_a = (float *)((char*)devPtr + tidy * pitch);
        if (tidx == tidy) {
            row_a[tidx] = 0.0;
        } else {

            row_a[tidx] = row_a[tidx] * tidx * tidy;
        }
    }
}



int testCudaMatrixMemory()
{
    float hostPtr[Nrows][Ncols];
    float *devPtr;
    size_t pitch;

    for (int i = 0; i < Nrows; i++)
        for (int j = 0; j < Ncols; j++) {
            hostPtr[i][j] = 1.f;
            //printf("row %i column %i value %f \n", i, j, hostPtr[i][j]);
        }

    // --- 2D pitched allocation and host->device memcopy
    HANDLE_ERROR(cudaMallocPitch(&devPtr, &pitch, Ncols * sizeof(float), Nrows));
    HANDLE_ERROR(cudaMemcpy2D(devPtr, pitch, hostPtr, Ncols*sizeof(float), Ncols*sizeof(float), Nrows, cudaMemcpyHostToDevice));

    dim3 gridSize(iDivUp(Ncols, BLOCKSIZE_x), iDivUp(Nrows, BLOCKSIZE_y));
    dim3 blockSize(BLOCKSIZE_y, BLOCKSIZE_x);

    test_kernel_2D<<<gridSize, blockSize>>>(devPtr, pitch);

    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    HANDLE_ERROR(cudaMemcpy2D(hostPtr, Ncols * sizeof(float), devPtr, pitch, Ncols * sizeof(float), Nrows, cudaMemcpyDeviceToHost));

    for (int i = 0; i < Nrows; i++)
        for (int j = 0; j < Ncols; j++)
            printf("row %i column %i value %f \n", i, j, hostPtr[i][j]);

    return 0;
}
bool AreSame(double a, double b)
{
    return fabs(a - b) < EPSILON;
}

void testCudaLinearMatrixMemoryRealTest() {
    // Generate test data

    Metrics m;

    json gcq = generateTestData(9);
    m = testCudaLinearMatrixMemory(gcq, gcq);

    assert(m.similarity == 1);
    assert(m.recommendation == 1);

    json gcq2 = generateTestData(2040);
    m = testCudaLinearMatrixMemory(gcq2, gcq2);

    assert(m.similarity == 1);
    assert(m.recommendation == 1);


    nlohmann::json gcq3;
    gcq3["dictionary"] = { "head", "body", "foot"};
    gcq3["matrix"] = {{1,1,0}, {0,1,0}, {0,1,1}};

    nlohmann::json gcq4;
    gcq4["dictionary"] = { "head", "body", "foot"};
    gcq4["matrix"] = {{1,2,0}, {0,1,0}, {0,0,1}};

    Metrics m2 = testCudaLinearMatrixMemory(gcq3, gcq4);

    assert(AreSame(m2.similarity,(float) 3./3.));
    assert(m2.recommendation == .5);
    assert(m2.inferencing == 0);




}

//
//Metrics testCudaLinearMatrixMemory(json json1, json json2) {
//
//    json gc1Dictionary;
//    int numberOfElements1;
//    int items1;
//    int *inputMatrix1;
//
//    convertGc2Cuda(json1, gc1Dictionary, numberOfElements1, items1, inputMatrix1);
//
//
//    json gc2Dictionary;
//    int numberOfElements2;
//    int items2;
//    int *inputMatrix2;
//    convertGc2Cuda(json2, gc2Dictionary, numberOfElements2, items2, inputMatrix2);
//
//    // Prep for cuda
//
//
//    int *gpu_inputMatrix1;
//    int *gpu_inputMatrix2;
//    int *darr_edge_metric_count;
//    int *darr_num_of_non_zero_edges;
//    int *darr_edge_type;
//    // Allocate device memory for inputMatrix1
//    //cudaMalloc((void**)&gpu_inputMatrix1, sizeof(int) );
//
//    auto start = std::chrono::system_clock::now();
//
//    HANDLE_ERROR(cudaMalloc((void**)&gpu_inputMatrix1, sizeof(int) * items1) );
//    HANDLE_ERROR(cudaMalloc((void**)&gpu_inputMatrix2, sizeof(int) * items1) );
//    HANDLE_ERROR(cudaMalloc((void**)&darr_num_of_non_zero_edges, sizeof(int) * items1) );
//    HANDLE_ERROR(cudaMalloc((void**)&darr_edge_metric_count, sizeof(int) * items1) );
//    HANDLE_ERROR(cudaMalloc((void**)&darr_edge_type, sizeof(int) * items1) );
//    /*
//    cudaMemcpy2DToArray (dst,
//                         0,
//                         0,
//                         matrix1,
//                         sizeof(int),
//                         gc1Dictionary.size() * sizeof(int),
//                         gc1Dictionary.size(),
//                         cudaMemcpyHostToDevice );
//
//    */
//
//    // Transfer data from host to device memory
//    HANDLE_ERROR(cudaMemcpy(gpu_inputMatrix1, inputMatrix1, sizeof(int) * items1, cudaMemcpyHostToDevice));
//    HANDLE_ERROR(cudaMemcpy(gpu_inputMatrix2, inputMatrix2, sizeof(int) * items1, cudaMemcpyHostToDevice));
//
//    int gridSize;
//    int blockSize;
//    dim3 block;
//    dim3 grid;
//    if (items1 > 1024) {
//        gridSize = ceil(items1 / 1024.0);
//        blockSize = findLargestDivisor(1024);
//        block = (blockSize, ceil(1024/(float) blockSize));
//        grid = (gridSize);
//
//    } else {
//        gridSize = findLargestDivisor(items1);
//        blockSize = findLargestDivisor(gridSize);
//        if (blockSize == 0) {
//            if (isPrime(gridSize)) {
//                //blockSize= findLargestDivisor(gridSize+1);
//                gridSize += 1;
//                blockSize = findLargestDivisor(gridSize);
//            }
//        }
//        block = (items1);
//        //grid = (ceil(items1/(float) gridSize));
//        grid = (1);
//    }
//
//
//
//
//
//
//    //HANDLE_ERROR(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calcMetrices, 0, 0));
//
//    // calculation
//    auto loaded = std::chrono::system_clock::now();
//    std::chrono::duration<double> elapsed_seconds = loaded - start;
//
//    std::cout << "elapsed time: " << elapsed_seconds.count()
//              << std::endl;
//
//    calcMetrices<<<grid, block>>>(gpu_inputMatrix1, gpu_inputMatrix2, items1,
//                                  darr_num_of_non_zero_edges,
//                                  darr_edge_metric_count,
//                                  darr_edge_type
//    );
//
//
//
//    //printf("CUDA error %s\n",cudaGetErrorString(cudaPeekAtLastError()));
//    HANDLE_ERROR(cudaPeekAtLastError());
//    HANDLE_ERROR(cudaDeviceSynchronize());
//    auto end = std::chrono::system_clock::now();
//
//    elapsed_seconds = end - start;
//    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
//
//    std::cout << "finished computation at " << std::ctime(&end_time)
//              << "elapsed time: " << elapsed_seconds.count() << "s\n";
//    elapsed_seconds = end - loaded;
//    std::cout << "Computation time: " << elapsed_seconds.count() << "s\n";
//
//    // Retrieve results
//    //int arr_edge_metric_count[items1];
//    int *arrEdgeTypeMetricCount = (int *) malloc(sizeof(int) * items1);
//
//    int *arr_edge_metric_count = (int *) malloc(sizeof(int) * items1);
//
//    //int arr_num_of_non_zero_edges[items1];
//    int *arr_num_of_non_zero_edges = (int *) malloc(sizeof(int) * items1);
//
//    HANDLE_ERROR(cudaMemcpy(arr_num_of_non_zero_edges, darr_num_of_non_zero_edges, sizeof (int) * items1, cudaMemcpyDeviceToHost));
//    HANDLE_ERROR(cudaMemcpy(arr_edge_metric_count, darr_edge_metric_count, sizeof (int) * items1, cudaMemcpyDeviceToHost));
//    HANDLE_ERROR(cudaMemcpy(arrEdgeTypeMetricCount, darr_edge_type, sizeof (int) * items1, cudaMemcpyDeviceToHost));
//
//    HANDLE_ERROR(cudaFree(gpu_inputMatrix1));
//    HANDLE_ERROR(cudaFree(gpu_inputMatrix2));
//    HANDLE_ERROR(cudaFree(darr_edge_metric_count));
//    HANDLE_ERROR(cudaFree(darr_num_of_non_zero_edges));
//    HANDLE_ERROR(cudaFree(darr_edge_type));
//
//
//    free(inputMatrix1);
//    free(inputMatrix2);
//
//    // Result reduction
//    int num_of_non_zero_edges = 0;
//    int edge_metric_count = 0;
//    int edgeTypeCount = 0;
//    for(int i = 0; i < items1; i++) {
//        //  std::cout << "pos: " << i
//        //    << " arr_edge_metric_count: " << arr_edge_metric_count[i]
//        //    << " arr_num_of_non_zero_edges: " << arr_num_of_non_zero_edges[i]
//        //    << std::endl;
//        if (arr_edge_metric_count[i] == 1) {
//            edge_metric_count++;
//        }
//        if (arr_num_of_non_zero_edges[i] == 1) {
//            num_of_non_zero_edges++;
//        }
//        if (arrEdgeTypeMetricCount[i] == 1) {
//            edgeTypeCount++;
//        }
//    }
//
//    std::string gc1Dict[gc1Dictionary.size()];
//
//    int sim = 0;
//    int n = 0;
//    for (const auto &item: gc1Dictionary.items()) {
//        //std::cout << item.value() << "\n";
//        std::string str = item.value().get<std::string>();
//        gc1Dict[n++] = str;
//
//
//        for (const auto &item2: gc2Dictionary.items()) {
//            if (str == item2.value()) {
//                //std::cout << "Match" << std::endl;
//                sim++;
//            }
//        }
//    }
//
//    // Calculate metrices
//    //float node_metric = (float) numberOfElements1 / (float) gc1Dictionary.size();
//    float node_metric = (float) sim / (float) gc1Dictionary.size();
//
//
//    float edge_metric = 0.0;
//    if (num_of_non_zero_edges > 0)
//        edge_metric = (float) edge_metric_count / (float) num_of_non_zero_edges;
//
//    float edge_type_metric = 0.0;
//    if (edge_metric_count > 0)
//        edge_type_metric = (float) edgeTypeCount / (float) edge_metric_count;
//
//    std::cout << "Similarity: " << " value: " << node_metric << std::endl;
//    std::cout << "Recommendation: " << " value: " << edge_metric << std::endl;
//
//    Metrics m;
//    m.similarity = node_metric;
//    m.recommendation = edge_metric;
//    m.inferencing = edge_type_metric;
//
//    free(arrEdgeTypeMetricCount);
//    free(arr_num_of_non_zero_edges);
//    free(arr_edge_metric_count);
//
//    return m;
//
//}


/********/
/* MAIN */
/********/
int main(int, char**)
{
    testFindLargestDivisor();
    testCudaMatrixMemory();
    testConvertGc4Cuda();
    //testCudaLinearMatrixMemory();
    testCudaLinearMatrixMemoryRealTest();

}

void testConvertGc4Cuda() {

    nlohmann::json gcq3;
    gcq3["dictionary"] = { "head", "body", "foot"};
    gcq3["matrix"] = {{1,1,0}, {0,1,0}, {0,1,1}};

    json dict;
    int numberOfElements1;
    int items1;
    int *inputMatrix1;
    convertGc2Cuda(gcq3, dict, numberOfElements1, items1, inputMatrix1);

    assert(inputMatrix1[0] == 1);
    assert(inputMatrix1[1] == 1);
    assert(inputMatrix1[2] == 0);
    assert(inputMatrix1[3] == 0);
    assert(inputMatrix1[4] == 1);
    assert(inputMatrix1[5] == 0);
    assert(inputMatrix1[6] == 0);
    assert(inputMatrix1[7] == 1);
    assert(inputMatrix1[8] == 1);

}

void testFindLargestDivisor() {
    // Note that this loop runs till square root
    int d = findLargestDivisor(513);
    assert(d == 171);
    d = findLargestDivisor(73);
    assert(d == 1);
    d = findLargestDivisor(4000000);
    assert(d == 2000000);

}

