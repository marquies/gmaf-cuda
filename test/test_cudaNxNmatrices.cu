// Ttt
#include <cstdlib>

#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>

#include <math.h>

#include "../src/graphcode.h"
#include "../src/cudahelper.cuh"
#include "../src/helper.h"


#define BLOCKSIZE_x 16
#define BLOCKSIZE_y 16

#define Nrows 4
#define Ncols 4

/*******************/
/* iDivUp FUNCTION */
/*******************/
int iDivUp(int hostPtr, int b){ return ((hostPtr % b) != 0) ? (hostPtr / b + 1) : (hostPtr / b); }



__global__ void check(int *data, int *comparedata, unsigned long matrixSize, int *pInt, int *pInt1) {
    int tid = blockIdx.x;

     //int q = sqrt((float)matrixSize);

     for (int i = 0; i < matrixSize; i++) {
        if (tid == i*matrixSize+i) {
            //Can be used to debug
            //pInt[tid] = -1;
            return;
        }
     }

    if (data[tid] != 0 ) {
        if (comparedata[tid] != 0) {
            pInt1[tid] = 1;
            if (data[tid] == comparedata[tid]) {
                pInt[tid] = 1;
            }

        }
    }

}



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

/********/
/* MAIN */
/********/
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

void testCudaLinearMatrixMemory(){
    nlohmann::json gcq;
    gcq["dictionary"] = { "head", "body"};
    gcq["matrix"] = {{1,1}, {0,1}};

    json gc1Dictionary = gcq["dictionary"];

    int matrix1[gc1Dictionary.size()][gc1Dictionary.size()];
    convertDict2Matrix(gc1Dictionary.size(), (int *) matrix1, gcq["matrix"]);

    int inputMatrix[gcq.size() * gcq.size()];
    int count = 0;
    for (int i = 0; i < gcq.size(); i++)
        for (int j = 0; j < gcq.size(); j++) {
            inputMatrix[count++] = matrix1[i][j];
        }

    int *gpu_inputMatrix;
    int *darr_edge_metric_count;
    int *darr_num_of_non_zero_edges;
    // Allocate device memory for inputMatrix
    //cudaMalloc((void**)&gpu_inputMatrix, sizeof(int) );

    int items = 4;


    HANDLE_ERROR(cudaMalloc((void**)&gpu_inputMatrix, sizeof(int) * items) );
    HANDLE_ERROR(cudaMalloc((void**)&darr_edge_metric_count, sizeof(int) * items) );
    HANDLE_ERROR(cudaMalloc((void**)&darr_num_of_non_zero_edges, sizeof(int) * items) );
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
    HANDLE_ERROR(cudaMemcpy(gpu_inputMatrix, inputMatrix, sizeof(int) * gcq.size() * gcq.size(), cudaMemcpyHostToDevice));


    check<<<items, items>>>(gpu_inputMatrix, gpu_inputMatrix, gcq.size(), darr_edge_metric_count,
                            darr_num_of_non_zero_edges);


    int arr_edge_metric_count[items];
    int arr_num_of_non_zero_edges[items];

    HANDLE_ERROR(cudaMemcpy(arr_edge_metric_count, darr_edge_metric_count, sizeof (int) * gcq.size() * gcq.size(), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(arr_num_of_non_zero_edges, darr_num_of_non_zero_edges, sizeof (int) * gcq.size() * gcq.size(), cudaMemcpyDeviceToHost));

    int num_of_non_zero_edges = 0;
    int edge_metric_count = 0;
    for(int i = 0; i < items; i++) {
        std::cout << "pos: " << i << " value: " << arr_edge_metric_count[i] << std::endl;
        if (arr_edge_metric_count[i] == 1) {
            edge_metric_count++;
        }
        if (arr_num_of_non_zero_edges[i] == 1) {
            num_of_non_zero_edges++;
        }
    }
    float node_metric = (float) items/2 / (float) gc1Dictionary.size();

    float edge_metric = 0.0;
    if (num_of_non_zero_edges > 0)
        edge_metric = (float) edge_metric_count / (float) num_of_non_zero_edges;


    std::cout << "Similarity: " << " value: " << node_metric << std::endl;
    std::cout << "Recommendation: " << " value: " << edge_metric << std::endl;



    HANDLE_ERROR(cudaFree(gpu_inputMatrix));
}


int main(int, char**)
{
    int q = sqrt((float)4);

    for (int i = 0; i < q; i++) {

            std::cout <<i*q+i << std::endl;
    }

    testCudaMatrixMemory();
    testCudaLinearMatrixMemory();

}
