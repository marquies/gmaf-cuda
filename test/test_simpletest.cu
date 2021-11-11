//
// Created by breucking on 06.11.21.
//

#include <cstdlib>

#include "../src/graphcode.h"
#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCKSIZE_x 16
#define BLOCKSIZE_y 16

#define Nrows 4
#define Ncols 4

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %dn", cudaGetErrorString(code), file, line);
        if (abort) { exit(code); }
    }
}

/*******************/
/* iDivUp FUNCTION */
/*******************/
int iDivUp(int hostPtr, int b){ return ((hostPtr % b) != 0) ? (hostPtr / b + 1) : (hostPtr / b); }

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
int testCuda()
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
    gpuErrchk(cudaMallocPitch(&devPtr, &pitch, Ncols * sizeof(float), Nrows));
    gpuErrchk(cudaMemcpy2D(devPtr, pitch, hostPtr, Ncols*sizeof(float), Ncols*sizeof(float), Nrows, cudaMemcpyHostToDevice));

    dim3 gridSize(iDivUp(Ncols, BLOCKSIZE_x), iDivUp(Nrows, BLOCKSIZE_y));
    dim3 blockSize(BLOCKSIZE_y, BLOCKSIZE_x);

    test_kernel_2D<<<gridSize, blockSize>>>(devPtr, pitch);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy2D(hostPtr, Ncols * sizeof(float), devPtr, pitch, Ncols * sizeof(float), Nrows, cudaMemcpyDeviceToHost));

    for (int i = 0; i < Nrows; i++)
        for (int j = 0; j < Ncols; j++)
            printf("row %i column %i value %f \n", i, j, hostPtr[i][j]);

    return 0;
}
void testBasic()
{
    nlohmann::json gcq;
    gcq["dictionary"] = { "head", "body"};
    gcq["matrix"] = {{1,1}, {0,1}};

    std::vector<json> others;
    others.push_back(gcq);

    gmaf::GraphCode gc;

    gc.calculateSimilarityV(0,&gcq,&others,0,1);
}

int main(int, char**)
{
    testBasic();
    testCuda();
}
