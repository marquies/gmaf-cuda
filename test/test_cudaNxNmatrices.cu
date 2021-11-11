// Ttt
#include <cstdlib>

#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>

#include "../src/graphcode.h"
#include "../src/cudahelper.cuh"
#include "../src/helper.h"


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



__global__ void check(int *data, unsigned long matrixSize, int *pInt) {
    int tid = blockIdx.x;

    //if (tid % matrixSize != 0) {
    if (data[tid] != 0) {
        pInt[tid] = 1;
    }
    //}

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

void testCudaLinearMatrixMemory(){
    nlohmann::json gcq;
    gcq["dictionary"] = { "head", "body"};
    gcq["matrix"] = {{1,1}, {0,1}};

    json gc1Dictionary = gcq["dictionary"];

    int matrix1[gc1Dictionary.size()][gc1Dictionary.size()];
    convertDict2Matrix(gc1Dictionary.size(), (int *) matrix1, gcq["matrix"]);

    int a[gcq.size() * gcq.size()];
    int count = 0;
    for (int i = 0; i < gcq.size(); i++)
        for (int j = 0; j < gcq.size(); j++) {
            a[count++] = matrix1[i][j];
        }

    int *d_a;
    int *founds;
    // Allocate device memory for a
    //cudaMalloc((void**)&d_a, sizeof(int) );

    int items = 4;


    HANDLE_ERROR(cudaMalloc((void**)&d_a, sizeof(int) * items) );
    HANDLE_ERROR(cudaMalloc((void**)&founds, sizeof(int) * items) );
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
    HANDLE_ERROR(cudaMemcpy(d_a, a, sizeof(int) * gcq.size() * gcq.size(), cudaMemcpyHostToDevice));


    check<<<items, 1>>>(d_a, gcq.size(), founds);


    int f[items];
    HANDLE_ERROR(cudaMemcpy(f, founds, sizeof (int)* gcq.size() * gcq.size(), cudaMemcpyDeviceToHost));

    for(int i = 0; i < items; i++) {
        std::cout << "pos: " << i << " value: " << f[i] << std::endl;
    }

    HANDLE_ERROR(cudaFree(d_a));
}


int main(int, char**)
{
    testCudaMatrixMemory();
    testCudaLinearMatrixMemory();

}
