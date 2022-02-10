/*
 * CUDA-Quicksort.cu
 *
 * Copyright © 2012-2015 Emanuele Manca
 *
 **********************************************************************************************
 **********************************************************************************************
 *
 	This file is part of CUDA-Quicksort.

    CUDA-Quicksort is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    CUDA-Quicksort is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with CUDA-Quicksort.

    If not, see http://www.gnu.org/licenses/gpl-3.0.txt and http://www.gnu.org/copyleft/gpl.html


  **********************************************************************************************
  **********************************************************************************************
 *
 * Contact: Ing. Emanuele Manca
 *
 * Department of Electrical and Electronic Engineering,
 * University of Cagliari,
 * P.zza D’Armi, 09123, Cagliari, Italy
 *
 * email: emanuele.manca@diee.unica.it
 *
 *
 * This software contains source code provided by NVIDIA Corporation
 * license: http://developer.download.nvidia.com/licenses/general_license.txt
 *
 * this software uses the library of NVIDIA CUDA SDK and the Cederman and Tsigas' GPU Quick Sort
 *
 */



#include <thrust/scan.h>
#include "../graphcode.h"
//#include <helper_cuda.h>
//#include <helper_timer.h>
#include "scan.h"
#include "scan_common.h"
#include "CUDA-Quicksort.h"
#include "../cudahelper.cuh"

extern __shared__ uint sMemory[];

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
                          __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ static float atomicMin(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
                          __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ inline double atomicMax(double *address, double val) {
    unsigned long long int *address_as_ull = (unsigned long long int *) address;
    unsigned long long int assumed;
    unsigned long long int old = *address_as_ull;

    assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    __double_as_longlong(max(val, __longlong_as_double(assumed))));

    while (assumed != old) {
        assumed = old;
        old = atomicCAS(address_as_ull,
                        assumed,
                        __double_as_longlong(max(val, __longlong_as_double(assumed))));
    }
    return __longlong_as_double(old);
}


__device__ inline double atomicMin(double *address, double val) {
    unsigned long long int *address_as_ull = (unsigned long long int *) address;
    unsigned long long int old = *address_as_ull, assumed;

    assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    __double_as_longlong(min(val, __longlong_as_double(assumed))));
    while (assumed != old) {
        assumed = old;
        old = atomicCAS(address_as_ull,
                        assumed,
                        __double_as_longlong(min(val, __longlong_as_double(assumed))));
    }
    return __longlong_as_double(old);
}


template<typename Metrics>
__device__ inline void Comparator(

        Metrics &valA,
        Metrics &valB,
        uint dir
) {
    Metrics t;
    if ((valA.compareValue > valB.compareValue) == dir) {
        t = valA;
        valA = valB;
        valB = t;
    }
}


static __device__ __forceinline__ unsigned int __qsflo(unsigned int word) {
    unsigned int ret;
    asm volatile("bfind.u32 %0, %1;" : "=r"(ret) : "r"(word));
    return ret;
}

template<typename Metrics>
__global__ void globalBitonicSort(Metrics *indata, Metrics *outdata, Block<Metrics> *bucket, bool inputSelect) {
    __shared__ Metrics shared[1024];


    Metrics *data;

    Block<Metrics> cord = bucket[blockIdx.x];


    uint size = cord.end - cord.begin;
    bool select = !(cord.select);

    if (cord.end - cord.begin > 1024 || cord.end - cord.begin == 0)
        return;

    unsigned int bitonicSize = 1 << (__qsflo(size - 1U) + 1);


    if (select)
        data = indata;
    else
        data = outdata;

    //__syncthreads();

    for (int i = threadIdx.x; i < size; i += blockDim.x)
        shared[i] = data[i + cord.begin];


    for (int i = threadIdx.x + size; i < bitonicSize; i += blockDim.x)
        shared[i].compareValue = 0xffffffff;

    __syncthreads();


    for (uint size = 2; size < bitonicSize; size <<= 1) {
        //Bitonic merge
        uint ddd = 1 ^ ((threadIdx.x & (size / 2)) != 0);
        for (uint stride = size / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            //if(pos <bitonicSize){
            Comparator(
                    shared[pos + 0],
                    shared[pos + stride],
                    ddd
            );
            // }
        }
    }


    //ddd == dir for the last bitonic merge step

    for (uint stride = bitonicSize / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        // if(pos <bitonicSize){
        Comparator(
                shared[pos + 0],
                shared[pos + stride],
                1
        );
        // }
    }

    __syncthreads();

    // Write back the sorted data to its correct position
    for (int i = threadIdx.x; i < size; i += blockDim.x)
        indata[i + cord.begin] = shared[i];

}


template<typename Metrics>
__global__ void quick(Metrics *indata, Metrics *buffer, Partition<Metrics> *partition, Block<Metrics> *bucket) {

    __shared__ Metrics sh_out[1024];

    __shared__ uint start1, end1;
    __shared__ uint left, right;

    int tix = threadIdx.x;

    uint start = partition[blockIdx.x].from;
    uint end = partition[blockIdx.x].end;
    Metrics pivot = partition[blockIdx.x].pivot;
    uint nseq = partition[blockIdx.x].ibucket;

    uint lo = 0;
    uint hi = 0;

    Metrics lmin;
    lmin.compareValue = 0xffffffff;
    Metrics rmax;
    rmax.compareValue = 0;

    Metrics d;


    // start read on 1° tile and store the coordinates of the items that must
    // be moved on the left or on the right of the pivot

    if (tix + start < end) {
        d = indata[tix + start];

        //count items smaller or bigger than the pivot
        // if d<pivot then ll++ else ll
        lo = (d.compareValue < pivot.compareValue) * (lo + 1) + (d.compareValue >= pivot.compareValue) * lo;
        // if d>pivot then lr++ else lr
        hi = (d.compareValue <= pivot.compareValue) * (hi) + (d.compareValue > pivot.compareValue) * (hi + 1);

        lmin = d;
        rmax = d;
    }

    //read and store the coordinates on next tiles for each block
    for (uint i = tix + start + blockDim.x; i < end; i += blockDim.x) {
        Metrics d = indata[i];

        //count items smaller or bigger than the pivot
        lo = (d.compareValue < pivot.compareValue) * (lo + 1) + (d.compareValue >= pivot.compareValue) * lo;
        hi = (d.compareValue <= pivot.compareValue) * (hi) + (d.compareValue > pivot.compareValue) * (hi + 1);

        //compute max and min of tile items
        lmin.compareValue = min(lmin.compareValue, d.compareValue);
        rmax.compareValue = max(rmax.compareValue, d.compareValue);

    }

    //compute max and min of every partition

    compareInclusive(rmax, lmin, (Metrics *) sh_out, blockDim.x);

    __syncthreads();

    if (tix == blockDim.x - 1) {
        //compute absolute max and min for the bucket
        atomicMax(&bucket[nseq].maxPiv.compareValue, rmax.compareValue);
        atomicMin(&bucket[nseq].minPiv.compareValue, lmin.compareValue);
    }

    __syncthreads();


    /*
     * calculate the coordinates of its assigned item to each thread,
     * which are necessary to known in which subsequences the item must be copied
     *
     */
    scan1Inclusive2(lo, hi, (uint *) sh_out, blockDim.x);
    lo = lo - 1;
    hi = SHARED_LIMIT - hi;


    if (tix == blockDim.x - 1) {
        left = lo + 1;
        right = SHARED_LIMIT - hi;

        start1 = atomicAdd(&bucket[nseq].nextbegin, left);
        end1 = atomicSub(&bucket[nseq].nextend, right);
    }

    __syncthreads();


    //thread blocks write on the shared memory the items smaller and bigger than the first tile's pivot
    if (tix + start < end) {
        //items smaller than pivot
        if (d.compareValue < pivot.compareValue) {
            sh_out[lo] = d;
            lo--;
        }

        //items bigger than pivot
        if (d.compareValue > pivot.compareValue) {
            sh_out[hi] = d;
            hi++;
        }

    }

    //thread blocks write on the shared memory the items smaller and bigger than next tiles' pivot
    for (uint i = start + tix + blockDim.x; i < end; i += blockDim.x) {

        Metrics d = indata[i];
        //items smaller than the pivot
        if (d.compareValue < pivot.compareValue) {
            sh_out[lo] = d;
            lo--;
        }

        //items bigger than the pivot
        if (d.compareValue > pivot.compareValue) {
            sh_out[hi] = d;
            hi++;
        }

    }

    __syncthreads();

    //items smaller and bigger than the pivot already sorted in the shared memory are coalesced written on the global memory
    //partial results of each thread block stored on the shared memory are merged together in two subsequences within the global memory
    //coalesced writing of next tiles on the global memory
    for (uint i = tix; i < SHARED_LIMIT; i += blockDim.x) {
        if (i < left)
            buffer[start1 + i] = sh_out[i];

        if (i >= SHARED_LIMIT - right)
            buffer[end1 + i - SHARED_LIMIT] = sh_out[i];
    }

}




//this function assigns the attributes to each partition of each bucket
//a thread block is assigned to a specific partition
template<typename Metrics>
__global__ void partitionAssign(struct Block<Metrics> *bucket, uint *npartitions, struct Partition<Metrics> *partition) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    uint beg = bucket[bx].nextbegin;
    uint end = bucket[bx].nextend;
    Metrics pivot = bucket[bx].pivot;

    uint from;
    uint to;

    if (bx > 0) {
        from = npartitions[bx - 1];
        to = npartitions[bx];
    } else {
        from = 0;
        to = npartitions[bx];
    }


    uint i = tx + from;

    if (i < to) {
        uint begin = beg + SHARED_LIMIT * tx;
        partition[i].from = begin;
        partition[i].end = begin + SHARED_LIMIT;
        partition[i].pivot = pivot;
        partition[i].ibucket = bx;

    }


    for (uint i = tx + from + blockDim.x; i < to; i += blockDim.x) {
        uint begin = beg + SHARED_LIMIT * (i - from);
        partition[i].from = begin;
        partition[i].end = begin + SHARED_LIMIT;
        partition[i].pivot = pivot;
        partition[i].ibucket = bx;
    }
    __syncthreads();
    if (tx == 0 && to - from > 0) partition[to - 1].end = end;


}

//this function enters the pivot value in the central bucket's items
template<typename Metrics>
__global__ void insertPivot(Metrics *data, struct Block<Metrics> *bucket, int nbucket) {

    Metrics pivot = bucket[blockIdx.x].pivot;
    uint start = bucket[blockIdx.x].nextbegin;
    uint end = bucket[blockIdx.x].nextend;
    bool is_altered = bucket[blockIdx.x].done;

    if (is_altered && blockIdx.x < nbucket)
        for (uint j = start + threadIdx.x; j < end; j += blockDim.x)
            data[j] = pivot;


}


//this function assigns the new attributes of each bucket
template<typename Metrics>
__global__ void bucketAssign(Block<Metrics> *bucket, uint *npartitions, int nbucket, int select) {

    uint i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nbucket) {
        bool is_altered = bucket[i].done;
        if (is_altered) {
            //read on i node
            uint orgbeg = bucket[i].begin;
            uint from = bucket[i].nextbegin;
            uint orgend = bucket[i].end;
            uint end = bucket[i].nextend;
            Metrics pivot = bucket[i].pivot;
            Metrics minPiv = bucket[i].minPiv;
            Metrics maxPiv = bucket[i].maxPiv;

            //compare each bucket's max and min to the pivot
            Metrics lmaxpiv;
            lmaxpiv.compareValue = min(pivot.compareValue, maxPiv.compareValue);
            Metrics rminpiv;
            rminpiv.compareValue = max(pivot.compareValue, minPiv.compareValue);

            //write on i+nbucket node
            bucket[i + nbucket].begin = orgbeg;
            bucket[i + nbucket].nextbegin = orgbeg;
            bucket[i + nbucket].nextend = from;
            bucket[i + nbucket].end = from;
            bucket[i + nbucket].pivot.compareValue = (minPiv.compareValue + lmaxpiv.compareValue) / 2;

            //if(select)
            //	bucket[i+nbucket].done   = (from-orgbeg)>1024;// && (minPiv!=maxPiv);
            //else
            bucket[i + nbucket].done = (from - orgbeg) > 1024 && (minPiv.compareValue != maxPiv.compareValue);
            bucket[i + nbucket].select = select;
            bucket[i + nbucket].minPiv.compareValue = 0xffffffffffffffff;
            bucket[i + nbucket].maxPiv.compareValue = 0;
            //bucket[i+nbucket].finish=false;

            //calculate the number of partitions (npartitions) necessary to the i+nbucket bucket
            if (!bucket[i + nbucket].done)
                npartitions[i + nbucket] = 0;
            else npartitions[i + nbucket] = (from - orgbeg + SHARED_LIMIT - 1) / SHARED_LIMIT;

            //write on i node
            bucket[i].begin = end;
            bucket[i].nextbegin = end;
            bucket[i].nextend = orgend;
            bucket[i].pivot.compareValue = (rminpiv.compareValue + maxPiv.compareValue) / 2 + 1;

            //if(select)
            //bucket[i].done   = (orgend-end)>1024;// && (minPiv!=maxPiv);
            //	else
            bucket[i].done = (orgend - end) > 1024 && (minPiv.compareValue != maxPiv.compareValue);
            bucket[i].select = select;
            bucket[i].minPiv.compareValue = 0xffffffffffffffff;
            bucket[i].maxPiv.compareValue = 0;
            //bucket[i].finish=false;

            //calculate the number of partitions (npartitions) necessary to the i-bucket bucket
            if (!bucket[i].done)
                npartitions[i] = 0;
            else
                npartitions[i] = (orgend - end + SHARED_LIMIT - 1) / SHARED_LIMIT;

        }
    }


}


template<typename Metrics>
__global__ void init(Metrics *data, Block<Metrics> *bucket, uint *npartitions, int size, int nblocks) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nblocks) {
        bucket[i].nextbegin = 0;
        bucket[i].begin = 0;

        bucket[i].nextend = 0 + size * (i == 0);
        bucket[i].end = 0 + size * (i == 0);
        npartitions[i] = 0;
        bucket[i].done = false + i == 0;
        bucket[i].select = false;
        bucket[i].maxPiv.compareValue = 0x0;
        bucket[i].minPiv.compareValue = 0xffffffffffffffff;
        //bucket[i].pivot  = 0+ (i==0)*((min(min(data[0],data[size/2]),data[size-1]) + max(max(data[0],data[size/2]),data[size-1]))/2);
        bucket[i].pivot = data[size / 2];
    }

}


void sort(Metrics *inputData, Metrics *outputData, uint size, uint threadCount, int device) {

    cudaSetDevice(device);

    cudaGetLastError();
    //cudaDeviceReset();

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

//	StopWatchInterface* htimer=NULL;
    Metrics *ddata;
    Metrics *dbuffer;

    Block<Metrics> *dbucket;
    struct Partition<Metrics> *partition;
    uint *npartitions1, *npartitions2;

    uint *cudaBlocks = (uint *) malloc(4);

    uint blocks = (size + SHARED_LIMIT - 1) / SHARED_LIMIT;
    uint nblock = 10 * blocks;
    int partition_max = 262144;

    unsigned long long int total = partition_max * sizeof(Block<Metrics>) + nblock * sizeof(Partition<Metrics>) +
                                   2 * partition_max * sizeof(uint) + 3 * (size) * sizeof(Metrics);

//    printf("\nINFO: Device Memory consumed is %.3f GB out of %.3f GB of available memory\n", ((double) total / GIGA),
//           (double) deviceProp.totalGlobalMem / GIGA);

    //Allocating and initializing CUDA arrays
//	sdkCreateTimer(&htimer);
    HANDLE_ERROR(cudaMalloc((void **) &dbucket, partition_max * sizeof(Block<Metrics>)));
    HANDLE_ERROR(cudaMalloc((void **) &partition, nblock * sizeof(Partition<Metrics>))); //nblock

    HANDLE_ERROR(cudaMalloc((void **) &npartitions1, partition_max * sizeof(uint)));
    HANDLE_ERROR(cudaMalloc((void **) &npartitions2, partition_max * sizeof(uint)));

    HANDLE_ERROR(cudaMalloc((void **) &dbuffer, (size) * sizeof(Metrics)));
    HANDLE_ERROR(cudaMalloc((void **) &ddata, (size) * sizeof(Metrics)));

    HANDLE_ERROR(cudaMemcpy(ddata, inputData, size * sizeof(Metrics), cudaMemcpyHostToDevice));

    initScan();

    //setting GPU Cache
    cudaFuncSetCacheConfig(init<Metrics>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(insertPivot<Metrics>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(bucketAssign<Metrics>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(partitionAssign<Metrics>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(quick<Metrics>, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(globalBitonicSort<Metrics>, cudaFuncCachePreferShared);


    HANDLE_ERROR(cudaDeviceSynchronize());
//    sdkResetTimer(&htimer);
//    sdkStartTimer(&htimer);

    //initializing bucket array: initial attributes for each bucket
    init<Metrics><<<(nblock + 255) / 256, 256>>>(ddata, dbucket, npartitions1, size, partition_max);


    uint nbucket = 1;
    uint numIterations = 0;
    bool inputSelect = true;

    *cudaBlocks = blocks;
    HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaPeekAtLastError());

//    getLastCudaError("init() execution FAILED\n");
    HANDLE_ERROR(cudaMemcpy(&npartitions2[0], cudaBlocks, sizeof(uint), cudaMemcpyHostToDevice));


    // beginning of the first phase
    // this phase goes on until the size of the buckets is comparable to the SHARED_LIMIT size
    while (1) {

        /*
         *       	---------------------    Pre-processing: Partitioning    ---------------------
         *
         * buckets are further divided in partitions based on their size
         * the number of partitions needed for each subsequence is determined by the number of elements which can be
         * processed by each thread block.
         *
         * the number of partitions (npartitions) for each block will depend on the shared memory size (SHARED_LIMIT)
         *
         */

        if (numIterations > 0) {    //1024 is the shared memory limit of scanInclusiveShort()
            if (nbucket <= 1024)
                scanInclusiveShort(npartitions2, npartitions1, 1, nbucket);
            else
                scanInclusiveLarge(npartitions2, npartitions1, 1, nbucket);

            HANDLE_ERROR(cudaMemcpy(cudaBlocks, &npartitions2[nbucket - 1], sizeof(uint), cudaMemcpyDeviceToHost));
        }

        if (*cudaBlocks == 0)
            break;


        /*
         *  ---------------------     step 1    ---------------------
         *
         * 	A thread block is assigned to each different partition
         * 	each partition is assigned coordinates, pivot and ....
         */


        partitionAssign<Metrics><<<nbucket, 1024>>>(dbucket, npartitions2, partition);
        cudaDeviceSynchronize();
//		getLastCudaError("partitionAssign() execution FAILED\n");
        HANDLE_ERROR(cudaPeekAtLastError());

        /*
              ---------------------    step 2a    ---------------------

              in this function each thread block creates two subsequences
              to divide the items in the partition whose value is lower than
             the pivot value, from the items whose value is higher than the pivot value
         */

        if (inputSelect)
            quick<Metrics><<<*cudaBlocks, threadCount>>>(ddata, dbuffer, partition, dbucket);
        else
            quick<Metrics><<<*cudaBlocks, threadCount>>>(dbuffer, ddata, partition, dbucket);
        cudaDeviceSynchronize();
        HANDLE_ERROR(cudaPeekAtLastError());

//        getLastCudaError("quick() execution FAILED\n");

        //step 2b: this function enters the pivot value in the central bucket's items
        insertPivot<Metrics><<<nbucket, 512>>>(ddata, dbucket, nbucket);


        //step 3: parameters are assigned, linked to the two new buckets created in step 2
        bucketAssign<Metrics><<<(nbucket + 255) / 256, 256>>>(dbucket, npartitions1, nbucket, inputSelect);
        cudaDeviceSynchronize();
        HANDLE_ERROR(cudaPeekAtLastError());

//        getLastCudaError("insertPivot() or bucketAssign() execution FAILED\n");

        nbucket *= 2;

        inputSelect = !inputSelect;
        numIterations++;
        if (nbucket > deviceProp.maxGridSize[0])
            break;
        //if(numIterations==18) break;
    }

    /*
     * start second phase:
     * now the size of the buckets is such that they can be entirely processed by a thread block
     *
     */

    if (nbucket > deviceProp.maxGridSize[0])
        fprintf(stderr,
                "ERROR: CUDA-Quicksort can't terminate sorting as the block threads needed to finish it are more than the Maximum x-dimension of FERMI GPU thread blocks. Please use Kepler GPUs as the Maximum x-dimension of their thread blocks is much higher\n");
    else
        globalBitonicSort<Metrics><<<nbucket, 512, 0>>>(ddata, dbuffer, dbucket, inputSelect);

    HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaPeekAtLastError());

//    getLastCudaError("globalBitonicSort() execution FAILED\n");

//	sdkStopTimer(&htimer);
//    *wallClock=sdkGetTimerValue(&htimer);


    // Copy the final result to the CPU in the outputData array
    HANDLE_ERROR(cudaMemcpy(outputData, ddata, size * sizeof(Metrics), cudaMemcpyDeviceToHost));

    // release resources
    HANDLE_ERROR(cudaFree(ddata));
    HANDLE_ERROR(cudaFree(dbuffer));
    HANDLE_ERROR(cudaFree(dbucket));
    HANDLE_ERROR(cudaFree(npartitions2));
    HANDLE_ERROR(cudaFree(npartitions1));
    free(cudaBlocks);

    closeScan();
    return;
}



void sortOnMem(Metrics *ddata, Metrics *outputData, uint size, uint threadCount, int device) {

    cudaSetDevice(device);

    cudaGetLastError();
    //cudaDeviceReset();

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

//	StopWatchInterface* htimer=NULL;
//    Metrics *ddata;
    Metrics *dbuffer;

    Block<Metrics> *dbucket;
    struct Partition<Metrics> *partition;
    uint *npartitions1, *npartitions2;

    uint *cudaBlocks = (uint *) malloc(4);

    uint blocks = (size + SHARED_LIMIT - 1) / SHARED_LIMIT;
    uint nblock = 10 * blocks;
    int partition_max = 262144;

    unsigned long long int total = partition_max * sizeof(Block<Metrics>) + nblock * sizeof(Partition<Metrics>) +
                                   2 * partition_max * sizeof(uint) + 3 * (size) * sizeof(Metrics);

//    printf("\nINFO: Device Memory consumed is %.3f GB out of %.3f GB of available memory\n", ((double) total / GIGA),
//           (double) deviceProp.totalGlobalMem / GIGA);

    //Allocating and initializing CUDA arrays
//	sdkCreateTimer(&htimer);
    HANDLE_ERROR(cudaMalloc((void **) &dbucket, partition_max * sizeof(Block<Metrics>)));
    HANDLE_ERROR(cudaMalloc((void **) &partition, nblock * sizeof(Partition<Metrics>))); //nblock

    HANDLE_ERROR(cudaMalloc((void **) &npartitions1, partition_max * sizeof(uint)));
    HANDLE_ERROR(cudaMalloc((void **) &npartitions2, partition_max * sizeof(uint)));

    HANDLE_ERROR(cudaMalloc((void **) &dbuffer, (size) * sizeof(Metrics)));
    HANDLE_ERROR(cudaMalloc((void **) &ddata, (size) * sizeof(Metrics)));

//    HANDLE_ERROR(cudaMemcpy(ddata, inputData, size * sizeof(Metrics), cudaMemcpyHostToDevice));

    initScan();

    //setting GPU Cache
    cudaFuncSetCacheConfig(init<Metrics>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(insertPivot<Metrics>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(bucketAssign<Metrics>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(partitionAssign<Metrics>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(quick<Metrics>, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(globalBitonicSort<Metrics>, cudaFuncCachePreferShared);


    HANDLE_ERROR(cudaDeviceSynchronize());
//    sdkResetTimer(&htimer);
//    sdkStartTimer(&htimer);

    //initializing bucket array: initial attributes for each bucket
    init<Metrics><<<(nblock + 255) / 256, 256>>>(ddata, dbucket, npartitions1, size, partition_max);


    uint nbucket = 1;
    uint numIterations = 0;
    bool inputSelect = true;

    *cudaBlocks = blocks;
    HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaPeekAtLastError());

//    getLastCudaError("init() execution FAILED\n");
    HANDLE_ERROR(cudaMemcpy(&npartitions2[0], cudaBlocks, sizeof(uint), cudaMemcpyHostToDevice));


    // beginning of the first phase
    // this phase goes on until the size of the buckets is comparable to the SHARED_LIMIT size
    while (1) {

        /*
         *       	---------------------    Pre-processing: Partitioning    ---------------------
         *
         * buckets are further divided in partitions based on their size
         * the number of partitions needed for each subsequence is determined by the number of elements which can be
         * processed by each thread block.
         *
         * the number of partitions (npartitions) for each block will depend on the shared memory size (SHARED_LIMIT)
         *
         */

        if (numIterations > 0) {    //1024 is the shared memory limit of scanInclusiveShort()
            if (nbucket <= 1024)
                scanInclusiveShort(npartitions2, npartitions1, 1, nbucket);
            else
                scanInclusiveLarge(npartitions2, npartitions1, 1, nbucket);

            HANDLE_ERROR(cudaMemcpy(cudaBlocks, &npartitions2[nbucket - 1], sizeof(uint), cudaMemcpyDeviceToHost));
        }

        if (*cudaBlocks == 0)
            break;


        /*
         *  ---------------------     step 1    ---------------------
         *
         * 	A thread block is assigned to each different partition
         * 	each partition is assigned coordinates, pivot and ....
         */


        partitionAssign<Metrics><<<nbucket, 1024>>>(dbucket, npartitions2, partition);
        cudaDeviceSynchronize();
//		getLastCudaError("partitionAssign() execution FAILED\n");
        HANDLE_ERROR(cudaPeekAtLastError());

        /*
              ---------------------    step 2a    ---------------------

              in this function each thread block creates two subsequences
              to divide the items in the partition whose value is lower than
             the pivot value, from the items whose value is higher than the pivot value
         */

        if (inputSelect)
            quick<Metrics><<<*cudaBlocks, threadCount>>>(ddata, dbuffer, partition, dbucket);
        else
            quick<Metrics><<<*cudaBlocks, threadCount>>>(dbuffer, ddata, partition, dbucket);
        cudaDeviceSynchronize();
        HANDLE_ERROR(cudaPeekAtLastError());

//        getLastCudaError("quick() execution FAILED\n");

        //step 2b: this function enters the pivot value in the central bucket's items
        insertPivot<Metrics><<<nbucket, 512>>>(ddata, dbucket, nbucket);


        //step 3: parameters are assigned, linked to the two new buckets created in step 2
        bucketAssign<Metrics><<<(nbucket + 255) / 256, 256>>>(dbucket, npartitions1, nbucket, inputSelect);
        cudaDeviceSynchronize();
        HANDLE_ERROR(cudaPeekAtLastError());

//        getLastCudaError("insertPivot() or bucketAssign() execution FAILED\n");

        nbucket *= 2;

        inputSelect = !inputSelect;
        numIterations++;
        if (nbucket > deviceProp.maxGridSize[0])
            break;
        //if(numIterations==18) break;
    }

    /*
     * start second phase:
     * now the size of the buckets is such that they can be entirely processed by a thread block
     *
     */

    if (nbucket > deviceProp.maxGridSize[0])
        fprintf(stderr,
                "ERROR: CUDA-Quicksort can't terminate sorting as the block threads needed to finish it are more than the Maximum x-dimension of FERMI GPU thread blocks. Please use Kepler GPUs as the Maximum x-dimension of their thread blocks is much higher\n");
    else
        globalBitonicSort<Metrics><<<nbucket, 512, 0>>>(ddata, dbuffer, dbucket, inputSelect);

    HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaPeekAtLastError());

//    getLastCudaError("globalBitonicSort() execution FAILED\n");

//	sdkStopTimer(&htimer);
//    *wallClock=sdkGetTimerValue(&htimer);


    // Copy the final result to the CPU in the outputData array
    HANDLE_ERROR(cudaMemcpy(outputData, ddata, size * sizeof(Metrics), cudaMemcpyDeviceToHost));

    // release resources
    HANDLE_ERROR(cudaFree(ddata));
    HANDLE_ERROR(cudaFree(dbuffer));
    HANDLE_ERROR(cudaFree(dbucket));
    HANDLE_ERROR(cudaFree(npartitions2));
    HANDLE_ERROR(cudaFree(npartitions1));
    free(cudaBlocks);

    closeScan();
    return;
}


extern "C"
//void CUDA_Quicksort(uint* inputData, uint* outputData, uint dataSize, uint threadCount, int Device, double* wallClock)
void CUDA_Quicksort(Metrics *inputData, Metrics *outputData, unsigned int dataSize, unsigned int threadCount) {

    int Device = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, Device);

    if (deviceProp.major < 2) {
        fprintf(stderr,
                "Error: the GPU device %d has a Compute Capability of %d.%d, while a Compute Capability of 2.x is required to run the code\n",
                Device, deviceProp.major, deviceProp.minor);

        int deviceCount;
        cudaGetDeviceCount(&deviceCount);

        fprintf(stderr, "       the Host system has the following GPU devices:\n");

        for (int device = 0; device < deviceCount; device++) {

            fprintf(stderr, "\t  the GPU device %d is a %s, with Compute Capability %d.%d\n",
                    device, deviceProp.name, deviceProp.major, deviceProp.minor);
        }

        return;
    }

    sort(inputData, outputData, dataSize, threadCount, Device);
}
//
//extern "C"
//void CUDA_Quicksort_64(double* inputData,double* outputData, uint dataSize, uint threadCount, int Device, double* wallClock)
//{
//
//	cudaDeviceProp deviceProp;
//	cudaGetDeviceProperties(&deviceProp, Device);
//
//	if(deviceProp.major<2)
//	{
//		fprintf(stderr, "Error: the GPU device %d has a Compute Capability of %d.%d, while a Compute Capability of 2.x is required to run the code\n",
//				Device, deviceProp.major, deviceProp.minor);
//
//		int deviceCount;
//		cudaGetDeviceCount(&deviceCount);
//
//		fprintf(stderr, "       the Host system has the following GPU devices:\n");
//
//		for (int device = 0; device < deviceCount; device++) {
//
//				fprintf(stderr, "\t  the GPU device %d is a %s, with Compute Capability %d.%d\n",
//						device, deviceProp.name, deviceProp.major, deviceProp.minor);
//		}
//
//		return;
//	}
//
//	sort<double>(inputData,outputData, dataSize,threadCount,Device,wallClock);
//
//}
