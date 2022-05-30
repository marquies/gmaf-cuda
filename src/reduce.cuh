//
// Created by breucking on 29.12.21.
//

#ifndef GCSIM_REDUCE_CUH
#define GCSIM_REDUCE_CUH


#define MAX_BLOCK_SZ 1024

/**
 * Method to reduce the sums on cuda in an array.
 * @param d_in input array with integer.
 * @param d_in_len length of the input array.
 * @return the final sum of the array elements.
 */
unsigned int gpu_sum_reduce(unsigned int* d_in,  long d_in_len);

#endif //GCSIM_REDUCE_CUH
