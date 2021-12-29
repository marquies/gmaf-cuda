//
// Created by breucking on 29.12.21.
//

#ifndef GCSIM_REDUCE_CUH
#define GCSIM_REDUCE_CUH


#define MAX_BLOCK_SZ 1024

unsigned int gpu_sum_reduce(unsigned int* d_in,  long d_in_len);

#endif //GCSIM_REDUCE_CUH
