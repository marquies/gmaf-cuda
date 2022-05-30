//
// Created by breucking on 11.11.21.
//

#ifndef GCSIM_CUDAHELPER_CUH
#define GCSIM_CUDAHELPER_CUH

#include <stdio.h>

/**
 * helper function to check for error codes in return values.
 * @param err error as return value.
 * @param file file of the function definition.
 * @param line line of the function definition.
 */
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
// Define the macro to wrap CUDA functions.
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#endif //GCSIM_CUDAHELPER_CUH
