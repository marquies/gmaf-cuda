
#include <cuda_runtime.h>

#include <c++/9/iostream>
#include <helper.h>
#include "../src/cuda_algorithms.cuh"
#include "../src/cudahelper.cuh"


void testCudaSort();
//
//int main() {
//    testCudaSort();
//}

////////////////////////////////////////////////////////////////////////////////
// Initialize data on the host.
////////////////////////////////////////////////////////////////////////////////
void initialize_data(unsigned int *dst, unsigned int nitems) {
    // Fixed seed for illustration
    srand(2047);

    // Fill dst with random values
    for (unsigned i = 0; i < nitems; i++)
        dst[i] = rand() % nitems;
}

////////////////////////////////////////////////////////////////////////////////
// Main entry point.
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    int num_items = 1000;
    bool verbose = true;

    Metrics *inData = new Metrics[num_items];

    for (int i = 0; i < num_items; i++) {
        inData[i].idx = i;
        inData[i].similarity =  i ;
        inData[i].recommendation =  i ;
        inData[i].recommendation =  i ;
    }
    for (int i = 1; i < num_items; i++) {
        assert(compare(inData + i, inData + (i - 1)) >0);
    }

    // Get device properties
    int device_count = 0, device = 0;
    cudaDeviceProp properties;
    HANDLE_ERROR(cudaGetDeviceProperties(&properties, device));

    if (properties.major > 3 || (properties.major == 3 && properties.minor >= 5)) {
        std::cout << "Running on GPU " << device << " (" << properties.name << ")" << std::endl;
    } else {
        std::cout << "ERROR: cdpsimpleQuicksort requires GPU devices with compute SM 3.5 or higher." << std::endl;
        std::cout << "Current GPU device has compute SM" << properties.major << "." << properties.minor
                  << ". Exiting..." << std::endl;
        exit(EXIT_FAILURE);
    }


    if (device == -1) {
        std::cerr << "cdpSimpleQuicksort requires GPU devices with compute SM 3.5 or higher.  Exiting..." << std::endl;
        exit(EXIT_SUCCESS);
    }

    cudaSetDevice(device);

    // Create input data
    Metrics *h_data = inData;
    Metrics *d_data = 0;

    // Allocate CPU memory and initialize data.
    std::cout << "Initializing data:" << std::endl;
//    h_data = (Metrics *) malloc(num_items * sizeof(Metrics));
//    initialize_data(h_data, num_items);

//    if (verbose) {
        for (int i = 0; i < num_items; i++)
            std::cout << "Data [" << i << "]: " << h_data[i].similarity << std::endl;
//    }

    // Allocate GPU memory.
    HANDLE_ERROR(cudaMalloc((void **) &d_data, num_items * sizeof(Metrics)));
    HANDLE_ERROR(cudaMemcpy(d_data, h_data, num_items * sizeof(Metrics), cudaMemcpyHostToDevice));

    // Execute
    std::cout << "Running quicksort on " << num_items << " elements" << std::endl;
    run_qsort(d_data, num_items);

    HANDLE_ERROR(cudaMemcpy(h_data, d_data, num_items * sizeof(Metrics), cudaMemcpyDeviceToHost));


    // Check result
    std::cout << "Validating results: ";
    //check_results(num_items, d_data);
    for (int i = 1; i < num_items; i++) {
        std::cout << "Data [" << i-1 << "]: " << h_data[i-1].similarity << std::endl;
        assert(compare(h_data + i, h_data + (i - 1)) < 0);
    }

    free(h_data);
    HANDLE_ERROR(cudaFree(d_data));

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
    exit(EXIT_SUCCESS);
}

//
void testCudaSort() {
//
//    const unsigned int N = 100000;
//    Metrics *inData = new Metrics[N];
//
//    for (int i = 0; i < N; i++) {
//        inData[i].idx = N - i;
//    }
//
//    for (int i = 0; i < N; i++) {
//        // std::cout << "(" << i <<") " << inData[i] << " " ;
//    }
//    Metrics *outData = new Metrics[N];
//    unsigned int num_items = N;
////    CUDA_Quicksort(inData,outData, N,128);
//
//
//    for (int i = 0; i < N; i++) {
//        std::cout << "(" << i << ") " << outData[i].idx << " ";
//    }


}