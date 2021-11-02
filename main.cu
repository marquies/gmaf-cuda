#include <iostream>
#include <unistd.h>
#include <cuda.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "main.cpp"

__global__ void kernel( void ) {}

/*
int main() {

             kernel<<<1,1>>>();
    std::vector<json> arr;

    thrust::host_vector<int> h_vec(32 << 20);
    loadGraphCodes((char *) "../graphcodes/", &arr);


    std::cout << "loaded " << arr.size() << " graph code files." << std::endl;

//icudaMalloc( (void**)&dev_c, sizeof(int) )

   // while(true) {
        for (int i = 1; i < arr.size(); i++) {

            float resultMetrics[3];
            calculateSimilarity(arr.at(0), arr.at(i), resultMetrics);



            std::cout << "Similarity " << resultMetrics[0] << std::endl;
            std::cout << "Recommendation " << resultMetrics[1] << std::endl;
            std::cout << "Inferencing " << resultMetrics[2] << std::endl;
        }
   // }

    return 0;
}
*/
