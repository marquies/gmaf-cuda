#include <nlohmann/json.hpp>
#include<string.h>
// for convenience
using json = nlohmann::json;

#include <iostream>
#include <fstream>

#include <vector>


#ifdef __CUDACC__
#warning using nvcc
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

#elif __GNUC__
#  include <features.h>
#  if __GNUC_PREREQ(8,0)
//      If  gcc_version >= 8.0
#include <filesystem>
namespace fs = std::filesystem;


#  else
//       Else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#  endif
#else
//    If not gcc
#include <filesystem>
namespace fs = std::__fs::filesystem;
#endif



#include "graphcode.cpp"

int getPosition(std::string string, std::vector<std::string> dictionary);

//__global__ void kernel( void ) {}




int main() {

    auto start = std::chrono::system_clock::now();
             kernel<<<1,1>>>();

    std::vector<json> arr;


    loadGraphCodes((char *) "/Users/breucking/dev/data/GraphCodes/WAPO_CG_Collection", &arr);

    auto loaded = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = loaded - start;
    std::time_t inter_time = std::chrono::system_clock::to_time_t(loaded);

    //std::cout << "finished computation at " << std::ctime(&inter_time)
    //          << "elapsed time: " << elapsed_seconds.count() << "s\n";

    std::cout << "loaded " << arr.size() << " graph code files. (" << "elapsed time: " << elapsed_seconds.count() << ")"
              << std::endl;


    int s = 4;
    int x = arr.size() / s;
    std::vector<std::thread> threads;


    for (int i = 0; i < s; i++) {
        //std::vector<std::string>   sub(&arr[i*x+1],&arr[(i+1)*x]);

        //calculateSimilarity(i, arr.at(0), arr,  i*x+1, (i+1)*x);
        //std::thread(calculateSimilarityV, i, arr.at(0), arr, (i*x+1), ((i+1)*x));
        int end = i == s - 1 ? arr.size() : (i + 1) * x;

        threads.push_back(std::thread(calculateSimilarityV, i, &arr.at(0), &arr, i * x + 1, end));
//        threads.push_back(std::thread(myThreadFun, i, sub, &tmp_jsons[i]));

    }

    for (auto &th: threads) {
        th.join();
    }

    /*
    //while(true) {
        for (int i = 1; i < arr.size(); i++) {

            float resultMetrics[3];
            calculateSimilarity(arr.at(0), arr.at(i), resultMetrics);



            std::cout << "Similarity " << resultMetrics[0] << std::endl;
            std::cout << "Recommendation " << resultMetrics[1] << std::endl;
            std::cout << "Inferencing " << resultMetrics[2] << std::endl;
        }
   // }*/
    auto end = std::chrono::system_clock::now();

    elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "finished computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n";

}


