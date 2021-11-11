//
// Created by Patrick Steinert on 16.10.21.

#include "graphcode.h"

#include "cudahelper.cuh"
#include "helper.h"

#include <thread>
#include <chrono>
#include <ctime>

#include <iostream>
#include <fstream>

#include <nlohmann/json.hpp>


#ifdef __CUDACC__
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;
#elif __GNUC__
  #include <features.h>
  #if __GNUC_PREREQ(8,0)
  //      If  gcc_version >= 8.0
    #include <filesystem>
    namespace fs = std::filesystem;

  #  else
  //       Else gcc_version < 8.0
    #include <experimental/filesystem>
    namespace fs = std::experimental::filesystem;
  #endif
#else
//    If not gcc
  #include <filesystem>
  namespace fs = std::__fs::filesystem;
#endif

//0. Ein File haben wir (gcQuery), dazu die similarity berehcnen
//
//1. Für jedes Element in der Collection:
//1.a Einlesen der Files in einem Directory (for each file)
//1.b Für jedes File similarity berechnen
//   2. Für jedes Element im Dictionary (Annotation) der gcQuery
//   2. a-> für jedes element in der Matrix
//        wenn der wert in QCquery != 0
//            num_of_non_zero_edges = 1
//        -> check anderes GC element:
//            wenn es die Begriffe im Dict nicht gibt skip
//            else
//                -> wenn matrix wert != 0 dann edge_metric_count = 1
//                -> wenn matrix wert == gcQuery Wert (beziehungstyp identisch) edge_type =
//  2. b -> Kalkulation der
//1.c Sortieren


using json = nlohmann::json;


int getPosition(std::string string, std::vector<std::string> dictionary);

//void convertDict2Matrix(int size, int *destMatrix, json jsonMatrix);
#define N 1000


//copy of other check function
__global__ void check2(int *data, unsigned long matrixSize, int *pInt) {
    int tid = blockIdx.x;

    //if (tid % matrixSize != 0) {
        if (data[tid] != 0) {
            pInt[tid] = 1;
        }
    //}

}
/*
__global__ void check( cudaArray_t data) {
    ((cudaArray *)data)
}
*/
//
//__global__ void
//vector_add(int *a, int *b, int *c, std::basic_string<char> *gc1Dict, int n, std::vector<std::string> dict2,
//           int * matrix1) {
//    int tid = blockIdx.x;
//    //if(tid < N) {
//    //    c[tid] = a[tid] + b[tid];
//    //}
//
//
//    for (int i = 0; i < n; i++) {
//        for (int j = 0; j < n; j++) {
//
//            if (i != j && matrix1[i][j] != 0) {
//                //num_of_non_zero_edges++;
//
//                //int position1 = getPosition(gc1Dict[i], dict2);
//                int position1 = -1;
//                for (int i = 0; i < dict2.size(); i++) {
//                    if (dict2.at(i) == gc1Dict[i]) {
//                        position1 = i;
//                        break;
//                    }
//                }
//
//
//
//                //int position2 = getPosition(gc1Dict[j], dict2);
//                int position2 = -1;
//                for (int i = 0; i < dict2.size(); i++) {
//                    if (dict2.at(i) == gc1Dict[j]) {
//                        position2 = i;
//                        break;
//                    }
//                }
//
//                //std::cout << "Pos " << position1 << " " << position2 << std::endl;
//                if (position1 == -1 || position2 == -1) {
//                    continue;
//                }
//                /*
//                int edge = matrix2[position1][position2];
//                if (edge != 0) {
//                    edge_metric_count++;
//                }
//                if (edge == matrix1[i][j]) {
//                    edge_type++;
//                }
//                 */
//
//            }
//        }
//    }
//}




void threadLoader(int i, const std::vector<std::string> &files, std::vector<json> *arr)
{


    for (int j = 0; j < files.size(); j++) {

        //std::cout << entry.path() << std::endl;
        try {
            //std::ifstream ifs(entry.path());
            std::ifstream ifs(files.at(j));
            json jf = json::parse(ifs);

            //std::cout << jf["dictionary"] << std::endl;
            arr->push_back(jf);

        } catch (json::exception &e) {
            std::cerr << e.what() << '\n';
        }
        if (j % 10000 == 0)
            std::cout << "Thread " << i << ": Status " << j << std::endl;
    }


}

void gmaf::GraphCode::loadGraphCodes(char *directory, int limit,  std::vector<json> *arr) {



    std::vector<std::string> files;


    for (const auto &entry: fs::directory_iterator(directory)) {
        files.push_back(entry.path().string());
    }

    int s = 4;
    int x;
    if (limit > files.size()) {
        x = files.size() / s;
    } else {
        x = limit / s;
    }
    std::vector<std::thread> threads;

    std::vector<json> tmp_jsons[s];

    for (int i = 0; i < s; i++) {
        std::vector<std::string> sub(&files[i * x + 1], &files[(i + 1) * x]);

        threads.push_back(std::thread(threadLoader, i, sub, &tmp_jsons[i]));
    }

    for (auto &th: threads) {
        th.join();
    }

    arr->reserve(tmp_jsons[0].size() + tmp_jsons[1].size() + tmp_jsons[2].size() + tmp_jsons[3].size());
    arr->insert(arr->end(), tmp_jsons[0].begin(), tmp_jsons[0].end());
    arr->insert(arr->end(), tmp_jsons[1].begin(), tmp_jsons[1].end());
    arr->insert(arr->end(), tmp_jsons[2].begin(), tmp_jsons[2].end());
    arr->insert(arr->end(), tmp_jsons[3].begin(), tmp_jsons[3].end());



//        if (i++ > limit) {
//            break;
//        }


}


int calculateSimilaritySequential(json gc1, json gc2, float *results) {
    json gc1Dictionary = gc1["dictionary"];
    json gc2Dictionary = gc2["dictionary"];

    std::string gc1Dict[gc1Dictionary.size()];

    int n = 0;

    int sim = 0;



    int matrix1[gc1Dictionary.size()][gc1Dictionary.size()];
    convertDict2Matrix(gc1Dictionary.size(), (int *) matrix1, gc1["matrix"]);

    int matrix2[gc2Dictionary.size()][gc2Dictionary.size()];
    convertDict2Matrix(gc2Dictionary.size(), (int *) matrix2, gc2["matrix"]);


    std::vector<std::string> dict2;
    for (const auto &item2: gc2Dictionary.items()) {
        dict2.push_back(item2.value().get<std::string>());
    }


    for (const auto &item: gc1Dictionary.items()) {
        //std::cout << item.value() << "\n";
        std::string str = item.value().get<std::string>();
        gc1Dict[n++] = str;


        for (const auto &item2: gc2Dictionary.items()) {
            if (str == item2.value()) {
                //std::cout << "Match" << std::endl;
                sim++;
            }
        }
    }
    int num_of_non_zero_edges = 0;
    int edge_metric_count = 0;
    int edge_type = 0;
    for (int i = 0; i < gc1Dictionary.size(); i++) {
        for (int j = 0; j < gc1Dictionary.size(); j++) {

            if (i != j && matrix1[i][j] != 0) {
                num_of_non_zero_edges++;

                int position1 = getPosition(gc1Dict[i], dict2);
                int position2 = getPosition(gc1Dict[j], dict2);
                //std::cout << "Pos " << position1 << " " << position2 << std::endl;
                if (position1 == -1 || position2 == -1) {
                    continue;
                }

                int edge = matrix2[position1][position2];
                if (edge != 0) {
                    edge_metric_count++;
                }
                if (edge == matrix1[i][j]) {
                    edge_type++;
                }

            }
        }
    }

    float node_metric = (float) sim / (float) gc1Dictionary.size();
    float edge_metric = 0.0;
    if (num_of_non_zero_edges > 0)
        edge_metric = (float) edge_metric_count / (float) num_of_non_zero_edges;
    float edge_type_metric = 0.0;
    if (edge_metric_count > 0)
        edge_type_metric = (float) edge_type / (float) edge_metric_count;

    results[0] = node_metric;
    results[1] = edge_metric;
    results[2] = edge_type_metric;
    return 0;

}



int calculateSimilarityCuda(json gc1, json gc2, float *results) {
    json gc1Dictionary = gc1["dictionary"];
    json gc2Dictionary = gc2["dictionary"];



    std::string gc1Dict[gc1Dictionary.size()];

    int n = 0;

    int sim = 0;




    int matrix1[gc1Dictionary.size()][gc1Dictionary.size()];
    convertDict2Matrix(gc1Dictionary.size(), (int *) matrix1, gc1["matrix"]);

    int matrix2[gc2Dictionary.size()][gc2Dictionary.size()];
    convertDict2Matrix(gc2Dictionary.size(), (int *) matrix2, gc2["matrix"]);



    std::vector<std::string> dict2;
    for (const auto &item2: gc2Dictionary.items()) {
        dict2.push_back(item2.value().get<std::string>());
    }


    for (const auto &item: gc1Dictionary.items()) {
        //std::cout << item.value() << "\n";
        std::string str = item.value().get<std::string>();
        gc1Dict[n++] = str;


        for (const auto &item2: gc2Dictionary.items()) {
            if (str == item2.value()) {
                //std::cout << "Match" << std::endl;
                sim++;
            }
        }
    }
    int num_of_non_zero_edges = 0;
    int edge_metric_count = 0;
    int edge_type = 0;

    int a[gc1Dictionary.size() * gc1Dictionary.size()];
    int count = 0;
    for (int i = 0; i < gc1Dictionary.size(); i++)
        for (int j = 0; j < gc1Dictionary.size(); j++) {
            a[count++] = matrix1[i][j];
        }

    int items = gc1Dictionary.size() * gc1Dictionary.size();
    for(int i = 0; i < items; i++) {
        std::cout << "pos: " << i << " value: " << a[i] << std::endl;
    }

    //cudaArray_t dst;
    int *d_a;
    int *founds;
    // Allocate device memory for a
    //cudaMalloc((void**)&d_a, sizeof(int) );


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
    HANDLE_ERROR(cudaMemcpy(d_a, a, sizeof(int) * gc1Dictionary.size() * gc1Dictionary.size(), cudaMemcpyHostToDevice));


    /*
    cudaMemcpy(d_b, b, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, sizeof(int) * N, cudaMemcpyHostToDevice);
    */

    //vector_add<<<N, 1>>>(d_a, d_b, d_c, gc1Dict, gc1Dictionary.size(), dict2, (int *) matrix1);
    check2<<<items, 1>>>(d_a, gc1Dictionary.size(), founds);


    int f[items];
    HANDLE_ERROR(cudaMemcpy(f, founds, sizeof (int)* gc1Dictionary.size() * gc1Dictionary.size(), cudaMemcpyDeviceToHost));

    for(int i = 0; i < items; i++) {
        std::cout << "pos: " << i << " value: " << f[i] << std::endl;
    }

    // Cleanup after kernel execution
    HANDLE_ERROR(cudaFree(d_a));
    //cudaFree(d_b);
    //cudaFree(d_c);
    //cudaFree(dst);



    float node_metric = (float) sim / (float) gc1Dictionary.size();
    float edge_metric = 0.0;
    if (num_of_non_zero_edges > 0)
        edge_metric = (float) edge_metric_count / (float) num_of_non_zero_edges;
    float edge_type_metric = 0.0;
    if (edge_metric_count > 0)
        edge_type_metric = (float) edge_type / (float) edge_metric_count;

    results[0] = node_metric;
    results[1] = edge_metric;
    results[2] = edge_type_metric;
    return 0;

}


int getPosition(std::string string, std::vector<std::string> dictionary) {
    for (int i = 0; i < dictionary.size(); i++) {
        if (dictionary.at(i) == string) {
            return i;
        }
    }
    return -1;
}



std::vector<Metrics>
gmaf::GraphCode::calculateSimilarityV(int index, json *gcQuery, std::vector<json> *compares, int start, int end) {
    if (compares == NULL) {
        std::cout << "Argument compare is NULL" << std::endl;
        exit(1);
    }

    std::vector<Metrics> metrics;

    for (int i = start; i < end; i++) {

        std::cout << "Idx " << index << " i " << i << " limit(" << end << ")" << std::endl;

        float resultMetrics[3];
        //calculateSimilaritySequential(*gcQuery, compares->at(i), resultMetrics);
        calculateSimilarityCuda(*gcQuery, compares->at(i), resultMetrics);

        std::cout << "Similarity " << resultMetrics[0] << std::endl;
        std::cout << "Recommendation " << resultMetrics[1] << std::endl;
        std::cout << "Inferencing " << resultMetrics[2] << std::endl;

        Metrics m;
        m.similarity = resultMetrics[0];
        m.recommendation = resultMetrics[1];
        m.inferencing = resultMetrics[2];

        metrics.push_back(m);

    }

    return metrics;
}

