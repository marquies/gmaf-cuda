//
// Created by Patrick Steinert on 16.10.21.

#include "graphcode.h"


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

void convertDict2Matrix(int size, int *destMatrix, json jsonMatrix);
#define N 1000

__global__ void vector_add(int *a, int *b, int *c) {
    int tid = blockIdx.x;
    if(tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}




void myThreadFun(int i, const std::vector<std::string> &files, std::vector<json> *arr)
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

        threads.push_back(std::thread(myThreadFun, i, sub, &tmp_jsons[i]));
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

void convertDict2Matrix(int size, int *destMatrix, json jsonMatrix) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {

           //destMatrix[i][j] = jsonMatrix.at(i).at(j);
            *((destMatrix+i*size) + j) = jsonMatrix.at(i).at(j);
        }
    }
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
    for (int i = 0; i < gc1Dictionary.size(); i++) {
        for (int j = 0; j < gc1Dictionary.size(); j++) {
            int a[N], b[N], c[N];
            int *d_a, *d_b, *d_c;
            //float *d_a;

            //a = (float*)malloc(sizeof(float) * N);

            // Allocate device memory for a
            cudaMalloc((void**)&d_a, sizeof(int) * N);
            cudaMalloc((void**)&d_b, sizeof(int) * N);
            cudaMalloc((void**)&d_c, sizeof(int) * N);

            for(int z=0; z<N; z++) {
                a[z] = -z;
                b[z] = z * z;
            }

            // Transfer data from host to device memory
            cudaMemcpy(d_a, a, sizeof(int) * N, cudaMemcpyHostToDevice);
            cudaMemcpy(d_b, b, sizeof(int) * N, cudaMemcpyHostToDevice);
            cudaMemcpy(d_c, c, sizeof(int) * N, cudaMemcpyHostToDevice);


            vector_add<<<N,1>>>(d_a, d_b, d_c);

            cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);



            // Cleanup after kernel execution
            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_c);

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


int getPosition(std::string string, std::vector<std::string> dictionary) {
    for (int i = 0; i < dictionary.size(); i++) {
        if (dictionary.at(i) == string) {
            return i;
        }
    }
    return -1;
}

void gmaf::GraphCode::foo() {}

void gmaf::GraphCode::calculateSimilarityV(int index, json *gcQuery, std::vector<json> *compares, int start, int end) {
    for (int i = start; i < end; i++) {

        std::cout << "Idx " << index << " i " << i << " limit(" << end << ")" << std::endl;

        float resultMetrics[3];
        calculateSimilaritySequential(*gcQuery, compares->at(i), resultMetrics);

        std::cout << "Similarity " << resultMetrics[0] << std::endl;
        std::cout << "Recommendation " << resultMetrics[1] << std::endl;
        std::cout << "Inferencing " << resultMetrics[2] << std::endl;
    }
}

