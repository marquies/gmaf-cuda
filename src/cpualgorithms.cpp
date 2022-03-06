//
// Created by breucking on 26.01.22.
//

#include "helper.h"
#include <stdlib.h>
#include <thread>
#include<string.h>
#include <iostream>
#include "cpualgorithms.h"


std::vector<Metrics>
demoCalculateCpuThreaded(GraphCode &gcQuery, std::vector<GraphCode> &compares, int numberOfThreads) {

    unsigned long x = compares.size() / numberOfThreads;
    std::vector<std::thread> threads;

    std::vector<Metrics> metrics = std::vector<Metrics>(compares.size());

    for (int i = 0; i < numberOfThreads; i++) {
        unsigned long start = i * x;
        unsigned long end = i == numberOfThreads - 1 ? compares.size() - 1 : start + x -1; //(i + 1) * x;
        threads.push_back(std::thread(calculateSimilarityV, &gcQuery, &compares, start, end, &metrics, i));
    }

    for (auto &th: threads) {
        th.join();
    }
    return metrics;
}


void
calculateSimilarityV(GraphCode *gcQuery, std::vector<GraphCode> *compares, unsigned long start, unsigned long end,
                     std::vector<Metrics> *metrics, int index) {
    if (compares == NULL) {
        std::cout << "Argument compare is NULL" << std::endl;
        exit(1);
    }


    for (unsigned long i = start; i <= end; i++) {

        if (G_DEBUG)
            std::cout << "Idx " << index << " i " << i << " limit(" << end << ")" << std::endl;

//        float resultMetrics[3];
        Metrics res = demoCalculateSimilaritySequentialOrdered(*gcQuery, compares->at(i));
        //calculateSimilaritySequential(*gcQuery, compares->at(i), resultMetrics);
        //calculateSimilarityCuda(*gcQuery, compares->at(i), resultMetrics);
        res.idx = i;
        if (G_DEBUG) {
            std::cout << "Similarity " << res.similarity << std::endl;
            std::cout << "Recommendation " << res.recommendation << std::endl;
            std::cout << "Inferencing " << res.inferencing << std::endl;
        }

        metrics->at(i) = res;

    }

//    return metrics;
}


Metrics demoCalculateSimilaritySequentialOrdered(GraphCode gcQuery, GraphCode gcCompare) {

    int sim = 0;

    unsigned short *matrix1 = gcQuery.matrix;
    unsigned short *matrix2 = gcCompare.matrix;

    for (const auto &item: *gcQuery.dict) {
        for (const auto &item2: *gcCompare.dict) {
            if (item == item2) {
                sim++;
            }
        }
    }
    int num_of_non_zero_edges = 0;
    int edge_metric_count = 0;
    int edge_type = 0;

    for (unsigned long i = 0; i < gcQuery.dict->size(); i++) {
        for (unsigned long j = 0; j < gcQuery.dict->size(); j++) {

            if (i != j && matrix1[i * gcQuery.dict->size() + j] != 0) {
                num_of_non_zero_edges++;

                unsigned long position1 = i;
                unsigned long position2 = j;
                //TODO: Was soll das?
                if (position1 == -1 || position2 == -1) {
                    continue;
                }

                std::basic_string<char> &v1 = gcQuery.dict->at(i);
                std::basic_string<char> &v2 = gcQuery.dict->at(j);

                const std::vector<std::string>::iterator &it1 = std::find(gcCompare.dict->begin(), gcCompare.dict->end(), v1);
                if (it1 == gcCompare.dict->end()) {
                    continue;
                }
                long transX = std::distance(gcCompare.dict->begin(), it1);

                const std::vector<std::string>::iterator &it2 = std::find(gcCompare.dict->begin(), gcCompare.dict->end(), v2);
                if (it2 == gcCompare.dict->end()) {
                    continue;
                }
                long transY = std::distance(gcCompare.dict->begin(), it2);


                
                unsigned short edge = matrix2
                        [transX * gcCompare.dict->size()+transY];
                        //matrix2[position1 * gcQuery.dict->size() + position2];//matrix2[position1][position2];
                if (edge != 0) {
                    edge_metric_count++;
                }
                if (edge == matrix1[i * gcQuery.dict->size() + j]) {
                    edge_type++;
                }

            }
        }
    }

    float node_metric = (float) sim / (float) gcQuery.dict->size();
    float edge_metric = 0.0;
    if (num_of_non_zero_edges > 0)
        edge_metric = (float) edge_metric_count / (float) num_of_non_zero_edges;
    float edge_type_metric = 0.0;
    if (edge_metric_count > 0)
        edge_type_metric = (float) edge_type / (float) edge_metric_count;

    Metrics metrics;

    metrics.similarity = node_metric;
    metrics.recommendation = edge_metric;
    metrics.inferencing = edge_type_metric;
    return metrics;

}

Metrics demoCalculateSimilaritySequentialOrdered(json gc1, json gc2) {
    int sim = 0;

    json gc1Dictionary;
    unsigned long numberOfElements1;
    unsigned long items1;
    unsigned short int *matrix1;

    convertJsonGc2GcDataStructure(gc1, gc1Dictionary, numberOfElements1, items1, matrix1);

    json gc2Dictionary;
    unsigned long numberOfElements2;
    unsigned long items2;
    unsigned short int *matrix2;
    convertJsonGc2GcDataStructure(gc2, gc2Dictionary, numberOfElements2, items2, matrix2);

    std::vector<std::basic_string<char>> dict2;
    for (const auto &item2: gc2Dictionary.items()) {
        dict2.push_back(item2.value().get<std::basic_string<char>>());
    }


    for (const auto &item: gc1Dictionary.items()) {

        std::basic_string<char> str = item.value().get<std::basic_string<char>>();

        for (const auto &item2: gc2Dictionary.items()) {
            if (str == item2.value()) {
                sim++;
            }
        }

    }
    int num_of_non_zero_edges = 0;
    int edge_metric_count = 0;
    int edge_type = 0;

    for (unsigned long i = 0; i < gc1Dictionary.size(); i++) {
        for (unsigned long j = 0; j < gc1Dictionary.size(); j++) {

            if (i != j && matrix1[i * gc1Dictionary.size() + j] != 0) {
                num_of_non_zero_edges++;

                unsigned long position1 = i;
                unsigned long position2 = j;
                // TODO: Was soll das?
                if (position1 == -1 || position2 == -1) {
                    continue;
                }

                int edge = matrix2[position1 * gc1Dictionary.size() + position2];//matrix2[position1][position2];
                if (edge != 0) {
                    edge_metric_count++;
                }
                if (edge == matrix1[i * gc1Dictionary.size() + j]) {
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

    Metrics metrics;
    metrics.similarity = node_metric;
    metrics.recommendation = edge_metric;
    metrics.inferencing = edge_type_metric;
    return metrics;

}