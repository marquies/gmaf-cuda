//
// Created by breucking on 26.01.22.
//

#include "helper.h"
#include "gcloadunit.cuh"
#include <stdlib.h>
#include <thread>
#include<string.h>
#include <iostream>
#include "cpualgorithms.h"


std::vector<Metrics>
demoCalculateCpuThreaded(GraphCode &gcQuery, std::vector<GraphCode> &compares, int numberOfThreads) {

    int x = compares.size() / numberOfThreads;
    std::vector<std::thread> threads;

    std::vector<Metrics> metrics = std::vector<Metrics>(compares.size());

    for (int i = 0; i < numberOfThreads; i++) {
        int start = i * x;
        int end = i == numberOfThreads - 1 ? compares.size() - 1 : start + x -1; //(i + 1) * x;
        threads.push_back(std::thread(calculateSimilarityV, &compares.at(0), &compares, start, end, &metrics, i));
    }

    for (auto &th: threads) {
        th.join();
    }
    return metrics;
}


void
calculateSimilarityV(GraphCode *gcQuery, std::vector<GraphCode> *compares, int start, int end,
                     std::vector<Metrics> *metrics, int index) {
    if (compares == NULL) {
        std::cout << "Argument compare is NULL" << std::endl;
        exit(1);
    }


    for (int i = start; i <= end; i++) {

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

    for (int i = 0; i < gcQuery.dict->size(); i++) {
        for (int j = 0; j < gcQuery.dict->size(); j++) {

            if (i != j && matrix1[i * gcQuery.dict->size() + j] != 0) {
                num_of_non_zero_edges++;

                int position1 = i;
                int position2 = j;
                if (position1 == -1 || position2 == -1) {
                    continue;
                }

                int edge = matrix2[position1 * gcQuery.dict->size() + position2];//matrix2[position1][position2];
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
    int numberOfElements1;
    long items1;
    unsigned short int *matrix1;

    convertGc2Cuda(gc1, gc1Dictionary, numberOfElements1, items1, matrix1);

    json gc2Dictionary;
    int numberOfElements2;
    long items2;
    unsigned short int *matrix2;
    convertGc2Cuda(gc2, gc2Dictionary, numberOfElements2, items2, matrix2);

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

    for (int i = 0; i < gc1Dictionary.size(); i++) {
        for (int j = 0; j < gc1Dictionary.size(); j++) {

            if (i != j && matrix1[i * gc1Dictionary.size() + j] != 0) {
                num_of_non_zero_edges++;

                int position1 = i;
                int position2 = j;
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