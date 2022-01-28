//
// Created by breucking on 26.01.22.
//

#ifndef GCSIM_CPUALGORITHMS_H
#define GCSIM_CPUALGORITHMS_H

#include <vector>
#include "graphcode.h"


std::vector<Metrics> demoCalculateCpuThreaded(std::vector<GraphCode> &arr, GraphCode &gc, int numberOfThreads = 2);

void calculateSimilarityV(int index, GraphCode *gcQuery, std::vector<GraphCode> *compares, int start, int end, std::vector<Metrics> *metrics);

Metrics demoCalculateSimilaritySequentialOrdered(GraphCode gc1, GraphCode gc2);

Metrics demoCalculateSimilaritySequentialOrdered(json gc1, json gc2);

#endif //GCSIM_CPUALGORITHMS_H
