//
// Created by breucking on 26.01.22.
//

#ifndef GCSIM_CPUALGORITHMS_H
#define GCSIM_CPUALGORITHMS_H

#include <vector>
#include "graphcode.h"



/**
 * Calculates the metrics for the graph codes in a thread parallel way.
 * @param compares the elements to calculate.
 * @param gcQuery the query Graph Code.
 * @param numberOfThreads number of threads to create - best is number of cores -1 or -2.
 * @return an unsorted list of the calculated metrics
 */
std::vector<Metrics>
demoCalculateCpuThreaded(GraphCode &gcQuery, std::vector<GraphCode> &compares, int numberOfThreads = 2);

/**
 * Calculates the metrics for a chunk of a vector of Graph Codes.
 * @param index id for the job, can be used as a thread id.
 * @param gcQuery  the query Graph Code
 * @param compares  the vector of comparables
 * @param start start position in the vector
 * @param end end position in the vector
 * @param metrics return vector for the calculated metrics (need to be same size than compares)
 */
void calculateSimilarityV(GraphCode *gcQuery, std::vector<GraphCode> *compares, unsigned long start, unsigned long end,
                          std::vector<Metrics> *metrics, int index = 0);

/**
 * Calculates the Metrics for to Graph Codes.
 * @param gcQuery the query Graph Code
 * @param gcCompare  the compareUUID Graph Code
 * @return the calculated metrics
 */
Metrics demoCalculateSimilaritySequentialOrdered(GraphCode gcQuery, GraphCode gcCompare);

/**
 * Calculates the Metrics for to Graph Codes.
 * @param gcQuery the query Graph Code as json object
 * @param gcCompare  the compareUUID Graph Code as json object
 * @return the calculated metrics
 */
Metrics demoCalculateSimilaritySequentialOrdered(json gcQuery, json gcCompare);



#endif //GCSIM_CPUALGORITHMS_H
