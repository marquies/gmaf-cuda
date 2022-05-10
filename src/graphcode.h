//
// Created by Patrick Steinert on 16.10.21.
//

#ifndef GRAPHCODE_H
#define GRAPHCODE_H

#include <nlohmann/json.hpp>
#include <vector>

/**
 * Object to contain Graph Code elements.
 *
 */
typedef struct GraphCode {
    /**
     * Pointer to a vector with the dictionary elements.
     */
    std::vector<std::string> *dict;

    /**
     * Pointer to a linear 2D matrix.
     */
    unsigned short *matrix;
} GraphCode;


using json = nlohmann::json;

/**
 * Object to contain Metrics of a query Graph Code.
 *
 * The query Graph Code need to be recorded somewhere else.
 * The compare Graph Code can be found via the idx field.
 */
struct Metrics {
    /**
     * Index of the compare Graph Code in the collection.
     */
    unsigned long idx;

    /**
     * Similarity metric value of the query Graph Code and the compare Graph Code.
     */
    float similarity;

    /**
     * Recommendation metric value of the query Graph Code and the compare Graph Code.
     */
    float recommendation;

    /**
     * Inferencing metric value of the query Graph Code and the compare Graph Code.
     */
    float inferencing;

    /**
     * Calculated value to compare two metrics.
     */
    float compareValue;
};

#endif //GRAPHCODE_H
