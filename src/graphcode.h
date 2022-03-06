//
// Created by Patrick Steinert on 16.10.21.
//

#ifndef GRAPHCODE_H
#define GRAPHCODE_H

#include <nlohmann/json.hpp>
#include <vector>


typedef struct GraphCode {
    std::vector<std::string> *dict;
    unsigned short *matrix;
} GraphCode;


using json = nlohmann::json;

struct Metrics {
    unsigned long idx;
    float similarity;
    float recommendation;
    float inferencing;
    float compareValue;
};


namespace gmaf {

    class GraphCode {
    public:
        void loadGraphCodes(char *directory, int limit, std::vector<json> *arr);

    };
}
#endif //GRAPHCODE_H
