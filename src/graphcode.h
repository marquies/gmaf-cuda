//
// Created by Patrick Steinert on 16.10.21.
//

#ifndef GRAPHCODE_H
#define GRAPHCODE_H

#include <nlohmann/json.hpp>
#include <vector>

using json = nlohmann::json;
namespace gmaf {

    class GraphCode {
    public:
        void loadGraphCodes(char *directory, int limit, std::vector<json> *arr);

        void calculateSimilarityV(int index, json *gcQuery, std::vector<json> *compares, int start, int end);

        void foo();
    };

/*
class graphcode {

};
*/
}
#endif //GRAPHCODE_H