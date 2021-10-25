#include <nlohmann/json.hpp>
#include<string.h>
// for convenience
using json = nlohmann::json;

#include <iostream>
#include <fstream>

#include <vector>
#include <filesystem>

int getPosition(std::string string, std::vector<std::string> dictionary);

namespace fs = std::__fs::filesystem;




//__global__ void kernel( void ) {}



int *loadGraphCodes(char *directory, std::vector<json> *arr) {

    for (const auto &entry: fs::directory_iterator(directory)) {
        std::cout << entry.path() << std::endl;
        try {
            std::ifstream ifs(entry.path());
            json jf = json::parse(ifs);

            std::cout << jf["dictionary"] << std::endl;

            arr->push_back(jf);

        } catch (json::exception &e) {
            std::cerr << e.what() << '\n';
        }
    }

}


int calculateSimilarity(json gc1, json gc2, float *results) {
    json gc1Dictionary = gc1["dictionary"];
    json gc2Dictionary = gc2["dictionary"];

    std::string gc1Dict[gc1Dictionary.size()];

    int n = 0;

    int sim = 0;


    int matrix1[gc1Dictionary.size()][gc1Dictionary.size()];
    int matrix2[gc2Dictionary.size()][gc2Dictionary.size()];

    json jsonMatrix1 = gc1["matrix"];
    json jsonMatrix2 = gc2["matrix"];

    //std::cout << jsonMatrix1.at(0).at(0) << std::endl;

    for (int i = 0; i < gc1Dictionary.size(); i++) {
        for (int j = 0; j < gc1Dictionary.size(); j++) {

            matrix1[i][j] = jsonMatrix1.at(i).at(j);
        }
    }

    for (int i = 0; i < gc2Dictionary.size(); i++) {
        for (int j = 0; j < gc2Dictionary.size(); j++) {

            matrix2[i][j] = jsonMatrix2.at(i).at(j);
        }
    }


    std::vector<std::string> dict2;
    for (const auto &item2: gc2Dictionary.items()) {
        dict2.push_back(item2.value().get<std::string>());
    }


    for (const auto &item: gc1Dictionary.items()) {
        std::cout << item.value() << "\n";
        std::string str = item.value().get<std::string>();
        gc1Dict[n++] = str;


        for (const auto &item2: gc2Dictionary.items()) {
            if (str == item2.value()) {
                std::cout << "Match" << std::endl;
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
                std::cout << "Pos " << position1 << " " << position2 << std::endl;
                if (position1 == -1 || position2 == -1) {
                    continue;
                }

                int edge = matrix2[position1][position2];
                if (edge != 0) {
                    edge_metric_count++;
                }
                if(edge == matrix1[i][j]) {
                    edge_type++;
                }

            }
        }
    }

    float node_metric = (float) sim / (float) gc1Dictionary.size();
    float edge_metric = 0.0;
    if (num_of_non_zero_edges > 0)
        edge_metric = (float) edge_metric_count / (float) num_of_non_zero_edges;
    float edge_type_metric =  0.0;
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

int main() {


    std::vector<json> arr;


    loadGraphCodes((char *) "../graphcodes/", &arr);


    std::cout << "loaded " << arr.size() << " graph code files." << std::endl;

    while(true) {
        for (int i = 1; i < arr.size(); i++) {

            float resultMetrics[3];
            calculateSimilarity(arr.at(0), arr.at(i), resultMetrics);

            // kernel<<<1,1>>>();


            std::cout << "Similarity " << resultMetrics[0] << std::endl;
            std::cout << "Recommendation " << resultMetrics[1] << std::endl;
            std::cout << "Inferencing " << resultMetrics[2] << std::endl;
        }
    }

    return 0;
}
