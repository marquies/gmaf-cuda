//
// Created by breucking on 06.11.21.
//



#include "../src/graphcode.h"
#include "testhelper.cpp"


void testAnother();

void testBasic()
{
    nlohmann::json gcq;
    gcq["dictionary"] = { "head", "body"};
    gcq["matrix"] = {{1,1}, {0,1}};

    std::vector<json> others;
    others.push_back(gcq);

    gmaf::GraphCode gc;

    const std::vector<Metrics> &metrics = gc.calculateSimilarityV(0, &gcq, &others, 0, 1);
    assert(metrics.size() == 1);
    Metrics m = metrics[0];
    assert(m.similarity == 1);
    assert(m.inferencing == 1);
    assert(m.recommendation == 1);
}

void testBasic2()
{
    nlohmann::json gcq;
    gcq["dictionary"] = { "head", "body", "foot"};
    gcq["matrix"] = {{1,1,0}, {0,1,0}, {0,1,1}};

    nlohmann::json gce;
    gce["dictionary"] = { "head", "body", "torso"};
    gce["matrix"] = {{1,2,0}, {0,1,0}, {0,0,1}};

    std::vector<json> others;
    others.push_back(gce);

    gmaf::GraphCode gc;

    const std::vector<Metrics> &metrics = gc.calculateSimilarityV(0, &gcq, &others, 0, 1);
    assert(metrics.size() == 1);
    Metrics m = metrics[0];
    //std::cout << "!!!Similarity " <<m.similarity << "==" << 2./3. << std::endl;
    assert(AreSame(m.similarity,(float) 2./3.));
    assert(m.inferencing == 0);
//    assert(m.recommendation == .5);
}

int main(int, char**)
{
    //testBasic();
    testBasic2();
    testAnother();
}

void testAnother() {


    long addedElements = 0;
    long addedDictItems = 0;
    int count = 10000;
    int dimension = 100;

    for (int n = 0; n < count; n++) {


        unsigned short *data = (unsigned short *) malloc(sizeof(unsigned short) * dimension * dimension);

        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                if (i == j) {
                    //data[i][j] = 1;
                    data[i * dimension + j] = 1;
                } else {
                    data[i * dimension + j] = i + j % 2;
                }
            }
        }
    }
}
