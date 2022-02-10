//
// Created by breucking on 06.11.21.
//



#include <helper.h>
#include "../src/graphcode.h"
#include "testhelper.h"


void testAnother();

void testCompare();

void testBasic() {
//    nlohmann::json gcq;
//    gcq["dictionary"] = {"head", "body"};
//    gcq["matrix"] = {{1, 1},
//                     {0, 1}};
//
//    std::vector<json> others;
//    others.push_back(gcq);
//
//    gmaf::GraphCode gc;
//
//    const std::vector<Metrics> &metrics = gc.calculateSimilarityV(0, &gcq, &others, 0, 1);
//    assert(metrics.size() == 1);
//    Metrics m = metrics[0];
//    assert(m.similarity == 1);
//    assert(m.inferencing == 1);
//    assert(m.recommendation == 1);
}

void testBasic2() {
//    nlohmann::json gcq;
//    gcq["dictionary"] = {"head", "body", "foot"};
//    gcq["matrix"] = {{1, 1, 0},
//                     {0, 1, 0},
//                     {0, 1, 1}};
//
//    nlohmann::json gce;
//    gce["dictionary"] = {"head", "body", "torso"};
//    gce["matrix"] = {{1, 2, 0},
//                     {0, 1, 0},
//                     {0, 0, 1}};
//
//    std::vector<json> others;
//    others.push_back(gce);
//
//    gmaf::GraphCode gc;
//
//    const std::vector<Metrics> &metrics = gc.calculateSimilarityV(0, &gcq, &others, 0, 1);
//    assert(metrics.size() == 1);
//    Metrics m = metrics[0];
//    //std::cout << "!!!Similarity " <<m.similarity << "==" << 2./3. << std::endl;
//    assert(AreSame(m.similarity, (float) 2. / 3.));
//    assert(m.inferencing == 0);
////    assert(m.recommendation == .5);
}

int main(int, char **) {
    //testBasic();
    testBasic2();
    testAnother();
    testCompare();
}

void testCompare() {
    Metrics a;
    a.idx = 50009;
    a.similarity = 0.500090003;
    a.inferencing = 0.500090003;
    a.recommendation = 0.500090003;

    Metrics b;
    b.idx = 50008;
    b.similarity = 0.500079989;
    b.inferencing = 0.500079989;
    b.recommendation = 0.500079989;

    float cmp = compare(&a, &b);
    assert(cmp > 0);
    cmp = compare(&b, &a);
    assert(cmp < 0);


    a.idx = 50011;
    a.similarity = 0.500109971;
    a.inferencing = 0.500109971;
    a.recommendation = 0.500109971;

    b.idx = 50010;
    b.similarity = 0.500100017;
    b.inferencing = 0.500100017;
    b.recommendation = 0.500100017;

    cmp = compare(&a, &b);
    assert(cmp > 0);
    cmp = compare(&b, &a);
    assert(cmp < 0);

}

void testAnother() {


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
