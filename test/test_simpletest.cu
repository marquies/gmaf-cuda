//
// Created by breucking on 06.11.21.
//



#include "../src/graphcode.h"

double EPSILON = 0.000001;

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
bool AreSame(double a, double b)
{
    return fabs(a - b) < EPSILON;
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
    assert(m.recommendation == .5);
}

int main(int, char**)
{
    //testBasic();
    testBasic2();
}
