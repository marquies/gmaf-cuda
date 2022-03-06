//
// Created by breucking on 26.01.22.
//

#include <cassert>
#include <cpualgorithms.h>
#include "testhelper.h"

void testBasic();

void testMassTest();

int main(int, char**)
{
    testBasic();
    testMassTest();
}

void testMassTest() {
    std::vector<GraphCode> arr;

    const GraphCode &m = generateTestDataGc(10);

    for (int i = 0; i < 1000; i++) {
        arr.push_back(m);
    }


    const std::vector<Metrics> &results1 = demoCalculateCpuThreaded(arr.at(1), arr, 4);

    assert(results1.size() == arr.size());

    for (int i = 0; i < 1000; i++) {
        assert(results1.at(i).idx == i);
    }

    const GraphCode &misfit = generateTestDataGc(9);
    arr.push_back(misfit);
    const std::vector<Metrics> &results2 = demoCalculateCpuThreaded(arr.at(1000), arr, 4);

    assert(results2.size() == arr.size());
    assert(results2.at(1000).idx == 1000);
    assert(results2.at(1000).inferencing == 1);


    for (int i = 0; i < 1000; i++) {
        assert(results2.at(i).idx == i);
    }

    const std::vector<Metrics> &results3 = demoCalculateCpuThreaded(arr.at(0), arr, 4);

    assert(results3.size() == arr.size());
    assert(results3.at(1000).idx == 1000);
    assert(results3.at(1000).inferencing != 0);

}

void testBasic() {

    GraphCode gc_query;
    gc_query.dict = new std::vector<std::string>();
    gc_query.dict->push_back("My1");
    gc_query.dict->push_back("My2");

    gc_query.matrix = (unsigned short *) malloc(sizeof(unsigned short) * 4);
    gc_query.matrix[0] = 1;
    gc_query.matrix[1] = 2;
    gc_query.matrix[2] = 3;
    gc_query.matrix[3] = 4;

    std::vector<GraphCode> arr;
    arr.push_back(gc_query);

    const GraphCode &m = generateTestDataGc(10);
    arr.push_back(m);
    arr.push_back(m);
    arr.push_back(m);
    arr.push_back(m);

    const std::vector<Metrics> &results = demoCalculateCpuThreaded(arr.at(0), arr, 2);

    assert(results.size() == arr.size());

    assert(results.at(0).idx == 0);
    assert(results.at(1).idx == 1);
    assert(results.at(2).idx == 2);
    assert(results.at(3).idx == 3);
    assert(results.at(4).idx == 4);

    assert(results.at(0).recommendation == 1);
    assert(results.at(0).similarity == 1);
    assert(results.at(0).inferencing == 1);

}
