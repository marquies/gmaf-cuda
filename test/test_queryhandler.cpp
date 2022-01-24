//
// Created by breucking on 19.01.22.
//



#include <queryhandler.cuh>
#include <cassert>
#include <gcloadunit.cuh>

void testErrorQuery();

void testValidation();

void testSimpleQuery();

void testOrdering();

int main() {
    testErrorQuery();
    testValidation();
    testSimpleQuery();
    testOrdering();

}

void testOrdering() {

    Metrics m1;
    m1.idx = 1;
    m1.similarity = 1;
    m1.recommendation = 1;
    m1.inferencing = 1;

    Metrics m2;
    m2.idx = 2;
    m2.similarity = 2;
    m2.recommendation = 2;
    m2.inferencing = 2;

    Metrics m3;
    m3.idx = 3;
    m3.similarity = 3;
    m3.recommendation = 3;
    m3.inferencing = 3;


    Metrics *metrics = (Metrics *) malloc(sizeof(Metrics) * 3);
    metrics[0] = m1;
    metrics[1] = m2;
    metrics[2] = m3;

    QueryHandler::selectionSort(metrics, 3);

    assert(metrics[0].idx == 3);

    free(metrics);

}

void testSimpleQuery() {
    GcLoadUnit loadUnit;
    loadUnit.loadArtificialGcs(10, 1);
    int value = QueryHandler::processQuery("Query by Example: 6.gc", loadUnit);
    assert(value == 0);
}

void testValidation() {
    bool valid = QueryHandler::validate("");
    assert(valid == false);
    valid = QueryHandler::validate("Query by Example: xoxo.png");
    assert(valid);
}

void testErrorQuery() {


    try {
        QueryHandler::processQuery("", GcLoadUnit());
        assert(false);
    } catch (std::invalid_argument) {

    }

}

