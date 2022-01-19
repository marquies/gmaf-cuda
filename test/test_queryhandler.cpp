//
// Created by breucking on 19.01.22.
//



#include <queryhandler.cuh>
#include <cassert>
#include <gcloadunit.cuh>

void testErrorQuery();

void testValidation();

void testSimpleQuery();

int main() {
    testErrorQuery();
    testValidation();
    testSimpleQuery();

}

void testSimpleQuery() {
    GcLoadUnit loadUnit;
    loadUnit.loadArtificialGcs(10,1);
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
    } catch(std::invalid_argument) {

    }

}

