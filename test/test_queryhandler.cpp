//
// Created by breucking on 19.01.22.
//



#include <queryhandler.h>
#include <cassert>

void testSimpleQuery();

void testValidation();

int main() {
    testSimpleQuery();
    testValidation();
}

void testValidation() {
    bool valid = QueryHandler::validate("");
    assert(valid == false);
    valid = QueryHandler::validate("Query by Example: xoxo.png");
    assert(valid);
}

void testSimpleQuery() {

    try {
        QueryHandler::processQuery("");

        assert(false);
    } catch(std::invalid_argument) {

    }

}

