//
// Created by breucking on 19.01.22.
//


#include <gcloadunit.cuh>
#include <stdlib.h>
#include <cassert>

void testLoadSimple();

int main() {
    testLoadSimple();
}

void testLoadSimple() {
    GcLoadUnit loadUnit;
    loadUnit.loadArtificialGcs(10,1);

    int *ptr = loadUnit.getGcPtr();
    int size = loadUnit.getSize();

    assert(size == 10);
    assert(ptr[0] == 1);

    loadUnit.loadArtificialGcs(11,1);

    ptr = loadUnit.getGcPtr();
    size = loadUnit.getSize();

    assert(size == 11);
    assert(ptr[0] == 1);

}
