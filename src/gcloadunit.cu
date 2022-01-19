//
// Created by breucking on 19.01.22.
//
#include <iostream>

#include "gcloadunit.cuh"

void GcLoadUnit::loadArtificialGcs(int count, int dimension)
{
    std::cout << "Hello" << std::endl;
    if (init) {
        free(gcPtr);
    }
    gcPtr = (int *) malloc(sizeof(int) * count);
    init = true;
    gcPtr[0] = 1;
    gcSize = count;
}

int* GcLoadUnit::getGcPtr() {
    return gcPtr;
}

int GcLoadUnit::getSize() {
    return gcSize;
}
