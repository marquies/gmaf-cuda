//
// Created by breucking on 19.01.22.
//
#include <iostream>
#include <algorithm>

#include "gcloadunit.cuh"

void GcLoadUnit::loadArtificialGcs(int count, int dimension) {
    std::cout << "Hello" << std::endl;
    if (init) {
        free(gcMatrixDataPtr);
        free(gcMatrixSizesPtr);
        free(gcMatrixOffsetsPtr);
    }
    int matrixSize = dimension * dimension;
    gcMatrixDataPtr = (unsigned short *) malloc(sizeof(unsigned short) * count * matrixSize);
    gcMatrixSizesPtr = (unsigned int *) malloc(sizeof(unsigned int) * count);
    gcMatrixOffsetsPtr = (unsigned int *) malloc(sizeof(unsigned int) * count);

    init = true;

    gcNames.clear();

    for (int i = 0; i < count; i++) {
        gcNames.push_back(std::to_string(i) + ".gc");
        for (int j = 0; j < matrixSize; j++) {
            gcMatrixDataPtr[i * matrixSize + j] = i;
        }
        gcMatrixSizesPtr[i] = matrixSize;
        gcMatrixOffsetsPtr[i] = matrixSize*i;
    }
    gcSize = count;

}

unsigned short *GcLoadUnit::getGcMatrixDataPtr() {
    return gcMatrixDataPtr;
}

int GcLoadUnit::getNumberOfGc() {
    return gcSize;
}

bool GcLoadUnit::hasGc(std::string basicString) {
    return std::find(gcNames.begin(), gcNames.end(), basicString) != gcNames.end();
}

unsigned int *GcLoadUnit::getGcMatrixOffsetsPtr() {
    return gcMatrixOffsetsPtr;
}

unsigned int *GcLoadUnit::getMatrixSizesPtr() {
    return gcMatrixSizesPtr;
}
