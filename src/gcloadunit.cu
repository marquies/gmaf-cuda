//
// Created by breucking on 19.01.22.
//
#include <iostream>
#include <algorithm>

#include "gcloadunit.cuh"
#include "helper.h"

void GcLoadUnit::loadArtificialGcs(int count, int dimension) {
    reinit();
    int matrixSize = dimension * dimension;
    gcMatrixDataPtr = (unsigned short *) malloc(sizeof(unsigned short) * count * matrixSize);
    gcMatrixSizesPtr = (unsigned int *) malloc(sizeof(unsigned int) * count);
    gcMatrixOffsetsPtr = (unsigned int *) malloc(sizeof(unsigned int) * count);
    gcDictDataPtr = (unsigned int *) malloc(sizeof(unsigned int) * count * dimension);
    gcDictOffsetsPtr = (unsigned int *) malloc(sizeof(unsigned int) * count);

    init = true;

    gcNames.clear();


    for (int i = 0; i < count; i++) {
        gcNames.push_back(std::to_string(i) + ".gc");
        for (int j = 0; j < matrixSize; j++) {
            gcMatrixDataPtr[i * matrixSize + j] = i;
        }
        gcMatrixSizesPtr[i] = dimension;
        gcMatrixOffsetsPtr[i] = matrixSize * i;
        for (int j = 0; j < dimension; j++) {
            gcDictDataPtr[i * dimension + j] = j;
        }
        gcDictOffsetsPtr[i] = dimension;
    }
    gcSize = count;

}

void GcLoadUnit::reinit() const {
    if (init) {
        if(G_DEBUG) {
            std::cout << "Reinitialize LoadUnit (free allocated space)" << std::endl;
        }
        free(gcMatrixDataPtr);
        free(gcMatrixSizesPtr);
        free(gcMatrixOffsetsPtr);
        free(gcDictDataPtr);
        free(gcDictOffsetsPtr);
    }
}

void GcLoadUnit::loadMultipleByExample(int count, GraphCode code) {
    reinit();

    gcMatrixDataPtr = (unsigned short *) malloc(0);
    gcDictDataPtr = (unsigned int *) malloc(0);

    //unsigned int *gcMatrixOffsets = (unsigned int *) malloc(0);
    //unsigned int *gcMatrixSizesPtr = (unsigned int *) malloc(0);

    gcMatrixOffsetsPtr = (unsigned int *) malloc(sizeof(unsigned int) * count);
    gcDictOffsetsPtr = (unsigned int *) malloc(sizeof(unsigned int) * count);
    gcMatrixSizesPtr = (unsigned int *) malloc(sizeof(unsigned int) * count);

    unsigned int lastDictOffset = 0;
    unsigned int dictCounter = 0;


    int lastOffset = 0;
    int lastPosition = 0;
    std::vector<std::string> *vect = code.dict;

    for (int ii = 0; ii < count; ii++) {
        //------
        // Adding
        //------

        //std::vector<std::string> *vect = new std::vector<std::string>{"head", "body", "foot", "head"};
        //unsigned short mat[] = {1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1};


        unsigned short mat[code.dict->size() * code.dict->size()];
        for (int i = 0; i < vect->size() * vect->size(); i++) {
            mat[i] = code.matrix[i];
//            if (i == 2)
//                mat[i] = rand() % 10;
        }

        int dc1 = 0;
        // Create dict map for first vector
        for (std::string str: *vect) {
            if (dict_map.find(str) == dict_map.end()) {
                dict_map[str] = dict_map.size();
            }
            dc1++;
        }

        // Expand global dict array
        int newS = dc1 + dictCounter;
        unsigned int *gcDictData_n = (unsigned int *) realloc(gcDictDataPtr, newS * sizeof(unsigned int));
        //unsigned int *gcDictData_n = (unsigned int *) realloc(gcDictData, dc1 * sizeof(unsigned int));
        if (gcDictData_n) {
            gcDictDataPtr = gcDictData_n;
        } else {
            // deal with realloc failing because memory could not be allocated.
            exit(98);
        }

        // Add dict to the global dict data array
        for (std::string str: *vect) {
            gcDictDataPtr[dictCounter++] = dict_map[str];
        }

        gcDictOffsetsPtr[ii] = lastDictOffset;
        lastDictOffset = dictCounter;

        // Expand global matrix array
        int matSize = sizeof(mat) / sizeof(unsigned short);

        newS = lastOffset + matSize;
        //for (int i = 0; i <= lastPosition; i++) {
        //    newS += gcMatrixSizesPtr[i];
        //}// ;
        size_t newSize = newS * sizeof(unsigned short);
        unsigned short *gcMatrixData_n = (unsigned short *) realloc(gcMatrixDataPtr, newSize);
        if (gcMatrixData_n) {
            gcMatrixDataPtr = gcMatrixData_n;
        } else {
            // deal with realloc failing because memory could not be allocated.
            exit(99);
        }

        // Add matrix to the global matrix array
        appendMatrix(mat, matSize, gcMatrixDataPtr, gcDictDataPtr, gcMatrixOffsetsPtr, gcMatrixSizesPtr, &lastOffset,
                     lastPosition);

        lastPosition++;

        gcSize++;
    }
}

void GcLoadUnit::appendMatrix(const unsigned short *mat1, unsigned short sizeofMat, unsigned short *gcMatrixData,
                  unsigned int *gcDictData, unsigned int *gcMatrixOffsets, unsigned int *gcMatrixSizes,
                  int *lastOffset,
                  int position) {
    gcMatrixOffsets[position] = *lastOffset;
    gcMatrixSizes[position] = sizeofMat; // / sizeof (unsigned short )

    for (int i = 0; i < gcMatrixSizes[position]; i++) {
        if (G_DEBUG)
            std::cout << i << " ; position: " << gcMatrixOffsets[position] + i << std::endl;
        gcMatrixData[gcMatrixOffsets[position] + i] = mat1[i];
    }
    *lastOffset += sizeofMat;
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

unsigned int *GcLoadUnit::getGcDictDataPtr() {
    return gcDictDataPtr;
}

unsigned int *GcLoadUnit::getDictOffsetPtr() {
    return gcDictOffsetsPtr;
}

unsigned int GcLoadUnit::getDictCode(std::string key) {
    return dict_map.at(key);
}
