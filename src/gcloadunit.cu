//
// Created by breucking on 19.01.22.
//
#include "queryhandler.cuh"
#include "graphcode.h"
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <experimental/filesystem>
#include <thread>
#include<string.h>
#include <dirent.h>
#include <unistd.h>
#include <iostream>
#include <algorithm>
#include <fstream>

#include "gcloadunit.cuh"
#include "helper.h"
#include "cudahelper.cuh"


GcLoadUnit::GcLoadUnit(const Modes opMode) {
    this->opMode = opMode;
}

void GcLoadUnit::loadArtificialGcs(int count, int dimension) {
    reinit();
    if (opMode == MODE_MEMORY_MAP) {
        loadArtificialGcsMemMap(count, dimension);
    } else {

        gcCodes.clear();
        gcCodes.reserve(count);

        init = true;


        long addedElements = 0;
        long addedDictItems = 0;

        for (int n = 0; n < count; n++) {


            std::vector<std::string> *newVec = new std::vector<std::string>(dimension);


            unsigned short *data = (unsigned short *) malloc(sizeof(unsigned short) * dimension * dimension);

            //newVec->push_back(std::to_string(n));

            for (int i = 0; i < dimension; i++) {
                gcNames.push_back(std::to_string(i).append(".gc"));
                newVec->at(i) = std::to_string(i);
                std::vector<int> x;
                for (int j = 0; j < dimension; j++) {
                    //long idx = (long) i * matrixSize + j;
                    //gcMatrixDataPtr[idx] = (unsigned short) 1;

                    addedElements++;
                    if (i == j) {
                        //data[i][j] = 1;
                        data[i * dimension + j] = 1;
                    } else {
                        data[i * dimension + j] = i + j % 2;
                    }
                }
//            gcMatrixSizesPtr[i] = matrixSize;
//            gcMatrixOffsetsPtr[i] = matrixSize * i;
//            for (int j = 0; j < dimension; j++) {
//                gcDictDataPtr[i * dimension + j] = j;
//                addedDictItems++;
//            }
//            gcDictOffsetsPtr[i] = i * dimension;
            }
            if (G_DEBUG) {
                std::cout << "Added " << count << " GCs with total " << addedElements << " elements" << std::endl;
                std::cout << "Added " << count << " GCs with total " << addedDictItems << " dict elements" << std::endl;
            }
            GraphCode gc;
            gc.dict = newVec;
            gc.matrix = data;

            gcCodes.push_back(gc);

            gcSize = count;
//            dictCounter = count * dimension;
        }
    }
}

void GcLoadUnit::loadArtificialGcsMemMap(int count, int dimension) {
    const int matrixSize = dimension * dimension;
    gcMatrixDataPtr = (unsigned short *) malloc(sizeof(unsigned short) * count * matrixSize);
    gcMatrixSizesPtr = (unsigned int *) malloc(sizeof(unsigned int) * count);
    gcMatrixOffsetsPtr = (unsigned int *) malloc(sizeof(unsigned int) * count);
    gcDictDataPtr = (unsigned int *) malloc(sizeof(unsigned int) * count * dimension);
    gcDictOffsetsPtr = (unsigned int *) malloc(sizeof(unsigned int) * count);

    init = true;

    gcNames.clear();
    long addedElements = 0;
    long addedDictItems = 0;

    for (int i = 0; i < count; i++) {
        gcNames.push_back(std::to_string(i) + ".gc");
        for (int j = 0; j < matrixSize; j++) {
            long idx = (long) i * matrixSize + j;
            gcMatrixDataPtr[idx] = (unsigned short) 1;
            addedElements++;
        }
        gcMatrixSizesPtr[i] = matrixSize;
        gcMatrixOffsetsPtr[i] = matrixSize * i;
        for (int j = 0; j < dimension; j++) {
            gcDictDataPtr[i * dimension + j] = j;
            addedDictItems++;
        }
        gcDictOffsetsPtr[i] = i * dimension;
    }
    if (G_DEBUG) {
        std::cout << "Added " << count << " GCs with total " << addedElements << " elements" << std::endl;
        std::cout << "Added " << count << " GCs with total " << addedDictItems << " dict elements" << std::endl;
    }
    gcSize = count;
    dictCounter = count * dimension;
}

void GcLoadUnit::reinit() {
    if (init) {
        if (G_DEBUG) {
            std::cout << "Reinitialize LoadUnit (free allocated space)" << std::endl;
        }
        if (gcMatrixDataPtr)
            free(gcMatrixDataPtr);
        if (gcMatrixSizesPtr)
            free(gcMatrixSizesPtr);
        if (gcMatrixOffsetsPtr)
            free(gcMatrixOffsetsPtr);
        if (gcDictDataPtr)
            free(gcDictDataPtr);
        if (gcDictOffsetsPtr)
            free(gcDictOffsetsPtr);


    }
    gcNames.clear();
    if (opMode == MODE_MEMORY_MAP) {
        lastOffset = 0;
        lastPosition = 0;
        gcSize = 0;
        lastDictOffset = 0;
        dictCounter = 0;
        gcMatrixDataPtr = (unsigned short *) malloc(0);
        gcDictDataPtr = (unsigned int *) malloc(0);

        gcMatrixOffsetsPtr = (unsigned int *) malloc(0);
        gcDictOffsetsPtr = (unsigned int *) malloc(0);
        gcMatrixSizesPtr = (unsigned int *) malloc(0);

        init = true;
    }
}

void GcLoadUnit::loadMultipleByExample(int count, GraphCode code) {
    reinit();

    if (opMode == MODE_MEMORY_MAP) {
        loadMultipleByExampleMemMap(count, code);
    } else {

    }
}

void GcLoadUnit::loadMultipleByExampleMemMap(int count, GraphCode &code) {
    reallocPtrBySize(count);

    std::vector<std::string> *vect = code.dict;

    for (int ii = 0; ii < count; ii++) {
        //------
        // Adding
        //------
        gcNames.push_back(std::to_string(ii) + ".gc");


        // Copy matrix data form source to own structure
//        unsigned short mat[code.dict->size() * code.dict->size()];
//        for (int i = 0; i < vect->size() * vect->size(); i++) {
//            mat[i] = code.matrix[i];
////            if (i == 2)
////                mat[i] = rand() % 10;
//        }
        int matNumberOfElements = code.dict->size() * code.dict->size();

        int elementsAdded = addVectorToDictMap(vect);

        appendVectorToDict(vect, elementsAdded);



        // Expand global matrix array
//        int matSize = sizeof(mat) / sizeof(unsigned short);
        appendGCMatrixToMatrix(code, matNumberOfElements);


        gcSize++;
    }
}

void GcLoadUnit::reallocPtrBySize(int count) {
    unsigned int *tmp = (unsigned int *) realloc(gcMatrixOffsetsPtr, sizeof(unsigned int) * count);
    if (tmp) {
        gcMatrixOffsetsPtr = tmp;
    } else {
        // deal with realloc failing because memory could not be allocated.
        exit(99);
    }

    tmp = (unsigned int *) realloc(gcDictOffsetsPtr, sizeof(unsigned int) * count);
    if (tmp) {
        gcDictOffsetsPtr = tmp;
    } else {
        // deal with realloc failing because memory could not be allocated.
        exit(99);
    }
    tmp = (unsigned int *) realloc(gcMatrixSizesPtr, sizeof(unsigned int) * count);
    if (tmp) {
        gcMatrixSizesPtr = tmp;
    } else {
        // deal with realloc failing because memory could not be allocated.
        exit(99);
    }
}

void GcLoadUnit::appendGCMatrixToMatrix(const GraphCode &code, int matNumberOfElements) {
    int matSize = matNumberOfElements;

    int newS;
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
    appendMatrix(code.matrix, matSize, gcMatrixDataPtr, gcDictDataPtr, gcMatrixOffsetsPtr, gcMatrixSizesPtr,
                 &lastOffset,
                 lastPosition);

    lastPosition++;
}

void
GcLoadUnit::appendVectorToDict(const std::vector<std::string> *vect, int elementsAdded) {// Expand global dict array

    int newS = elementsAdded + dictCounter;
    unsigned int *gcDictData_n = (unsigned int *) realloc(gcDictDataPtr, newS * sizeof(unsigned int));
    //unsigned int *gcDictData_n = (unsigned int *) realloc(gcDictData, elementsAdded * sizeof(unsigned int));
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
    gcDictOffsetsPtr[lastPosition] = lastDictOffset;
    lastDictOffset = dictCounter;

}

int GcLoadUnit::addVectorToDictMap(const std::vector<std::string> *vect) {
    int dc1 = 0;

    // Create dict map for first vector
// -> Add source dict data to own structure
    for (std::string str: *vect) {
        if (dict_map.find(str) == dict_map.end()) {
            dict_map[str] = dict_map.size();
        }
        dc1++;
    }
    return dc1;
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

void GcLoadUnit::addGcFromFile(std::string filepath) {
    if (opMode == MODE_MEMORY_MAP) {
        addGcFromFileMemMap(filepath);
    }
}

void GcLoadUnit::addGcFromFileMemMap(const std::string &filepath) {
    std::ifstream ifs(filepath);
    json jf = json::parse(ifs);

    if (!init) {
        reinit();
    }

    gcNames.push_back(std::string(basename(filepath.c_str())));


    // TODO: xthis could be replaced by direct conversions
//    GraphCode gc = GcLoadUnit::convertJsonToGraphCode(jf);

    // TODO: Replace with non-static member
    json gc1Dictionary = jf["dictionary"];
    json jsonMatrix = jf["matrix"];
    std::vector<std::string> *dict = new std::vector<std::string>;
    int size = gc1Dictionary.size();
    dict->reserve(size);

    for (const auto &item2: gc1Dictionary.items()) {
        dict->push_back(item2.value().get<std::string>());
    }

    unsigned short *matrix = (unsigned short *) malloc(sizeof(unsigned short) * size * size);


    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {

            matrix[i * size + j] = jsonMatrix.at(i).at(j);
        }
    }

    GraphCode gc;
    gc.matrix = matrix;
    gc.dict = dict;

    reallocPtrBySize(gcSize + 1);
    int numberOfElementsAdded = addVectorToDictMap(gc.dict);
    appendVectorToDict(gc.dict, numberOfElementsAdded);

    appendGCMatrixToMatrix(gc, gc.dict->size() * gc.dict->size());

    free(&jf);

    gcSize++;
}

GraphCode GcLoadUnit::convertJsonToGraphCode(json jsonGraphCode) {
    json gc1Dictionary = jsonGraphCode["dictionary"];
    json jsonMatrix = jsonGraphCode["matrix"];
    std::vector<std::string> *dict = new std::vector<std::string>;
    int size = gc1Dictionary.size();
    dict->reserve(size);

    for (const auto &item2: gc1Dictionary.items()) {
        dict->push_back(item2.value().get<std::string>());
    }

    unsigned short *matrix = (unsigned short *) malloc(sizeof(unsigned short) * size * size);


    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {

            matrix[i * size + j] = jsonMatrix.at(i).at(j);
        }
    }

    GraphCode gc;
    gc.matrix = matrix;
    gc.dict = dict;

    return gc;
}

unsigned int GcLoadUnit::getNumberOfDictElements() {
    return dictCounter;
}

void GcLoadUnit::loadIntoCudaMemory() {

    long md_size = 0;
    for (int i = 0; i < gcSize; i++) {
        md_size += gcMatrixSizesPtr[i];
    }// ;



    HANDLE_ERROR(cudaMalloc((void **) &d_gcMatrixData, md_size * sizeof(unsigned short)));
    HANDLE_ERROR(cudaMalloc((void **) &d_gcDictData, dictCounter * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void **) &d_gcMatrixOffsets, gcSize * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void **) &d_gcMatrixSizes, gcSize * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void **) &d_gcDictOffsets, gcSize * sizeof(unsigned int)));


    HANDLE_ERROR(
            cudaMemcpy(d_gcMatrixData, gcMatrixDataPtr, md_size * sizeof(unsigned short), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_gcDictData, gcDictDataPtr, dictCounter * sizeof(unsigned int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_gcMatrixOffsets, gcDictOffsetsPtr, gcSize * sizeof(unsigned int),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(
            cudaMemcpy(d_gcMatrixSizes, gcMatrixSizesPtr, gcSize * sizeof(unsigned int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(
            cudaMemcpy(d_gcDictOffsets, gcDictOffsetsPtr, gcSize * sizeof(unsigned int), cudaMemcpyHostToDevice));

}

void GcLoadUnit::freeAll() {
    if (d_gcMatrixData)
        HANDLE_ERROR(cudaFree(d_gcMatrixData));
    if (d_gcDictData)
        HANDLE_ERROR(cudaFree(d_gcDictData));
    if (d_gcMatrixOffsets)
        HANDLE_ERROR(cudaFree(d_gcMatrixOffsets));
    if (d_gcMatrixSizes)
        HANDLE_ERROR(cudaFree(d_gcMatrixSizes));
}

unsigned short *GcLoadUnit::getGcMatrixDataCudaPtr() {
    return d_gcMatrixData;
}

unsigned int *GcLoadUnit::getGcDictDataCudaPtr() {
    return d_gcDictData;
}

unsigned int *GcLoadUnit::getGcMatrixOffsetsCudaPtr() {
    return d_gcMatrixOffsets;
}

unsigned int *GcLoadUnit::getDictOffsetCudaPtr() {
    return d_gcDictOffsets;
}

unsigned int *GcLoadUnit::getMatrixSizesCudaPtr() {
    return d_gcMatrixSizes;
}

void GcLoadUnit::loadGraphCodes(const char *cvalue, int limit) {

    int n = 0;


    for (const auto &entry: std::experimental::filesystem::v1::__cxx11::directory_iterator(cvalue)) {
        try {

            addGcFromFile(entry.path().string());
        } catch (json::exception &e) {
            std::cerr << entry.path().string() << '\n';
            std::cerr << e.what() << '\n';
        }
        n++;
        if (n > limit) {
            break;
        }
    }
}

void GcLoadUnit::setOperationMode(const Modes mode) {
    opMode = mode;
    if (init) {
        reinit();
    }
}

std::vector<GraphCode> GcLoadUnit::getGcCodes() {
    return gcCodes;

}
