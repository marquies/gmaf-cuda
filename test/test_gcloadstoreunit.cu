//
// Created by breucking on 19.01.22.
//


#include <gcloadunit.cuh>
#include <cassert>
#include <cuda_algorithms.cuh>
#include <helper.h>
#include "testhelper.cpp"


void testLoadSimple();

void testLoadMultipleByExample();

int main() {
    testLoadSimple();
    testLoadMultipleByExample();
}

void testLoadMultipleByExample() {
    int count = 2;
    int dim = 3;
    G_DEBUG = true;
    GraphCode gc1 = generateTestDataGc(dim);

    GcLoadUnit loadUnit;
    loadUnit.loadMultipleByExample(count, gc1);

    int size = loadUnit.getNumberOfGc();
    unsigned short *gcMatrixDataPtr = loadUnit.getGcMatrixDataPtr();
    unsigned int *gcMatrixOffsets = loadUnit.getGcMatrixOffsetsPtr();
    unsigned int *gcMatrixSizes = loadUnit.getMatrixSizesPtr();
    unsigned int *gcDictData = loadUnit.getGcDictDataPtr();
    unsigned int *gcDictOffsets = loadUnit.getDictOffsetPtr();

    assert(size == count);
    assert(gcMatrixDataPtr[0] == gc1.matrix[0]);



    assert(gcMatrixSizes[0] == dim*dim);
    assert(gcMatrixSizes[count-1] == dim * dim);

    assert(gcMatrixOffsets[0] == 0);
    assert(gcMatrixOffsets[1] == dim *dim); //Error I assume

    assert(gcDictOffsets[0] == 0);
    assert(gcDictOffsets[1] == dim);

    assert(gcMatrixDataPtr[gcMatrixOffsets[0]+0] == gc1.matrix[0]);
    assert(gcMatrixDataPtr[gcMatrixOffsets[0]+1] == gc1.matrix[1]);
    assert(gcMatrixDataPtr[gcMatrixOffsets[0]+2] == gc1.matrix[2]);

    assert(gcMatrixDataPtr[gcMatrixOffsets[0]+gcMatrixSizes[0]-1] == gc1.matrix[dim*dim-1]);


    assert(gcMatrixDataPtr[gcMatrixOffsets[1]] == gc1.matrix[0]);
    assert(gcMatrixDataPtr[gcMatrixOffsets[1]+1] == gc1.matrix[1]);
    assert(gcMatrixDataPtr[gcMatrixOffsets[1]+2] == gc1.matrix[2]);

    assert(gcMatrixDataPtr[gcMatrixOffsets[1]+gcMatrixSizes[1]-1] == gc1.matrix[dim*dim-1]);


    unsigned int intCodedWord = loadUnit.getDictCode(gc1.dict->at(0));
    assert(gcDictData[0] == intCodedWord);

    intCodedWord = loadUnit.getDictCode(gc1.dict->at(2));
    assert(gcDictData[2] == intCodedWord);

}

void testLoadSimple() {
    int dim = 3;
    GcLoadUnit loadUnit;
    loadUnit.loadArtificialGcs(10,dim);


//    gcMatrixData, md_size * sizeof(unsigned short)
//    gcMatrixOffsets, NUMBER_OF_GCS * sizeof(unsigned int)
//    gcMatrixSizesPtr, NUMBER_OF_GCS * sizeof(unsigned int)
//    gcDictData, dictCounter * sizeof(unsigned int)
//    gcDictOffsets, NUMBER_OF_GCS * sizeof(unsigned int)


    int size = loadUnit.getNumberOfGc();
    unsigned short *gcMatrixDataPtr = loadUnit.getGcMatrixDataPtr();
    unsigned int *gcMatrixOffsets = loadUnit.getGcMatrixOffsetsPtr();
    unsigned int *gcMatrixSizes = loadUnit.getMatrixSizesPtr();
    unsigned int *gcDictData = loadUnit.getGcDictDataPtr();
    unsigned int *gcDictOffsets = loadUnit.getDictOffsetPtr();

    assert(size == 10);
    assert(gcMatrixDataPtr[0] == 0);
    assert(gcMatrixSizes[0] == dim);
    assert(gcMatrixSizes[5] == dim);
    assert(gcMatrixOffsets[0] == 0);
    assert(gcMatrixOffsets[1] == 9);
    assert(gcDictData[0] == 0);
    assert(gcDictData[1] == 1);
    assert(gcDictOffsets[0] = 3);


    dim = 2;
    loadUnit.loadArtificialGcs(11,dim);

    gcMatrixDataPtr = loadUnit.getGcMatrixDataPtr();
    size = loadUnit.getNumberOfGc();
    gcMatrixOffsets = loadUnit.getGcMatrixOffsetsPtr();
    gcMatrixSizes = loadUnit.getMatrixSizesPtr();

    assert(size == 11);
    assert(loadUnit.hasGc(std::to_string(12)+".gc") == false);
    for(int i = 0; i < 11; i++) {
        assert(loadUnit.hasGc(std::to_string(i)+".gc"));
        assert(gcMatrixDataPtr[i * dim * dim] == i);
        unsigned int msize = gcMatrixSizes[i];
        assert(msize == dim);

    }

}
