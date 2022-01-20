//
// Created by breucking on 19.01.22.
//


#include <gcloadunit.cuh>
#include <cassert>

void testLoadSimple();

int main() {
    testLoadSimple();
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
    unsigned short *ptr = loadUnit.getGcMatrixDataPtr();
    unsigned int *gcMatrixOffsets = loadUnit.getGcMatrixOffsetsPtr();
    unsigned int *gcMatrixSizes = loadUnit.getMatrixSizesPtr();
    unsigned int *gcDictData = loadUnit.getGcDictDataPtr();
    unsigned int *gcDictOffsets = loadUnit.getDictOffsetPtr();

    assert(size == 10);
    assert(ptr[0] == 0);
    assert(gcMatrixSizes[0] == dim*dim);
    assert(gcMatrixSizes[5] == dim*dim);
    assert(gcMatrixOffsets[0] == 0);
    assert(gcMatrixOffsets[1] == 9);
    assert(gcDictData[0] == 0);
    assert(gcDictData[1] == 1);
    assert(gcDictOffsets[0] = 3);


    dim = 2;
    loadUnit.loadArtificialGcs(11,dim);

    ptr = loadUnit.getGcMatrixDataPtr();
    size = loadUnit.getNumberOfGc();
    gcMatrixOffsets = loadUnit.getGcMatrixOffsetsPtr();
    gcMatrixSizes = loadUnit.getMatrixSizesPtr();

    assert(size == 11);
    assert(loadUnit.hasGc(std::to_string(12)+".gc") == false);
    for(int i = 0; i < 11; i++) {
        assert(loadUnit.hasGc(std::to_string(i)+".gc"));
        assert(ptr[i*dim*dim] == i);
        unsigned int msize = gcMatrixSizes[i];
        assert(msize == dim*dim);

    }

}
