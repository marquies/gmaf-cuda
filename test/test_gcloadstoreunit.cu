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
//    gcDictData, dictCounter * sizeof(unsigned int)
//    gcMatrixOffsets, NUMBER_OF_GCS * sizeof(unsigned int)
//    gcMatrixSizesPtr, NUMBER_OF_GCS * sizeof(unsigned int)
//    gcDictOffsets, NUMBER_OF_GCS * sizeof(unsigned int)


    unsigned short *ptr = loadUnit.getGcMatrixDataPtr();
    int size = loadUnit.getNumberOfGc();
    unsigned int *gcMatrixOffsets = loadUnit.getGcMatrixOffsetsPtr();
    unsigned int *gcMatrixSizes = loadUnit.getMatrixSizesPtr();

    assert(size == 10);
    assert(ptr[0] == 0);
    assert(gcMatrixSizes[0] == dim*dim);
    assert(gcMatrixSizes[5] == dim*dim);
    assert(gcMatrixOffsets[0] == 0);
    assert(gcMatrixOffsets[1] == 9);


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
