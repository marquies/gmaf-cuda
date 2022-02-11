//
// Created by breucking on 19.01.22.
//


#include <queryhandler.cuh>
#include <gcloadunit.cuh>
#include <cassert>
#include <helper.h>
#include <fstream>
#include "testhelper.h"


void testLoadSimple();

void testLoadMultipleByExample();

void testLoadTwoRealGcs();

void testConvertJsonToGraphCode();

void testPosition();

int main() {
    testLoadSimple();
    testLoadMultipleByExample();
    testLoadTwoRealGcs();
    testConvertJsonToGraphCode();
    testPosition();

}

void testConvertJsonToGraphCode() {
    std::string file = "/home/breucking/CLionProjects/gmaf-cuda/GMAF_TMP_17316548361524909203.png.gc";

    std::ifstream ifs(file);
    json jf = json::parse(ifs);

    const GraphCode &gc = GcLoadUnit::convertJsonToGraphCode(jf);

    //["root-asset","fish","cat","person","smurf"]
    assert(gc.dict->at(0).compare("root-asset") == 0);
    assert(gc.dict->at(1).compare("fish") == 0);
    assert(gc.dict->at(2).compare("cat") == 0);
    assert(gc.dict->at(3).compare("person") == 0);
    assert(gc.dict->at(4).compare("smurf") == 0);

    //[[1,1,1,1,1],[0,2,0,0,0],[0,0,2,0,0],[0,0,0,2,0],[0,0,0,0,2]]
    assert(gc.matrix[0] == 1);
    assert(gc.matrix[1] == 1);
    assert(gc.matrix[2] == 1);
    assert(gc.matrix[3] == 1);
    assert(gc.matrix[4] == 1);

    assert(gc.matrix[5] == 0);
    assert(gc.matrix[6] == 2);
    assert(gc.matrix[7] == 0);
    assert(gc.matrix[8] == 0);
    assert(gc.matrix[9] == 0);

    assert(gc.matrix[10] == 0);
    assert(gc.matrix[11] == 0);
    assert(gc.matrix[12] == 2);
    assert(gc.matrix[13] == 0);
    assert(gc.matrix[14] == 0);

}


void testLoadTwoRealGcs() {
    GcLoadUnit loadUnit = GcLoadUnit(GcLoadUnit::MODE_MEMORY_MAP);

    std::string file = "/home/breucking/CLionProjects/gmaf-cuda/GMAF_TMP_17316548361524909203.png.gc";
    loadUnit.addGcFromFile(file);

    unsigned int *gcDictData = loadUnit.getGcDictDataPtr();

    unsigned short *gcMatrixDataPtr = loadUnit.getGcMatrixDataPtr();
    unsigned int *gcMatrixOffsets = loadUnit.getGcMatrixOffsetsPtr();
    unsigned int *gcMatrixSizes = loadUnit.getMatrixSizesPtr();


    //["root-asset","fish","cat","person","smurf"]
    unsigned int word0 = loadUnit.getDictCode("root-asset");
    unsigned int word1 = loadUnit.getDictCode("fish");
    unsigned int word2 = loadUnit.getDictCode("cat");
    unsigned int word3 = loadUnit.getDictCode("person");
    unsigned int word4 = loadUnit.getDictCode("smurf");

    assert(gcDictData[0] == word0);
    assert(gcDictData[1] == word1);
    assert(gcDictData[2] == word2);
    assert(gcDictData[3] == word3);
    assert(gcDictData[4] == word4);

    assert(gcMatrixSizes[0] == 25);
    assert(gcMatrixOffsets[0] == 0);

//    //[[1,1,1,1,1],[0,2,0,0,0],[0,0,2,0,0],[0,0,0,2,0],[0,0,0,0,2]]
    assert(gcMatrixDataPtr[0] == 1);
    assert(gcMatrixDataPtr[1] == 1);
    assert(gcMatrixDataPtr[2] == 1);
    assert(gcMatrixDataPtr[3] == 1);
    assert(gcMatrixDataPtr[4] == 1);

    assert(gcMatrixDataPtr[5] == 0);
    assert(gcMatrixDataPtr[6] == 2);
    assert(gcMatrixDataPtr[7] == 0);
    assert(gcMatrixDataPtr[8] == 0);
    assert(gcMatrixDataPtr[9] == 0);

    assert(gcMatrixDataPtr[10] == 0);
    assert(gcMatrixDataPtr[11] == 0);
    assert(gcMatrixDataPtr[12] == 2);
    assert(gcMatrixDataPtr[13] == 0);
    assert(gcMatrixDataPtr[14] == 0);


    file = "/home/breucking/CLionProjects/gmaf-cuda/example2.gc";
    loadUnit.addGcFromFile(file);

    gcDictData = loadUnit.getGcDictDataPtr();

    gcMatrixDataPtr = loadUnit.getGcMatrixDataPtr();
    gcMatrixOffsets = loadUnit.getGcMatrixOffsetsPtr();
    gcMatrixSizes = loadUnit.getMatrixSizesPtr();


    //["root-asset","hammer","drill","schrowetrecker","nail"]
    unsigned int word5 = loadUnit.getDictCode("root-asset");
    unsigned int word6 = loadUnit.getDictCode("hammer");
    unsigned int word7 = loadUnit.getDictCode("drill");
    unsigned int word8 = loadUnit.getDictCode("schrowetrecker");

    assert(word0 == word5);

    assert(gcDictData[5] == word5);
    assert(gcDictData[6] == word6);
    assert(gcDictData[7] == word7);
    assert(gcDictData[8] == word8);

    assert(gcMatrixSizes[1] == 16);
    assert(gcMatrixOffsets[1] == 25);

//    [[12,12,12,12],[10,12,10,10],[10,10,12,10],[10,10,10,12]]
    assert(gcMatrixDataPtr[25] == 12);
    assert(gcMatrixDataPtr[26] == 12);
    assert(gcMatrixDataPtr[27] == 12);
    assert(gcMatrixDataPtr[28] == 12);

    assert(gcMatrixDataPtr[29] == 10);
    assert(gcMatrixDataPtr[30] == 12);
    assert(gcMatrixDataPtr[31] == 10);
    assert(gcMatrixDataPtr[32] == 10);

    assert(gcMatrixDataPtr[33] == 10);
    assert(gcMatrixDataPtr[34] == 10);
    assert(gcMatrixDataPtr[35] == 12);
    assert(gcMatrixDataPtr[36] == 10);

    assert(gcMatrixDataPtr[37] == 10);
    assert(gcMatrixDataPtr[38] == 10);
    assert(gcMatrixDataPtr[39] == 10);
    assert(gcMatrixDataPtr[40] == 12);

    assert(loadUnit.hasGc("example2.gc"));
    assert(loadUnit.hasGc("GMAF_TMP_17316548361524909203.png.gc"));
}

void testLoadMultipleByExample() {
    int count = 2;
    int dim = 3;
    G_DEBUG = true;
    GraphCode gc1 = generateTestDataGc(dim);

    GcLoadUnit loadUnit = GcLoadUnit(GcLoadUnit::MODE_MEMORY_MAP);
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
    GcLoadUnit loadUnit = GcLoadUnit(GcLoadUnit::MODE_MEMORY_MAP);
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
    assert(gcMatrixDataPtr[0] == 1);
    assert(gcMatrixSizes[0] == dim*dim);
    assert(gcMatrixSizes[5] == dim*dim);
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
    assert(loadUnit.getNumberOfDictElements() == 11*dim);
    for(int i = 0; i < 11; i++) {
        assert(loadUnit.hasGc(std::to_string(i)+".gc"));
        assert(gcMatrixDataPtr[i * dim * dim] == 1);
        unsigned int msize = gcMatrixSizes[i];
        assert(msize == dim*dim);

    }

}

void testPosition() {
    GcLoadUnit gc = GcLoadUnit(GcLoadUnit::MODE_VECTOR_MAP);
    gc.loadArtificialGcs(10, 5);
    assert(gc.hasGc("0.gc") == true);
    assert(gc.getGcPosition("0.gc") == 0);
    assert(gc.hasGc("1.gc") == true);
    assert(gc.getGcPosition("1.gc") == 1);
    assert(gc.hasGc("2.gc") == true);
    assert(gc.getGcPosition("2.gc") == 2);

    gc = GcLoadUnit(GcLoadUnit::MODE_MEMORY_MAP);
    std::basic_string<char> file = "/home/breucking/CLionProjects/gmaf-cuda/GMAF_TMP_17316548361524909203.png.gc";
    gc.addGcFromFile(file);
    file = "/home/breucking/CLionProjects/gmaf-cuda/example2.gc";
    gc.addGcFromFile(file);

    assert(gc.hasGc("example2.gc") == true);
    assert(gc.getGcPosition("example2.gc") == 1);
    assert(gc.hasGc("GMAF_TMP_17316548361524909203.png.gc") == true);
    assert(gc.getGcPosition("GMAF_TMP_17316548361524909203.png.gc") == 0);

}