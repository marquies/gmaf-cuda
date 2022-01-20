//
// Created by breucking on 19.01.22.
//

#ifndef GCSIM_GCLOADUNIT_CUH
#define GCSIM_GCLOADUNIT_CUH


//typedef struct GraphCodeBlock {
//    int *ptr;
//    int size;
//} GraphCodeBlock;

#include <vector>
#include <string>

class GcLoadUnit {

public:
    void loadArtificialGcs(int count, int dimension);

    unsigned short *getGcMatrixDataPtr();

    int getNumberOfGc();

    bool hasGc(std::string basicString);

    unsigned int *getGcMatrixOffsetsPtr();

    unsigned int *getMatrixSizesPtr();

    unsigned int *getGcDictDataPtr();

    unsigned int *getDictOffsetPtr();

private:
    unsigned short *gcMatrixDataPtr;

    int gcSize;

    bool init = false;

    std::vector<std::string> gcNames;
    unsigned int *gcMatrixSizesPtr;
    unsigned int *gcMatrixOffsetsPtr;
    unsigned int *gcDictDataPtr;
    unsigned int *gcDictOffsetsPtr;
};


#endif //GCSIM_GCLOADUNIT_CUH
