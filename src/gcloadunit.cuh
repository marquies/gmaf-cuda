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
#include "cuda_algorithms.cuh"

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

    void loadMultipleByExample(int count, GraphCode code);

    unsigned int getDictCode(std::string key);

private:
    unsigned short *gcMatrixDataPtr;

    int gcSize = 0;

    bool init = false;

    std::vector<std::string> gcNames;
    unsigned int *gcMatrixSizesPtr;
    unsigned int *gcMatrixOffsetsPtr;
    unsigned int *gcDictDataPtr;
    unsigned int *gcDictOffsetsPtr;

    std::map<std::string, unsigned int> dict_map;

    void reinit() const;

    void appendMatrix(const unsigned short *mat1, unsigned short sizeofMat, unsigned short *gcMatrixData,
                      unsigned int *gcDictData, unsigned int *gcMatrixOffsets, unsigned int *gcMatrixSizes,
                      int *lastOffset,
                      int position);

};


#endif //GCSIM_GCLOADUNIT_CUH
