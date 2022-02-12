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
    enum Modes {
        MODE_MEMORY_MAP,
        MODE_VECTOR_MAP
    };
//    static  int const MODE_MEMORY_MAP = 0;
//    static  int const MODE_VECTOR_MAP = 1;

    GcLoadUnit(const Modes opMode);


    void loadArtificialGcs(int count, int dimension);

    void loadMultipleByExample(int count, GraphCode code);

    void addGcFromFile(std::string filepath);

    void loadGraphCodes(const char *cvalue, int limit);

    unsigned short *getGcMatrixDataPtr();

    int getNumberOfGc();

    bool hasGc(std::string basicString);

    unsigned int *getGcMatrixOffsetsPtr();

    unsigned int *getMatrixSizesPtr();

    unsigned int *getGcDictDataPtr();

    unsigned int *getDictOffsetPtr();

    unsigned int getDictCode(std::string key);

    GraphCode static convertJsonToGraphCode(json jsonGraphCode);

    unsigned int getNumberOfDictElements();

    void matchMetricToGc(Metrics *pMetrics);

    void loadIntoCudaMemory();

    void freeAll();

    unsigned short *getGcMatrixDataCudaPtr();

    unsigned int *getGcDictDataCudaPtr();

    unsigned int *getGcMatrixOffsetsCudaPtr();

    unsigned int *getDictOffsetCudaPtr();

    unsigned int *getMatrixSizesCudaPtr();


    void setOperationMode(const Modes mode);

    std::vector<GraphCode> getGcCodes();

    int getGcPosition(std::string gcFileName);

private:

    int opMode;

    std::vector<GraphCode> gcCodes;

    unsigned short *gcMatrixDataPtr;

    int gcSize = 0;
    bool init = false;
    bool isInCudaMemory = false;
    long lastOffset = 0;
    int lastPosition = 0;
    unsigned int lastDictOffset = 0;
    unsigned int dictCounter = 0;

    std::vector<std::string> gcNames;
    unsigned int *gcMatrixSizesPtr;
    unsigned int *gcMatrixOffsetsPtr;
    unsigned int *gcDictDataPtr;
    unsigned int *gcDictOffsetsPtr;


    // CUDA Pointer ------------------------
    unsigned short *d_gcMatrixData;
    unsigned int *d_gcDictData;
    unsigned int *d_gcMatrixOffsets;
    unsigned int *d_gcMatrixSizes;
    unsigned int *d_gcDictOffsets;
    // -------------------------------------

    std::map<std::string, unsigned int> dict_map;

    void reinit();

    void appendMatrix(const unsigned short *mat1, unsigned long sizeofMat, unsigned short *gcMatrixData,
                      unsigned int *gcDictData, unsigned int *gcMatrixOffsets, unsigned int *gcMatrixSizes,
                      long *lastOffset,
                      int position);

    int addVectorToDictMap(const std::vector<std::string> *vect);

    void appendVectorToDict(const std::vector<std::string> *vect, int elementsAdded);

    void appendGCMatrixToMatrix(const GraphCode &code, int matNumberOfElements);

    void reallocPtrBySize(int count);

    void loadMultipleByExampleMemMap(int count, GraphCode &code);

    void loadArtificialGcsMemMap(int count, int dimension);

    void addGcFromFileMemMap(const std::string &filepath);

    void addGcFromFileVecMap(std::string basicString);
};


#endif //GCSIM_GCLOADUNIT_CUH
