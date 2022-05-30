//
// Created by breucking on 19.01.22.
//

#ifndef GCSIM_GCLOADUNIT_CUH
#define GCSIM_GCLOADUNIT_CUH


#include <vector>
#include <string>
#include "cudaalgorithms.cuh"

/**
 * Load unit for loading Graph Codes and manage them in memory.
 * This class can be used in two modes, MEMORY for CUDA algorithms
 * and VECTOR for CPU algorithms.
 *
 * <p>If vector mode is used, data will be stored in an std::vector.
 * Memory mode will use memory pointers with 1D arrays. To access the data,
 * the get__Ptr methods should be used. Offset and size information can be used to
 * work through the array.</p>
 *
 */
class GcLoadUnit {

public:
    /**
     * Operation modes.
     */
    enum Modes {
        /**
         * Used for CUDA pointers.
         */
        MODE_MEMORY_MAP,

        /**
         * Used for VECTOR pointers.
         */
        MODE_VECTOR_MAP
    };
//    static  int const MODE_MEMORY_MAP = 0;
//    static  int const MODE_VECTOR_MAP = 1;

    /**
    * Constructor.
    * @param opMode select operation mode.
    */
    GcLoadUnit(const Modes opMode);

    /**
     * Generate artificial data.
     *
     * @param count number of items to generate.
     * @param dimension shape of the Graph Code.
     */
    void loadArtificialGcs(int count, int dimension);

    /**
     * Load Graph Code data multiple times.
     *
     * @param count number of items to generate.
     * @param code Graph Code to multiply.
     */
    void loadMultipleByExample(int count, GraphCode code);

    /**
     * Load a Graph Code from a single json file.
     * @param filepath path to file (json)
     */
    void addGcFromFile(std::string filepath);

    /**
     * Load Graph Codes from a directory.
     * @param cvalue path to directory of Graph Code json files.
     * @param limit limit to a certain number of Graph Codes.
     */
    void loadGraphCodes(const char *cvalue, int limit);

    /**
     * Returns the pointer to the loaded graph code matrix data in main memory.
     * @return a pointer to a one  dimensional array
     */
    unsigned short *getGcMatrixDataPtr();

    /**
     * Return the total number of loaded graph codes.
     *
     * @return number of loaded graph codes.
     */
    int getNumberOfGc();

    /**
     * Tests if the given graph code file name has been loaded.
     * Just checks file names, not full paths.
     *
     * @param basicString the base file name to test.
     * @return true if the file name matches with a loaded file.
     */
    bool hasGc(std::string basicString);

    /**
     * Returns the pointer to the offset array of the matrix data in main memory.
     * The offset array stores the offsets as index of the graph code matrix data.
     * To access the data of the 2nd graph code, you can lookup the value in the array of this pointer.
     * <p>E.g. <code>dataPtr[offsetPrt[1]]</code></p>
     * @return a pointer to a one dimensional array
     */
    unsigned int *getGcMatrixOffsetsPtr();

    /**
     * Returns a pointer to an array with the graph code matrix size information in main memory.
     * The sizes array stores the sizes of the graph code matrix data.
     * To access the last data point of the 2nd graph code matrix, you can lookup the value in the array of this pointer.
     * <p>E.g. <code>dataPtr[offsetPrt[1] + sizePrt[1] - 1]</code></p>
     * @return a pointer to a one dimensional array
     */
    unsigned int *getMatrixSizesPtr();

    /**
     * Returns the pointer to the loaded graph code dictionary data
     * (mapped to an integer) in main memory.
     * @return a pointer to a one dimensional array
     */
    unsigned int *getGcDictDataPtr();

    /**
     * Returns the pointer to the offset array of the matrix dictionary in main memory.
     * The offset array stores the offsets as index of the graph code dictionary data.
     * To access the data of the 2nd graph code, you can lookup the value in the array of this pointer.
     * <p>E.g. <code>dictPtr[dictOffsetPrt[1]]</code></p>
     * @return a pointer to a one dimensional array
     */
    unsigned int *getDictOffsetPtr();

    /**
     * Returns an integer for a dictionary word
     * @param key the word to find
     * @return integer if word exists
     * @throws std::out_of_range If no such data is present.
     */
    unsigned int getDictCode(std::string key);

    /**
     * Helper method to convert a graph code in json representation to a GraphCode-
     * @param jsonGraphCode input json object with matrix (2D int) and dictionary (1D string array).
     * @return output GraphCode object
     */
    GraphCode static convertJsonToGraphCode(json jsonGraphCode);

    /**
     * Returns the total number of items in the dict array.
     *
     * @return the total number of items in the dict array.
     */
    unsigned int getNumberOfDictElements();

    /**
     * Copies all data of the loaded graph codes into the CUDA memory.
     * It also initializes (or updates) the cudaPtr which can be used in the algorithms.
     */
    void loadIntoCudaMemory();

    /**
     * Frees all loaded data from the CUDA memory.
     */
    void freeAll();

    /**
     * Returns the pointer to the loaded graph code matrix data in CUDA memory.
     * @return a pointer to a one dimensional array
     */
    unsigned short *getGcMatrixDataCudaPtr();

    /**
     * Returns the pointer to the loaded graph code dictionary data
     * (mapped to an integer) in CUDA memory.
     * @return a pointer to a one dimensional array
     */
    unsigned int *getGcDictDataCudaPtr();

    /**
     * Returns the pointer to the offset array of the matrix data in CUDA memory.
     * The offset array stores the offsets as index of the graph code matrix data.
     * To access the data of the 2nd graph code, you can lookup the value in the array of this pointer.
     * <p>E.g. <code>dataPtr[offsetPrt[1]]</code></p>
     * @return a pointer to a one dimensional array
     */
    unsigned int *getGcMatrixOffsetsCudaPtr();

    /**
     * Returns the pointer to the offset array of the matrix dictionary in CUDA memory.
     * The offset array stores the offsets as index of the graph code dictionary data.
     * To access the data of the 2nd graph code, you can lookup the value in the array of this pointer.
     * <p>E.g. <code>dictPtr[dictOffsetPrt[1]]</code></p>
     * @return a pointer to a one dimensional array
     */
    unsigned int *getDictOffsetCudaPtr();

    /**
     * Returns a pointer to an array with the graph code matrix size information in CUDA memory.
     * The sizes array stores the sizes of the graph code matrix data.
     * To access the last data point of the 2nd graph code matrix, you can lookup the value in the array of this pointer.
     * <p>E.g. <code>dataPtr[offsetPrt[1] + sizePrt[1] - 1]</code></p>
     * @return a pointer to a one dimensional array
     */
    unsigned int *getMatrixSizesCudaPtr();

    /**
     * Re-maps the file name of the graph code collection index
     * @param index the position of the file.
     * @return the filename.
     */
    std::string getGcNameOnPosition(unsigned long index);

    /**
     * Changes the operation mode. Use with caution, free all data before.
     * @param mode mode enum
     */
    void setOperationMode(const Modes mode);

    /**
     * Get all the Graph Codes data as a vector (VECTOR MODE).
     *
     * @return a std::vector with the loaded graph codes in main memory.
     */
    std::vector<GraphCode> getGcCodes();

    /**
     * Get the position of a graph code in the loaded collection.
     * @param gcFileName base filename
     * @return the position of the graph code
     */
    int getGcPosition(std::string gcFileName);

private:

    /**
     * Operational mode
     */
    int opMode;

    /**
     * Vector with graph codes.
     */
    std::vector<GraphCode> gcCodes;

    /**
     * Pointer to array with the graph code matrix data.
     */
    unsigned short *gcMatrixDataPtr;

    /**
     * Number of loaded Graph Codes
     */
    int gcSize = 0;

    /**
     * State for initialization.
     */
    bool init = false;

    /**
     * Sate if is data copied to CUDA
     */
    bool isInCudaMemory = false;

    /**
     * Internal helper for the last offset postion
     */
    long lastOffset = 0;

    /**
     * Internal helper for the last position in the matrix array
     */
    int lastPosition = 0;

    /**
     * Internal helper for the last dictionary offset.
     */
    unsigned int lastDictOffset = 0;

    /**
     * Number of dictionary elements.
     */
    unsigned int dictCounter = 0;

    /**
     * Internal storage for the file names in the collection.
     */
    std::vector<std::string> gcNames;

    /**
     * Pointer to the matrix size array.
     */
    unsigned int *gcMatrixSizesPtr;

    /**
     * Pointer to the matrix offsets array.
     */
    unsigned int *gcMatrixOffsetsPtr;

    /**
     * Pointer to the dictionary array.
     */
    unsigned int *gcDictDataPtr;

    /**
     * Pointer to the dictionary offset array.
     */
    unsigned int *gcDictOffsetsPtr;


    // CUDA Pointer ------------------------
    /**
     * Internal pointer to the CUDA matrix data.
     */
    unsigned short *d_gcMatrixData;

    /**
     * Internal pointer to the CUDA dictionary data.
     */
    unsigned int *d_gcDictData;

    /**
     * Internal pointer to the CUDA matrix offsets.
     */
    unsigned int *d_gcMatrixOffsets;

    /**
     * Internal pointer to the CUDA matrix sizes.
     */
    unsigned int *d_gcMatrixSizes;

    /**
     * Internal pointer to the CUDA dictionaries offsets.
     */
    unsigned int *d_gcDictOffsets;
    // -------------------------------------

    /**
     * Internal map to convert between dictionary word and integer.
     */
    std::map<std::string, unsigned int> dict_map;

    /**
     * Internal helper to reinitialize the class instance.
     */
    void reinit();

    /**
     * Internal helper to append the matrix pointer
     * @param mat1 1d serialized array of the 2d matrix to add
     * @param sizeofMat number of items in the matrix
     * @param gcMatrixData the matrix pointer to append the mat1
     * @param gcDictData  the dictionary
     * @param gcMatrixOffsets the matrix offsets
     * @param gcMatrixSizes the matrix sizes
     * @param lastOffset helper information with the last offset value
     * @param position internal helper with the position of the matrix to add.
     */
    void appendMatrix(const unsigned short *mat1, unsigned long sizeofMat, unsigned short *gcMatrixData,
                      unsigned int *gcDictData, unsigned int *gcMatrixOffsets, unsigned int *gcMatrixSizes,
                      long *lastOffset,
                      int position);

    /**
     * Adds a vector of dictionary items to the internal dict map.
     * @param vect the words to add to the dict map.
     * @return the number of added items
     */
    int addVectorToDictMap(const std::vector<std::string> *vect);

    /**
     * Append the vector to the dict array in memory.
     * @param vect the graph code dictionary to add
     * @param elementsAdded number of items in dict.
     */
    void appendVectorToDict(const std::vector<std::string> *vect, int elementsAdded);

    /**
     * Append the matrix of a graph code object to the graph codes
     * @param code pointer to the data
     * @param matNumberOfElements number of elements in the matrix.
     */
    void appendGCMatrixToMatrix(const GraphCode &code, int matNumberOfElements);

    /**
     * Helper method to reallocate all pointer with number of items.
     * @param count number of items.
     */
    void reallocPtrBySize(int count);

    /**
     * Loads a sample graph code multiple times.
     * @param count how many times the code should be loaded
     * @param code sample graph code
     */
    void loadMultipleByExampleMemMap(int count, GraphCode &code);

    /**
     * Loads artificial graph code data into memory map.
     * @param count number of elements
     * @param dimension size of the dimension of the graph code.
     */
    void loadArtificialGcsMemMap(int count, int dimension);

    /**
     * Loads a graph code from the filesystem in the memory map.
     *
     * @param filepath full path to the file to load.
     */
    void addGcFromFileMemMap(const std::string &filepath);

    /**
     *Loads a graph code from the filesystem in the vector memory.
     *
     * @param basicString full path to the file to load.
     */
    void addGcFromFileVecMap(std::string basicString);
};


#endif //GCSIM_GCLOADUNIT_CUH
