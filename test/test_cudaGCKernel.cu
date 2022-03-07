//
// Created by breucking on 30.12.21.
//

#include "testhelper.h"
#include "../src/cudaalgorithms.cuh"
#include <cuda_runtime.h>
#include <uuid/uuid.h>
#include <string>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <cudahelper.cuh>
#include <c++/9/chrono>
#include "helper.h"
#include "cudaalgorithms.cuh"
#include <time.h>
#include <stdlib.h>
#include <gcloadunit.cuh>


void testGcSimilarityKernelWith100x100();

void testGcSimilarityKernelWith3x3();

void appendMatrix(const unsigned short *mat1, unsigned short sizeofMat, unsigned short *gcMatrixData,
                  unsigned int *gcDictData, unsigned int *gcMatrixOffsets, unsigned int *gcMatrixSizes, int *lastOffset,
                  int position);


void testGcSimilarityKernelWithMany3x3();

void testGcSimilarityKernelWith10000();

struct uuid {
    uint32_t time_low;
    uint16_t time_mid;
    uint16_t time_hi_and_version;
    uint16_t clock_seq;
    uint8_t node[6];
};


/*
 * prototypes
 */
//void uuid_pack(const struct uuid *uu, uuid_t ptr);
//void uuid_unpack(const uuid_t in, struct uuid *uu);


__device__ int cuda_memcmp(const void *s1, const void *s2, size_t n) {
    const unsigned char *p1 = static_cast<const unsigned char *>(s1);
    const unsigned char *end1 = p1 + n;
    const unsigned char *p2 = static_cast<const unsigned char *>(s2);
    int d = 0;
    for (;;) {
        if (d || p1 >= end1) break;
        d = (int) *p1++ - (int) *p2++;
        if (d || p1 >= end1) break;
        d = (int) *p1++ - (int) *p2++;
        if (d || p1 >= end1) break;
        d = (int) *p1++ - (int) *p2++;
        if (d || p1 >= end1) break;
        d = (int) *p1++ - (int) *p2++;
    }
    return d;
}

__device__ void cuda_uuid_unpack(const uuid_t in, struct uuid *uu) {
    const uint8_t *ptr = in;
    uint32_t tmp;

    tmp = *ptr++;
    tmp = (tmp << 8) | *ptr++;
    tmp = (tmp << 8) | *ptr++;
    tmp = (tmp << 8) | *ptr++;
    uu->time_low = tmp;

    tmp = *ptr++;
    tmp = (tmp << 8) | *ptr++;
    uu->time_mid = tmp;

    tmp = *ptr++;
    tmp = (tmp << 8) | *ptr++;
    uu->time_hi_and_version = tmp;

    tmp = *ptr++;
    tmp = (tmp << 8) | *ptr++;
    uu->clock_seq = tmp;

    memcpy(uu->node, ptr, 6);
}

#define UUCMP(u1, u2) if (u1 != u2) return((u1 < u2) ? -1 : 1);

__device__ int cuda_uuid_compare(const uuid_t uu1, const uuid_t uu2) {
    struct uuid uuid1, uuid2;

    cuda_uuid_unpack(uu1, &uuid1);
    cuda_uuid_unpack(uu2, &uuid2);

    UUCMP(uuid1.time_low, uuid2.time_low);
    UUCMP(uuid1.time_mid, uuid2.time_mid);
    UUCMP(uuid1.time_hi_and_version, uuid2.time_hi_and_version);
    UUCMP(uuid1.clock_seq, uuid2.clock_seq);
    return cuda_memcmp(uuid1.node, uuid2.node, 6);
}


__global__ void compareUUID(UUIDGraphCode *gc1, UUIDGraphCode *gc2, Metrics *metrics) {
//
//    uuid_t word1; //= gc1->dict[0];
//    uuid_t word2;// = gc2->dict[0];
//    for (int i = 0; i < 16; i++) {
//        word1[i] = gc1->dict[0][i];
//    }
//
//    for (int i = 0; i < 16; i++) {
//        word2[i] = gc1->dict[1][i];
//    }
//
//
//    if (cuda_uuid_compare(gc1->dict[0], gc2->dict[0]) == 0) {
//        *metrics = 100 + *metrics;
//    } else {
//        *metrics = 200 + *metrics;
//    }

    int sim = 0;
    int elements = sizeof(gc1->dict) / sizeof(uuid_t);

    unsigned short *matrix1 = gc1->matrix;
    unsigned short *matrix2 = gc2->matrix;

//    for (const auto &item: *gc1->dict) {
    for (int i = 0; i < elements; i++) {
//        for (const auto &item2: *gc2->dict) {
        for (int j = 0; j < elements; j++) {
            if (cuda_uuid_compare(gc1->dict[i], gc2->dict[j]) == 0) {
                sim++;
            }
        }
    }
    int num_of_non_zero_edges = 0;
    int edge_metric_count = 0;
    int edge_type = 0;


    for (int i = 0; i < elements; i++) {
        for (int j = 0; j < elements; j++) {

            if (i != j && matrix1[i * elements + j] != 0) {
                num_of_non_zero_edges++;

                int position1 = i;
                int position2 = j;
                if (position1 == -1 || position2 == -1) {
                    continue;
                }

                int edge = matrix2[position1 * elements + position2];//matrix2[position1][position2];
                if (edge != 0) {
                    edge_metric_count++;
                }
                if (edge == matrix1[i * elements + j]) {
                    edge_type++;
                }

            }
        }
    }

    metrics->similarity = 0.0;
    metrics->recommendation = 0.0;
    metrics->inferencing = 0.0;
    //float node_metric = (float) sim / (float) elements;
    metrics->similarity = (float) sim / (float) elements;
    //float edge_metric = 0.0;
    if (num_of_non_zero_edges > 0) {
        /*edge_metric*/ metrics->recommendation = (float) edge_metric_count / (float) num_of_non_zero_edges;
    }
//    float edge_type_metric = 0.0;
    if (edge_metric_count > 0) {
        /*edge_type_metric*/ metrics->inferencing = (float) edge_type / (float) edge_metric_count;
    }
    //*metrics = node_metric;
//    return metrics;
}


/**
* Main
*/
int main() {
    testGcSimilarityKernelWith100x100();
    testGcSimilarityKernelWith3x3();
    testGcSimilarityKernelWithMany3x3();
    testGcSimilarityKernelWith10000();
}

void testGcSimilarityKernelWith10000() {
    int numberOfGcs = 100;
    GcLoadUnit loadUnit(GcLoadUnit::MODE_MEMORY_MAP);
    loadUnit.loadArtificialGcs(numberOfGcs, 100);
    loadUnit.loadIntoCudaMemory();
    auto start = std::chrono::system_clock::now();

    Metrics *result = demoCalculateGCsOnCuda(loadUnit.getNumberOfGc(),
                                             loadUnit.getNumberOfDictElements(),
                                             loadUnit.getGcMatrixDataCudaPtr(),
                                             loadUnit.getGcDictDataCudaPtr(),
                                             loadUnit.getGcMatrixOffsetsCudaPtr(),
                                             loadUnit.getDictOffsetCudaPtr(),
                                             loadUnit.getMatrixSizesCudaPtr()
    );
    auto endOfCalc = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_secondsCalc = endOfCalc - start;
    std::cout << loadUnit.getNumberOfGc() << "\t" << elapsed_secondsCalc.count()
              << "\n";

    for (int i = 0; i < numberOfGcs; i++) {
        assert(result[i].similarity == 1);
    }

}


void testGcSimilarityKernelWithMany3x3() {

    G_DEBUG = true;
//    int NUMBER_OF_GCS = 10;
    int NUMBER_OF_GCS = 10000;
    srand(time(NULL));   // Initialization, should only be called once.


    unsigned short *gcMatrixData = (unsigned short *) malloc(0);
    unsigned int *gcDictData = (unsigned int *) malloc(0);


    unsigned int *gcMatrixOffsets = (unsigned int *) malloc(sizeof(unsigned int) * NUMBER_OF_GCS);
    unsigned int *gcDictOffsets = (unsigned int *) malloc(sizeof(unsigned int) * NUMBER_OF_GCS);
    unsigned int *gcMatrixSizes = (unsigned int *) malloc(sizeof(unsigned int) * NUMBER_OF_GCS);

    unsigned int lastDictOffset = 0;
    unsigned int dictCounter = 0;

    std::map<std::string, unsigned int> dict_map;

    int lastOffset = 0;
    int lastPosition = 0;
    GraphCode gc1 = generateTestDataGc(250);
    std::vector<std::string> *vect = gc1.dict;

    for (int ii = 0; ii < NUMBER_OF_GCS; ii++) {
        //------
        // Adding
        //------

        //std::vector<std::string> *vect = new std::vector<std::string>{"head", "body", "foot", "head"};
        //unsigned short mat[] = {1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1};


        unsigned short mat[gc1.dict->size() * gc1.dict->size()];
        for (int i = 0; i < vect->size(); i++) {
            mat[i] = gc1.matrix[i];
            if (i == 2)
                mat[i] = rand() % 10;
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
        unsigned int *gcDictData_n = (unsigned int *) realloc(gcDictData, newS * sizeof(unsigned int));
        //unsigned int *gcDictData_n = (unsigned int *) realloc(gcDictData, dc1 * sizeof(unsigned int));
        if (gcDictData_n) {
            gcDictData = gcDictData_n;
        } else {
            // deal with realloc failing because memory could not be allocated.
            exit(98);
        }

        // Add dict to the global dict data array
        for (std::string str: *vect) {
            gcDictData[dictCounter++] = dict_map[str];
        }

        gcDictOffsets[ii] = lastDictOffset;
        lastDictOffset = dictCounter;

        // Expand global matrix array
        int matSize = sizeof(mat) / sizeof(unsigned short);

        newS = lastOffset + matSize;
        //for (int i = 0; i <= lastPosition; i++) {
        //    newS += gcMatrixSizesPtr[i];
        //}// ;
        size_t newSize = newS * sizeof(unsigned short);
        unsigned short *gcMatrixData_n = (unsigned short *) realloc(gcMatrixData, newSize);
        if (gcMatrixData_n) {
            gcMatrixData = gcMatrixData_n;
        } else {
            // deal with realloc failing because memory could not be allocated.
            exit(99);
        }

        // Add matrix to the global matrix array
        appendMatrix(mat, matSize, gcMatrixData, gcDictData, gcMatrixOffsets, gcMatrixSizes, &lastOffset,
                     lastPosition);

        lastPosition++;


    }
    //delete vect;
    delete gc1.dict;
    delete gc1.matrix;


    assert(gcMatrixSizes[0] == 250 * 250);
    assert(gcMatrixSizes[NUMBER_OF_GCS - 1] == 250 * 250);

    std::cout << gcMatrixOffsets[0] << std::endl;
    assert(gcMatrixOffsets[0] == 0);
    assert(gcMatrixOffsets[1] == 250 * 250); //Error I assume

    assert(gcDictOffsets[0] == 0);
    assert(gcDictOffsets[1] == 250);
    demoCalculateGCsOnCudaWithCopy(NUMBER_OF_GCS, dictCounter, gcMatrixData, gcDictData, gcMatrixOffsets, gcDictOffsets,
                                   gcMatrixSizes);


}

void testGcSimilarityKernelWith3x3() {

    int NUMBER_OF_GCS = 3;


    unsigned short *gcMatrixData = (unsigned short *) malloc(0);
    unsigned int *gcDictData = (unsigned int *) malloc(0);

    //unsigned int *gcMatrixOffsets = (unsigned int *) malloc(0);
    //unsigned int *gcMatrixSizesPtr = (unsigned int *) malloc(0);

    unsigned int *gcMatrixOffsets = (unsigned int *) malloc(sizeof(unsigned int) * NUMBER_OF_GCS);
    unsigned int *gcDictOffsets = (unsigned int *) malloc(sizeof(unsigned int) * NUMBER_OF_GCS);
    unsigned int *gcMatrixSizes = (unsigned int *) malloc(sizeof(unsigned int) * NUMBER_OF_GCS);

    unsigned int lastDictOffset = 0;
    unsigned int dictCounter = 0;

    std::map<std::string, unsigned int> dict_map;

    int lastOffset = 0;
    int lastPosition = 0;

    //------
    // Adding
    //------

    std::vector<std::string> *vect = new std::vector<std::string>{"head", "body", "foot"};
    unsigned short mat1[] = {1, 1, 0, 0, 1, 0, 0, 1, 1};


    int dc1 = 0;
    // Create dict map for first vector
    for (std::string str: *vect) {
        if (dict_map.find(str) == dict_map.end()) {
            dict_map[str] = dict_map.size();
        }
        dc1++;
    }

    // Expand global dict array
    unsigned int *gcDictData_n = (unsigned int *) realloc(gcDictData, dc1 * sizeof(unsigned int));
    if (gcDictData_n) {
        gcDictData = gcDictData_n;
    } else {
        // deal with realloc failing because memory could not be allocated.
        exit(99);
    }

    // Add dict to the global dict data array
    for (std::string str: *vect) {
        gcDictData[dictCounter++] = dict_map[str];
    }

    gcDictOffsets[0] = lastDictOffset;
    lastDictOffset = dictCounter;

    // Expand global matrix array
    int matSize = sizeof(mat1) / sizeof(unsigned short);

    unsigned int newS = 0;
    for (int i = 0; i <= lastPosition; i++) {
        newS += gcMatrixSizes[i];
    }// ;
    size_t newSize = newS * sizeof(unsigned short);
    unsigned short *gcMatrixData_n = (unsigned short *) realloc(gcMatrixData, newSize);
    if (gcMatrixData_n) {
        gcMatrixData = gcMatrixData_n;
    } else {
        // deal with realloc failing because memory could not be allocated.
        exit(99);
    }

    // Add matrix to the global matrix array
    appendMatrix(mat1, matSize, gcMatrixData, gcDictData, gcMatrixOffsets, gcMatrixSizes, &lastOffset,
                 lastPosition);

    lastPosition++;

    assert(gcMatrixData[0] == 1);
    assert(gcMatrixData[1] == 1);
    assert(gcMatrixData[2] == 0);
    assert(gcMatrixData[3] == 0);
    assert(gcMatrixData[4] == 1);
    assert(gcMatrixData[5] == 0);
    assert(gcMatrixData[6] == 0);
    assert(gcMatrixData[7] == 1);
    assert(gcMatrixData[8] == 1);

    assert(gcDictData[0] == 0);
    assert(gcDictData[1] == 1);
    assert(gcDictData[2] == 2);

    //------
    // Adding
    //------


    std::vector<std::string> *vect2 = new std::vector<std::string>{"head", "body", "foot"};
    unsigned short mat2[] = {1, 2, 0, 0, 1, 0, 0, 0, 1};

    dc1 = 0;
    // Create dict map for first vector
    for (std::string str: *vect2) {
        if (dict_map.find(str) == dict_map.end()) {
            dict_map[str] = dict_map.size();
        }
        dc1++;
    }


    // Expand the global dict array
    newS = dc1 + dictCounter;
    unsigned int *gcDictData_n2 = (unsigned int *) realloc(gcDictData, newS * sizeof(unsigned int));
    if (gcDictData_n2) {
        gcDictData = gcDictData_n2;
    } else {
        // deal with realloc failing because memory could not be allocated.
        exit(99);
    }

    // Add dict to global dict array
    for (std::string str: *vect2) {
        gcDictData[dictCounter++] = dict_map[str];
    }

    gcDictOffsets[lastPosition] = lastDictOffset;
    lastDictOffset = dictCounter;


    // Expand global matrix
    matSize = sizeof(mat2) / sizeof(unsigned short);

    for (int i = 0; i <= lastPosition; i++) {
        newS += gcMatrixSizes[lastPosition];
    }// ;
    newSize = newS * sizeof(unsigned short);
    gcMatrixData_n = (unsigned short *) realloc(gcMatrixData, newSize);
    if (gcMatrixData_n) {
        gcMatrixData = gcMatrixData_n;
    } else {
        // deal with realloc failing because memory could not be allocated.
        exit(99);
    }

    // Add to global dict
    appendMatrix(mat2, matSize, gcMatrixData, gcDictData, gcMatrixOffsets, gcMatrixSizes, &lastOffset,
                 lastPosition);

    lastPosition++;

    assert(gcMatrixData[0] == 1);
    assert(gcMatrixData[1] == 1);
    assert(gcMatrixData[2] == 0);
    assert(gcMatrixData[3] == 0);
    assert(gcMatrixData[4] == 1);
    assert(gcMatrixData[5] == 0);
    assert(gcMatrixData[6] == 0);
    assert(gcMatrixData[7] == 1);
    assert(gcMatrixData[8] == 1);

    assert(gcMatrixData[9] == 1);
    assert(gcMatrixData[10] == 2);
    assert(gcMatrixData[11] == 0);
    assert(gcMatrixData[12] == 0);
    assert(gcMatrixData[13] == 1);
    assert(gcMatrixData[14] == 0);
    assert(gcMatrixData[15] == 0);
    assert(gcMatrixData[16] == 0);
    assert(gcMatrixData[17] == 1);

    assert(gcDictData[0] == 0);
    assert(gcDictData[1] == 1);
    assert(gcDictData[2] == 2);
    assert(gcDictData[3] == 0);
    assert(gcDictData[4] == 1);
    assert(gcDictData[5] == 2);

    assert(gcMatrixOffsets[0] == 0);
    assert(gcMatrixOffsets[1] == 9);

    assert(gcDictOffsets[0] == 0);
    assert(gcDictOffsets[1] == 3);


    //------
    // Adding
    //------

    std::vector<std::string> *vect3 = new std::vector<std::string>{"head", "body", "foot"};
    unsigned short mat3[] = {1, 2, 0, 0, 1, 1, 1, 1, 1};
//                     mat2 {1, 2, 0, 0, 1, 0, 0, 0, 1}

    dc1 = 0;
    // Create dict map for first vector
    for (std::string str: *vect3) {
        if (dict_map.find(str) == dict_map.end()) {
            dict_map[str] = dict_map.size();
        }
        dc1++;
    }


    // Expand the global dict array
    newS = dc1 + dictCounter;
    gcDictData_n2 = (unsigned int *) realloc(gcDictData, newS * sizeof(unsigned int));
    if (gcDictData_n2) {
        gcDictData = gcDictData_n2;
    } else {
        // deal with realloc failing because memory could not be allocated.
        exit(99);
    }

    // Add dict to global dict array
    for (std::string str: *vect3) {
        gcDictData[dictCounter++] = dict_map[str];
    }

    gcDictOffsets[lastPosition] = lastDictOffset;
    lastDictOffset = dictCounter;


    // Expand global matrix
    matSize = sizeof(mat3) / sizeof(unsigned short);

    for (int i = 0; i <= lastPosition; i++) {
        newS += gcMatrixSizes[lastPosition];
    }// ;
    newSize = newS * sizeof(unsigned short);
    gcMatrixData_n = (unsigned short *) realloc(gcMatrixData, newSize);
    if (gcMatrixData_n) {
        gcMatrixData = gcMatrixData_n;
    } else {
        // deal with realloc failing because memory could not be allocated.
        exit(99);
    }

    // Add to global dict
    appendMatrix(mat3, matSize, gcMatrixData, gcDictData, gcMatrixOffsets, gcMatrixSizes, &lastOffset,
                 lastPosition);

    lastPosition++;


    //------------
    // CUDA prep
    //------------


    unsigned short *d_gcMatrixData;
    unsigned int *d_gcDictData;
    unsigned int *d_gcMatrixOffsets;
    unsigned int *d_gcMatrixSizes;
    unsigned int *d_gcDictOffsets;
    Metrics *d_result;

    int md_size = 0;
    for (int i = 0; i < lastPosition; i++) {
        md_size += gcMatrixSizes[i];
    }// ;

    HANDLE_ERROR(cudaMalloc((void **) &d_gcMatrixData, md_size * sizeof(unsigned short)));
    HANDLE_ERROR(cudaMalloc((void **) &d_gcDictData, dictCounter * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void **) &d_gcMatrixOffsets, NUMBER_OF_GCS * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void **) &d_gcMatrixSizes, NUMBER_OF_GCS * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void **) &d_gcDictOffsets, NUMBER_OF_GCS * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void **) &d_result, NUMBER_OF_GCS * sizeof(Metrics)));


    HANDLE_ERROR(
            cudaMemcpy(d_gcMatrixData, gcMatrixData, md_size * sizeof(unsigned short), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_gcDictData, gcDictData, dictCounter * sizeof(unsigned int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_gcMatrixOffsets, gcMatrixOffsets, NUMBER_OF_GCS * sizeof(unsigned int),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(
            cudaMemcpy(d_gcMatrixSizes, gcMatrixSizes, NUMBER_OF_GCS * sizeof(unsigned int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(
            cudaMemcpy(d_gcDictOffsets, gcDictOffsets, NUMBER_OF_GCS * sizeof(unsigned int), cudaMemcpyHostToDevice));


    cudaGcCompute<<<1, NUMBER_OF_GCS>>>(d_gcMatrixData, d_gcDictData, d_gcMatrixOffsets, d_gcMatrixSizes,
                                        d_gcDictOffsets, 0,
                                        NUMBER_OF_GCS,
                                        d_result);

    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    Metrics result[NUMBER_OF_GCS];
    HANDLE_ERROR(cudaMemcpy(&result, d_result, NUMBER_OF_GCS * sizeof(Metrics), cudaMemcpyDeviceToHost));

    for (int i = 0; i < NUMBER_OF_GCS; i++) {


        std::cout << "Result (" << i << ") "
                  << "Similarity " << result[i].similarity << "; "
                  << "Recommendation " << result[i].recommendation << "; "
                  << "Inference " << result[i].inferencing << "; "
                  << std::endl;

    }

    assert(result[1].similarity == 1);
    assert(result[1].recommendation == 0.5);
    assert(result[1].inferencing == 0);

    HANDLE_ERROR(cudaFree(d_gcMatrixData));
    HANDLE_ERROR(cudaFree(d_gcDictData));
    HANDLE_ERROR(cudaFree(d_gcMatrixOffsets));
    HANDLE_ERROR(cudaFree(d_gcMatrixSizes));
    HANDLE_ERROR(cudaFree(d_result));

}


void appendMatrix(const unsigned short *mat1, unsigned short sizeofMat, unsigned short *gcMatrixData,
                  unsigned int *gcDictData, unsigned int *gcMatrixOffsets, unsigned int *gcMatrixSizes,
                  int *lastOffset,
                  int position) {
    gcMatrixOffsets[position] = *lastOffset;
    gcMatrixSizes[position] = sizeofMat; // / sizeof (unsigned short )

    for (int i = 0; i < gcMatrixSizes[position]; i++) {
        //if (G_DEBUG)
        //    std::cout << i << " ; position: " << gcMatrixOffsets[position] + i << std::endl;
        gcMatrixData[gcMatrixOffsets[position] + i] = mat1[i];
    }
    *lastOffset += sizeofMat;
}

void testGcSimilarityKernelWith100x100() {
    const GraphCode &gc = generateTestDataGc(100);

    UUIDGraphCode newGc;

    for (int i = 0; i < 100 * 100; i++) {
        newGc.matrix[i] = gc.matrix[i];
    }

    int count = 0;
    for (std::string str: *gc.dict) {
        uuid_t binuuid;
        uuid_generate_random(binuuid);

        for (int i = 0; i < 16; i++) {
            newGc.dict[count][i] = binuuid[i];
        }


    }

    Metrics *dResult;
    UUIDGraphCode *dGC1;
    UUIDGraphCode *dGC2;
    HANDLE_ERROR(cudaMalloc((void **) &dResult, sizeof(Metrics)));
    HANDLE_ERROR(cudaMalloc((void **) &dGC1, sizeof(newGc)));
    HANDLE_ERROR(cudaMalloc((void **) &dGC2, sizeof(newGc)));

    HANDLE_ERROR(cudaMemcpy(dGC1, &newGc, sizeof(newGc), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dGC2, &newGc, sizeof(newGc), cudaMemcpyHostToDevice));


    compareUUID<<<1, 1>>>(dGC1, dGC2, dResult);
    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    Metrics myresult;

    HANDLE_ERROR(cudaMemcpy(&myresult, dResult, sizeof(Metrics), cudaMemcpyDeviceToHost));

    std::cout << "Result"
              << "Similarity " << myresult.similarity << "; "
              << "Recommendation " << myresult.recommendation << "; "
              << "Inference " << myresult.inferencing << "; "
              << std::endl;

    assert(myresult.similarity == 1);
    assert(myresult.recommendation == 1);
    assert(myresult.inferencing == 1);

    HANDLE_ERROR(cudaFree(dResult));
    HANDLE_ERROR(cudaFree(dGC1));
    HANDLE_ERROR(cudaFree(dGC2));

}


