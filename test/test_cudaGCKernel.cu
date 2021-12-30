//
// Created by breucking on 30.12.21.
//

#include "testhelper.cpp"
#include "../src/cuda_algorithms.cuh"
#include <cuda_runtime.h>
#include <uuid/uuid.h>
#include <string>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <cudahelper.cuh>


void testGcSimilarityKernelWith100x100();

void testGcSimilarityKernelWith3x3();

void appendMatrix(const unsigned short *mat1, unsigned short sizeofMat, unsigned short *gcMatrixData,
                  unsigned int *gcDictData, unsigned int *gcMatrixOffsets, unsigned int *gcMatrixSizes, int *lastOffset,
                  int position);


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

__global__ void compare2(unsigned short *gcMatrixData,
                         unsigned int *gcDictData,
                         unsigned int gcMatrixOffsets[2],
                         unsigned int gcMatrixSizes[2],
                         unsigned int gcDictOffsets[2],
                         Metrics *metrics) {
    int sim = 0;
    int elements = sqrtf((float) gcMatrixSizes[0]);

    for (int i = 0; i < elements; i++) {
        for (int j = 0; j < elements; j++) {
            if (gcDictData[gcDictOffsets[0] + i] == gcDictData[gcDictOffsets[1] + j]) {
                sim++;
            }
        }
    }

    int num_of_non_zero_edges = 0;
    int edge_metric_count = 0;
    int edge_type = 0;


    for (int i = 0; i < elements; i++) {
        for (int j = 0; j < elements; j++) {

            if (i != j && gcMatrixData[gcMatrixOffsets[0] + i * elements + j] != 0) {
                num_of_non_zero_edges++;

                int position1 = i;
                int position2 = j;
                if (position1 == -1 || position2 == -1) {
                    continue;
                }

                int edge = gcMatrixData[gcMatrixOffsets[1] + position1 * elements +
                                        position2];
                if (edge != 0) {
                    edge_metric_count++;
                }
                if (edge == gcMatrixData[gcMatrixOffsets[0] + i * elements + j]) {
                    edge_type++;
                }

            }
        }
    }
    metrics->similarity = 0.0;
    metrics->recommendation = 0.0;
    metrics->inferencing = 0.0;
    metrics->similarity = (float) sim / (float) elements;
    if (num_of_non_zero_edges > 0) {
        /*edge_metric*/ metrics->recommendation = (float) edge_metric_count / (float) num_of_non_zero_edges;
    }
    if (edge_metric_count > 0) {
        /*edge_type_metric*/ metrics->inferencing = (float) edge_type / (float) edge_metric_count;
    }
}


__global__ void compare(GraphCode2 *gc1, GraphCode2 *gc2, Metrics *metrics) {
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
}

void testGcSimilarityKernelWith3x3() {

    int dictSize = 3;

    GraphCode gcq3;
    std::vector<std::string> *vect = new std::vector<std::string>{"head", "body", "foot"};
    unsigned short mat1[] = {1, 1, 0, 0, 1, 0, 0, 1, 1};
    unsigned short *mat = mat1;
    gcq3.dict = vect;
    gcq3.matrix = mat;


    GraphCode gcq4;
    std::vector<std::string> *vect2 = new std::vector<std::string>{"head", "body", "foot"};
    unsigned short mat2[] = {1, 2, 0, 0, 1, 0, 0, 0, 1};
    unsigned short *mata = mat2;
    gcq4.dict = vect2;
    gcq4.matrix = mata;


    unsigned short *gcMatrixData = (unsigned short *) malloc(0);
    unsigned int *gcDictData = (unsigned int *) malloc(0);

    //unsigned int *gcMatrixOffsets = (unsigned int *) malloc(0);
    //unsigned int *gcMatrixSizes = (unsigned int *) malloc(0);

    unsigned int gcMatrixOffsets[2];
    unsigned int gcDictOffsets[2];
    unsigned int gcMatrixSizes[2];


    unsigned int dictCounter = 0;
    std::map<std::string, unsigned int> dict_map;
    for (std::string str: *vect) {
        dict_map[str] = dict_map.size();
    }

    int aS = dictSize;
    gcDictOffsets[0] = 0;
    gcDictOffsets[1] = aS;
    unsigned int *gcDictData_n = (unsigned int *) realloc(gcDictData, aS * sizeof(unsigned int));
    if (gcDictData_n) {
        gcDictData = gcDictData_n;
    } else {
        // deal with realloc failing because memory could not be allocated.
        exit(99);
    }
    for (std::string str: *vect) {
        gcDictData[dictCounter++] = dict_map[str];
    }


    int lastOffset = 0;
    int lastPosition = 0;

    int matSize = sizeof(mat1) / sizeof(unsigned short);

//    size_t currentSize = sizeof(gcMatrixData);
//    std::cout << currentSize << std::endl;

    unsigned int newS = 0;
    for (int i = 0; i <= lastPosition; i++) {
        newS += gcMatrixSizes[lastPosition];
    }// ;
    size_t newSize = newS * sizeof(unsigned short);
    unsigned short *gcMatrixData_n = (unsigned short *) realloc(gcMatrixData, newSize);
    if (gcMatrixData_n) {
        gcMatrixData = gcMatrixData_n;
    } else {
        // deal with realloc failing because memory could not be allocated.
        exit(99);
    }

    appendMatrix(mat1, matSize, gcMatrixData, gcDictData, gcMatrixOffsets, gcMatrixSizes, &lastOffset,
                 lastPosition);

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

    lastPosition++;

    newS = dictSize + dictCounter;

    unsigned int *gcDictData_n2 = (unsigned int *) realloc(gcDictData, newS * sizeof(unsigned int));
    if (gcDictData_n2) {
        gcDictData = gcDictData_n2;
    } else {
        // deal with realloc failing because memory could not be allocated.
        exit(99);
    }
    for (std::string str: *vect) {
        gcDictData[dictCounter++] = dict_map[str];
    }
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

    appendMatrix(mat2, matSize, gcMatrixData, gcDictData, gcMatrixOffsets, gcMatrixSizes, &lastOffset,
                 lastPosition);

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


    unsigned short *d_gcMatrixData;
    unsigned int *d_gcDictData;
    unsigned int *d_gcMatrixOffsets;
    unsigned int *d_gcMatrixSizes;
    unsigned int *d_gcDictOffsets;
    Metrics *d_result;

    int md_size = 0;
    for (int i = 0; i <= lastPosition; i++) {
        md_size += gcMatrixSizes[lastPosition];
    }// ;

    HANDLE_ERROR(cudaMalloc((void **) &d_gcMatrixData, md_size * sizeof(unsigned short)));
    HANDLE_ERROR(cudaMalloc((void **) &d_gcDictData, dictCounter * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void **) &d_gcMatrixOffsets, 2 * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void **) &d_gcMatrixSizes, 2 * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void **) &d_gcDictOffsets, 2 * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void **) &d_result, sizeof(Metrics)));


    HANDLE_ERROR(
            cudaMemcpy(d_gcMatrixData, gcMatrixData, md_size * sizeof(unsigned short), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_gcDictData, gcDictData, dictCounter * sizeof(unsigned int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_gcMatrixOffsets, gcMatrixOffsets, 2 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_gcMatrixSizes, gcMatrixSizes, 2 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_gcDictOffsets, gcDictOffsets, 2 * sizeof(unsigned int), cudaMemcpyHostToDevice));


    //for(int i = 0; i <= lastPosition; i++) {
    compare2<<<1, 1>>>(d_gcMatrixData, d_gcDictData, d_gcMatrixOffsets, d_gcMatrixSizes, d_gcDictOffsets, d_result);
    // }
    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    Metrics result;
    HANDLE_ERROR(cudaMemcpy(&result, d_result, sizeof(Metrics), cudaMemcpyDeviceToHost));
    std::cout << "Result"
              << "Similarity " << result.similarity << "; "
              << "Recommendation " << result.recommendation << "; "
              << "Inference " << result.inferencing << "; "
              << std::endl;

    assert(result.similarity == 1);
    assert(result.recommendation == 0.5);
    assert(result.inferencing == 0);

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
    int newS = 0;

    for (int i = 0; i < gcMatrixSizes[position]; i++) {
        std::cout << i << " ; position: " << gcMatrixOffsets[position] + i << std::endl;
        gcMatrixData[gcMatrixOffsets[position] + i] = mat1[i];
    }
    *lastOffset += sizeofMat;
}

void testGcSimilarityKernelWith100x100() {
    const GraphCode &gc = generateTestDataGc(100);

    GraphCode2 newGc;

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
    GraphCode2 *dGC1;
    GraphCode2 *dGC2;
    HANDLE_ERROR(cudaMalloc((void **) &dResult, sizeof(Metrics)));
    HANDLE_ERROR(cudaMalloc((void **) &dGC1, sizeof(newGc)));
    HANDLE_ERROR(cudaMalloc((void **) &dGC2, sizeof(newGc)));

    HANDLE_ERROR(cudaMemcpy(dGC1, &newGc, sizeof(newGc), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dGC2, &newGc, sizeof(newGc), cudaMemcpyHostToDevice));


    compare<<<1, 1>>>(dGC1, dGC2, dResult);
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


