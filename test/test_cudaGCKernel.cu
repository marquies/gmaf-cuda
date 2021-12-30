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


void testGcSimilarityKernel();

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


__global__ void compare(GraphCode2 *gc1, GraphCode2 *gc2, float *metrics) {
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
    for(int i = 0; i < elements; i++) {
//        for (const auto &item2: *gc2->dict) {
        for(int j = 0; j < elements; j++) {
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

    metrics[0] = 0.0;
    metrics[1] = 0.0;
    metrics[2] = 0.0;
    //float node_metric = (float) sim / (float) elements;
    metrics[0] = (float) sim / (float) elements;
    //float edge_metric = 0.0;
    if (num_of_non_zero_edges > 0) {
        /*edge_metric*/ metrics[1] = (float) edge_metric_count / (float) num_of_non_zero_edges;
    }
//    float edge_type_metric = 0.0;
    if (edge_metric_count > 0) {
        /*edge_type_metric*/ metrics[2] = (float) edge_type / (float) edge_metric_count;
    }
    //*metrics = node_metric;
//    return metrics;
}


/**
* Main
*/
int main() {
    testGcSimilarityKernel();
}

void testGcSimilarityKernel() {
    const GraphCode &gc = generateTestDataGc(100);

    GraphCode2 newGc;

    for(int i = 0; i < 100*100; i++) {
        newGc.matrix[i] = gc.matrix[i];
    }

    //newGc.dict = (unsigned char *) malloc(gc.dict->size() * sizeof(uuid_t));

    int count = 0;
    for (std::string str: *gc.dict) {
        uuid_t binuuid;
        uuid_generate_random(binuuid);

//        newGc.dict[count++] = uuid_t;
        for (int i = 0; i < 16; i++) {
            newGc.dict[count][i] = binuuid[i];
        }

//        unsigned char foo[16];
//        for(int i = 0; i < 16; i++) {
//            foo[i] = newGc.dict[count + i];
//        }
//
//        char *myuuid = static_cast<char *>(malloc(37));
//        uuid_unparse(foo, myuuid);
//        std::cout << "UUID" << myuuid << std::endl;
//        uuid_unparse(binuuid, myuuid);
//        std::cout << "UUID" << myuuid << std::endl;


    }
    std::cout << "SizeOfMetrics " << sizeof (Metrics) << std::endl;
    std::cout << "Compare: " << uuid_compare(newGc.dict[1], newGc.dict[0]) << std::endl;

    float *dResult;
    GraphCode2 *dGC1;
    GraphCode2 *dGC2;
    HANDLE_ERROR(cudaMalloc((void **) &dResult, sizeof(float) * 3));
    HANDLE_ERROR(cudaMalloc((void **) &dGC1, sizeof(newGc)));
    HANDLE_ERROR(cudaMalloc((void **) &dGC2, sizeof(newGc)));

    HANDLE_ERROR(cudaMemcpy(dGC1, &newGc, sizeof(newGc), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dGC2, &newGc, sizeof(newGc), cudaMemcpyHostToDevice));


    compare<<<1, 1>>>(dGC1, dGC2, dResult);
    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    float myresult[3];

    HANDLE_ERROR(cudaMemcpy(&myresult, dResult, sizeof(float) * 3, cudaMemcpyDeviceToHost));

    std::cout << "Result"
    << "Similarity " << myresult[0]
    << "Recommendation " << myresult[0+1]
    << "Inference " << myresult[0+2]
    << std::endl;

    HANDLE_ERROR(cudaFree(dResult));
    HANDLE_ERROR(cudaFree(dGC1));
    HANDLE_ERROR(cudaFree(dGC2));


    std::cout << "Done!" << std::endl;
}
