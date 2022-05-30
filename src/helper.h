//
// Created by breucking on 11.11.21.
//


#ifndef HELPER_H_
#define HELPER_H_

#include <nlohmann/json.hpp>
#include "graphcode.h"

extern bool G_DEBUG;
extern bool G_BENCHMARK;
extern short G_BENCHMARK_REPEAT;

/**
 * Converts a graph code as json representation to a serialized array.
 * @param size size of the graph code
 * @param destMatrix pointer to a allocated space with size of size param.
 * @param jsonMatrix the source matrix with a 2D int array.
 */
void convertDict2Matrix(unsigned long size, unsigned short *destMatrix, nlohmann::json jsonMatrix);

//void convertDict2Matrix(int size, int **destMatrix, nlohmann::json jsonMatrix);
/**
 * Checks if a number is prime.
 * @param number number to test
 * @return true is number is prime.
 */
bool isPrime(int number);

/**
 * Find the largest divisor of a number.
 * @param n number to test.
 * @return the largest divisor.
 */
int findLargestDivisor(int n);

/**
 * Sorts an array with selection sort algorithm.
 * @param array input data array to sort
 * @param size number of elements in the array.
 */
void selectionSort(Metrics *array, const int size);

/**
 * Sorts an array with Introsort algorithm
 * @param array input array to sort.
 * @param begin first element
 * @param end last element.
 */
void Introsort(Metrics *array, int begin, int end);

/**
 * Sorts an array with heap sort algorithm.
 * @param data input data array to sort
 * @param size number of elements in the array.
 */
void HeapSort(Metrics *data, int size);

/**
 * Swaps the data of the pointer.
 * @param x pointer
 * @param y pointer
 */
void swap(Metrics *const x, Metrics *const y);

/**
 * Compares the metric values based on the orderd values
 * @param positionA a metric
 * @param positionB b metric
 * @return difference of a - b
 */
float compare(Metrics *positionA, Metrics *positionB);

/**
 * Helper to write the metrics to a file 'output.csv' in the working directory.
 *
 * @param metrics array of the metrics.
 * @param n numbe rof elements in the array.
 */
void writeMetricsToFile(Metrics *metrics, int n);

/**
 * Helper to write the metrics to a file 'output.csv' in the working directory.
 * @param metrics vector of metrics.
 */
void writeMetricsToFile(std::vector<Metrics> metrics);

/**
 * Converts a json graph code to a memory data structure.
 * @param gcq the graph code as json 2int d array
 * @param gc1Dictionary  the dictionary as json string array
 * @param numberOfElements number of elements in the graph code.
 * @param items return value of the number of items in the matrix
 * @param inputMatrix return value of the converted matrix
 */
void convertJsonGc2GcDataStructure(const json &gcq, json &gc1Dictionary, unsigned long &numberOfElements,
                                   unsigned long &items, unsigned short int *&inputMatrix);


#endif