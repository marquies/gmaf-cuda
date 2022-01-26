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

void convertDict2Matrix(int size, int *destMatrix, nlohmann::json jsonMatrix);
//void convertDict2Matrix(int size, int **destMatrix, nlohmann::json jsonMatrix);
bool isPrime(int number);

int findLargestDivisor(int n);

void selectionSort(Metrics *array, const int size);



#endif