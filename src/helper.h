//
// Created by breucking on 11.11.21.
//


#ifndef HELPER_H_
#define HELPER_H_

#include <nlohmann/json.hpp>

extern bool G_DEBUG;

void convertDict2Matrix(int size, int *destMatrix, nlohmann::json jsonMatrix);
//void convertDict2Matrix(int size, int **destMatrix, nlohmann::json jsonMatrix);
bool isPrime(int number);

int findLargestDivisor(int n);



#endif