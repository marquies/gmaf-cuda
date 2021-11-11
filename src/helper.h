//
// Created by breucking on 11.11.21.
//


#ifndef HELPER_H_
#define HELPER_H_

#include <nlohmann/json.hpp>


void convertDict2Matrix(int size, int *destMatrix, nlohmann::json jsonMatrix);

#endif