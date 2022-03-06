//
// Created by breucking on 10.02.22.
//

#ifndef GCSIM_TESTHELPER_H
#define GCSIM_TESTHELPER_H

#include <vector>
#include <nlohmann/json.hpp>
#include "graphcode.h"
#include <stdio.h>

extern std::vector<std::string> DICT;
extern std::vector<std::string> DICT2;
extern double EPSILON;
/*
 * Decleration of methods.
 */
json generateTestData(int n);

GraphCode generateTestDataGc(int dimension);

bool AreSame(double a, double b);

#endif //GCSIM_TESTHELPER_H
