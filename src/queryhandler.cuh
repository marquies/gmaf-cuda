//
// Created by breucking on 19.01.22.
//

#ifndef GCSIM_QUERYHANDLER_CUH
#define GCSIM_QUERYHANDLER_CUH

#include <iostream>
#include <stdlib.h>
#include "gcloadunit.cuh"

class QueryHandler {

public:
    static int processQuery(std::string query, GcLoadUnit loadUnit);

    static bool validate(std::string query);

    static void selectionSort(Metrics *pMetrics, int i);
};


#endif //GCSIM_QUERYHANDLER_CUH
