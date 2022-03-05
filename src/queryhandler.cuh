//
// Created by breucking on 19.01.22.
//

#ifndef GCSIM_QUERYHANDLER_CUH
#define GCSIM_QUERYHANDLER_CUH

#include <iostream>
#include <stdlib.h>
#include "gcloadunit.cuh"
#include "helper.h"
#include "algorithmstrategies.cuh"


class QueryHandler {

    std::unique_ptr<Strategy> strat_;

public:
    int processQuery(const std::string& query, GcLoadUnit *loadUnit);

    static bool validate(const std::string& query);

//    static void selectionSort(Metrics *pMetrics, int i);

    Metrics *runQuery(GcLoadUnit &loadUnit);

    void setStrategy(std::unique_ptr<Strategy> strat) { strat_ = std::move(strat); }


};


#endif //GCSIM_QUERYHANDLER_CUH
