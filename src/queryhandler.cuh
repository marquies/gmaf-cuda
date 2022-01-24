//
// Created by breucking on 19.01.22.
//

#ifndef GCSIM_QUERYHANDLER_CUH
#define GCSIM_QUERYHANDLER_CUH

#include <iostream>
#include <stdlib.h>
#include "gcloadunit.cuh"
#include "helper.h"


class Strategy {
public:
    virtual void performQuery(GcLoadUnit loadUnit) = 0;

    virtual ~Strategy() = default;
};

class QueryHandler {

    std::unique_ptr<Strategy> strat_;

public:
    int processQuery(std::string query, GcLoadUnit loadUnit);

    bool validate(std::string query);

    void selectionSort(Metrics *pMetrics, int i);

    Metrics *runQuery(GcLoadUnit &loadUnit);

    void setStrategy(std::unique_ptr<Strategy> strat) { strat_ = std::move(strat); }


};


class CudaTask1OnGpuMemory : public Strategy {
public:
    void performQuery(GcLoadUnit loadUnit) override;
};

class CudaTask1MemCopy : public Strategy {
public:
    void performQuery(GcLoadUnit loadUnit) override;
};

class CpuSequentialTask1 : public Strategy {
public:
    void performQuery(GcLoadUnit loadUnit) override;
};

#endif //GCSIM_QUERYHANDLER_CUH
