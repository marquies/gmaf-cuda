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
    virtual void performQuery(GcLoadUnit *loadUnit, int gcPosition) = 0;

    virtual ~Strategy() = default;
};

class QueryHandler {

    std::unique_ptr<Strategy> strat_;

public:
    int processQuery(const std::string& query, GcLoadUnit *loadUnit);

    static bool validate(const std::string& query);

//    static void selectionSort(Metrics *pMetrics, int i);

    Metrics *runQuery(GcLoadUnit &loadUnit);

    void setStrategy(std::unique_ptr<Strategy> strat) { strat_ = std::move(strat); }


};


class CudaTask1OnGpuMemory : public Strategy {
public:
    void performQuery(GcLoadUnit *loadUnit, int gcPosition) override;
};

class CudaTask1MemCopy : public Strategy {
public:
    void performQuery(GcLoadUnit *loadUnit, int gcPosition) override;
};

class CpuSequentialTask1 : public Strategy {
public:
    void performQuery(GcLoadUnit *loadUnit, int gcPosition) override;
};
class CpuParallelTask1 : public Strategy {
public:
    void performQuery(GcLoadUnit *loadUnit, int gcPosition) override;
};

class CudaTask2a : public Strategy {
public:
    void performQuery(GcLoadUnit *loadUnit, int gcPosition) override;
};
class CudaTask2ab : public Strategy {
public:
    void performQuery(GcLoadUnit *loadUnit, int gcPosition) override;
};
class CudaTask13 : public Strategy {
public:
    void performQuery(GcLoadUnit *loadUnit, int gcPosition) override;
};



#endif //GCSIM_QUERYHANDLER_CUH
