//
// Created by breucking on 13.02.22.
//

#ifndef GCSIM_ALGORITHMSTRATEGIES_CUH
#define GCSIM_ALGORITHMSTRATEGIES_CUH


#include <iostream>
#include <stdlib.h>
#include "gcloadunit.cuh"
#include "helper.h"
#include "queryhandler.cuh"

enum Algorithms {
    Algo_Invalid,
    Algo_pc_cuda,
    Algo_pc_cpu_seq,
    Algo_pc_cpu_par,
    Algo_pm_cuda,
    Algo_pmr_cuda,
    Algo_pcs_cuda
    //others...
};

class Strategy {
public:
    virtual void performQuery(GcLoadUnit *loadUnit, int gcPosition) = 0;

    virtual ~Strategy() = default;
};

class QueryHandler;

class StrategyFactory {
public:
    GcLoadUnit * setupStrategy(Algorithms algorithm, QueryHandler *qh);
    std::tuple<GcLoadUnit::Modes , Strategy*> setupStrategy1(Algorithms algorithm);
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

#endif //GCSIM_ALGORITHMSTRATEGIES_CUH
