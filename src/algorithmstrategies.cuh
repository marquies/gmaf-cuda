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

/**
 * Abstract Strategy class.
 */
class Strategy {
public:
    /**
     * Performs the implemented algorithm.
     * @param loadUnit initialized LoadUnit
     * @param gcPosition position of the query Graph Code
     */
    virtual void performQuery(GcLoadUnit *loadUnit, int gcPosition) = 0;

    virtual ~Strategy() = default;
};

class QueryHandler;

/**
 * Abstract Factory for algorithm strategies.
 */
class StrategyFactory {
public:
    /**
     * Sets the algorithm to the query handler and returns a loadUnit with appropriate mode set.
     * @param algorithm the desired algorithm.
     * @param qh query handler.
     * @return a loadUnit with mode set according to the algorithm.
     */
    GcLoadUnit *setupStrategy(Algorithms algorithm, QueryHandler *qh);

    /**
    * Initializes a query handler and a loadUnit with appropriate mode set.
    * @param algorithm the desired algorithm.
    * @return returns a tuple of a loadUnit with mode set according to the algorithm and queryhandler.
    */
    std::tuple<GcLoadUnit::Modes, Strategy *> setupStrategy1(Algorithms algorithm);
};

/**
 * Strategy for the algorithm PC on CUDA with data in GPU memory.
 */
class CudaTask1OnGpuMemory : public Strategy {
public:
    /**
     * Performs a query with the algorithm PC on CUDA
     * @param loadUnit loadUnit with the data pointers.
     * @param gcPosition the position of the query Graph Code in the loaded data.
     */
    void performQuery(GcLoadUnit *loadUnit, int gcPosition) override;
};

/**
 * Strategy for the algorithm PC on CUDA loading the data in GPU memory.
 */
class CudaTask1MemCopy : public Strategy {
public:
    /**
     * Performs a query with the algorithm PC on CUDA
     * @param loadUnit loadUnit with the data pointers.
     * @param gcPosition the position of the query Graph Code in the loaded data.
     */
    void performQuery(GcLoadUnit *loadUnit, int gcPosition) override;
};

/**
 * Strategy for the algorithm PC on a CPU without parallelism.
 */
class CpuSequentialTask1 : public Strategy {
public:
    /**
     * Performs a query with the algorithm PC on CPU.
     * @param loadUnit loadUnit with the data pointers.
     * @param gcPosition the position of the query Graph Code in the loaded data.
     */
    void performQuery(GcLoadUnit *loadUnit, int gcPosition) override;
};

/**
 * Strategy for the algorithm PC on a CPU with thread parallelism.
 */
class CpuParallelTask1 : public Strategy {
public:
    /**
     * Performs a query with the algorithm PC on CPU with threads.
     * @param loadUnit loadUnit with the data pointers.
     * @param gcPosition the position of the query Graph Code in the loaded data.
     */
    void performQuery(GcLoadUnit *loadUnit, int gcPosition) override;
};

/**
 * Strategy for the algorithm PM on CUDA.
 */
class CudaTask2a : public Strategy {
public:
    /**
     * Perfoms the algorithm PM on CUDA.
     * @param loadUnit loadUnit with the data pointers.
     * @param gcPosition the position of the query Graph Code in the loaded data.
     */
    void performQuery(GcLoadUnit *loadUnit, int gcPosition) override;
};

/**
 * Strategy for the algorithm PMPR on CUDA.
 */
class CudaTask2ab : public Strategy {
public:
    /**
     * Perfoms the algorithm PMPR on CUDA.
     * @param loadUnit loadUnit with the data pointers.
     * @param gcPosition the position of the query Graph Code in the loaded data.
     */
    void performQuery(GcLoadUnit *loadUnit, int gcPosition) override;
};

/**
 * Strategy for the algorithm PCPS on CUDA.
 */
class CudaTask13 : public Strategy {
public:
    void performQuery(GcLoadUnit *loadUnit, int gcPosition) override;
};

#endif //GCSIM_ALGORITHMSTRATEGIES_CUH
