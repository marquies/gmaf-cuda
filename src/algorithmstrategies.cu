//
// Created by breucking on 13.02.22.
//

#include <chrono>
#include <algorithm>
#include "cpualgorithms.h"
#include "helper.h"
#include "gcloadunit.cuh"
#include "queryhandler.cuh"
#include <regex>
#include "algorithmstrategies.cuh"

bool comp(Metrics e1, Metrics e2);

GcLoadUnit *StrategyFactory::setupStrategy(Algorithms algorithm, QueryHandler *qh) {
    GcLoadUnit *loadUnit;
    switch (algorithm) {
        case Algo_pc_cuda:
            qh->setStrategy(std::unique_ptr<Strategy>(new CudaTask1OnGpuMemory));
            loadUnit = new GcLoadUnit(GcLoadUnit::Modes::MODE_MEMORY_MAP);
            break;
        case Algo_pc_cpu_seq:
            qh->setStrategy(std::unique_ptr<Strategy>(new CpuSequentialTask1));
            loadUnit = new GcLoadUnit(GcLoadUnit::Modes::MODE_VECTOR_MAP);
            break;
        case Algo_pc_cpu_par:
            qh->setStrategy(std::unique_ptr<Strategy>(new CpuParallelTask1));
            loadUnit = new GcLoadUnit(GcLoadUnit::Modes::MODE_VECTOR_MAP);
            break;
        case Algo_pm_cuda:
            qh->setStrategy(std::unique_ptr<Strategy>(new CudaTask2a));
            loadUnit = new GcLoadUnit(GcLoadUnit::Modes::MODE_VECTOR_MAP);
            break;
        case Algo_pmr_cuda:
            qh->setStrategy(std::unique_ptr<Strategy>(new CudaTask2ab));
            loadUnit = new GcLoadUnit(GcLoadUnit::Modes::MODE_VECTOR_MAP);
            break;
        case Algo_pcs_cuda:
            qh->setStrategy(std::unique_ptr<Strategy>(new CudaTask13));
            loadUnit = new GcLoadUnit(GcLoadUnit::Modes::MODE_MEMORY_MAP);
            break;

        case Algo_Invalid:
        default:
            throw new std::runtime_error("Invalid Algorithm");
    }
    return loadUnit;
}


std::tuple<GcLoadUnit::Modes, Strategy *> StrategyFactory::setupStrategy1(Algorithms algorithm) {
    GcLoadUnit::Modes mode;
    Strategy *strategy;
    switch (algorithm) {
        case Algo_pc_cuda:

            strategy = (Strategy *) new CudaTask1OnGpuMemory();
            //qh.setStrategy(std::unique_ptr<Strategy>(new CudaTask1MemCopy));
            mode = GcLoadUnit::Modes::MODE_MEMORY_MAP;
            break;
        case Algo_pc_cpu_seq:
            strategy = (Strategy *) new CpuSequentialTask1();
            mode = GcLoadUnit::Modes::MODE_VECTOR_MAP;
            break;
        case Algo_pc_cpu_par:
            strategy = (Strategy *) new CpuParallelTask1();
            mode = GcLoadUnit::Modes::MODE_VECTOR_MAP;
            break;
        case Algo_pm_cuda:
            strategy = (Strategy *) new CudaTask2a();
            mode = GcLoadUnit::Modes::MODE_VECTOR_MAP;
            break;
        case Algo_pmr_cuda:
            strategy = (Strategy *) new CudaTask2ab();
            mode = GcLoadUnit::Modes::MODE_VECTOR_MAP;
            break;
        case Algo_pcs_cuda:
            strategy = (Strategy* ) new CudaTask13();
            mode = GcLoadUnit::Modes::MODE_MEMORY_MAP;
            break;

        case Algo_Invalid:
        default:
            throw new std::runtime_error("Invalid Algorithm");
    }
    return {mode,strategy};
}

void CudaTask1OnGpuMemory::performQuery(GcLoadUnit *loadUnit, int gcPosition) {
// call algorithm in form of
    std::cout << "-----------------------------------" << std::endl;

    loadUnit->loadIntoCudaMemory();
    int times = G_BENCHMARK ? G_BENCHMARK_REPEAT : 1;

    for (int i = 0; i < times; i++) {
        auto start = std::chrono::_V2::system_clock::now();

        Metrics *result = demoCalculateGCsOnCuda(loadUnit->getNumberOfGc(),
                                                 loadUnit->getNumberOfDictElements(),
                                                 loadUnit->getGcMatrixDataCudaPtr(),
                                                 loadUnit->getGcDictDataCudaPtr(),
                                                 loadUnit->getGcMatrixOffsetsCudaPtr(),
                                                 loadUnit->getDictOffsetCudaPtr(),
                                                 loadUnit->getMatrixSizesCudaPtr(),
                                                 gcPosition
        );

        auto endOfCalc = std::chrono::_V2::system_clock::now();


//        selectionSort(result, loadUnit->getNumberOfGc());
        Introsort(result, 0, loadUnit->getNumberOfGc());

        if (G_DEBUG) {


            for (int i = 0; i < loadUnit->getNumberOfGc(); i++) {
                std::cout << "-----------------------------------" << std::endl;
                std::cout << result[i].idx << std::endl;
                std::cout << result[i].similarity << std::endl;
                std::cout << result[i].recommendation << std::endl;
                std::cout << result[i].inferencing << std::endl;
            }
        }

        auto end = std::chrono::_V2::system_clock::now();


        std::chrono::duration<double> elapsed_secondsCalc = endOfCalc - start;
        std::chrono::duration<double> elapsed_secondsTotal = end - start;
        time_t end_time = std::chrono::_V2::system_clock::to_time_t(end);
        if (i == times - 1) {
            writeMetricsToFile(result, loadUnit->getNumberOfGc());
        }
        if (G_DEBUG) {


            std::cout << "run " << ctime(&end_time)
                      << "elapsed time: " << elapsed_secondsTotal.count() << "s\n";
        }
        if (!G_BENCHMARK) {}
        else {
            std::cout << typeid(this).name() << "\t" << loadUnit->getNumberOfGc() << "\t" << elapsed_secondsCalc.count()
                      << "\t" << elapsed_secondsTotal.count() << "\n";
        }
        free(result);
    }
}

void CudaTask1MemCopy::performQuery(GcLoadUnit *loadUnit, int gcPosition) {
// call algorithm in form of
    std::cout << "-----------------------------------" << std::endl;

    int times = G_BENCHMARK ? G_BENCHMARK_REPEAT : 1;
    for (int i = 0; i < times; i++) {
        auto start = std::chrono::_V2::system_clock::now();

        Metrics *result = demoCalculateGCsOnCudaWithCopy(loadUnit->getNumberOfGc(),
                                                         loadUnit->getNumberOfDictElements(),
                                                         loadUnit->getGcMatrixDataPtr(),
                                                         loadUnit->getGcDictDataPtr(),
                                                         loadUnit->getGcMatrixOffsetsPtr(),
                                                         loadUnit->getDictOffsetPtr(),
                                                         loadUnit->getMatrixSizesPtr(),
                                                         gcPosition
        );


        auto endOfCalc = std::chrono::_V2::system_clock::now();


        //selectionSort(result, loadUnit->getNumberOfGc());
        Introsort(result, 0, loadUnit->getNumberOfGc());
        if (G_DEBUG) {


            for (int i = 0; i < loadUnit->getNumberOfGc(); i++) {
                std::cout << "-----------------------------------" << std::endl;
                std::cout << result[i].idx << std::endl;
                std::cout << result[i].similarity << std::endl;
                std::cout << result[i].recommendation << std::endl;
                std::cout << result[i].inferencing << std::endl;
            }
        }

        auto end = std::chrono::_V2::system_clock::now();


        std::chrono::duration<double> elapsed_secondsCalc = endOfCalc - start;
        std::chrono::duration<double> elapsed_secondsTotal = end - start;
        time_t end_time = std::chrono::_V2::system_clock::to_time_t(end);

        if (G_DEBUG) {


            std::cout << "run " << ctime(&end_time)
                      << "elapsed time: " << elapsed_secondsTotal.count() << "s\n";
        }
        if (!G_BENCHMARK) {}
        else {
            std::cout << typeid(this).name() << "\t" << loadUnit->getNumberOfGc() << "\t" << elapsed_secondsCalc.count()
                      << "\t" << elapsed_secondsTotal.count() << "\n";
        }
        free(result);
    }
}

void CpuSequentialTask1::performQuery(GcLoadUnit *loadUnit, int gcPosition) {
    int times = G_BENCHMARK ? G_BENCHMARK_REPEAT : 1;
    std::cout << "-----------------------------------" << std::endl;

    for (int i = 0; i < times; i++) {
        std::vector<GraphCode> codes = loadUnit->getGcCodes();
        auto start = std::chrono::_V2::system_clock::now();
        std::vector<Metrics> results;
        for (int j = 0; j < codes.size(); j++) {
            Metrics res = demoCalculateSimilaritySequentialOrdered(codes.at(gcPosition), codes.at(j));
            res.idx = j;
            results.push_back(res);
        }
        auto endOfCalc = std::chrono::_V2::system_clock::now();

        std::sort(results.begin(), results.end(), comp);

        auto end = std::chrono::_V2::system_clock::now();


        std::chrono::duration<double> elapsed_secondsCalc = endOfCalc - start;
        std::chrono::duration<double> elapsed_secondsTotal = end - start;
        time_t end_time = std::chrono::_V2::system_clock::to_time_t(end);

        if (G_DEBUG) {
            std::cout << "run " << ctime(&end_time)
                      << "elapsed time: " << elapsed_secondsTotal.count() << "s\n";
        }

        if (i == times - 1) {
            writeMetricsToFile(results);
        }

        if (!G_BENCHMARK) {}
        else {
            std::cout << typeid(this).name() << "\t" << loadUnit->getNumberOfGc() << "\t" << elapsed_secondsCalc.count()
                      << "\t" << elapsed_secondsTotal.count() << "\n";
        }
    }

}

void CpuParallelTask1::performQuery(GcLoadUnit *loadUnit, int gcPosition) {
    int times = G_BENCHMARK ? G_BENCHMARK_REPEAT : 1;
    std::cout << "-----------------------------------" << std::endl;

    for (int i = 0; i < times; i++) {
        std::vector<GraphCode> codes = loadUnit->getGcCodes();
        auto start = std::chrono::_V2::system_clock::now();

        std::vector<Metrics> results = demoCalculateCpuThreaded(codes.at(gcPosition), codes, 5);

        auto endOfCalc = std::chrono::_V2::system_clock::now();

        std::sort(results.begin(), results.end(), comp);

        auto end = std::chrono::_V2::system_clock::now();


        std::chrono::duration<double> elapsed_secondsCalc = endOfCalc - start;
        std::chrono::duration<double> elapsed_secondsTotal = end - start;
        time_t end_time = std::chrono::_V2::system_clock::to_time_t(end);

        if (G_DEBUG) {
            std::cout << "run " << ctime(&end_time)
                      << "elapsed time: " << elapsed_secondsTotal.count() << "s\n";
        }
        if (i == times - 1) {
            writeMetricsToFile(results);
        }

        if (!G_BENCHMARK) {}
        else {
            std::cout << typeid(this).name() << "\t" << loadUnit->getNumberOfGc() << "\t" << elapsed_secondsCalc.count()
                      << "\t" << elapsed_secondsTotal.count() << "\n";
        }
    }

}

void CudaTask2a::performQuery(GcLoadUnit *loadUnit, int gcPosition) {
    int times = G_BENCHMARK ? G_BENCHMARK_REPEAT : 1;
    std::cout << "-----------------------------------" << std::endl;

    for (int i = 0; i < times; i++) {
        std::vector<GraphCode> codes = loadUnit->getGcCodes();
        auto start = std::chrono::_V2::system_clock::now();
        std::vector<Metrics> results;
        for (int j = 1; j < codes.size(); j++) {
            Metrics res = demoCudaLinearMatrixMemoryWithCopy(codes.at(gcPosition), codes.at(j));
            results.push_back(res);
        }
        auto endOfCalc = std::chrono::_V2::system_clock::now();

        std::sort(results.begin(), results.end(), comp);

        auto end = std::chrono::_V2::system_clock::now();


        std::chrono::duration<double> elapsed_secondsCalc = endOfCalc - start;
        std::chrono::duration<double> elapsed_secondsTotal = end - start;
        time_t end_time = std::chrono::_V2::system_clock::to_time_t(end);

        if (G_DEBUG) {
            std::cout << "run " << ctime(&end_time)
                      << "elapsed time: " << elapsed_secondsTotal.count() << "s\n";
        }

        if (!G_BENCHMARK) {}
        else {
            std::cout << typeid(this).name() << "\t" << loadUnit->getNumberOfGc() << "\t" << elapsed_secondsCalc.count()
                      << "\t" << elapsed_secondsTotal.count() << "\n";
        }
    }
}

void CudaTask2ab::performQuery(GcLoadUnit *loadUnit, int gcPosition) {
    int times = G_BENCHMARK ? G_BENCHMARK_REPEAT : 1;
    std::cout << "-----------------------------------" << std::endl;

    for (int i = 0; i < times; i++) {
        std::vector<GraphCode> codes = loadUnit->getGcCodes();
        auto start = std::chrono::_V2::system_clock::now();
        std::vector<Metrics> results;
        for (int j = 1; j < codes.size(); j++) {
            Metrics res = demoCudaLinearMatrixMemoryCudaReduceSum(codes.at(gcPosition), codes.at(j));
            results.push_back(res);
        }
        auto endOfCalc = std::chrono::_V2::system_clock::now();

        std::sort(results.begin(), results.end(), comp);

        auto end = std::chrono::_V2::system_clock::now();


        std::chrono::duration<double> elapsed_secondsCalc = endOfCalc - start;
        std::chrono::duration<double> elapsed_secondsTotal = end - start;
        time_t end_time = std::chrono::_V2::system_clock::to_time_t(end);

        if (G_DEBUG) {
            std::cout << "run " << ctime(&end_time)
                      << "elapsed time: " << elapsed_secondsTotal.count() << "s\n";
        }

        if (!G_BENCHMARK) {}
        else {
            std::cout << typeid(this).name() << "\t" << loadUnit->getNumberOfGc() << "\t" << elapsed_secondsCalc.count()
                      << "\t" << elapsed_secondsTotal.count() << "\n";
        }
    }
}

void CudaTask13::performQuery(GcLoadUnit *loadUnit, int gcPosition) {
// call algorithm in form of
    std::cout << "-----------------------------------" << std::endl;

    loadUnit->loadIntoCudaMemory();
    int times = G_BENCHMARK ? G_BENCHMARK_REPEAT : 1;

    for (int i = 0; i < times; i++) {
        auto start = std::chrono::_V2::system_clock::now();

        Metrics *devicePtr = demoCalculateGCsOnCudaAndKeepMetricsInMem(loadUnit->getNumberOfGc(),
                                                                       loadUnit->getNumberOfDictElements(),
                                                                       loadUnit->getGcMatrixDataCudaPtr(),
                                                                       loadUnit->getGcDictDataCudaPtr(),
                                                                       loadUnit->getGcMatrixOffsetsCudaPtr(),
                                                                       loadUnit->getDictOffsetCudaPtr(),
                                                                       loadUnit->getMatrixSizesCudaPtr(),
                                                                       gcPosition
        );

        auto endOfCalc = std::chrono::_V2::system_clock::now();
        Metrics *result = demoSortAndRetrieveMetrics(devicePtr, loadUnit->getNumberOfGc());


        if (G_DEBUG) {


            for (int i = 0; i < loadUnit->getNumberOfGc(); i++) {
                std::cout << "-----------------------------------" << std::endl;
                std::cout << result[i].idx << std::endl;
                std::cout << result[i].similarity << std::endl;
                std::cout << result[i].recommendation << std::endl;
                std::cout << result[i].inferencing << std::endl;
            }
        }

        auto end = std::chrono::_V2::system_clock::now();

        if (i == times - 1) {
            writeMetricsToFile(result, loadUnit->getNumberOfGc());
        }

        std::chrono::duration<double> elapsed_secondsCalc = endOfCalc - start;
        std::chrono::duration<double> elapsed_secondsTotal = end - start;
        time_t end_time = std::chrono::_V2::system_clock::to_time_t(end);

        if (G_DEBUG) {


            std::cout << "run " << ctime(&end_time)
                      << "elapsed time: " << elapsed_secondsTotal.count() << "s\n";
        }
        if (!G_BENCHMARK) {}
        else {
            std::cout << typeid(this).name() << "\t" << loadUnit->getNumberOfGc() << "\t" << elapsed_secondsCalc.count()
                      << "\t" << elapsed_secondsTotal.count() << "\n";
        }
        free(result);
    }
}


bool comp(Metrics e1, Metrics e2) {
//    Metrics e1 = *((Metrics*)elem1);
//    Metrics e2 = *((Metrics*)elem2);
    float a = e1.similarity * 100000.0f + e1.recommendation * 100.0f +
              e1.inferencing;


    float b = e2.similarity * 100000.0f + e2.recommendation * 100.0f +
              e2.inferencing;
    return b - a < 0;
}

