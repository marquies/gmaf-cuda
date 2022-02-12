//
// Created by breucking on 19.01.22.
//

#include <regex>
#include "queryhandler.cuh"
#include "gcloadunit.cuh"
#include "helper.h"
#include "cpualgorithms.h"
#include <algorithm>
#include <chrono>


int QueryHandler::processQuery(const std::string& query, GcLoadUnit *loadUnit) {
    if (!validate(query)) {
        throw std::invalid_argument("Empty String");
    }

    std::regex regexp("Query by Example: (.*)$");

    std::smatch m;

    regex_search(query, m, regexp);

    if (m.size() > 1) {
        std::string gcQueryName = m.str(1);
        std::cout << m.str(1) << " ";
        if (loadUnit->hasGc(gcQueryName)) {

            if (strat_) {
                strat_->performQuery(loadUnit, loadUnit->getGcPosition(gcQueryName));
            } else {
                throw std::runtime_error("Algorithm strategy not set");
            }

            return 0;
        } else {
            std::cout << "GC not found." << std::endl;

        }
    }

    return 1;
}


bool QueryHandler::validate(const std::string& query) {
    if (query.empty()) {
        return false;
    }
    if (!regex_match(query, std::regex("(Query by Example: )(.*)$")))
        return false;
    return true;
}


void CudaTask1OnGpuMemory::performQuery(GcLoadUnit *loadUnit, int gcPosition) {
// call algorithm in form of
    std::cout << "-----------------------------------" << std::endl;

    loadUnit->loadIntoCudaMemory();
    int times = G_BENCHMARK ? G_BENCHMARK_REPEAT : 1;

    for (int i = 0; i < times; i++) {
        auto start = std::chrono::system_clock::now();

        Metrics *result = demoCalculateGCsOnCuda(loadUnit->getNumberOfGc(),
                                                 loadUnit->getNumberOfDictElements(),
                                                 loadUnit->getGcMatrixDataCudaPtr(),
                                                 loadUnit->getGcDictDataCudaPtr(),
                                                 loadUnit->getGcMatrixOffsetsCudaPtr(),
                                                 loadUnit->getDictOffsetCudaPtr(),
                                                 loadUnit->getMatrixSizesCudaPtr(),
                                                 gcPosition
        );

        auto endOfCalc = std::chrono::system_clock::now();


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

        auto end = std::chrono::system_clock::now();


        std::chrono::duration<double> elapsed_secondsCalc = endOfCalc - start;
        std::chrono::duration<double> elapsed_secondsTotal = end - start;
        time_t end_time = std::chrono::system_clock::to_time_t(end);
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
        auto start = std::chrono::system_clock::now();

        Metrics *result = demoCalculateGCsOnCudaWithCopy(loadUnit->getNumberOfGc(),
                                                         loadUnit->getNumberOfDictElements(),
                                                         loadUnit->getGcMatrixDataPtr(),
                                                         loadUnit->getGcDictDataPtr(),
                                                         loadUnit->getGcMatrixOffsetsPtr(),
                                                         loadUnit->getDictOffsetPtr(),
                                                         loadUnit->getMatrixSizesPtr(),
                                                         gcPosition
        );


        auto endOfCalc = std::chrono::system_clock::now();


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

        auto end = std::chrono::system_clock::now();


        std::chrono::duration<double> elapsed_secondsCalc = endOfCalc - start;
        std::chrono::duration<double> elapsed_secondsTotal = end - start;
        time_t end_time = std::chrono::system_clock::to_time_t(end);

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

void CpuSequentialTask1::performQuery(GcLoadUnit *loadUnit, int gcPosition) {
    int times = G_BENCHMARK ? G_BENCHMARK_REPEAT : 1;
    std::cout << "-----------------------------------" << std::endl;

    for (int i = 0; i < times; i++) {
        std::vector<GraphCode> codes = loadUnit->getGcCodes();
        auto start = std::chrono::system_clock::now();
        std::vector<Metrics> results;
        for (int j = 0; j < codes.size(); j++) {
            Metrics res = demoCalculateSimilaritySequentialOrdered(codes.at(gcPosition), codes.at(j));
            res.idx = j;
            results.push_back(res);
        }
        auto endOfCalc = std::chrono::system_clock::now();

        std::sort(results.begin(), results.end(), comp);

        auto end = std::chrono::system_clock::now();


        std::chrono::duration<double> elapsed_secondsCalc = endOfCalc - start;
        std::chrono::duration<double> elapsed_secondsTotal = end - start;
        time_t end_time = std::chrono::system_clock::to_time_t(end);

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
        auto start = std::chrono::system_clock::now();

        std::vector<Metrics> results = demoCalculateCpuThreaded(codes.at(gcPosition), codes, 5);

        auto endOfCalc = std::chrono::system_clock::now();

        std::sort(results.begin(), results.end(), comp);

        auto end = std::chrono::system_clock::now();


        std::chrono::duration<double> elapsed_secondsCalc = endOfCalc - start;
        std::chrono::duration<double> elapsed_secondsTotal = end - start;
        time_t end_time = std::chrono::system_clock::to_time_t(end);

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
        auto start = std::chrono::system_clock::now();
        std::vector<Metrics> results;
        for (int j = 1; j < codes.size(); j++) {
            Metrics res = demoCudaLinearMatrixMemoryWithCopy(codes.at(gcPosition), codes.at(j));
            results.push_back(res);
        }
        auto endOfCalc = std::chrono::system_clock::now();

        std::sort(results.begin(), results.end(), comp);

        auto end = std::chrono::system_clock::now();


        std::chrono::duration<double> elapsed_secondsCalc = endOfCalc - start;
        std::chrono::duration<double> elapsed_secondsTotal = end - start;
        time_t end_time = std::chrono::system_clock::to_time_t(end);

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
        auto start = std::chrono::system_clock::now();
        std::vector<Metrics> results;
        for (int j = 1; j < codes.size(); j++) {
            Metrics res = demoCudaLinearMatrixMemoryCudaReduceSum(codes.at(gcPosition), codes.at(j));
            results.push_back(res);
        }
        auto endOfCalc = std::chrono::system_clock::now();

        std::sort(results.begin(), results.end(), comp);

        auto end = std::chrono::system_clock::now();


        std::chrono::duration<double> elapsed_secondsCalc = endOfCalc - start;
        std::chrono::duration<double> elapsed_secondsTotal = end - start;
        time_t end_time = std::chrono::system_clock::to_time_t(end);

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
        auto start = std::chrono::system_clock::now();

        Metrics *devicePtr = demoCalculateGCsOnCudaAndKeepMetricsInMem(loadUnit->getNumberOfGc(),
                                                                       loadUnit->getNumberOfDictElements(),
                                                                       loadUnit->getGcMatrixDataCudaPtr(),
                                                                       loadUnit->getGcDictDataCudaPtr(),
                                                                       loadUnit->getGcMatrixOffsetsCudaPtr(),
                                                                       loadUnit->getDictOffsetCudaPtr(),
                                                                       loadUnit->getMatrixSizesCudaPtr(),
                                                                       gcPosition
        );

        auto endOfCalc = std::chrono::system_clock::now();
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

        auto end = std::chrono::system_clock::now();

        if (i == times - 1) {
            writeMetricsToFile(result, loadUnit->getNumberOfGc());
        }

        std::chrono::duration<double> elapsed_secondsCalc = endOfCalc - start;
        std::chrono::duration<double> elapsed_secondsTotal = end - start;
        time_t end_time = std::chrono::system_clock::to_time_t(end);

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
