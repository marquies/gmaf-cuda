//
// Created by breucking on 27.12.21.
//

#include <iostream>
#include <graphcode.h>
#include <chrono>
#include <ctime>

/********/
/* MAIN */
void testCpuCalc();

/********/
int main(int, char**)
{
    testCpuCalc();
   std::cout << "foo";

}

void testCpuCalc() {

    int n = 10000;
    nlohmann::json gcq;
    gcq["dictionary"] = { "head", "body", "foot"};
    gcq["matrix"] = {{1,1,0}, {0,1,0}, {0,1,1}};

    nlohmann::json gce;
    gce["dictionary"] = { "head", "body", "torso"};
    gce["matrix"] = {{1,2,0}, {0,1,0}, {0,0,1}};

    std::vector<json> others;

    for(int i = 0; i < n; i++) {
        others.push_back(gce);
    }

    gmaf::GraphCode gc;

    auto start = std::chrono::system_clock::now();

    const std::vector<Metrics> &metrics = gc.calculateSimilarityV(0, &gcq, &others, 0, n);

    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "finished computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n";

}
