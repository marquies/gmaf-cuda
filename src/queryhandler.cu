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


