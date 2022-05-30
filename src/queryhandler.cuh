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

/**
 * Class to check and execute a query.
 */
class QueryHandler {

    /**
     * Internal strategy pointer.
     */
    std::unique_ptr<Strategy> strat_;

public:
    /**
     * Process a user input query.
     * @param query query string
     * @param loadUnit load unit with the loaded graph code collection.
     * @return the calculated metric values.
     */
    Metrics * processQuery(const std::string &query, GcLoadUnit *loadUnit);

    /**
     * Validates a query string for correct syntax.
     * @param query string.
     * @return true if string is valid.
     */
    static bool validate(const std::string& query);

    /**
     * Sets the strategy with the algorithm to use for calculation.
     *
     * @param strat the desired strategy.
     */
    void setStrategy(std::unique_ptr<Strategy> strat) { strat_ = std::move(strat); }


};


#endif //GCSIM_QUERYHANDLER_CUH
