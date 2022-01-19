//
// Created by breucking on 19.01.22.
//

#ifndef GCSIM_QUERYHANDLER_H
#define GCSIM_QUERYHANDLER_H

#include <iostream>
#include <stdlib.h>

class QueryHandler {

public:
    static void processQuery(std::string basicString);

    static bool validate(std::string query);
};


#endif //GCSIM_QUERYHANDLER_H
