//
// Created by breucking on 19.01.22.
//

#include "queryhandler.h"


 void QueryHandler::processQuery(std::string basicString) {
    if(!validate(basicString)) {
        throw std::invalid_argument( "Empty String" );
    }
}

bool QueryHandler::validate(std::string query) {
    if(query.empty() || query.compare("") == 0) {
        return false;
    }
    return true;
}
