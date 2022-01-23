//
// Created by breucking on 19.01.22.
//

#include <regex>
#include "queryhandler.cuh"
#include "gcloadunit.cuh"


int QueryHandler::processQuery(std::string query, GcLoadUnit loadUnit) {
    if (!validate(query)) {
        throw std::invalid_argument("Empty String");
    }

    std::regex regexp("Query by Example: (.*)$");

    std::smatch m;

    regex_search(query, m, regexp);

    if (m.size() > 1) {
        std::string gcQueryName = m.str(1);
        std::cout << m.str(1) << " ";
        if(loadUnit.hasGc(gcQueryName)) {
            // call algorithm in form of

            //result = calculateMetrics(get_position_of_gc_query_from_loadunit, loadUnit.ptr, loadUnit.size, offsets, dictOffsets);
            demoCalculateGCsOnCuda(loadUnit.getNumberOfGc(),
                                   loadUnit.getNumberOfDictElements(),
                                   loadUnit.getGcMatrixDataPtr(),
                                   loadUnit.getGcDictDataPtr(),
                                   loadUnit.getGcMatrixOffsetsPtr(),
                                   loadUnit.getDictOffsetPtr(),
                                   loadUnit.getMatrixSizesPtr()
                                   /* , */
                                   );
            return 0;
        }
    }

    return 1;
}

bool QueryHandler::validate(std::string query) {
    if (query.empty() || query.compare("") == 0) {
        return false;
    }
    if (!regex_match(query, std::regex("(Query by Example: )(.*)$")))
        return false;
    return true;
}
