//
// Created by breucking on 19.01.22.
//

#include <regex>
#include "queryhandler.cuh"
#include "gcloadunit.cuh"
#include "helper.h"
#include <algorithm>


// swap values at memory locations to which
// x and y point
void swap(Metrics *const x, Metrics *const y) {
    Metrics temp = *x;
    *x = *y;
    *y = temp;
}

void QueryHandler::selectionSort(Metrics *const array, const int size) {
    int smallest; // index of smallest element

    for (int i = 0; i < size - 1; i++) {
        smallest = i; // first index of remaining array

        // loop to find index of smallest element

        for (int index = i + 1; index < size; index++) {


            //if ( array[ index ].similarity < array[ smallest ].similarity )
            float a = array[index].similarity * 100000.0f + array[index].recommendation * 100.0f +
                      array[index].inferencing;


            float b = array[smallest].similarity * 100000.0f + array[smallest].recommendation * 100.0f +
                      array[smallest].inferencing;

            if (G_DEBUG) {
                std::cout << "a: " << array[index].similarity << ";" << array[index].recommendation << ";"
                          << array[index].inferencing << std::endl;
                std::cout << a << std::endl;
                std::cout << "b: " << array[smallest].similarity << ";" << array[smallest].recommendation << ";"
                          << array[smallest].inferencing << std::endl;
                std::cout << b << std::endl;

                std::cout << "b-a" << b-a << std::endl;
            }

            if (b - a < 0) {
                smallest = index;
            }
        }


        swap(&array[i], &array[smallest]);
    }
}

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
        if (loadUnit.hasGc(gcQueryName)) {
            // call algorithm in form of

            //result = calculateMetrics(get_position_of_gc_query_from_loadunit, loadUnit.ptr, loadUnit.size, offsets, dictOffsets);
            Metrics *result = demoCalculateGCsOnCuda(loadUnit.getNumberOfGc(),
                                                     loadUnit.getNumberOfDictElements(),
                                                     loadUnit.getGcMatrixDataPtr(),
                                                     loadUnit.getGcDictDataPtr(),
                                                     loadUnit.getGcMatrixOffsetsPtr(),
                                                     loadUnit.getDictOffsetPtr(),
                                                     loadUnit.getMatrixSizesPtr()
                    /* , */
            );

            //std::sort(std::begin(result), std::end(result));
//            for(int i = 0; i < loadUnit.getNumberOfGc(); i++) {
//                while(result[result[i - 1] != result[i])
//                    std::swap(result[result[i] - 1], result[i]);
//            }
            selectionSort(result, loadUnit.getNumberOfGc());
            for (int i = 0; i < loadUnit.getNumberOfGc(); i++) {
                std::cout << "-----------------------------------" << std::endl;
                std::cout << result[i].idx << std::endl;
                std::cout << result[i].similarity << std::endl;
                std::cout << result[i].recommendation << std::endl;
                std::cout << result[i].inferencing << std::endl;
            }

            free(result);


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
