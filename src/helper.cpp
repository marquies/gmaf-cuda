//
// Created by breucking on 11.11.21.
//

#include "helper.h"

void convertDict2Matrix(int size, int *destMatrix, nlohmann::json jsonMatrix) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {

            //destMatrix[i][j] = jsonMatrix.at(i).at(j);
            *((destMatrix+i*size) + j) = jsonMatrix.at(i).at(j);
        }
    }
}

/*
void convertDict2Matrix(int size, int *destMatrix, nlohmann::json jsonMatrix) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {

            //destMatrix[i][j] = jsonMatrix.at(i).at(j);
            destMatrix[i*size + j] = 1;
                 //   (int*) (jsonMatrix.at(i).at(j).get<int>());
        }
    }
}
*/