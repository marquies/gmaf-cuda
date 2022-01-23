//
// Created by breucking on 11.11.21.
//

#include "helper.h"

bool G_DEBUG = true;

void convertDict2Matrix(int size, int *destMatrix, nlohmann::json jsonMatrix) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {

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
*/bool isPrime(int number) {
    bool isPrime = true;

    // 0 and 1 are not prime numbers
    if (number == 0 || number == 1) {
        isPrime = false;
    }
    else {
        for (int i = 2; i <= number / 2; ++i) {
            if (number % i == 0) {
                isPrime = false;
                break;
            }
        }

    }
    return isPrime;
}

int findLargestDivisor(int n) {

    int i;
    for (i = n / 2; i >= 1; --i) {
        // if i divides n, then i is the largest divisor of n
        // return i
        if (n % i == 0)
            return i;
    }

    return 0;

    for (int i=sqrt(n); i < n; i++)
    {
        if (n%i == 0)
            return i;
    }
    return 0;
}
