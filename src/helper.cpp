//
// Created by breucking on 11.11.21.
//

#include <iostream>
#include "helper.h"
#include "graphcode.h"

bool G_DEBUG = false;
bool G_BENCHMARK = false;

void convertDict2Matrix(int size, int *destMatrix, nlohmann::json jsonMatrix) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {

            *((destMatrix+i*size) + j) = jsonMatrix.at(i).at(j);
        }
    }

}


// swap values at memory locations to which
// x and y point
void swap(Metrics *const x, Metrics *const y) {
    Metrics temp = *x;
    *x = *y;
    *y = temp;
}

void selectionSort(Metrics *const array, const int size) {
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

                std::cout << "b-a" << b - a << std::endl;
            }

            if (b - a < 0) {
                smallest = index;
            }
        }


        swap(&array[i], &array[smallest]);
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
