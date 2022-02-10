//
// Created by breucking on 11.11.21.
//

#include <iostream>
#include "helper.h"
#include "graphcode.h"
#include <iomanip>

bool G_DEBUG = false;
bool G_BENCHMARK = false;
short G_BENCHMARK_REPEAT = 10;

void convertDict2Matrix(int size, int *destMatrix, nlohmann::json jsonMatrix) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {

            *((destMatrix + i * size) + j) = jsonMatrix.at(i).at(j);
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

float compare(Metrics *const array, int positionA, int positionB) {
    //if ( array[ index ].similarity < array[ smallest ].similarity )
    float a = array[positionA].similarity * 100000.0f + array[positionA].recommendation * 100.0f +
              array[positionA].inferencing;


    float b = array[positionB].similarity * 100000.0f + array[positionB].recommendation * 100.0f +
              array[positionB].inferencing;

    if (G_DEBUG) {
        std::cout << "a: " << array[positionA].similarity << ";" << array[positionA].recommendation << ";"
                  << array[positionA].inferencing << std::endl;
        std::cout << a << std::endl;
        std::cout << "b: " << array[positionB].similarity << ";" << array[positionB].recommendation << ";"
                  << array[positionB].inferencing << std::endl;
        std::cout << b << std::endl;

        std::cout << "b-a" << b - a << std::endl;
    }
    return b - a;
}

float compare(Metrics *positionA, Metrics *positionB) {
    //if ( array[ index ].similarity < array[ smallest ].similarity )
    float a = positionA->similarity * 100000.0f + positionA->recommendation * 100.0f +
              positionA->inferencing;


    float b = positionB->similarity * 100000.0f + positionB->recommendation * 100.0f +
              positionB->inferencing;

    if (G_DEBUG) {
        std::cout << std::fixed << std::setprecision(10) << "a: " << positionA->similarity << ";" << positionA->recommendation << ";"
                  << positionA->inferencing << std::endl;
        std::cout << a << std::endl;
        std::cout << "b: " << positionB->similarity << ";" << positionB->recommendation << ";"
                  << positionB->inferencing << std::endl;
        std::cout << b << std::endl;

        std::cout << "a-b=" << a - b << std::endl;
    }
    return a - b;
}


void selectionSort(Metrics *const array, const int size) {
    int smallest; // index of smallest element

    for (int i = 0; i < size - 1; i++) {
        smallest = i; // first index of remaining array

        // loop to find index of smallest element

        for (int index = i + 1; index < size; index++) {
            if (compare(array, index, smallest) < 0) {
                smallest = index;
            }
        }
        swap(&array[i], &array[smallest]);
    }
}

//
///* Function to sort an array using insertion sort*/
//void InsertionSort(Metrics *arr, int begin, int end) {
//    // Get the left and the right index of the subarray
//    // to be sorted
//    int left = b    egin - arr;
//    int right = end - arr;
//
//    for (int i = left + 1; i <= right; i++) {
//        Metrics key = arr[i];
//        int j = i - 1;
//
//        /* Move elements of arr[0..i-1], that are
//           greater than key, to one position ahead
//           of their current position */
//        while (j >= left && compare(arr, j, i) > 0) {
//            arr[j + 1] = arr[j];
//            j = j - 1;
//        }
//        arr[j + 1] = key;
//    }
//
//    return;
//}

void InsertionSort(Metrics *data, int count) {
    for (int i = 1; i < count; i++) {
        Metrics key = data[i];
        int j = i - 1;

        while (j >= 0 && compare(&data[j], &key) < 0) {
            data[j + 1] = data[j];
            j = j - 1;
        }
        data[j + 1] = key;
    }
}


//// A function to partition the array and return
//// the partition point
//Metrics *Partition(Metrics *arr, int low, int high) {
//    Metrics pivot = arr[high];    // pivot
//    int i = (low - 1);  // Index of smaller element
//
//    for (int j = low; j <= high - 1; j++) {
//        // If current element is smaller than or
//        // equal to pivot
////        if (arr[j] <= pivot) {
//        if (compare(&arr[j], &pivot) <= 0) {
//
//            // increment index of smaller element
//            i++;
//
//            swap(&arr[i], &arr[j]);
//        }
//    }
//    swap(&arr[i + 1], &arr[high]);
//    return (arr + i + 1);
//}


int Partition(Metrics *data, int left, int right) {
    Metrics pivot = data[right];
    Metrics temp;
    int i = left;

    for (int j = left; j < right; ++j) {
//        if (data[j] <= pivot)
        if (compare(&data[j], &pivot) <= 0) {
            temp = data[j];
            data[j] = data[i];
            data[i] = temp;
            i++;
        }
    }

    data[right] = data[i];
    data[i] = pivot;

    return i;
}

void MaxHeapify(Metrics *data, int heapSize, int index) {
//    int left = (index + 1) * 2 - 1;
//    int right = (index + 1) * 2;
    int left = 2 * index + 1;
    int right = 2 * index + 2;
    int smallest = index;

//    if (left < heapSize && data[left] > data[index])
    if (left < heapSize && compare(&data[left], &data[smallest]) > 0)
        smallest = left;
    if (right < heapSize && compare(&data[right], &data[smallest]) > 0)
        smallest = right;

    if (smallest != index) {
        swap(&data[index], &data[smallest]);
        MaxHeapify(data, heapSize, smallest);
    }
}


void MinHeapify(Metrics *data, int heapSize, int index) {
//    int left = (index + 1) * 2 - 1;
//    int right = (index + 1) * 2;
    int left = 2 * index + 1;
    int right = 2 * index + 2;
    int smallest = index;

//    if (left < heapSize && data[left] > data[index])
    if (left < heapSize && compare(&data[left], &data[index]) < 0)
        smallest = left;
    if (right < heapSize && compare(&data[right], &data[smallest]) < 0)
        smallest = right;

    if (smallest != index) {
        swap(&data[index], &data[smallest]);
        MinHeapify(data, heapSize, smallest);
    }
}

void HeapSort(Metrics *data, int size) {

    for (int p = (size) / 2-1 ; p >= 0; p--)
        MinHeapify(data, size, p);

    for (int i = size - 1; i >= 0; i--) {
        swap(&data[0], &data[i]);
        MinHeapify(data, i, 0);
    }
}

// A function that find the middle of the
// values pointed by the pointers a, b, c
// and return that pointer
//int *MedianOfThree(int a, int b, int *c) {
//    if (*a < *b && *b < *c)
////    if (compare(a, b) < 0 && compare(b, c) < 0)
//        return (b);
//
//    if (*a < *c && *c <= *b)
////    if (compare(a, c) < 0 && compare(c, b) <= 0)
//        return (c);
//
//    if (*b <= *a && *a < *c)
////    if (compare(b, a) <= 0 && compare(a, c) < 0)
//        return (a);
//
//    if (*b < *c && *c <= *a)
////    if (compare(b, c) < 0 && compare(c, a) <= 0)
//        return (c);
//
//    if (*c <= *a && *a < *b)
////    if (compare(c, a) <= 0 && compare(a, b) < 0)
//        return (a);
//
//    if (*c <= *b && *b <= *a)
////    if (compare(c, b) < 0 && compare(b, a) <= 0)
//        return (b);
//}

void QuickSortRecursive(Metrics *data, int left, int right) {
    if (left < right) {
        int q = Partition(data, left, right);
        QuickSortRecursive(data, left, q - 1);
        QuickSortRecursive(data, q + 1, right);
    }
}

// A Utility function to perform intro sort
void IntrosortUtil(Metrics *arr, int begin,
                   int end, int depthLimit) {
    // Count the number of elements

    int size = end - begin;
    int partitionSize = Partition(arr, 0, size - 1);
    // If partition size is low then do insertion sort
    if (partitionSize < 16) {
        InsertionSort(arr, size);
    }

        // If the depth is zero use heapsort
    else if (partitionSize > (2 * log(size))) {
        HeapSort(arr, size);
    } else {
        QuickSortRecursive(arr, 0, size - 1);
    }
}

/* Implementation of introsort*/
void Introsort(Metrics *array, int begin, int end) {
    int depthLimit = 2 * log(end - begin);

    // Perform a recursive Introsort
    IntrosortUtil(array, begin, end, depthLimit);

    return;
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
    } else {
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
        if (n % i == 0)
            return i;
    }

    return 0;

}
