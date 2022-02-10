//
// Created by breucking on 19.01.22.
//



#include <queryhandler.cuh>
#include <cassert>
#include <gcloadunit.cuh>

void testErrorQuery();

void testValidation();

void testSimpleQuery();

void testSelectionSort();

void testIntroSort();

void testHeapSort();

int main() {

    testErrorQuery();
    testValidation();
    testSimpleQuery();
    testSelectionSort();
    testHeapSort();
    testIntroSort();

}

// Function to swap the the position of two elements
void swap(float *a, float *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void heapify(float *arr, int n, int i) {
    // Find largest among root, left child and right child
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < n && arr[left] > arr[largest])
        largest = left;

    if (right < n && arr[right] > arr[largest])
        largest = right;

    // Swap and continue heapifying if root is not largest
    if (largest != i) {
        swap(&arr[i], &arr[largest]);
        heapify(arr, n, largest);
    }
}

// Main function to do heap sort
void heapSort(float *arr, int n) {
    // Build max heap
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);

    // Heap sort
    for (int i = n - 1; i >= 0; i--) {
        swap(&arr[0], &arr[i]);

        // Heapify root element to get highest element at root again
        heapify(arr, i, 0);
    }
}

// Print an array
void printArray(float *arr, int n) {
    for (int i = 0; i < n; ++i)
        printf("%f ", arr[i]);
    printf("\n");
}

void testHeapSort() {
    int n = 100000;
//    srand(time(NULL));
//    float arr[n];// = arr[n];
//    for (int i = 0; i <n; i++) {
//        //arr[i] = rand();// Initialization, should only be called once.
//        arr[i] = (n-i) / (float) n;
//    }
//
////    int arr[] = {1, 12, 9, 5, 6, 10};
//    int c = sizeof(arr) / sizeof(arr[0]);
//
//    heapSort(arr, c);
//
//    printf("Sorted array is \n");
//    printArray(arr, c);
//
//    for (int i = 0; i < n; i++) {
//        assert(arr[i] == (i+1) / (float) n);
//    }
//
//    return;

    Metrics *metrics = (Metrics *) malloc(sizeof(Metrics) * n);

    for (int i = 0; i < n; i++) {
        float f_n = (float) n;
        float f_i = (float) i;
        Metrics m;
        m.idx = n - i;
        m.similarity = (f_n - f_i) / f_n;
        m.recommendation = (f_n - f_i) / f_n;
        m.inferencing = (f_n - f_i) / f_n;

        metrics[n - i - 1] = m;
    }


    HeapSort(metrics, n);
    std::cout << std::endl;

//    for (int i = 49993; i >= 49980; i--) {
//        float f_n = (float) n;
//        float f_i = (float) i;
//        std::cout << "i: " << i << ": " << metrics[i].idx << " " << metrics[i].similarity << " == " << (f_n - f_i) / f_n << std::endl;
//
//    }

    for (int i = n - 1; i >= 0; i--) {
        float f_n = (float) n;
        float f_i = (float) i;
        assert(metrics[i].similarity == (f_n - f_i) / f_n);
    }
}

void testIntroSort() {
    int n = 100000;
    Metrics *metrics = (Metrics *) malloc(sizeof(Metrics) * n);

    for (int i = 0; i < n; i++) {
        Metrics m;
        m.idx = i;
        m.similarity = (float) i / 1000.f;
        m.recommendation = (float) i / 1000.f;
        m.inferencing = (float) i / 1000.f;

        metrics[i] = m;
    }


    Introsort(metrics, 0, n);
    std::cout << std::endl;
    for (int i = 0; i < n; i++) {
        assert(metrics[i].similarity == (float) (n - i - 1) / 1000.f);
    }


    free(metrics);
}

void testSelectionSort() {

    Metrics m1;
    m1.idx = 1;
    m1.similarity = 1;
    m1.recommendation = 1;
    m1.inferencing = 1;

    Metrics m2;
    m2.idx = 2;
    m2.similarity = 2;
    m2.recommendation = 2;
    m2.inferencing = 2;

    Metrics m3;
    m3.idx = 3;
    m3.similarity = 3;
    m3.recommendation = 3;
    m3.inferencing = 3;


    Metrics *metrics = (Metrics *) malloc(sizeof(Metrics) * 3);
    metrics[0] = m1;
    metrics[1] = m2;
    metrics[2] = m3;

    selectionSort(metrics, 3);

    assert(metrics[0].idx == 3);
    assert(metrics[1].idx == 2);
    assert(metrics[2].idx == 1);

    free(metrics);

}

void testSimpleQuery() {
    GcLoadUnit *loadUnit = new GcLoadUnit(GcLoadUnit::MODE_MEMORY_MAP);
    QueryHandler queryHandler;
    loadUnit->loadArtificialGcs(10, 1);
    int value = queryHandler.processQuery("Query by Example: 6.gc", loadUnit);
    assert(value == 0);
}

void testValidation() {
    bool valid = QueryHandler::validate("");
    assert(valid == false);
    valid = QueryHandler::validate("Query by Example: xoxo.png");
    assert(valid);
}

void testErrorQuery() {


    try {
        QueryHandler queryHandler;
        queryHandler.processQuery("", new GcLoadUnit(GcLoadUnit::MODE_MEMORY_MAP));
        assert(false);
    } catch (std::invalid_argument) {

    }

}

