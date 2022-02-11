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

void testPosition();

int main() {

    testErrorQuery();
    testValidation();
    testSimpleQuery();
    testSelectionSort();
    testHeapSort();
    testIntroSort();
    testPosition();

}

//TODO: move me to gcloadUnit tests
void testPosition() {
    GcLoadUnit gc = GcLoadUnit(GcLoadUnit::MODE_VECTOR_MAP);
    gc.loadArtificialGcs(10, 5);
    assert(gc.hasGc("0.gc") == true);
    assert(gc.getGcPosition("0.gc") == 0);
    assert(gc.hasGc("1.gc") == true);
    assert(gc.getGcPosition("1.gc") == 1);
    assert(gc.hasGc("2.gc") == true);
    assert(gc.getGcPosition("2.gc") == 2);

    gc = GcLoadUnit(GcLoadUnit::MODE_MEMORY_MAP);
    std::string file = "/home/breucking/CLionProjects/gmaf-cuda/GMAF_TMP_17316548361524909203.png.gc";
    gc.addGcFromFile(file);
    file = "/home/breucking/CLionProjects/gmaf-cuda/example2.gc";
    gc.addGcFromFile(file);

    assert(gc.hasGc("example2.gc") == true);
    assert(gc.getGcPosition("example2.gc") == 1);
    assert(gc.hasGc("GMAF_TMP_17316548361524909203.png.gc") == true);
    assert(gc.getGcPosition("GMAF_TMP_17316548361524909203.png.gc") == 0);

}


// Print an array
void printArray(float *arr, int n) {
    for (int i = 0; i < n; ++i)
        printf("%f ", arr[i]);
    printf("\n");
}

void testHeapSort() {
    int n = 100000;


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

