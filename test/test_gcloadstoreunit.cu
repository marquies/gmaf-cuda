//
// Created by breucking on 19.01.22.
//


#include <gcloadunit.cuh>

void testLoadSimple();

int main() {
    testLoadSimple();
}

void testLoadSimple() {
    GcLoadUnit loadUnit;
    loadUnit.loadArtificialGcs(10,1);

}
