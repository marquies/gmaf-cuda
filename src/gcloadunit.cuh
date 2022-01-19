//
// Created by breucking on 19.01.22.
//

#ifndef GCSIM_GCLOADUNIT_CUH
#define GCSIM_GCLOADUNIT_CUH


//typedef struct GraphCodeBlock {
//    int *ptr;
//    int size;
//} GraphCodeBlock;

class GcLoadUnit {

public:
    void loadArtificialGcs(int count, int dimension);

    int *getGcPtr();

    int getSize();

private:
    int *gcPtr;

    int gcSize;

    bool init = false;
};


#endif //GCSIM_GCLOADUNIT_CUH
