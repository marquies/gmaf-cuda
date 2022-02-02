//
// Created by Patrick Steinert on 16.10.21.
//

#ifndef GRAPHCODE_H
#define GRAPHCODE_H

#include <nlohmann/json.hpp>
#include <vector>


typedef struct GraphCode {
    std::vector<std::string> *dict;
    unsigned short *matrix;
} GraphCode;


using json = nlohmann::json;

struct Metrics {
    int idx;
    float similarity;
    float recommendation;
    float inferencing;

    inline bool operator>(Metrics a) {
        exit(EXIT_FAILURE);

        return idx > a.idx;
    }

    inline bool operator<(Metrics a) {
        exit(EXIT_FAILURE);

        return idx < a.idx;
    }
//
//    inline Metrics operator/(int a) {
//        exit(EXIT_FAILURE);
//
//        //return idx < a.idx;
//    }
//
//    inline Metrics operator+(Metrics a) {
//        exit(EXIT_FAILURE);
//
//        //return idx < a.idx;
//    }
//
//    inline Metrics operator+(int a) {
//        exit(EXIT_FAILURE);
//
//        //return idx < a.idx;
//    }
//
//    inline bool operator>=(Metrics a) {
//        exit(EXIT_FAILURE);
//
//        //return idx < a.idx;
//    }
//
//    inline bool operator<=(Metrics a) {
//        exit(EXIT_FAILURE);
//
//        //return idx < a.idx;
//    }
//
////    inline Metrics operator=(volatile Metrics a) {
////        exit(EXIT_FAILURE);
////    }
//    inline Metrics &operator=(const Metrics &a) {
//        exit(EXIT_FAILURE);
//    }
//
//    inline Metrics *operator=(Metrics *a) volatile {
//        exit(EXIT_FAILURE);
//    }
//
//    inline Metrics &operator=(Metrics a) volatile {
//        exit(EXIT_FAILURE);
//    }
//
//    inline void operator=(int a) volatile {
//        exit(EXIT_FAILURE);
//    }
//
//    inline void operator=(int a) {
//        exit(EXIT_FAILURE);
//    }
//
//    inline Metrics(Metrics *a) {
//        exit(EXIT_FAILURE);
//    }
//
//    inline Metrics(const Metrics &a) {
//        exit(EXIT_FAILURE);
//    }
//
////    inline Metrics(volatile Metrics &a) {
////        exit(EXIT_FAILURE);
////    }
//
//    inline Metrics(int a) {
//        exit(EXIT_FAILURE);
//    }
//
//    inline Metrics() {
//        exit(EXIT_FAILURE);
//    }
//
//    inline bool operator!=(Metrics a) {
//        exit(EXIT_FAILURE);
//    }
//
//    inline bool min(Metrics a) {
//        exit(EXIT_FAILURE);
//
//        return idx < a.idx;
//    }

};

//Metrics *min(Metrics *a, Metrics *b) {
//    return a;
//}
//
//Metrics *max(Metrics *a, Metrics *b) {
//    return a;
//}
//
//Metrics min(Metrics a, Metrics b) {
//    return a;
//}
//
//Metrics max(Metrics a, Metrics b) {
//    return a;
//}

//void foo() {
//    Metrics* a;
//    a[0] = max( a, a);
//}


namespace gmaf {

    class GraphCode {
    public:
        void loadGraphCodes(char *directory, int limit, std::vector<json> *arr);

    };
}
#endif //GRAPHCODE_H
