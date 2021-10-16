//
// Created by Patrick Steinert on 16.10.21.
//

#include "graphcode.h"

//0. Ein File haben wir (gcQuery), dazu die similarity berehcnen
//
//1. Für jedes Element in der Collection:
//1.a Einlesen der Files in einem Directory (for each file)
//1.b Für jedes File similarity berechnen
//   2. Für jedes Element im Dictionary (Annotation) der gcQuery
//   2. a-> für jedes element in der Matrix
//        wenn der wert in QCquery != 0
//            num_of_non_zero_edges = 1
//        -> check anderes GC element:
//            wenn es die Begriffe im Dict nicht gibt skip
//            else
//                -> wenn matrix wert != 0 dann edge_metric_count = 1
//                -> wenn matrix wert == gcQuery Wert (beziehungstyp identisch) edge_type =
//  2. b -> Kalkulation der
//1.c Sortieren