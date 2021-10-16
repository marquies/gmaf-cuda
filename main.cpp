#include <nlohmann/json.hpp>
#include<string.h>
// for convenience
using json = nlohmann::json;

#include <iostream>
#include <fstream>


//__global__ void kernel( void ) {}

int main() {

    std::ifstream ifs("../GMAF_TMP_17316548361524909203.png.gc");
    json jf = json::parse(ifs);


   // kernel<<<1,1>>>();
    std::cout << jf["dictionary"] << std::endl;
    return 0;
}
