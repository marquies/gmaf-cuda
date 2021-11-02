cmake_minimum_required(VERSION 3.20)
enable_language(CUDA)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules")


project(test)

include(FetchContent)

FetchContent_Declare(json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.7.3)

FetchContent_GetProperties(json)
if(NOT json_POPULATED)
    FetchContent_Populate(json)
    add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()


find_package(CUDA 8.1 REQUIRED)
#find_package(Thrust REQUIRED CONFIG)
find_package(Thrust)

SET(GCC_COVERAGE_LINK_FLAGS    "-lstdc++fs")

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})




set(CMAKE_CXX_STANDARD 17)
#add_executable(test main.cpp main.cu )
add_executable(test  main.cu )

set_property(TARGET test PROPERTY CUDA_STANDARD 17)


target_link_libraries(test PRIVATE nlohmann_json::nlohmann_json)
#target_link_libraries(test PUBLIC std::__fs)
target_link_libraries(test PUBLIC stdc++fs)
#target_link_libraries(test PRIVATE Thrust)
target_link_libraries(test PUBLIC ${CUDA_LIBRARIES})

