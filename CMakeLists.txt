cmake_minimum_required(VERSION 3.20)
#SET(CUDA_SEPARABLE_COMPILATION ON)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)

set(CUDA_ALL_GPU_ARCHITECTURES "3.5" "5.0" "6.0" "7.0" "7.2" "7.5" "8.0")


#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

include(FindPkgConfig)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules")


project(gcsim)


file (GLOB_RECURSE Foo_SOURCES CONFIGURE_DEPENDS "src/*.cu")
file (GLOB_RECURSE Foo_HEADERS CONFIGURE_DEPENDS "src/*.h")

set (Foo_INCLUDE_DIRS "")
foreach (_headerFile ${Foo_HEADERS})
    get_filename_component(_dir ${_headerFile} PATH)
    list (APPEND Foo_INCLUDE_DIRS ${_dir})
endforeach()
list (REMOVE_DUPLICATES Foo_INCLUDE_DIRS)


include(FetchContent)
set(JSON_BuildTests OFF)
FetchContent_Declare(json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.2.0)

FetchContent_GetProperties(json)
if(NOT json_POPULATED)
    FetchContent_Populate(json)
    add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
#
#set(THRUST_INCLUDE_DIR "${thrust_SOURCE_DIR}")
#FetchContent_Declare(thrust
#        GIT_REPOSITORY https://github.com/NVIDIA/thrust.git
#        GIT_TAG 1.15.0)
#
#FetchContent_GetProperties(thrust)
#if(NOT thrust_POPULATED)
#    FetchContent_Populate(thrust)
#    add_subdirectory(${thrust_SOURCE_DIR} ${thrust_BINARY_DIR} EXCLUDE_FROM_ALL)
#endif()
#
#
#FetchContent_Declare(cub
#        GIT_REPOSITORY https://github.com/NVIDIA/cub.git
#        GIT_TAG 1.15.0)
#
#FetchContent_GetProperties(cub)
#if(NOT cub_POPULATED)
#    FetchContent_Populate(cub)
#    add_subdirectory(${cub_SOURCE_DIR} ${cub_BINARY_DIR} EXCLUDE_FROM_ALL)
#endif()

#find_package(CUDA 10.1 REQUIRED)

#find_package(CUB 1.5.0 REQUIRED)
#include_directories(${CUB_INCLUDE_DIR})
#
#message(STATUS "CUB_FOUND=${CUB_FOUND}")

find_package(PkgConfig REQUIRED)

pkg_search_module(UUID REQUIRED uuid)

#find_path(LIBUUID_INCLUDE_DIR uuid.h PATH_SUFFIXES uuid)
#find_library(LIBUUID_LIBRARY libuuid.a)
#include_directories(${LIBUUID_INCLUDE_DIR})

set( CMAKE_VERBOSE_MAKEFILE on )
find_package(CUDA 10 REQUIRED)
#find_package(Thrust REQUIRED CONFIG)
#find_package(Thrust)
SET(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_VERBOSE_BUILD ON)
CUDA_SELECT_NVCC_ARCH_FLAGS("3.5")
#include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
#SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
SET(CMAKE_CUDA_ARCHITECTURES 35)

#set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")

SET(GCC_COVERAGE_LINK_FLAGS    "-lstdc++fs -mavx2 -msse4.1 -march=native")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=nehalem")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1 -msse4.2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wall -Wextra -Wconversion -pedantic ")

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O1")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1")

#SET(GCC_COVERAGE_COMPILE_FLAGS "-mavx2 ")
#SET(GCC_COVERAGE_LINK_FLAGS    "-lgcov")

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})


set(CMAKE_CXX_STANDARD 17)
set (CUDA_PROPAGATE_HOST_FLAGS OFF)

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_61,code=sm_61 --generate-code=arch=compute_61,code=[compute_61,sm_61]")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --cudart shared -lcudadevrt -rdc=true")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -lineinfo -G -g ")




#include_directories(src)
#link_directories(src)

#add_executable(gcsim ${Foo_SOURCES})
add_library (GCS SHARED ${Foo_SOURCES} src/cudahelper.cu src/cudahelper.cuh src/helper.h src/helper.cpp src/queryhandler.cuh src/queryhandler.cu src/cpualgorithms.cpp src/cpualgorithms.h src/algorithmstrategies.cpp src/algorithmstrategies.h)
set_target_properties(GCS PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
add_executable (gcsim main2.cpp)

set_target_properties(gcsim PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(gcsim PUBLIC ${Foo_INCLUDE_DIRS} ${UUID_INCLUDE_DIRS})

target_sources(gcsim PRIVATE ${GMAF_SRC})

#set_property(TARGET gcsim PROPERTY CUDA_ARCHITECTURES 35 50 61 72)
#set_property(TARGET GCS PROPERTY CUDA_ARCHITECTURES 35 50 61 72)

ENABLE_TESTING()

ADD_SUBDIRECTORY( test )
cuda_select_nvcc_arch_flags(ARCH_FLAGS)

set_property(TARGET gcsim PROPERTY CUDA_STANDARD 17)


target_link_libraries(GCS PRIVATE nlohmann_json::nlohmann_json)
#target_link_libraries(GCS PRIVATE cub)


#target_compile_options(gcsim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>>)

target_link_libraries(gcsim PRIVATE GCS)
#target_link_libraries(gcsim PRIVATE avx2)
target_link_libraries(gcsim PRIVATE nlohmann_json::nlohmann_json)
#target_link_libraries(gcsim PRIVATE Thrust)
#target_link_libraries(gcsim PRIVATE cub)
#target_link_libraries(test PUBLIC std::__fs)
target_link_libraries(gcsim PUBLIC stdc++fs)
#target_link_libraries(test PRIVATE Thrust)
target_link_libraries(gcsim PUBLIC ${CUDA_LIBRARIES})
target_link_libraries(gcsim PUBLIC ${UUID_LIBRARIES})


#target_link_directories(test PUBLIC src)
