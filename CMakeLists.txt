cmake_minimum_required(VERSION 3.20)
enable_language(CUDA)


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


find_package(PkgConfig REQUIRED)

pkg_search_module(UUID REQUIRED uuid)

#find_path(LIBUUID_INCLUDE_DIR uuid.h PATH_SUFFIXES uuid)
#find_library(LIBUUID_LIBRARY libuuid.a)
#include_directories(${LIBUUID_INCLUDE_DIR})

find_package(CUDA 8.1 REQUIRED)
#find_package(Thrust REQUIRED CONFIG)
#find_package(Thrust)

SET(GCC_COVERAGE_LINK_FLAGS    "-lstdc++fs -mavx2 -msse4.1 -march=native")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=nehalem")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1 -msse4.2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O1")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1")

#SET(GCC_COVERAGE_COMPILE_FLAGS "-mavx2 ")
#SET(GCC_COVERAGE_LINK_FLAGS    "-lgcov")

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})

set(CMAKE_CXX_STANDARD 17)
#set (CUDA_PROPAGATE_HOST_FLAGS ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_61,code=sm_61")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -G -g")



#set_target_properties(test PROPERTIES CUDA_ARCHITECTURES "35;50;72")


#include_directories(src)
#link_directories(src)

#add_executable(gcsim ${Foo_SOURCES})
add_library (GCS ${Foo_SOURCES} src/cudahelper.cu src/cudahelper.cuh src/helper.h src/helper.cpp src/queryhandler.cuh src/queryhandler.cu)

add_executable (gcsim main2.cpp)

target_include_directories(gcsim PUBLIC ${Foo_INCLUDE_DIRS} ${UUID_INCLUDE_DIRS})

target_sources(gcsim PRIVATE ${GMAF_SRC})


ENABLE_TESTING()

ADD_SUBDIRECTORY( test )


set_property(TARGET gcsim PROPERTY CUDA_STANDARD 17)


target_link_libraries(GCS PRIVATE nlohmann_json::nlohmann_json)

target_link_libraries(gcsim PRIVATE GCS)
#target_link_libraries(gcsim PRIVATE avx2)
target_link_libraries(gcsim PRIVATE nlohmann_json::nlohmann_json)
#target_link_libraries(test PUBLIC std::__fs)
target_link_libraries(gcsim PUBLIC stdc++fs)
#target_link_libraries(test PRIVATE Thrust)
target_link_libraries(gcsim PUBLIC ${CUDA_LIBRARIES})
target_link_libraries(gcsim PUBLIC ${UUID_LIBRARIES})


#target_link_directories(test PUBLIC src)
