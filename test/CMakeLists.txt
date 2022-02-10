#CUDA_SELECT_NVCC_ARCH_FLAGS("6.1")

ADD_EXECUTABLE(test_simpletest test_simpletest.cu ${Foo_SOURCES} testhelper.h  testhelper.cpp)
ADD_EXECUTABLE(test_cudaNxNmatrices test_cudaNxNmatrices.cu ${Foo_SOURCES}  testhelper.h testhelper.cpp)
ADD_EXECUTABLE(test_speedNxNmatrices test_speedNxNmatrices.cu ${Foo_SOURCES} testhelper.h testhelper.cpp)
ADD_EXECUTABLE(test_cpuNxNmatrices test_cpuNxNmatrix.cu ${Foo_SOURCES}  testhelper.h  testhelper.cpp)
ADD_EXECUTABLE(test_cudaGCKernel test_cudaGCKernel.cu ${Foo_SOURCES}  testhelper.h testhelper.cpp)
ADD_EXECUTABLE(test_gcloadstoreunit test_gcloadstoreunit.cu ${Foo_SOURCES} testhelper.h testhelper.cpp)
ADD_EXECUTABLE(test_queryhandler test_queryhandler.cpp ${Foo_SOURCES} testhelper.h testhelper.cpp)
ADD_EXECUTABLE(test_cpuparallelgckernel test_cpuparallelgckernel.cu ${Foo_SOURCES} testhelper.h testhelper.cpp)
ADD_EXECUTABLE(test_cudasort test_cudasort.cu ${Foo_SOURCES} testhelper.h testhelper.cpp)


ADD_TEST(test_simpletest test_simpletest)
ADD_TEST(test_cudaNxNmatrices test_cudaNxNmatrices)
ADD_TEST(test_speedNxNmatrices test_speedNxNmatrices)
ADD_TEST(test_cpuNxNmatrices test_cpuNxNmatrices)
ADD_TEST(test_cudaGCKernel test_cudaGCKernel)
ADD_TEST(test_gcloadstoreunit test_gcloadstoreunit)
ADD_TEST(test_queryhandler test_queryhandler)
ADD_TEST(test_cpuparallelgckernel test_cpuparallelgckernel)
ADD_TEST(test_cudasort test_cudasort)

target_include_directories(test_simpletest PUBLIC ${Foo_INCLUDE_DIRS})
target_include_directories(test_cudaNxNmatrices PUBLIC ${Foo_INCLUDE_DIRS})
target_include_directories(test_cpuNxNmatrices PUBLIC ${Foo_INCLUDE_DIRS})
target_include_directories(test_speedNxNmatrices PUBLIC ${Foo_INCLUDE_DIRS})
target_include_directories(test_cudaGCKernel PUBLIC ${Foo_INCLUDE_DIRS})
target_include_directories(test_gcloadstoreunit PUBLIC ${Foo_INCLUDE_DIRS})
target_include_directories(test_queryhandler PUBLIC ${Foo_INCLUDE_DIRS})
target_include_directories(test_cpuparallelgckernel PUBLIC ${Foo_INCLUDE_DIRS})
target_include_directories(test_cudasort PUBLIC ${Foo_INCLUDE_DIRS})

set_target_properties(test_simpletest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(test_cudaNxNmatrices PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(test_cpuNxNmatrices PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(test_speedNxNmatrices PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(test_cudaGCKernel PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(test_gcloadstoreunit PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(test_queryhandler PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(test_cpuparallelgckernel PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(test_cudasort PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)





target_link_libraries(test_simpletest PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(test_simpletest PRIVATE GCS)
target_link_libraries(test_simpletest PUBLIC stdc++fs)
target_link_libraries(test_simpletest PUBLIC ${UUID_LIBRARIES})


target_link_libraries(test_cudaNxNmatrices PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(test_cudaNxNmatrices PRIVATE GCS)
target_link_libraries(test_cudaNxNmatrices PUBLIC stdc++fs)
target_link_libraries(test_cudaNxNmatrices PUBLIC ${UUID_LIBRARIES})


target_link_libraries(test_cpuNxNmatrices PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(test_cpuNxNmatrices PRIVATE GCS)
target_link_libraries(test_cpuNxNmatrices PUBLIC stdc++fs)
target_link_libraries(test_cpuNxNmatrices PUBLIC ${UUID_LIBRARIES})

target_link_libraries(test_speedNxNmatrices PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(test_speedNxNmatrices PRIVATE GCS)
target_link_libraries(test_speedNxNmatrices PUBLIC stdc++fs)
target_link_libraries(test_speedNxNmatrices PUBLIC ${UUID_LIBRARIES})

target_link_libraries(test_cudaGCKernel PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(test_cudaGCKernel PRIVATE GCS)
target_link_libraries(test_cudaGCKernel PUBLIC stdc++fs)
target_link_libraries(test_cudaGCKernel PUBLIC ${UUID_LIBRARIES})

target_link_libraries(test_gcloadstoreunit PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(test_gcloadstoreunit PRIVATE GCS)
target_link_libraries(test_gcloadstoreunit PUBLIC stdc++fs)
target_link_libraries(test_gcloadstoreunit PUBLIC ${UUID_LIBRARIES})

target_link_libraries(test_queryhandler PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(test_queryhandler PRIVATE GCS)
target_link_libraries(test_queryhandler PUBLIC stdc++fs)
target_link_libraries(test_queryhandler PUBLIC ${UUID_LIBRARIES})

target_link_libraries(test_cpuparallelgckernel PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(test_cpuparallelgckernel PRIVATE GCS)
target_link_libraries(test_cpuparallelgckernel PUBLIC stdc++fs)
target_link_libraries(test_cpuparallelgckernel PUBLIC ${UUID_LIBRARIES})

target_link_libraries(test_cudasort PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(test_cudasort PUBLIC GCS)
target_link_libraries(test_cudasort PUBLIC stdc++fs)
#target_link_libraries(test_cudasort PUBLIC ${CUDA_cudadevrt_LIBRARY})
#target_link_libraries(test_cudasort PUBLIC ${UUID_LIBRARIES})
#set_property(TEST test_cudasort PROPERTY CUDA_ARCHITECTURES 35 50 72)
#target_compile_options(test_cudasort PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_50,code=sm_50>)
