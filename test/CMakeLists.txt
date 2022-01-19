
ADD_EXECUTABLE( test_simpletest test_simpletest.cu ${Foo_SOURCES})
ADD_EXECUTABLE( test_cudaNxNmatrices test_cudaNxNmatrices.cu ${Foo_SOURCES})
ADD_EXECUTABLE( test_speedNxNmatrices test_speedNxNmatrices.cu ${Foo_SOURCES})
ADD_EXECUTABLE( test_cpuNxNmatrices test_cpuNxNmatrix.cu ${Foo_SOURCES})
ADD_EXECUTABLE( test_cudaGCKernel test_cudaGCKernel.cu ${Foo_SOURCES})
ADD_EXECUTABLE( test_gcloadstoreunit test_gcloadstoreunit.cu ${Foo_SOURCES})

ADD_TEST( io_gml test_simpletest    )
ADD_TEST( io_gml test_cudaNxNmatrices    )
ADD_TEST( io_gml test_speedNxNmatrices    )
ADD_TEST( io_gml test_cpuNxNmatrices    )
ADD_TEST( io_gml test_cudaGCKernel    )
ADD_TEST( io_gml test_gcloadstoreunit    )


target_include_directories(test_simpletest PUBLIC ${Foo_INCLUDE_DIRS})
target_include_directories(test_cudaNxNmatrices PUBLIC ${Foo_INCLUDE_DIRS})
target_include_directories(test_cpuNxNmatrices PUBLIC ${Foo_INCLUDE_DIRS})
target_include_directories(test_speedNxNmatrices PUBLIC ${Foo_INCLUDE_DIRS})
target_include_directories(test_cudaGCKernel PUBLIC ${Foo_INCLUDE_DIRS})
target_include_directories(test_gcloadstoreunit PUBLIC ${Foo_INCLUDE_DIRS})


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
