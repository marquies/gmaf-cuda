
ADD_EXECUTABLE( test_simpletest test_simpletest.cu ${Foo_SOURCES})
ADD_EXECUTABLE( test_cudaNxNmatrices test_cudaNxNmatrices.cu ${Foo_SOURCES})
ADD_EXECUTABLE( test_speedNxNmatrices test_speedNxNmatrices.cu ${Foo_SOURCES})
ADD_EXECUTABLE( test_cpuNxNmatrices test_cpuNxNmatrix.cpp ${Foo_SOURCES})

ADD_TEST(            io_gml test_simpletest    )
ADD_TEST(            io_gml test_cudaNxNmatrices    )
ADD_TEST(            io_gml test_speedNxNmatrices    )
ADD_TEST(            io_gml test_cpuNxNmatrices    )


target_include_directories(test_simpletest PUBLIC ${Foo_INCLUDE_DIRS})
target_include_directories(test_cudaNxNmatrices PUBLIC ${Foo_INCLUDE_DIRS})
target_include_directories(test_cpuNxNmatrices PUBLIC ${Foo_INCLUDE_DIRS})
target_include_directories(test_speedNxNmatrices PUBLIC ${Foo_INCLUDE_DIRS})


target_link_libraries(test_simpletest PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(test_simpletest PRIVATE GCS)
target_link_libraries(test_simpletest PUBLIC stdc++fs)

target_link_libraries(test_cudaNxNmatrices PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(test_cudaNxNmatrices PRIVATE GCS)
target_link_libraries(test_cudaNxNmatrices PUBLIC stdc++fs)

target_link_libraries(test_cpuNxNmatrices PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(test_cpuNxNmatrices PRIVATE GCS)
target_link_libraries(test_cpuNxNmatrices PUBLIC stdc++fs)

target_link_libraries(test_speedNxNmatrices PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(test_speedNxNmatrices PRIVATE GCS)
target_link_libraries(test_speedNxNmatrices PUBLIC stdc++fs)
