
ADD_EXECUTABLE( test_simpletest test_simpletest.cu ${Foo_SOURCES})
ADD_EXECUTABLE( test_cudaNxNmatrices test_cudaNxNmatrices.cu ${Foo_SOURCES})

ADD_TEST(            io_gml test_simpletest    )
ADD_TEST(            io_gml test_cudaNxNmatrices    )


target_include_directories(test_simpletest PUBLIC ${Foo_INCLUDE_DIRS})
target_include_directories(test_cudaNxNmatrices PUBLIC ${Foo_INCLUDE_DIRS})


target_link_libraries(test_simpletest PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(test_simpletest PRIVATE GCS)

target_link_libraries(test_simpletest PUBLIC stdc++fs)

target_link_libraries(test_cudaNxNmatrices PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(test_cudaNxNmatrices PRIVATE GCS)

target_link_libraries(test_cudaNxNmatrices PUBLIC stdc++fs)
