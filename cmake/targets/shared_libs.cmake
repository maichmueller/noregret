add_library(shared_test_libs INTERFACE)
target_link_libraries(
        shared_test_libs
        INTERFACE
        required_min_libs
        ${nor_lib}
        ${nor_lib}_envs
        gtest::gtest
)
target_include_directories(shared_test_libs INTERFACE "${PROJECT_TEST_DIR}/common_test_utils")
