add_library(common_testing_utils INTERFACE)

target_include_directories(common_testing_utils INTERFACE "${PROJECT_TEST_DIR}/common_test_utils")

add_library(shared_test_libs INTERFACE)
target_link_libraries(
    shared_test_libs
    INTERFACE ${nor_lib}
              ${nor_lib}_envs
              project_options
              project_warnings
              common_testing_utils
              gtest::gtest)
