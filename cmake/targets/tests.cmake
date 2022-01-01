
set(TEST_SOURCES
        tests.cpp
        )
list(TRANSFORM TEST_SOURCES PREPEND "${PROJECT_TEST_DIR}/")

add_executable(${nor_test} ${TEST_SOURCES})

#set_target_properties(${per_test} PROPERTIES
#        EXCLUDE_FROM_ALL True  # don't build tests when ALL is asked to be built. Only on demand.
#        )

target_link_libraries(${nor_test}
        PRIVATE
        project_warnings
        ${nor_lib}
        CONAN_PKG::gtest
        pybind11::module
        $<$<NOT:$<BOOL:USE_PYBIND11_FINDPYTHON>>:Python3::Module>
        )

add_test(
        NAME Test_${PROJECT_NAME}
        COMMAND ${nor_test}
)