# ######################################################################################################################
# A mere test-bed executable (hence sandbox)
# ######################################################################################################################

set(SANDBOX_TEST_SOURCES sandbox.cpp)
list(TRANSFORM SANDBOX_TEST_SOURCES PREPEND "${PROJECT_TEST_DIR}/sandbox/")

add_executable(sandbox ${SANDBOX_TEST_SOURCES})

set_target_properties(
        sandbox PROPERTIES EXCLUDE_FROM_ALL True # don't build the sandbox when ALL is asked to be built. Only on demand.
)

target_link_libraries(
        sandbox
        PRIVATE
#        ${nor_lib}
#        ${nor_lib}_envs
#        stratego
        required_min_libs
        common
        project_warnings
        range-v3::range-v3
        xtensor)
