add_library(
        common_testing_utils
        INTERFACE
)

target_include_directories(common_testing_utils INTERFACE "${PROJECT_TEST_DIR}/common_test_utils")

add_library(
        shared_test_libs
        INTERFACE
)
target_link_libraries(
        shared_test_libs
        INTERFACE
        ${nor_lib}
        ${nor_lib}_envs
        project_options
        project_warnings
        common_testing_utils
        CONAN_PKG::gtest
)

function(register_nor_target target_name)
    if (NOT ARGN)
        message(FATAL_ERROR "You must pass at least one source file to define the target '${target_name}'")
    endif ()
    set(${target_name}_sources_list ${ARGN})
    # this set needs to be set in the parent scope as well in order to remain present
    set(${target_name}_sources_list ${ARGN} PARENT_SCOPE)
    list(APPEND REGISTERED_TARGET_NAMES_LIST ${target_name})
    list(APPEND REGISTERED_TEST_SOURCES_LIST ${target_name}_sources_list)
    # update the lists in the parent scope, since function holds copies of these variables for its own scope
    set(REGISTERED_TARGET_NAMES_LIST ${REGISTERED_TARGET_NAMES_LIST} PARENT_SCOPE)
    set(REGISTERED_TEST_SOURCES_LIST ${REGISTERED_TEST_SOURCES_LIST} PARENT_SCOPE)

    message(STATUS "Target: ${target_name}")
    message(STATUS "Target Source Contents: ${${target_name}_sources_list}")

    list(TRANSFORM ${target_name}_sources_list PREPEND "${PROJECT_TEST_DIR}/libnor/")

    message(STATUS "Target Source Contents (pathed): ${${target_name}_sources_list}")

    add_executable(
            ${target_name}
            ${PROJECT_TEST_DIR}/main_tests.cpp
            ${${target_name}_sources_list})  # a list of source files to append

    target_link_libraries(
            ${target_name}
            PRIVATE
            shared_test_libs
    )

    add_test(
            NAME Test_${target_name}
            COMMAND ${target_name}
    )
endfunction()


function(register_game_target target_name lib_name game_folder_name)
    if (NOT ARGN)
        message(FATAL_ERROR "You must pass at least one source file to define the target '${target_name}'")
    endif ()
    # append the prefix 'game' and suffix 'test' to each target name
    set(target_name "game_${target_name}_test")
    set(${target_name}_sources_list ${ARGN})
    # this set needs to be set in the parent scope as well in order to remain present
    set(${target_name}_sources_list ${ARGN} PARENT_SCOPE)

    list(APPEND REGISTERED_GAME_TARGET_NAMES_LIST ${target_name})
    list(APPEND REGISTERED_GAME_TEST_SOURCES_LIST ${target_name}_sources_list)
    # update the lists in the parent scope, since function holds copies of these variables for its own scope
    set(REGISTERED_TARGET_NAMES_LIST ${REGISTERED_TARGET_NAMES_LIST} PARENT_SCOPE)
    set(REGISTERED_TEST_SOURCES_LIST ${REGISTERED_TEST_SOURCES_LIST} PARENT_SCOPE)

    message(STATUS "Target: ${target_name}")
    message(STATUS "Target Source Contents: ${${target_name}_sources_list}")
    list(TRANSFORM ${target_name}_sources_list PREPEND "${PROJECT_TEST_DIR}/games/${game_folder_name}/")
    message(STATUS "Target Source Contents (pathed): ${${target_name}_sources_list}")

    add_executable(
            ${target_name}
            ${PROJECT_TEST_DIR}/main_tests.cpp
            ${${target_name}_sources_list})  # a list of source files to append

    set_target_properties(${target_name} PROPERTIES
            EXCLUDE_FROM_ALL True  # don't build tests when ALL is asked to be built. Only on demand.
            )

    target_link_libraries(
            ${target_name}
            PRIVATE
            shared_test_libs
            ${lib_name}
    )

    add_test(
            NAME Test_${target_name}
            COMMAND ${target_name}
    )
endfunction()