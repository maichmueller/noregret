
##################################################
## common testing libraries for ease of linking ##
##################################################

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

###########################
# NOR Tests
###########################

set(
        TYPE_TRAITS_TEST_SOURCES
        test_fosg_traits.cpp
        test_type_traits.cpp
)
set(
        CONCEPTS_TEST_SOURCES
        test_fosg_concepts.cpp
)
set(
        CFR_VANILLA_TEST_SOURCES
        test_cfr.cpp
)
set(
        CFR_PLUS_TEST_SOURCES
        test_cfr_plus.cpp
)
set(
        CFR_LINEAR_TEST_SOURCES
        test_cfr_linear.cpp
)
set(
        CFR_DISCOUNTED_TEST_SOURCES
        test_cfr_discounted.cpp
)
set(
        CFR_EXP_TEST_SOURCES
        test_cfr_exponential.cpp
)
set(
        CFR_MONTE_CARLO_TEST_SOURCES
        test_cfr_monte_carlo.cpp
)
set(
        POLICY_TEST_SOURCES
        test_policy.cpp
)
set(
        RM_UTILS_TEST_SOURCES
        test_rm_utils.cpp
)

set(
        ALL_TEST_SOURCES_LIST
        # [this position will hold the element that joins all of the below entries]
        TYPE_TRAITS_TEST_SOURCES
        CONCEPTS_TEST_SOURCES
        RM_UTILS_TEST_SOURCES
        CFR_VANILLA_TEST_SOURCES
        CFR_PLUS_TEST_SOURCES
        CFR_LINEAR_TEST_SOURCES
        CFR_DISCOUNTED_TEST_SOURCES
        CFR_EXP_TEST_SOURCES
        CFR_MONTE_CARLO_TEST_SOURCES
        POLICY_TEST_SOURCES
)

# for the overall test executable we simply merge all other test files together
foreach (
        sources_list
        IN LISTS
        ALL_TEST_SOURCES_LIST
)
    list(APPEND NOR_TEST_SOURCES ${${sources_list}})
endforeach ()
#list(JOIN ALL_TEST_SOURCES_LIST ";" NOR_TEST_SOURCES)
# prepend it as the first element the target list
list(PREPEND ALL_TEST_SOURCES_LIST NOR_TEST_SOURCES)

set(
        TARGET_NAMES_LIST
        ${nor_test}  # the ALL nor tests exe name
        ${nor_test}_type_traits
        ${nor_test}_concepts
        ${nor_test}_rm_utils
        ${nor_test}_cfr_vanilla
        ${nor_test}_cfr_plus
        ${nor_test}_cfr_linear
        ${nor_test}_cfr_discounted
        ${nor_test}_cfr_exponential
        ${nor_test}_cfr_monte_carlo
        ${nor_test}_policy
)

foreach (
        source_files_list
        target_name
        IN ZIP_LISTS
        ALL_TEST_SOURCES_LIST
        TARGET_NAMES_LIST
)

    foreach (
            source_file
            IN LISTS
            ${source_files_list}
    )
        list(APPEND source_files ${source_file})
    endforeach ()

    message(STATUS "Target: ${target_name}")
    message(STATUS "Target Source List-Name: ${sources_list}")
    message(STATUS "Target Source Contents: ${source_files}")

    list(TRANSFORM source_files PREPEND "${PROJECT_TEST_DIR}/libnor/")

    message(STATUS "Target Source Contents (pathed): ${source_files}")

    add_executable(
            ${target_name}
            ${PROJECT_TEST_DIR}/main_tests.cpp
            ${source_files})  # a list of source files to append

    target_link_libraries(
            ${target_name}
            PRIVATE
            shared_test_libs
    )

    add_test(
            NAME Test_${target_name}
            COMMAND ${target_name}
    )
    # delete the list so that the next iteration can start with an empty list for that name.
    # If we dont do this then the path will be added recursively often.
    set(source_files)
endforeach ()

# the test of all parts needs an extra linkage for the pybind11 components and Python
target_link_libraries(
        ${nor_test}
        PRIVATE
        pybind11::module
        $<$<NOT:$<BOOL:USE_PYBIND11_FINDPYTHON>>:Python3::Module>
)


if (ENABLE_GAMES)
    #########################
    # Stratego Tests
    #########################
    set(
            STRATEGO_TEST_SOURCES
            test_logic.cpp
            test_config.cpp
            test_game.cpp
            test_state.cpp
            test_piece.cpp
    )
    list(TRANSFORM STRATEGO_TEST_SOURCES PREPEND "${PROJECT_TEST_DIR}/games/stratego/")

    add_executable(stratego_test ${PROJECT_TEST_DIR}/main_tests.cpp ${STRATEGO_TEST_SOURCES})

    set_target_properties(stratego_test PROPERTIES
            EXCLUDE_FROM_ALL True  # don't build tests when ALL is asked to be built. Only on demand.
            )

    target_link_libraries(
            stratego_test
            PRIVATE
            shared_test_libs
            stratego

    )

    add_test(
            NAME Test_stratego
            COMMAND stratego_test
    )

    #########################
    # Kuhn Poker Tests
    #########################
    set(
            KUHNPOKER_TEST_SOURCES
            test_state.cpp
    )
    list(TRANSFORM KUHNPOKER_TEST_SOURCES PREPEND "${PROJECT_TEST_DIR}/games/kuhn_poker/")

    add_executable(kuhn_poker_test ${PROJECT_TEST_DIR}/main_tests.cpp ${KUHNPOKER_TEST_SOURCES})

    set_target_properties(kuhn_poker_test PROPERTIES
            EXCLUDE_FROM_ALL True  # don't build tests when ALL is asked to be built. Only on demand.
            )

    target_link_libraries(
            kuhn_poker_test
            PRIVATE
            shared_test_libs
            kuhn_poker
    )

    add_test(
            NAME Test_kuhn_poker
            COMMAND kuhn_poker_test
    )

    #########################
    # Leduc Poker Tests
    #########################
    set(
            LEDUCPOKER_TEST_SOURCES
            test_state.cpp
    )
    list(TRANSFORM LEDUCPOKER_TEST_SOURCES PREPEND "${PROJECT_TEST_DIR}/games/leduc_poker/")

    add_executable(leduc_poker_test ${PROJECT_TEST_DIR}/main_tests.cpp ${LEDUCPOKER_TEST_SOURCES})

    set_target_properties(leduc_poker_test PROPERTIES
            EXCLUDE_FROM_ALL True  # don't build tests when ALL is asked to be built. Only on demand.
            )

    target_link_libraries(
            leduc_poker_test
            PRIVATE
            shared_test_libs
            leduc_poker
    )

    add_test(
            NAME Test_leduc_poker
            COMMAND leduc_poker_test
    )
endif ()