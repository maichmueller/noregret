

###########################
# NOR Tests
###########################
set(
        NOR_TEST_SOURCES
        test_cfr.cpp
        test_fosg_concept.cpp
        test_fosg_traits.cpp
)
list(TRANSFORM NOR_TEST_SOURCES PREPEND "${PROJECT_TEST_DIR}/libnor/")

add_executable(${nor_test} ${PROJECT_TEST_DIR}/main_tests.cpp ${NOR_TEST_SOURCES})

#set_target_properties(${per_test} PROPERTIES
#        EXCLUDE_FROM_ALL True  # don't build tests when ALL is asked to be built. Only on demand.
#        )

target_link_libraries(${nor_test}
        PRIVATE
        ${nor_lib}
        ${nor_lib}_wrappers
        project_warnings
        CONAN_PKG::gtest
        pybind11::module
        $<$<NOT:$<BOOL:USE_PYBIND11_FINDPYTHON>>:Python3::Module>
        stratego
        )

add_test(
        NAME Test_${PROJECT_NAME}
        COMMAND ${nor_test}
)

###########################
# NOR Type Traits Tests
###########################
set(
        NOR_TYPE_TRAITS_TEST_SOURCES
        test_fosg_traits.cpp
        test_type_traits.cpp
)
list(TRANSFORM NOR_TYPE_TRAITS_TEST_SOURCES PREPEND "${PROJECT_TEST_DIR}/libnor/")

add_executable(${nor_test}_type_traits ${PROJECT_TEST_DIR}/main_tests.cpp ${NOR_TYPE_TRAITS_TEST_SOURCES})

#set_target_properties(${per_test} PROPERTIES
#        EXCLUDE_FROM_ALL True  # don't build tests when ALL is asked to be built. Only on demand.
#        )

target_link_libraries(${nor_test}_type_traits
        PRIVATE
        ${nor_lib}
        ${nor_lib}_wrappers
        project_warnings
        CONAN_PKG::gtest
        )

add_test(
        NAME Test_${PROJECT_NAME}_type_traits
        COMMAND ${nor_test}_type_traits
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
            stratego
            project_warnings
            CONAN_PKG::gtest
    )

    add_test(
            NAME Test_stratego
            COMMAND stratego_test
    )
endif ()