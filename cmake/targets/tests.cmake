

###########################
# NOR Tests
###########################
set(
        TEST_SOURCES
        main_tests.cpp
)
list(TRANSFORM TEST_SOURCES PREPEND "${PROJECT_TEST_DIR}/")

add_executable(${nor_test} ${TEST_SOURCES})

#set_target_properties(${per_test} PROPERTIES
#        EXCLUDE_FROM_ALL True  # don't build tests when ALL is asked to be built. Only on demand.
#        )

target_link_libraries(${nor_test}
        PRIVATE
        ${nor_lib}
        project_warnings
        CONAN_PKG::gtest
        pybind11::module
        $<$<NOT:$<BOOL:USE_PYBIND11_FINDPYTHON>>:Python3::Module>
        )

add_test(
        NAME Test_${PROJECT_NAME}
        COMMAND ${nor_test}
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