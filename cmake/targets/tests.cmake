include("${_cmake_DIR}/settings/Utilities.cmake")


register_nor_target(
        ${nor_test}_type_traits
        test_fosg_traits.cpp
        test_type_traits.cpp
)
register_nor_target(
        ${nor_test}_concepts
        test_fosg_concepts.cpp
)
register_nor_target(
        ${nor_test}_utils
        test_utils.cpp
)
register_nor_target(
        ${nor_test}_rm_utils
        test_rm_utils.cpp
)
register_nor_target(
        ${nor_test}_cfr_vanilla
        test_cfr.cpp
)
register_nor_target(
        ${nor_test}_cfr_plus
        test_cfr_plus.cpp
)
register_nor_target(
        ${nor_test}_cfr_linear
        test_cfr_linear.cpp
)
register_nor_target(
        ${nor_test}_cfr_discounted
        test_cfr_discounted.cpp
)
register_nor_target(
        ${nor_test}_cfr_exponential
        test_cfr_exponential.cpp)
register_nor_target(
        ${nor_test}_cfr_monte_carlo
        test_cfr_monte_carlo.cpp
)
register_nor_target(
        ${nor_test}_policy
        test_policy.cpp
)
register_nor_target(
        ${nor_test}_helpers
        test_helpers.cpp
)
# for the overall test executable we simply merge all other test files together
foreach (
        sources_list
        IN LISTS
        REGISTERED_TEST_SOURCES_LIST
)
    list(APPEND NOR_TEST_SOURCES ${${sources_list}})
endforeach ()
register_nor_target(${nor_test} ${NOR_TEST_SOURCES})

# the test of all parts needs an extra linkage for the pybind11 components and Python
target_link_libraries(
        ${nor_test}
        PRIVATE
        pybind11::module
        $<$<NOT:$<BOOL:USE_PYBIND11_FINDPYTHON>>:Python3::Module>
)


if (ENABLE_GAMES)

    message(STATUS "Adding game targets.")

    register_game_target(
            stratego
            stratego
            stratego
            test_logic.cpp
            test_config.cpp
            test_game.cpp
            test_state.cpp
            test_piece.cpp
    )
    register_game_target(
            kuhn_poker
            kuhn_poker
            kuhn_poker
            test_state.cpp
    )
    register_game_target(
            leduc_poker
            leduc_poker
            leduc_poker
            test_state.cpp
    )
endif ()