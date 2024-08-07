include("${_cmake_DIR}/settings/Utilities.cmake")

register_nor_target(${nor_test}_type_traits test_fosg_traits.cpp test_type_traits.cpp)
register_nor_target(${nor_test}_concepts test_fosg_concepts.cpp)
register_nor_target(${nor_test}_utils test_utils.cpp)
register_nor_target(${nor_test}_rm_utils test_rm_utils.cpp)
register_nor_target(${nor_test}_cfr_vanilla test_cfr_vanilla.cpp)
register_nor_target(${nor_test}_cfr_plus test_cfr_plus.cpp)
register_nor_target(${nor_test}_cfr_linear test_cfr_linear.cpp)
register_nor_target(${nor_test}_cfr_discounted test_cfr_discounted.cpp)
register_nor_target(${nor_test}_cfr_exponential test_cfr_exponential.cpp)
register_nor_target(${nor_test}_cfr_monte_carlo test_cfr_monte_carlo.cpp)
register_nor_target(${nor_test}_policy test_policy.cpp)
register_nor_target(${nor_test}_helpers test_helpers.cpp)
register_nor_target(${nor_test}_exploitability test_exploitability.cpp)
# for the overall test executable we simply merge all other test files together
foreach(sources_list IN LISTS REGISTERED_TEST_SOURCES_LIST)
    list(APPEND NOR_TEST_SOURCES ${${sources_list}})
endforeach()
register_nor_target(${nor_test}_all ${NOR_TEST_SOURCES})

# the test of all parts needs an extra linkage for the pybind11 components and Python
target_link_libraries(${nor_test}_all PRIVATE # pybind11::module
                                              $<$<NOT:$<BOOL:USE_PYBIND11_FINDPYTHON>>:Python3::Module>)

if(ENABLE_GAMES)
    # this is a mere collector of all game test targets to build them via a single command to build all games (eg in
    # workflow)
    add_custom_target(game_test_all)

    message(STATUS "Adding game targets.")

    register_game_target(
        stratego
        INCLUDE_DIR
        stratego
        LINK_LIBRARY
        stratego
        SOURCE_FILES
        test_logic.cpp
        test_config.cpp
        test_game.cpp
        test_state.cpp
        test_piece.cpp)
    register_game_target(
        kuhn_poker
        INCLUDE_DIR
        kuhn_poker
        LINK_LIBRARY
        kuhn_poker
        SOURCE_FILES
        test_state.cpp)
    register_game_target(
        leduc_poker
        INCLUDE_DIR
        leduc_poker
        LINK_LIBRARY
        leduc_poker
        SOURCE_FILES
        test_state.cpp)
endif()
