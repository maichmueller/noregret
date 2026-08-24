include("${_cmake_DIR}/settings/Utilities.cmake")

register_nor_target(${nor_test}_type_traits test_fosg_traits.cpp test_type_traits.cpp)
register_nor_target(${nor_test}_concepts test_fosg_concepts.cpp)
register_nor_target(${nor_test}_utils test_utils.cpp)
register_nor_target(${nor_test}_meta test_meta.cpp)
register_nor_target(${nor_test}_rm_utils test_rm_utils.cpp)
register_nor_target(${nor_test}_cfr_vanilla test_cfr_vanilla.cpp)
register_nor_target(${nor_test}_cfr_plus test_cfr_plus.cpp)
register_nor_target(${nor_test}_cfr_pruning test_cfr_pruning.cpp)
register_nor_target(${nor_test}_cfr_linear test_cfr_linear.cpp)
register_nor_target(${nor_test}_cfr_discounted test_cfr_discounted.cpp)
register_nor_target(${nor_test}_cfr_exponential test_cfr_exponential.cpp)
register_nor_target(${nor_test}_cfr_predictive test_cfr_predictive.cpp)
register_nor_target(${nor_test}_cfr_predictive_variants test_cfr_predictive_variants.cpp)
register_nor_target(${nor_test}_cfr_scheduled test_cfr_scheduled.cpp)
register_nor_target(${nor_test}_cfr_monte_carlo test_cfr_monte_carlo.cpp)
register_nor_target(${nor_test}_cfr_vr_monte_carlo test_cfr_vr_monte_carlo.cpp)
register_nor_target(${nor_test}_cfr_escher test_cfr_escher.cpp)
register_nor_target(${nor_test}_mccfr_sampling_rules test_mccfr_sampling_rules.cpp)
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
        test_state.cpp
        test_state_3p.cpp)
    register_game_target(
        leduc_poker
        INCLUDE_DIR
        leduc_poker
        LINK_LIBRARY
        leduc_poker
        SOURCE_FILES
        test_state.cpp)
    register_game_target(
        big_leduc
        INCLUDE_DIR
        big_leduc
        LINK_LIBRARY
        leduc_poker
        SOURCE_FILES
        test_state.cpp
        test_payoff.cpp
        test_playouts.cpp
        test_convergence.cpp)
    register_game_target(
        texas_holdem_poker
        INCLUDE_DIR
        texas_holdem_poker
        LINK_LIBRARY
        texas_holdem_poker
        SOURCE_FILES
        test_state.cpp)
    register_game_target(
        goofspiel
        INCLUDE_DIR
        goofspiel
        LINK_LIBRARY
        goofspiel
        SOURCE_FILES
        test_state.cpp)
    register_game_target(
        liars_dice
        INCLUDE_DIR
        liars_dice
        LINK_LIBRARY
        liars_dice
        SOURCE_FILES
        test_state.cpp
        test_multiplayer.cpp)
    register_game_target(
        battleship
        INCLUDE_DIR
        battleship
        LINK_LIBRARY
        battleship
        SOURCE_FILES
        test_state.cpp)
    register_game_target(
        dark_hex
        INCLUDE_DIR
        dark_hex
        LINK_LIBRARY
        dark_hex
        SOURCE_FILES
        test_state.cpp)
    register_game_target(
        pursuit_evasion
        INCLUDE_DIR
        pursuit_evasion
        LINK_LIBRARY
        pursuit_evasion
        SOURCE_FILES
        test_state.cpp)
    register_game_target(
        oshi_zumo
        INCLUDE_DIR
        oshi_zumo
        LINK_LIBRARY
        oshi_zumo
        SOURCE_FILES
        test_state.cpp)
    register_game_target(
        shapley
        INCLUDE_DIR
        shapley
        LINK_LIBRARY
        shapley
        SOURCE_FILES
        test_state.cpp)
    register_game_target(
        centipede
        INCLUDE_DIR
        centipede
        LINK_LIBRARY
        centipede
        SOURCE_FILES
        test_state.cpp)
    register_game_target(
        colonel_blotto
        INCLUDE_DIR
        colonel_blotto
        LINK_LIBRARY
        colonel_blotto
        SOURCE_FILES
        test_state.cpp)
endif()
