# The runtime registry is deliberately compiled out of the public binding headers. Each game partition instantiates only
# that game's admitted solver/profile pairs; catalog.cpp assembles the resulting descriptors without instantiating the
# Cartesian product again.
set(NOR_BINDING_RUNTIME_DIR "${PROJECT_NOR_BINDING_DIR}/runtime")

set(NOR_BINDING_RUNTIME_PARTITION_SOURCES
    "${NOR_BINDING_RUNTIME_DIR}/catalog.cpp"
    "${NOR_BINDING_RUNTIME_DIR}/dynamic.cpp"
    "${NOR_BINDING_RUNTIME_DIR}/partitions/kuhn.cpp"
    "${NOR_BINDING_RUNTIME_DIR}/partitions/leduc.cpp"
    "${NOR_BINDING_RUNTIME_DIR}/partitions/rps.cpp"
    "${NOR_BINDING_RUNTIME_DIR}/partitions/stratego.cpp"
    "${NOR_BINDING_RUNTIME_DIR}/partitions/texas_holdem.cpp"
    "${NOR_BINDING_RUNTIME_DIR}/partitions/goofspiel.cpp"
    "${NOR_BINDING_RUNTIME_DIR}/partitions/three_player_goofspiel.cpp"
    "${NOR_BINDING_RUNTIME_DIR}/partitions/battleship.cpp"
    "${NOR_BINDING_RUNTIME_DIR}/partitions/battleship_gs.cpp"
    "${NOR_BINDING_RUNTIME_DIR}/partitions/dark_hex.cpp"
    "${NOR_BINDING_RUNTIME_DIR}/partitions/pursuit_evasion.cpp"
    "${NOR_BINDING_RUNTIME_DIR}/partitions/oshi_zumo.cpp"
    "${NOR_BINDING_RUNTIME_DIR}/partitions/shapley.cpp"
    "${NOR_BINDING_RUNTIME_DIR}/partitions/centipede.cpp"
    "${NOR_BINDING_RUNTIME_DIR}/partitions/colonel_blotto.cpp"
    "${NOR_BINDING_RUNTIME_DIR}/partitions/sheriff.cpp"
    "${NOR_BINDING_RUNTIME_DIR}/partitions/liars_dice.cpp")

add_library(nor_binding_runtime STATIC ${NOR_BINDING_RUNTIME_PARTITION_SOURCES})
set_property(GLOBAL APPEND PROPERTY JOB_POOLS nor_binding_runtime_compile=1)
set_property(TARGET nor_binding_runtime PROPERTY JOB_POOL_COMPILE nor_binding_runtime_compile)
set_target_properties(nor_binding_runtime PROPERTIES CXX_VISIBILITY_PRESET hidden POSITION_INDEPENDENT_CODE ON)
target_include_directories(nor_binding_runtime PUBLIC $<BUILD_INTERFACE:${PROJECT_SRC_DIR}>
                                                      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(
    nor_binding_runtime
    PUBLIC ${nor_lib}_envs
           stratego
           kuhn_poker
           leduc_poker
           rock_paper_scissors
           texas_holdem_poker
           goofspiel
           three_player_goofspiel
           battleship
           battleship_gs
           dark_hex
           pursuit_evasion
           oshi_zumo
           shapley
           centipede
           colonel_blotto
           sheriff
           liars_dice)
