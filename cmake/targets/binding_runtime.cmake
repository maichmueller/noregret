# The runtime registry is deliberately compiled out of the public binding headers. Each game partition instantiates only
# that game's admitted solver/profile pairs; catalog.cpp assembles the resulting descriptors without instantiating the
# Cartesian product again.
set(NOR_BINDING_RUNTIME_DIR "${PROJECT_NOR_BINDING_DIR}/runtime")

# The partition roster lives in catalog_partitions.hpp, which generates one declaration per game. Globbing the directory
# therefore cannot drift from it: a partition file that is missing is an unresolved declaration at link time, and a file
# for a game that is not in the static game list fails a static assertion. CONFIGURE_DEPENDS re-runs the glob when the
# directory changes.
file(
    GLOB
    NOR_BINDING_RUNTIME_GAME_PARTITIONS
    CONFIGURE_DEPENDS
    "${NOR_BINDING_RUNTIME_DIR}/partitions/*.cpp")
list(SORT NOR_BINDING_RUNTIME_GAME_PARTITIONS)
if(NOT NOR_BINDING_RUNTIME_GAME_PARTITIONS)
    message(FATAL_ERROR "no binding-runtime game partitions found in ${NOR_BINDING_RUNTIME_DIR}/partitions")
endif()

set(NOR_BINDING_RUNTIME_PARTITION_SOURCES
    "${NOR_BINDING_RUNTIME_DIR}/catalog.cpp" "${NOR_BINDING_RUNTIME_DIR}/dynamic.cpp"
    ${NOR_BINDING_RUNTIME_GAME_PARTITIONS})

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
