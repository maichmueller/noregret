cmake_minimum_required(VERSION 3.16)

#################################
# Stratego
#################################

add_library(stratego_core INTERFACE)

set(STRATEGO_CORE_INCLUDE_DIR ${PROJECT_GAMES_DIR}/stratego/include/stratego/core/include)

target_include_directories(
        stratego_core
        INTERFACE
        ${STRATEGO_CORE_INCLUDE_DIR}
)

target_link_libraries(
        stratego_core
        INTERFACE
        project_options
        CONAN_PKG::xtensor
)

set(
        STRATEGO_SOURCES
        Game.cpp
        Config.cpp
        Utils.cpp
        State.cpp
        Logic.cpp
)

list(TRANSFORM STRATEGO_SOURCES PREPEND "${PROJECT_GAMES_DIR}/stratego/impl/")

add_library(stratego SHARED ${STRATEGO_SOURCES})

target_include_directories(
        stratego
        PUBLIC
        ${PROJECT_GAMES_DIR}/stratego/include
)

target_link_libraries(
        stratego
        PUBLIC
        project_options
        common
        stratego_core
        CONAN_PKG::range-v3
        CONAN_PKG::namedtype
)

#################################
# Leduc Poker
#################################

set(
        KUHNPOKER_SOURCES
        state.cpp
)

list(TRANSFORM KUHNPOKER_SOURCES PREPEND "${PROJECT_GAMES_DIR}/leduc_poker/impl/")

add_library(leduc_poker SHARED ${KUHNPOKER_SOURCES})

target_include_directories(
        leduc_poker
        PUBLIC
        ${PROJECT_GAMES_DIR}/leduc_poker/include
)

target_link_libraries(
        leduc_poker
        PUBLIC
        project_options
        common
        CONAN_PKG::range-v3
)



#################################
# Kuhn Poker
#################################

set(
        KUHNPOKER_SOURCES
        state.cpp
)

list(TRANSFORM KUHNPOKER_SOURCES PREPEND "${PROJECT_GAMES_DIR}/kuhn_poker/impl/")

add_library(kuhn_poker SHARED ${KUHNPOKER_SOURCES})

target_include_directories(
        kuhn_poker
        PUBLIC
        ${PROJECT_GAMES_DIR}/kuhn_poker/include
)

target_link_libraries(
        kuhn_poker
        PUBLIC
        project_options
        common
        CONAN_PKG::range-v3
)

#################################
# Rock Paper Scissors
#################################

set(
        ROCKPAPERSCISSORS_SOURCES
        state.cpp
)

list(TRANSFORM ROCKPAPERSCISSORS_SOURCES PREPEND "${PROJECT_GAMES_DIR}/rock_paper_scissors/impl/")

add_library(rock_paper_scissors SHARED ${ROCKPAPERSCISSORS_SOURCES})

target_include_directories(
        rock_paper_scissors
        PUBLIC
        ${PROJECT_GAMES_DIR}/rock_paper_scissors/include
)

target_link_libraries(
        rock_paper_scissors
        PUBLIC
        project_options
        common
)