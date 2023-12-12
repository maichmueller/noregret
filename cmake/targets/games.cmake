cmake_minimum_required(VERSION 3.16)

# required dependencies for the games:
find_package(xtensor REQUIRED)


# ######################################################################################################################
# Stratego
# ######################################################################################################################

set(STRATEGO_SOURCES
        Game.cpp
        Config.cpp
        Utils.cpp
        State.cpp
        Logic.cpp)

list(TRANSFORM STRATEGO_SOURCES PREPEND "${PROJECT_GAMES_DIR}/stratego/impl/")

add_library(stratego SHARED ${STRATEGO_SOURCES})

target_include_directories(
        stratego
        PUBLIC
        ${PROJECT_GAMES_DIR}/stratego/include
        ${PROJECT_GAMES_DIR}/stratego/include/stratego/core/include
)

target_link_libraries(
        stratego
        PUBLIC
        required_min_libs
        common
        xtensor
        range-v3::range-v3
        namedtype::namedtype
)

# ######################################################################################################################
# Leduc Poker
# ######################################################################################################################

set(KUHNPOKER_SOURCES state.cpp)

list(TRANSFORM KUHNPOKER_SOURCES PREPEND "${PROJECT_GAMES_DIR}/leduc_poker/impl/")

add_library(leduc_poker SHARED ${KUHNPOKER_SOURCES})

target_include_directories(leduc_poker PUBLIC ${PROJECT_GAMES_DIR}/leduc_poker/include)

target_link_libraries(
        leduc_poker
        PUBLIC
        required_min_libs
        common
        range-v3::range-v3
)

# ######################################################################################################################
# Kuhn Poker
# ######################################################################################################################

set(KUHNPOKER_SOURCES state.cpp)

list(TRANSFORM KUHNPOKER_SOURCES PREPEND "${PROJECT_GAMES_DIR}/kuhn_poker/impl/")

add_library(kuhn_poker SHARED ${KUHNPOKER_SOURCES})

target_include_directories(kuhn_poker PUBLIC ${PROJECT_GAMES_DIR}/kuhn_poker/include)

target_link_libraries(
        kuhn_poker
        PUBLIC
        required_min_libs
        common
        range-v3::range-v3
)

# ######################################################################################################################
# Rock Paper Scissors
# ######################################################################################################################

set(ROCKPAPERSCISSORS_SOURCES state.cpp)

list(TRANSFORM ROCKPAPERSCISSORS_SOURCES PREPEND "${PROJECT_GAMES_DIR}/rock_paper_scissors/impl/")

add_library(rock_paper_scissors SHARED ${ROCKPAPERSCISSORS_SOURCES})

target_include_directories(rock_paper_scissors PUBLIC ${PROJECT_GAMES_DIR}/rock_paper_scissors/include)

target_link_libraries(
        rock_paper_scissors
        PUBLIC
        required_min_libs
        common
)
