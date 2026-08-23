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

target_include_directories(stratego PUBLIC ${PROJECT_GAMES_DIR}/stratego/include
                                           ${PROJECT_GAMES_DIR}/stratego/include/stratego/core/include)

target_link_libraries(
    stratego
    PUBLIC required_min_libs
           common
           xtensor
           range-v3::range-v3
           namedtype::namedtype)

# ######################################################################################################################
# Leduc Poker
# ######################################################################################################################

set(KUHNPOKER_SOURCES state.cpp)

list(TRANSFORM KUHNPOKER_SOURCES PREPEND "${PROJECT_GAMES_DIR}/leduc_poker/impl/")

add_library(leduc_poker SHARED ${KUHNPOKER_SOURCES})

target_include_directories(leduc_poker PUBLIC ${PROJECT_GAMES_DIR}/leduc_poker/include)

target_link_libraries(leduc_poker PUBLIC required_min_libs common range-v3::range-v3)

# ######################################################################################################################
# Kuhn Poker
# ######################################################################################################################

set(KUHNPOKER_SOURCES state.cpp)

list(TRANSFORM KUHNPOKER_SOURCES PREPEND "${PROJECT_GAMES_DIR}/kuhn_poker/impl/")

add_library(kuhn_poker SHARED ${KUHNPOKER_SOURCES})

target_include_directories(kuhn_poker PUBLIC ${PROJECT_GAMES_DIR}/kuhn_poker/include)

target_link_libraries(kuhn_poker PUBLIC required_min_libs common range-v3::range-v3)

# ######################################################################################################################
# Liar's Dice (header-only)
# ######################################################################################################################

add_library(liars_dice INTERFACE)

target_include_directories(liars_dice INTERFACE ${PROJECT_GAMES_DIR}/liars_dice/include)

target_link_libraries(liars_dice INTERFACE required_min_libs common range-v3::range-v3)

# ######################################################################################################################
# Texas Hold'em Poker
# ######################################################################################################################

set(TEXHOLDEM_SOURCES state.cpp)

list(TRANSFORM TEXHOLDEM_SOURCES PREPEND "${PROJECT_GAMES_DIR}/texas_holdem_poker/impl/")

add_library(texas_holdem_poker SHARED ${TEXHOLDEM_SOURCES})

target_include_directories(texas_holdem_poker PUBLIC ${PROJECT_GAMES_DIR}/texas_holdem_poker/include)

target_link_libraries(texas_holdem_poker PUBLIC required_min_libs common range-v3::range-v3)

# ######################################################################################################################
# Rock Paper Scissors
# ######################################################################################################################

set(ROCKPAPERSCISSORS_SOURCES state.cpp)

list(TRANSFORM ROCKPAPERSCISSORS_SOURCES PREPEND "${PROJECT_GAMES_DIR}/rock_paper_scissors/impl/")

add_library(rock_paper_scissors SHARED ${ROCKPAPERSCISSORS_SOURCES})

target_include_directories(rock_paper_scissors PUBLIC ${PROJECT_GAMES_DIR}/rock_paper_scissors/include)

target_link_libraries(rock_paper_scissors PUBLIC required_min_libs common)

# ######################################################################################################################
# Goofspiel
# ######################################################################################################################

# goofspiel is a small game and is implemented fully inline (header-only), hence an INTERFACE target

add_library(goofspiel INTERFACE)

target_include_directories(goofspiel INTERFACE ${PROJECT_GAMES_DIR}/goofspiel/include)

target_link_libraries(goofspiel INTERFACE required_min_libs common range-v3::range-v3)
# Battleship (header-only)
# ######################################################################################################################

add_library(battleship INTERFACE)

target_include_directories(battleship INTERFACE ${PROJECT_GAMES_DIR}/battleship/include)

target_link_libraries(battleship INTERFACE required_min_libs common range-v3::range-v3)

# ######################################################################################################################
# Dark Hex (header-only)
# ######################################################################################################################

add_library(dark_hex INTERFACE)

target_include_directories(dark_hex INTERFACE ${PROJECT_GAMES_DIR}/dark_hex/include)

target_link_libraries(dark_hex INTERFACE required_min_libs common range-v3::range-v3)

# ######################################################################################################################
# Pursuit Evasion (header-only)
# ######################################################################################################################

add_library(pursuit_evasion INTERFACE)

target_include_directories(pursuit_evasion INTERFACE ${PROJECT_GAMES_DIR}/pursuit_evasion/include)

target_link_libraries(pursuit_evasion INTERFACE required_min_libs common range-v3::range-v3)
