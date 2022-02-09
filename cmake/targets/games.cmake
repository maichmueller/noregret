cmake_minimum_required(VERSION 3.16)

#################################
# Stratego
#################################

add_library(core INTERFACE)

set(CORE_INCLUDE_DIR ${PROJECT_GAMES_DIR}/stratego/include/stratego/core/include)

target_include_directories(
        core
        INTERFACE
        ${CORE_INCLUDE_DIR}
)

target_link_libraries(
        core
        INTERFACE
        project_options
        CONAN_PKG::xtensor
)

set(
        STRATEGO_SOURCES
        main.cpp
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
        core
        CONAN_PKG::range-v3
        CONAN_PKG::namedtype
)


#################################
#
#################################