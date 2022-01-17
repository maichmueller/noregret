cmake_minimum_required(VERSION 3.15)

add_library(core INTERFACE)

set(CORE_INCLUDE_DIR core/include)

target_include_directories(core
		INTERFACE
		${CORE_INCLUDE_DIR}
		)

target_link_libraries(
		core
		INTERFACE
		project_options
		CONAN_PKG::xtensor
)

set(STRATEGO_SOURCES
        Game.cpp
		State.cpp
		Logic.cpp
		)

add_executable(stratego ${STRATEGO_SOURCES})

target_include_directories(stratego
		PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}
		)

target_link_libraries(stratego PRIVATE core CONAN_PKG::range-v3)