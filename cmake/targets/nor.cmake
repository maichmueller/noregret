
##########################################
### Common utilities for all libraries ###
##########################################

add_library(
        common
        INTERFACE
)

target_include_directories(
        common
        INTERFACE
        $<BUILD_INTERFACE:${PROJECT_COMMON_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(
        common
        INTERFACE
        project_options
        CONAN_PKG::range-v3
)

##########################################
###                NOR                 ###
##########################################

add_library(
        ${nor_lib}
        ${nor-lib-type}
)

target_include_directories(
        ${nor_lib}
        INTERFACE
        $<BUILD_INTERFACE:${PROJECT_NOR_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(
        ${nor_lib}
        INTERFACE
        project_options
        common
        CONAN_PKG::cppitertools
        CONAN_PKG::range-v3
)

set_target_properties(
        ${nor_lib}
        PROPERTIES
        CXX_VISIBILITY_PRESET hidden
)

##########################################
###      NOR Environment Wrappers      ###
##########################################

set(
        WRAPPER_SOURCES
        stratego_env.cpp
        kuhn_env.cpp
        rps_env.cpp
)
list(TRANSFORM WRAPPER_SOURCES PREPEND "${PROJECT_NOR_DIR}/impl/")

if (ENABLE_GAMES)
    add_library(
            ${nor_lib}_envs
            STATIC
    )

    target_sources(
            ${nor_lib}_envs
            PRIVATE
            ${WRAPPER_SOURCES}
    )

    target_link_libraries(
            ${nor_lib}_envs
            PUBLIC
            ${nor_lib}
            stratego
            kuhn_poker
            rock_paper_scissors
    )
endif ()
