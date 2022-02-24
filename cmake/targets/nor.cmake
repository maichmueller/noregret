
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
        CONAN_PKG::cppitertools
        CONAN_PKG::range-v3
)

set_target_properties(
        ${nor_lib}
        PROPERTIES
        CXX_VISIBILITY_PRESET hidden
)

set(
        WRAPPER_SOURCES
        stratego_env.cpp
)
list(TRANSFORM WRAPPER_SOURCES PREPEND "${PROJECT_NOR_DIR}/impl/")

if (ENABLE_GAMES)
    add_library(
            ${nor_lib}_wrappers
            STATIC
    )

    target_sources(
            ${nor_lib}_wrappers
            PRIVATE
            ${WRAPPER_SOURCES}
    )

    target_link_libraries(
            ${nor_lib}_wrappers
            PUBLIC
            ${nor_lib}
            stratego
    )
endif ()
