
add_library(${nor_lib} ${nor-lib-type})

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