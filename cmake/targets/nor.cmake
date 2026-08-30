# ######################################################################################################################
# Common utilities for all libraries ###
# ######################################################################################################################

add_library(common INTERFACE)

target_include_directories(common INTERFACE $<BUILD_INTERFACE:${PROJECT_COMMON_INCLUDE_DIR}>
                                            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(common INTERFACE ${required_min_libs})

# ######################################################################################################################
# NOR                 ###
# ######################################################################################################################

add_library(${nor_lib} ${nor-lib-type})

target_include_directories(${nor_lib} INTERFACE $<BUILD_INTERFACE:${PROJECT_NOR_INCLUDE_DIR}>
                                                $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(
    ${nor_lib}
    INTERFACE required_min_libs
              common
              namedtype::namedtype
              fmt::fmt-header-only)

set_target_properties(${nor_lib} PROPERTIES CXX_VISIBILITY_PRESET hidden)

# ######################################################################################################################
# NOR Environment Wrappers      ###
# ######################################################################################################################

set(WRAPPER_SOURCES
    stratego.cpp
    kuhn.cpp
    leduc.cpp
    rps.cpp)
list(TRANSFORM WRAPPER_SOURCES PREPEND "${PROJECT_NOR_DIR}/impl/")

if(ENABLE_GAMES)
    add_library(${nor_lib}_envs STATIC)

    # The environment wrappers are archived into the Python extension, which is a shared module. Every other game target
    # is already SHARED and therefore position independent; this static archive has to opt in explicitly or the
    # extension fails to link.
    set_target_properties(${nor_lib}_envs PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_sources(${nor_lib}_envs PRIVATE ${WRAPPER_SOURCES})

    target_link_libraries(
        ${nor_lib}_envs
        PUBLIC ${nor_lib}
               project_options
               stratego
               kuhn_poker
               leduc_poker
               rock_paper_scissors
               shapley)
endif()
