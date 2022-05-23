cmake_minimum_required(VERSION 3.16)

project(NOR VERSION 0.1 LANGUAGES CXX)

string(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWERCASE)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# options

option(ENABLE_BUILD_DOCS "Enable building the docs. Requires doxygen to be installed on the system" ON)
option(ENABLE_BUILD_PYTHON_EXTENSION "Enable building the python extension." ON)
option(ENABLE_BUILD_SANDBOX "Enable building of the sandbox testbed (Only for development purposes)." ON)
option(ENABLE_BUILD_WITH_TIME_TRACE "Enable -ftime-trace to generate time tracing .json files on clang" OFF)
option(ENABLE_CACHE "Enable cache if available" ON)
option(ENABLE_CLANG_TIDY "Enable static analysis with clang-tidy" OFF)
option(ENABLE_COVERAGE "Enable coverage reporting for gcc/clang" OFF)
option(ENABLE_CPPCHECK "Enable static analysis with cppcheck" OFF)
option(ENABLE_INCLUDE_WHAT_YOU_USE "Enable static analysis with include-what-you-use" OFF)
option(ENABLE_IPO "Enable Interprocedural Optimization, aka Link Time Optimization (LTO)" OFF)
option(ENABLE_PCH "Enable Precompiled Headers" ON)
option(ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" OFF)
option(ENABLE_SANITIZER_LEAK "Enable leak sanitizer" OFF)
option(ENABLE_SANITIZER_MEMORY "Enable memory sanitizer" OFF)
option(ENABLE_SANITIZER_THREAD "Enable thread sanitizer" OFF)
option(ENABLE_SANITIZER_UNDEFINED_BEHAVIOR "Enable undefined behavior sanitizer" OFF)
option(ENABLE_TESTING "Enable Test Builds" ON)
option(ENABLE_GAMES "Enable Game Builds" ON)
option(USE_PYBIND11_FINDPYTHON "Use pybind11 to search for the python library" OFF)
option(WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

# settings

# Use C++20
set(PROJ_CXX_STD_FEATURE cxx_std_20)
set(PROJ_CXX_STANDARD C++20)
if (MSVC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 19.29.30129 AND CMAKE_VERSION VERSION_GREATER 3.20.3)
    # this change happened in CMake 3.20.4
    set(PROJ_CXX_STD_FEATURE cxx_std_23)# /std:c++latest - unlocks the non stable cpp20 features. For new 16.11 versions
else ()
    set(PROJ_CXX_STD_FEATURE cxx_std_20) # /std:c++latest for msvc and -std=c++20 for everyone else.
endif ()

set(nor-lib-type INTERFACE)
set(nor_lib nor++)
set(nor_pymodule pynor)
set(nor_test nor_tests)

set(CONANFILE conanfile.txt)
set(DEPENDENCY_DIR deps)  # has to be relative to CMAKE_CURRENT_SOURCE_DIR
set(PROJECT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(PROJECT_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(PROJECT_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test")
set(PROJECT_NOR_DIR "${PROJECT_SRC_DIR}/libnor")
set(PROJECT_COMMON_DIR "${PROJECT_SRC_DIR}/common")
set(PROJECT_PYNOR_DIR "${PROJECT_SRC_DIR}/nor")
set(PROJECT_GAMES_DIR "${PROJECT_SRC_DIR}/games")
set(PROJECT_NOR_SRC_DIR "${PROJECT_NOR_DIR}/impl")
set(PROJECT_NOR_INCLUDE_DIR "${PROJECT_NOR_DIR}/include")
set(PROJECT_COMMON_INCLUDE_DIR "${PROJECT_SRC_DIR}/common/include")
set(PROJECT_NOR_BINDING_DIR "${PROJECT_PYNOR_DIR}/binding")
set(_cmake_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(_docs_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/docs/_build)
set(_docs_DOXYGEN_WORKING_DIR ${PROJECT_NOR_INCLUDE_DIR})  # has to be relative to CMAKE_CURRENT_SOURCE_DIR

message(STATUS "NOR project files directory: ${PROJECT_SRC_DIR}")
message(STATUS "NOR src directory: ${PROJECT_NOR_SRC_DIR}")
message(STATUS "NOR include directory: ${PROJECT_NOR_INCLUDE_DIR}")
message(STATUS "NOR bindings directory: ${PROJECT_NOR_BINDING_DIR}")

message(STATUS "NOR c++ library name: ${nor_lib}")
message(STATUS "NOR python extension name: ${nor_pymodule}")
message(STATUS "BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")

message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ Compiler Info: ${CMAKE_CXX_COMPILER_ID}, version ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CURRENT CMAKE BINARY DIR: ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "settings print end.\n")

# Set this path if the conan install is not in the standard directory
# This doesn't work due to a conan internal error when conan is not found in the standard directory (/usr/bin on linux)
# It doesn't work in general when CONAN_COMMAND is set it seems.
#execute_process(COMMAND which conan OUTPUT_VARIABLE CONAN_PATH)

# Link this 'library' to set the c++ standard / compile-time options requested
add_library(project_options INTERFACE)
# Link this 'library' to use the warnings specified in CompilerWarnings.cmake
add_library(project_warnings INTERFACE)

target_compile_features(project_options
        INTERFACE
        ${PROJ_CXX_STD_FEATURE}
        )
if (CMAKE_CXX_COMPILER_ID MATCHES ".*Clang" AND ENABLE_BUILD_WITH_TIME_TRACE)
    add_compile_definitions(project_options INTERFACE -ftime-trace)
endif ()
# string APPEND is needed for options in order to concatenate these strings AS IS,
# i.e. avoid the semicolon between every code line
string(
        APPEND _options
        "$<$<CXX_COMPILER_ID:MSVC>:"
        "/permissive-;"
        "$<$<CONFIG:RELEASE>:/O2>"
        ">"
        "$<$<CXX_COMPILER_ID:Clang>:"
        "$<$<CONFIG:RELEASE>:-O3>"
        "$<$<PLATFORM_ID:Darwin>:-stdlib=libc++>"
        ">"
        "$<$<CXX_COMPILER_ID:GNU>:"
        "$<$<CONFIG:RELEASE>:-O3>"
        ">"
)
set(
        _extra_options
        "-ftemplate-backtrace-limit=0"
        "$<$<CXX_COMPILER_ID:GNU>:-fconcepts-diagnostics-depth=5>"
)
target_compile_options(
        project_options
        INTERFACE
        $<$<CXX_COMPILER_ID:Clang>:-stdlib=libstdc++>
        "${_options}"
        "${_extra_options}"
)

# Set up some extra Conan dependencies based on our needs
# before loading Conan
set(CONAN_EXTRA_REQUIRES "")
set(CONAN_EXTRA_OPTIONS "")

include(${_cmake_DIR}/settings/Conan.cmake)
run_conan()
include(${PROJECT_BINARY_DIR}/conanbuildinfo.cmake)
include(${CMAKE_BINARY_DIR}/conan_paths.cmake)

# set the standard project settings
include(cmake/settings/StandardProjectSettings.cmake)

# load extra clang dev tools
include(cmake/settings/Clang-cxx-dev-tools.cmake)

# enable cache system
include(${_cmake_DIR}/settings/Cache.cmake)

# standard compiler warnings
include(${_cmake_DIR}/settings/CompilerWarnings.cmake)
set_project_warnings(project_warnings)

# sanitizer options if supported by compiler
include(${_cmake_DIR}/settings/Sanitizers.cmake)
enable_sanitizers(project_options)

# enable doxygen
if (ENABLE_BUILD_DOCS)
    include(${_cmake_DIR}/settings/Doxygen.cmake)
endif ()

# allow for static analysis options
include(${_cmake_DIR}/settings/StaticAnalyzers.cmake)

if (ENABLE_PCH)
    target_precompile_headers(project_options
            INTERFACE <vector> <string> <map> <utility> <memory> <array>)
endif ()

if (USE_PYBIND11_FINDPYTHON)
    set(PYBIND11_FINDPYTHON TRUE)
else ()
    set(PYBIND11_FINDPYTHON FALSE)
    find_package(Python3 COMPONENTS Interpreter Development)
endif ()
find_package(pybind11 REQUIRED)
find_package(range-v3 REQUIRED)

include(${_cmake_DIR}/targets/nor.cmake)
if (ENABLE_BUILD_PYTHON_EXTENSION)
    message(STATUS "Configuring Python Extension ${nor_pymodule}.")
    include(${_cmake_DIR}/targets/pynor.cmake)
endif ()
if (ENABLE_GAMES)
    message(STATUS "Configuring Game Builds.")
    include(${_cmake_DIR}/targets/games.cmake)
endif ()
if (ENABLE_BUILD_SANDBOX)
    message(STATUS "Configuring Sandbox Build.")
    include(${_cmake_DIR}/targets/sandbox.cmake)
endif ()

# use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" isSystemDir)
if ("${isSystemDir}" STREQUAL "-1")
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
endif ("${isSystemDir}" STREQUAL "-1")


if (SKBUILD)
    message("Building with scikit-build. Configuring only for python build.")
    if (APPLE)
        set(rpath_orig "'@executable_path'")
    else ()
        set(rpath_orig "'$ORIGIN'")
    endif ()
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath=$ORIGIN")
    # install locally so that sciki-build can correctly install it
    # The destination is relative to the path given to skbuild in
    # the cmake_install_dir parameter within setup in setup.py
    install(TARGETS ${nor_pymodule} LIBRARY DESTINATION pynor)
    install(TARGETS ${nor_lib} LIBRARY DESTINATION pynor)
else ()
    if (ENABLE_TESTING)
        find_package("GTest")
        enable_testing()
        message(STATUS "Configuring Tests.")
        include(${_cmake_DIR}/targets/tests.cmake)
    endif ()

    #
    # Install pkg-config file
    #

    set(NOR_PKGCONFIG ${CMAKE_CURRENT_BINARY_DIR}/nor.pc)

    configure_file(
            ${_cmake_DIR}/in/nor.pc.in
            ${NOR_PKGCONFIG}
            @ONLY
    )

    install(
            FILES ${NOR_PKGCONFIG}
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
    )

    write_basic_package_version_file(
            "${PROJECT_BINARY_DIR}/${PROJECT_NAME_LOWERCASE}ConfigVersion.cmake"
            VERSION ${PROJECT_VERSION}
            COMPATIBILITY AnyNewerVersion
    )

    configure_package_config_file(
            "${_cmake_DIR}/in/${PROJECT_NAME_LOWERCASE}Config.cmake.in"
            "${PROJECT_BINARY_DIR}/${PROJECT_NAME_LOWERCASE}Config.cmake"
            INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME_LOWERCASE}/cmake/
    )

    # CMake requires all interface dependencies of our library targets, including options and warnings, to be exported.
    install(
            TARGETS project_options
            EXPORT ${PROJECT_NAME_LOWERCASE}Options
    )
    install(
            TARGETS project_warnings
            EXPORT ${PROJECT_NAME_LOWERCASE}Warnings
    )
    install(
            TARGETS ${nor_lib} common
            EXPORT ${PROJECT_NAME_LOWERCASE}Targets
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Runtime
            BUNDLE DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Runtime
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Runtime
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Development
            PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT Development
    )

    if (ENABLE_BUILD_PYTHON_EXTENSION)
        # to use the latest build of the library for a development package install (pip install path/to/project -e),
        # we configure an optional install component of the python extension in-source
        install(
                TARGETS ${nor_pymodule}
                LIBRARY DESTINATION ${PROJECT_PYNOR_DIR}
                COMPONENT PyExtension_inplace
                OPTIONAL
        )
        # this target can be called conveniently from within IDEs to replace the current build of a development install
        add_custom_target(
                install_${nor_pymodule}_insource
                COMMAND ${CMAKE_COMMAND} --install . --component PyExtension_inplace
                DEPENDS ${nor_pymodule}
        )
    endif ()


    install(
            EXPORT ${PROJECT_NAME_LOWERCASE}Options
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME_LOWERCASE}/cmake/
            NAMESPACE ${PROJECT_NAME_LOWERCASE}::
    )
    install(
            EXPORT ${PROJECT_NAME_LOWERCASE}Warnings
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME_LOWERCASE}/cmake/
            NAMESPACE ${PROJECT_NAME_LOWERCASE}::
    )
    install(
            EXPORT ${PROJECT_NAME_LOWERCASE}Targets
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME_LOWERCASE}/cmake/
            NAMESPACE ${PROJECT_NAME_LOWERCASE}::
    )

    install(
            FILES
            "${PROJECT_BINARY_DIR}/${PROJECT_NAME_LOWERCASE}ConfigVersion.cmake"
            "${PROJECT_BINARY_DIR}/${PROJECT_NAME_LOWERCASE}Config.cmake"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME_LOWERCASE}/cmake/
    )
    # this installation assumes that the project has an eponymous include directory for the project c++ library
    # we install that sub-directory instead of PROJECT_nor_INCLUDE_DIR, in order to avoid an
    # include/include/proj_name situation and get the correct include/proj_name.
    install(DIRECTORY ${PROJECT_NOR_INCLUDE_DIR}/${PROJECT_NAME_LOWERCASE} DESTINATION include)

    export(PACKAGE nor)

endif ()
