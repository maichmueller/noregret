cmake_minimum_required(VERSION 3.17)

project(
    NOR
    VERSION 0.0.1
    LANGUAGES CXX)

string(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWERCASE)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# options
option(ENABLE_BUILD_DOCS "Enable building the docs. Requires doxygen to be installed on the system" OFF)
option(ENABLE_BUILD_PYTHON_EXTENSION "Enable building the python extension." ON)
option(ENABLE_BUILD_SANDBOX "Enable building of the sandbox testbed (Only for development purposes)." ON)
option(ENABLE_BUILD_BENCHMARK "Enable building of the benchmarks." OFF)
option(ENABLE_BUILD_WITH_TIME_TRACE "Enable -ftime-trace to generate time tracing .json files on clang" OFF)
option(ENABLE_CACHE "Enable cache if available" ON)
option(ENABLE_CLANG_TIDY "Enable static analysis with clang-tidy" OFF)
option(ENABLE_COVERAGE "Enable coverage reporting for gcc/clang" OFF)
option(ENABLE_CPPCHECK "Enable static analysis with cppcheck" OFF)
option(ENABLE_FAST_MATH "Enable Compiler specific breakway from IEEE compliance to speed up math computations" OFF)
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
option(ENABLE_WARNING_PADDING "Warn when structs are memory padded" OFF)
option(USE_PYBIND11_FINDPYTHON "Use pybind11 to search for the python library" OFF)
option(INSTALL_PYMODULE "Configure installation for python module." OFF)
option(WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

# settings

# Use C++20
set(PROJ_CXX_STD_FEATURE cxx_std_20)
set(PROJ_CXX_STANDARD C++20)
if(MSVC
   AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 19.29.30129
   AND CMAKE_VERSION VERSION_GREATER 3.20.3)
    # this change happened in CMake 3.20.4
    set(PROJ_CXX_STD_FEATURE cxx_std_23) # /std:c++latest - unlocks the non stable cpp20 features. For new 16.11
    # versions
else()
    set(PROJ_CXX_STD_FEATURE cxx_std_20) # /std:c++latest for msvc and -std=c++20 for everyone else.
endif()

set(nor-lib-type INTERFACE)
set(nor_lib nor++)
set(nor_pymodule noregret)
set(nor_test nor_tests)
set(nor_benchmark nor_benchmark)

set(CONANFILE conanfile.txt)
set(DEPENDENCY_DIR deps) # has to be relative to CMAKE_CURRENT_SOURCE_DIR
set(PROJECT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(PROJECT_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(PROJECT_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test")

set(PROJECT_NOR_DIR "${PROJECT_SRC_DIR}/libnor")
set(PROJECT_COMMON_DIR "${PROJECT_SRC_DIR}/common")
set(PROJECT_PYNOR_DIR "${PROJECT_SRC_DIR}/nor")
set(PROJECT_GAMES_DIR "${PROJECT_SRC_DIR}/games")
set(PROJECT_BENCHMARK_DIR "${PROJECT_DIR}/benchmark")

set(PROJECT_NOR_SRC_DIR "${PROJECT_NOR_DIR}/impl")
set(PROJECT_NOR_INCLUDE_DIR "${PROJECT_NOR_DIR}/include")
set(PROJECT_COMMON_INCLUDE_DIR "${PROJECT_SRC_DIR}/common/include")
set(PROJECT_NOR_BINDING_DIR "${PROJECT_PYNOR_DIR}/binding")
set(PROJECT_NOR_BENCHMARK_SRC_DIR "${PROJECT_BENCHMARK_DIR}")

set(_cmake_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(_docs_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/docs/_build)
set(_docs_DOXYGEN_WORKING_DIR ${PROJECT_NOR_INCLUDE_DIR}) # has to be relative to CMAKE_CURRENT_SOURCE_DIR

message(STATUS "SETTINGS PRINT start:")

message("CMake Generator: ${CMAKE_GENERATOR}")
message("C++ Compiler: ${CMAKE_CXX_COMPILER}")
message("C++ Compiler Info: ${CMAKE_CXX_COMPILER_ID}, version ${CMAKE_CXX_COMPILER_VERSION}")
message("CURRENT CMAKE BINARY DIR: ${CMAKE_CURRENT_BINARY_DIR}")

message("NOR project files directory: ${PROJECT_SRC_DIR}")
message("NOR src directory: ${PROJECT_NOR_SRC_DIR}")
message("NOR include directory: ${PROJECT_NOR_INCLUDE_DIR}")
message("NOR bindings directory: ${PROJECT_NOR_BINDING_DIR}")

message("NOR c++ library name: ${nor_lib}")
message("NOR python extension name: ${nor_pymodule}")
message("BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
message("FAST MATH SWITCH: ${ENABLE_FAST_MATH}")

message(STATUS "SETTINGS PRINT end.\n")

# Set this path if the conan install is not in the standard directory This doesn't work due to a conan internal error
# when conan is not found in the standard directory (/usr/bin on linux) It doesn't work in general when CONAN_COMMAND is
# set it seems. execute_process(COMMAND which conan OUTPUT_VARIABLE CONAN_PATH)

# Link this 'library' to set the c++ standard / compile-time options requested
add_library(project_options INTERFACE)
# Link this 'library' to use the warnings specified in CompilerWarnings.cmake
add_library(project_warnings INTERFACE)

target_compile_features(project_options INTERFACE ${PROJ_CXX_STD_FEATURE})
if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang" AND ENABLE_BUILD_WITH_TIME_TRACE)
    add_compile_definitions(project_options INTERFACE -ftime-trace)
endif()
# string APPEND is needed for options in order to concatenate these strings AS IS, i.e. avoid the semicolon between
# every code line

set(_options
    "$<$<CXX_COMPILER_ID:MSVC>:"
    "/permissive-;"
    "$<$<CONFIG:RELEASE>:/O2>"
    ">"
    "$<$<CXX_COMPILER_ID:Clang>:"
    "$<$<CONFIG:RELEASE>:-O3>"
    "$<$<PLATFORM_ID:Darwin>:-stdlib=libc++>"
    "$<$<AND:$<CONFIG:RELEASE>,$<BOOL:${ENABLE_FAST_MATH}>>:-ffast-math>"
    ">"
    "$<$<CXX_COMPILER_ID:GNU>:"
    "$<$<CONFIG:RELEASE>:-O3>"
    "$<$<AND:$<CONFIG:RELEASE>,$<BOOL:${ENABLE_FAST_MATH}>>:-ffast-math>"
    ">")
set(_extra_options "-ftemplate-backtrace-limit=0" "$<$<CXX_COMPILER_ID:GNU>:-fconcepts-diagnostics-depth=5>")
target_compile_options(project_options INTERFACE "${_options}" "${_extra_options}")

# Set up some extra Conan dependencies based on our needs before loading Conan
set(CONAN_EXTRA_REQUIRES "")
set(CONAN_EXTRA_OPTIONS "")

# the deps to be downloaded by cpm
include(${_cmake_DIR}/settings/CPM.cmake)
# let cpm try to find other deps locally first (find_package) to allow interplay with conan
set(CPM_USE_LOCAL_PACKAGES TRUE)

include("${_cmake_DIR}/settings/Utilities.cmake")
# define the conan settings and optional conan1 run if necessary
include(${_cmake_DIR}/settings/Conan.cmake)

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
if(ENABLE_BUILD_DOCS)
    include(${_cmake_DIR}/settings/Doxygen.cmake)
endif()

# allow for static analysis options
include(${_cmake_DIR}/settings/StaticAnalyzers.cmake)

if(USE_PYBIND11_FINDPYTHON)
    set(PYBIND11_FINDPYTHON TRUE)
else()
    set(PYBIND11_FINDPYTHON FALSE)
    find_package(Python3 COMPONENTS Interpreter Development)
endif()
find_package(pybind11 REQUIRED)
find_package(range-v3 REQUIRED)
find_package(namedtype REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(unordered_dense REQUIRED)
include(${DEPENDENCY_DIR}/cpm_deps.cmake)

add_library(required_min_libs INTERFACE)
target_link_libraries(
    required_min_libs
    INTERFACE project_options
              project_warnings
              fmt::fmt-header-only
              spdlog::spdlog
              unordered_dense::unordered_dense)
target_compile_definitions(
    required_min_libs
    INTERFACE # turn off logging (except info level and above) in release build, allow debug-level logging in debug
              # build
              SPDLOG_ACTIVE_LEVEL=$<$<CONFIG:RELEASE>:SPDLOG_LEVEL_INFO>$<$<CONFIG:DEBUG>:SPDLOG_LEVEL_DEBUG>)

include(${_cmake_DIR}/targets/shared_libs.cmake)
include(${_cmake_DIR}/targets/nor.cmake)
if(ENABLE_TESTING)
    find_package(GTest REQUIRED)
    enable_testing()
    message("Configuring Tests.")
    include(${_cmake_DIR}/targets/tests.cmake)
endif()
if(ENABLE_BUILD_PYTHON_EXTENSION)
    message("Configuring Python Extension ${nor_pymodule}.")
    include(${_cmake_DIR}/targets/pynor.cmake)
endif()
if(ENABLE_GAMES)
    message("Configuring Game Builds.")
    include(${_cmake_DIR}/targets/games.cmake)
endif()
if(ENABLE_BUILD_BENCHMARK)
    message("Configuring Benchmark Build.")
    include(${_cmake_DIR}/targets/benchmark.cmake)
endif()
if(ENABLE_BUILD_SANDBOX)
    message("Configuring Sandbox Build.")
    include(${_cmake_DIR}/targets/sandbox.cmake)
endif()

include(${_cmake_DIR}/settings/Installation.cmake)

if(ENABLE_PCH)
    target_precompile_headers(
        project_options
        INTERFACE
        <vector>
        <string>
        <map>
        <utility>
        <memory>
        <array>
        <iostream>
        <spdlog/spdlog.h>)
endif()

# This prints a summary of found dependencies
include(FeatureSummary)
feature_summary(WHAT ALL)
