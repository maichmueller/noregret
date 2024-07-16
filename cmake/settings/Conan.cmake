macro(run_conan)
    # Download automatically
    set(CONAN_VERSION 0.18.1)
    if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
        message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
        file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/${CONAN_VERSION}/conan.cmake"
             "${CMAKE_BINARY_DIR}/conan.cmake" TLS_VERIFY ON)
    endif()

    include(${CMAKE_BINARY_DIR}/conan.cmake)

    find_program(CONAN conan REQUIRED PATHS ${CONAN_PATH})

    conan_cmake_run(
        CONANFILE
        ${DEPENDENCY_DIR}/${CONANFILE}
        CONAN_COMMAND
        ${CONAN}
        ${CONAN_EXTRA_REQUIRES}
        OPTIONS
        ${CONAN_EXTRA_OPTIONS}
        BASIC_SETUP
        CMAKE_TARGETS # individual targets to link to
        KEEP_RPATHS
        BUILD
        missing
        PROFILE
        default
        PROFILE_AUTO
        ALL # ALL means that all the settings are taken from CMake's detection
    )

    include(${PROJECT_BINARY_DIR}/conanbuildinfo.cmake)
    include(${CMAKE_BINARY_DIR}/conan_paths.cmake)
endmacro()

if(DEFINED CONAN_PATH)
    message("Explicit Conan path specified by user: ${CONAN_PATH}. Using `find_program` searching ony in this path.")
    find_program(
        CONAN_CMD conan REQUIRED
        PATHS ${CONAN_PATH}
        NO_DEFAULT_PATH)
else()
    message("NO explicit Conan path specified by user. Using `find_program` with default settings.")
    find_program(CONAN_CMD conan REQUIRED)
endif()
execute_process(COMMAND ${CONAN_CMD} "--version" OUTPUT_VARIABLE _CONAN_VERSION_OUTPUT)
message(STATUS "Found conan: ${CONAN_CMD} - ${_CONAN_VERSION_OUTPUT}")
string(
    REGEX MATCH
          ".*Conan version ([0-9]+\\.[0-9]+\\.[0-9]+)"
          FOO
          "${_CONAN_VERSION_OUTPUT}")
if(${CMAKE_MATCH_1} VERSION_LESS "2.0.0")
    run_conan()
endif()
