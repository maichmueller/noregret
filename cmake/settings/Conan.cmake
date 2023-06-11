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
endmacro()
