# Install the Python package into a clean prefix and copy the pure-Python package sources beside the installed
# extension. This deliberately uses CMake's install rules, including the wheel install RPATH, instead of the build-tree
# staging helper.

if(NOT DEFINED CMAKE_COMMAND
   OR NOT DEFINED BUILD_DIR
   OR NOT DEFINED PACKAGE_SOURCE_DIR
   OR NOT DEFINED DESTINATION)
    message(
        FATAL_ERROR "install_python_package.cmake requires CMAKE_COMMAND, BUILD_DIR, PACKAGE_SOURCE_DIR and DESTINATION"
    )
endif()

file(REMOVE_RECURSE "${DESTINATION}")

set(_install_command "${CMAKE_COMMAND}" --install "${BUILD_DIR}" --prefix "${DESTINATION}")
if(DEFINED CONFIGURATION
   AND NOT
       CONFIGURATION
       STREQUAL
       "")
    list(
        APPEND
        _install_command
        --config
        "${CONFIGURATION}")
endif()
execute_process(
    COMMAND ${_install_command}
    RESULT_VARIABLE _install_result
    OUTPUT_VARIABLE _install_output
    ERROR_VARIABLE _install_error)
if(NOT
   _install_result
   EQUAL
   0)
    message(FATAL_ERROR "package install failed (${_install_result}): ${_install_error}")
endif()

file(
    GLOB
    _package_python_sources
    "${PACKAGE_SOURCE_DIR}/*.py"
    "${PACKAGE_SOURCE_DIR}/*.pyi")
if(NOT _package_python_sources)
    message(FATAL_ERROR "no Python package sources found in ${PACKAGE_SOURCE_DIR}")
endif()
file(MAKE_DIRECTORY "${DESTINATION}/nor")
file(COPY ${_package_python_sources} DESTINATION "${DESTINATION}/nor")

file(GLOB _installed_extensions "${DESTINATION}/nor/_noregret*")
if(NOT _installed_extensions)
    message(FATAL_ERROR "the install did not produce the _noregret extension")
endif()

message(STATUS "installed and assembled the relocatable nor package in ${DESTINATION}")
