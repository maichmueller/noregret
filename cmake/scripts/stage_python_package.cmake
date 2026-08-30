# Assembles the shipped `nor` package layout out of the build tree.
#
# The installed wheel is the package directory, the compiled extension, and the game libraries the extension links, all
# side by side. Staging those exercises the layout `import nor` depends on without rebuilding through pip.
#
# What this does NOT prove is relocatability: the staged extension is the build-tree binary and still carries its build
# RPATH, so it would resolve its siblings even if they were absent here. That property belongs to a real install.

if(NOT DEFINED EXTENSION
   OR NOT DEFINED PACKAGE_SOURCE_DIR
   OR NOT DEFINED DESTINATION)
    message(FATAL_ERROR "stage_python_package.cmake requires EXTENSION, PACKAGE_SOURCE_DIR and DESTINATION")
endif()

file(REMOVE_RECURSE "${DESTINATION}")
file(MAKE_DIRECTORY "${DESTINATION}/nor")

file(
    GLOB
    _package_python_sources
    "${PACKAGE_SOURCE_DIR}/*.py"
    "${PACKAGE_SOURCE_DIR}/*.pyi")
if(NOT _package_python_sources)
    message(FATAL_ERROR "no python sources found in ${PACKAGE_SOURCE_DIR}")
endif()

file(COPY ${_package_python_sources} DESTINATION "${DESTINATION}/nor")
file(COPY "${EXTENSION}" DESTINATION "${DESTINATION}/nor")

if(DEFINED RUNTIME_LIBRARIES)
    foreach(_runtime_library IN LISTS RUNTIME_LIBRARIES)
        if(EXISTS "${_runtime_library}")
            file(COPY "${_runtime_library}" DESTINATION "${DESTINATION}/nor")
        endif()
    endforeach()
endif()

message(STATUS "staged the nor package into ${DESTINATION}")
