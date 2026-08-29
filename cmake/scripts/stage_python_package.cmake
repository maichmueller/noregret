# Assembles the shipped `nor` package layout out of the build tree.
#
# The installed wheel is exactly the package directory plus the compiled extension next to it, so staging those two
# files is enough to exercise the layout `import nor` depends on without rebuilding the project through pip.

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

message(STATUS "staged the nor package into ${DESTINATION}")
