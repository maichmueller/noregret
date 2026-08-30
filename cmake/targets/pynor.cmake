set(PYTHON_MODULE_SOURCES module.cpp)
list(TRANSFORM PYTHON_MODULE_SOURCES PREPEND "${PROJECT_NOR_BINDING_DIR}/")

if(ENABLE_BUILD_PYTHON_EXTENSION AND INSTALL_PYMODULE)
    set(_pynor_exclude_from_all)
else()
    set(_pynor_exclude_from_all EXCLUDE_FROM_ALL)
endif()

nanobind_add_module(${nor_pymodule} ${LIBRARY_SOURCES} ${PYTHON_MODULE_SOURCES})

if(_pynor_exclude_from_all)
    set_target_properties(${nor_pymodule} PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

set_target_properties(${nor_pymodule} PROPERTIES LIBRARY_OUTPUT_NAME _${nor_pymodule} CXX_VISIBILITY_PRESET hidden)

# The project compiles with a strict warning set that third-party headers were never written against. Nanobind's and
# CPython's headers arrive through link interfaces, so mark those interfaces as system includes at the source rather
# than filtering the diagnostics afterwards; otherwise a single translation unit buries its own warnings under thousands
# of foreign ones.
foreach(_pynor_external_target nanobind-static Python::Module)
    if(TARGET ${_pynor_external_target})
        get_target_property(_pynor_external_includes ${_pynor_external_target} INTERFACE_INCLUDE_DIRECTORIES)
        if(_pynor_external_includes)
            set_target_properties(${_pynor_external_target} PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                                                                       "${_pynor_external_includes}")
        endif()
    endif()
endforeach()
unset(_pynor_external_target)
unset(_pynor_external_includes)
target_link_libraries(${nor_pymodule} PUBLIC ${nor_lib}_envs)
if(TARGET nor_binding_runtime)
    target_link_libraries(${nor_pymodule} PUBLIC nor_binding_runtime)
endif()

# ######################################################################################################################
# Python binding tests
# ######################################################################################################################

# The suite is plain unittest so it needs no dependency beyond the interpreter that already has to be present to build
# the extension at all. Build the ${nor_pymodule} target before running it.
if(ENABLE_TESTING)
    set(_pynor_test_dir "${PROJECT_TEST_DIR}/python")
    set(_pynor_package_dir "${CMAKE_BINARY_DIR}/python_package")

    # The wheel carries the game libraries next to the extension, so the staged layout does too.
    set(_pynor_runtime_libraries)
    foreach(_pynor_game IN LISTS NOR_GAME_LIBRARIES)
        if(TARGET ${_pynor_game})
            get_target_property(_pynor_game_type ${_pynor_game} TYPE)
            if(_pynor_game_type STREQUAL "SHARED_LIBRARY" OR _pynor_game_type STREQUAL "MODULE_LIBRARY")
                list(APPEND _pynor_runtime_libraries $<TARGET_FILE:${_pynor_game}>)
            endif()
        endif()
    endforeach()
    string(
        REPLACE ";"
                "\\;"
                _pynor_runtime_libraries_arg
                "${_pynor_runtime_libraries}")

    add_test(
        NAME Test_python_package_stage
        COMMAND
            ${CMAKE_COMMAND} -DEXTENSION=$<TARGET_FILE:${nor_pymodule}> -DPACKAGE_SOURCE_DIR=${PROJECT_PYNOR_DIR}
            -DRUNTIME_LIBRARIES=${_pynor_runtime_libraries_arg} -DDESTINATION=${_pynor_package_dir} -P
            ${_cmake_DIR}/scripts/stage_python_package.cmake)
    set_tests_properties(Test_python_package_stage PROPERTIES FIXTURES_SETUP python_package)

    set(_pynor_install_prefix "${CMAKE_BINARY_DIR}/python_package_install")
    add_test(
        NAME Test_python_package_install_stage
        COMMAND
            ${CMAKE_COMMAND} -DCMAKE_COMMAND=${CMAKE_COMMAND} -DBUILD_DIR=${CMAKE_BINARY_DIR}
            -DPACKAGE_SOURCE_DIR=${PROJECT_PYNOR_DIR} -DDESTINATION=${_pynor_install_prefix}
            -DCONFIGURATION=${CMAKE_BUILD_TYPE} -P ${_cmake_DIR}/scripts/install_python_package.cmake)
    set_tests_properties(Test_python_package_install_stage PROPERTIES FIXTURES_SETUP python_package_install)

    add_test(
        NAME Test_python_package_relocated
        COMMAND ${Python_EXECUTABLE} -m unittest test_packaging.RelocatedInstalledPackageTest --verbose
        WORKING_DIRECTORY ${_pynor_test_dir})
    set_tests_properties(
        Test_python_package_relocated
        PROPERTIES
            FIXTURES_REQUIRED
            python_package_install
            ENVIRONMENT
            "PYTHONPATH=${_pynor_test_dir};NOR_INSTALLED_PACKAGE_ROOT=${_pynor_install_prefix};PYTHON_EXECUTABLE=${Python_EXECUTABLE}"
    )

    add_test(
        NAME Test_python_bindings
        COMMAND ${Python_EXECUTABLE} -m unittest discover --start-directory ${_pynor_test_dir} --top-level-directory
                ${_pynor_test_dir} --verbose
        WORKING_DIRECTORY ${_pynor_test_dir})
    set_tests_properties(
        Test_python_bindings
        PROPERTIES
            FIXTURES_REQUIRED
            python_package
            ENVIRONMENT
            "PYTHONPATH=$<TARGET_FILE_DIR:${nor_pymodule}>;NOR_PACKAGE_ROOT=${_pynor_package_dir};PYTHON_EXECUTABLE=${Python_EXECUTABLE}"
    )

    unset(_pynor_test_dir)
    unset(_pynor_package_dir)
    unset(_pynor_install_prefix)
    unset(_pynor_game)
    unset(_pynor_game_type)
    unset(_pynor_runtime_libraries)
    unset(_pynor_runtime_libraries_arg)
endif()
