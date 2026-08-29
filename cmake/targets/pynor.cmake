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
target_link_libraries(${nor_pymodule} PUBLIC ${nor_lib}_envs)
if(TARGET nor_binding_runtime)
    target_link_libraries(${nor_pymodule} PUBLIC nor_binding_runtime)
endif()
