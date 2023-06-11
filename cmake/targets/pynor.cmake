set(PYTHON_MODULE_SOURCES module.cpp)
list(TRANSFORM PYTHON_MODULE_SOURCES PREPEND "${PROJECT_NOR_BINDING_DIR}/")

if(SKBUILD)
    set(_pynor_exclude_from_all)
else()
    set(_pynor_exclude_from_all EXCLUDE_FROM_ALL)
endif()

pybind11_add_module(
    ${nor_pymodule}
    MODULE
    ${_pynor_exclude_from_all}
    ${LIBRARY_SOURCES}
    ${PYTHON_MODULE_SOURCES})

set_target_properties(${nor_pymodule} PROPERTIES LIBRARY_OUTPUT_NAME _${nor_pymodule} CXX_VISIBILITY_PRESET hidden)
target_link_libraries(${nor_pymodule} PUBLIC ${nor_lib}_envs pybind11::module
                                             $<$<NOT:$<BOOL:USE_PYBIND11_FINDPYTHON>>:Python3::Module>)
