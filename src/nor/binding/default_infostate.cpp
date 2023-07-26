
#include <pybind11/pybind11.h>

#include "declarations.hpp"

namespace py = pybind11;

using namespace nor;

void init_default_infostate(py::module_& m)
{
   py::module def_module = py::module::import("default");

   py::class_< DefaultInfostate< py_info_state_type, py_observation_type > >(
      def_module, "DefaultInfostate"
   )
      .def(py::init< Player >());
}
