

#ifndef NOR_PYTHON_MODULE_BINDING_HPP
#define NOR_PYTHON_MODULE_BINDING_HPP

#include <pybind11/pybind11.h>

#include "declarations.hpp"
#include "nor/fosg_states.hpp"
#include "trampolines/action.hpp"
#include "trampolines/chance_outcome.hpp"
#include "trampolines/env.hpp"
#include "trampolines/infostate.hpp"
#include "trampolines/observation.hpp"
#include "trampolines/publicstate.hpp"
#include "trampolines/worldstate.hpp"

namespace py = pybind11;

using namespace nor;

PYBIND11_MODULE(_noregret, m)
{
   init_enums(m);
   init_env_polymorph(m);
   init_policy(m);
}

#endif  // NOR_PYTHON_MODULE_BINDING_HPP
