

#ifndef NOR_MODULE_NAME_HPP
#define NOR_MODULE_NAME_HPP

#include <pybind11/pybind11.h>

#include "trampolines/action.hpp"
#include "trampolines/chance_outcome.hpp"
#include "trampolines/infostate.hpp"
#include "trampolines/observation.hpp"
#include "trampolines/publicstate.hpp"
#include "trampolines/worldstate.hpp"

namespace py = pybind11;

PYBIND11_MODULE(noregret, m) {}

#endif  // NOR_MODULE_NAME_HPP
