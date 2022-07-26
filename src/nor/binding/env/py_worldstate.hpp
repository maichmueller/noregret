#ifndef NOR_PYWORLDSTATE_HPP
#define NOR_PYWORLDSTATE_HPP

#include <pybind11/pybind11.h>

#include <range/v3/all.hpp>
#include <string>
#include <vector>

#include "nor/fosg_states.hpp"
#include "nor/fosg_traits.hpp"
#include "nor/game_defs.hpp"
#include "py_observation.hpp"

namespace py = pybind11;

namespace pynor {

/// the current concept requirements on the c++ side for a worldstate are:
/// 1. move constructible
/// 2. cloneable

struct Worldstate {
   virtual ~Worldstate() = 0;
   virtual sptr< Worldstate > clone() const = 0;
};

struct PyWorldstate: public Worldstate {
   /* Inherit the constructors */
   using Worldstate::Worldstate;

   /* Trampoline */
   sptr< Worldstate > clone() const override
   {
      PYBIND11_OVERRIDE_PURE(
         sptr< Worldstate >, /* Return type */
         Worldstate, /* Parent class */
         clone /* Name of function in C++ (must match Python name) */
         /* Argument(s) */
      );
   }
};

}  // namespace pynor


#endif  // NOR_PYWORLDSTATE_HPP
