#ifndef NOR_PYACTION_HPP
#define NOR_PYACTION_HPP

#include <pybind11/pybind11.h>

#include <range/v3/all.hpp>
#include <string>
#include <vector>

#include "nor/fosg_states.hpp"
#include "nor/fosg_traits.hpp"
#include "nor/game_defs.hpp"

namespace py = pybind11;

namespace pynor {

/// the current concept requirement on the c++ side is for the action to be hashable, so that it can
/// be used in a hash map. This requires us to have a hash function and equality function to check
/// upon collision
struct Action {
   virtual ~Action() = 0;
   virtual size_t __hash__() const = 0;
   virtual bool __eq__(const Action&) const = 0;
};

struct PyAction: public Action {
   /* Inherit the constructors */
   using Action::Action;

   /* Trampoline */
   size_t __hash__() const override
   {
      PYBIND11_OVERRIDE_PURE(
         size_t, /* Return type */
         Action, /* Parent class */
         __hash__ /* Name of function in C++ (must match Python name) */
         /* Argument(s) */
      );
   }

   bool __eq__(const Action&) const override
   {
      PYBIND11_OVERRIDE_PURE(
         bool, /* Return type */
         Action, /* Parent class */
         __eq__ /* Name of function in C++ (must match Python name) */
         const Action& /* Argument(s) */
      );
   }
};

}  // namespace nor

namespace std {

template <>
struct hash< pynor::Action > {
   size_t operator()(const pynor::Action& action) const { return action.__hash__(); }
};

}  // namespace std

#endif  // NOR_PYACTION_HPP
