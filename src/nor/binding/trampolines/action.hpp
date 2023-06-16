
#ifndef NOR_PY_TRAMPOLINES_ACTION_HPP
#define NOR_PY_TRAMPOLINES_ACTION_HPP

#include "nor/env/polymorphic.hpp"

namespace nor::binding {

/* Trampoline Class */
struct PyAction: public nor::games::polymorph::Action {
   /* Inherit the constructors */
   using Action::Action;

   size_t hash() const override
   {
      PYBIND11_OVERRIDE_NAME(
         size_t, /* Return type */
         Action, /* Parent class */
         "__hash__", /* Name of function in Python */
         hash, /* Name of function in C++ */
         /* Argument(s) */
      );
   }

   bool operator==(const Action& other) const override
   {
      PYBIND11_OVERRIDE_NAME(
         bool, /* Return type */
         Action, /* Parent class */
         "__eq__", /* Name of function in Python */
         operator==, /* Name of function in C++ */
         other /* Argument(s) */
      );
   }
};

}  // namespace nor::binding

namespace std {

template <>
struct hash< nor::games::polymorph::Action > {
   size_t operator()(const nor::games::polymorph::Action& action) const { return action.hash(); }
};

}  // namespace std

#endif  // NOR_PY_TRAMPOLINES_ACTION_HPP
