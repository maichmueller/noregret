

#ifndef NOR_PY_TRAMPOLINES_CHANCE_OUTCOME_HPP
#define NOR_PY_TRAMPOLINES_CHANCE_OUTCOME_HPP

#include <pybind11/pybind11.h>

#include "nor/env/polymorphic_env.hpp"

namespace nor::py {

/* Trampoline Class */
struct PyChanceOutcome: public nor::games::polymorph::ChanceOutcome {
   /* Inherit the constructors */
   using ChanceOutcome::ChanceOutcome;

   size_t hash() const override
   {
      PYBIND11_OVERRIDE_NAME(
         size_t, /* Return type */
         ChanceOutcome, /* Parent class */
         "__hash__", /* Name of function in Python */
         hash, /* Name of function in C++ */
         /* Argument(s) */
      );
   }

   bool operator==(const ChanceOutcome& other) const override
   {
      PYBIND11_OVERRIDE_NAME(
         bool, /* Return type */
         ChanceOutcome, /* Parent class */
         "__eq__", /* Name of function in Python */
         operator==, /* Name of function in C++ */
         other /* Argument(s) */
      );
   }
};

}  // namespace nor::py

namespace std {

template <>
struct hash< nor::py::PyChanceOutcome > {
   size_t operator()(const nor::py::PyChanceOutcome& chance_outcome) const
   {
      return chance_outcome.hash();
   }
};

}  // namespace std

#endif  // NOR_PY_TRAMPOLINES_CHANCE_OUTCOME_HPP
