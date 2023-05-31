#ifndef NOR_PY_TRAMPOLINES_OBSERVATION_HPP
#define NOR_PY_TRAMPOLINES_OBSERVATION_HPP

#include "nor/env/polymorphic_env.hpp"
namespace nor::py {

/* Trampoline Class */
struct PyObservation: public nor::env::polymorph::Observation {
   /* Inherit the constructors */
   using Observation::Observation;

   size_t hash() const override
   {
      PYBIND11_OVERRIDE_NAME(
         size_t, /* Return type */
         Observation, /* Parent class */
         "__hash__", /* Name of function in Python */
         hash, /* Name of function in C++ */
         /* Argument(s) */
      );
   }

   bool operator==(const Observation& other) const override
   {
      PYBIND11_OVERRIDE_NAME(
         bool, /* Return type */
         Observation, /* Parent class */
         "__eq__", /* Name of function in Python */
         operator==, /* Name of function in C++ */
         other /* Argument(s) */
      );
   }
};

}  // namespace nor::py

namespace std {

template <>
struct hash< nor::env::polymorph::Observation > {
   size_t operator()(const nor::env::polymorph::Observation& observation) const
   {
      return observation.hash();
   }
};

}  // namespace std

#endif  // NOR_PY_TRAMPOLINES_OBSERVATION_HPP
