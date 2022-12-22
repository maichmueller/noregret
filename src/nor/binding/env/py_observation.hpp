#ifndef NOR_PYOBSERVATION_HPP
#define NOR_PYOBSERVATION_HPP

#include <pybind11/pybind11.h>

#include <range/v3/all.hpp>
#include <string>
#include <vector>

#include "nor/fosg_states.hpp"
#include "nor/fosg_traits.hpp"
#include "nor/game_defs.hpp"

namespace py = pybind11;

namespace pynor {

/// the current concept requirement on the c++ side is for an observation to be hashable, so that it
/// can be used in a hash map. This requires us to have a hash function and equality function to
/// check upon collision
struct Observation {
   virtual ~Observation() = 0;
   virtual size_t __hash__() const = 0;
   virtual bool __eq__(const Observation&) const = 0;
};

struct PyObservation: public Observation {
   /* Inherit the constructors */
   using Observation::Observation;

   /* Trampoline */
   size_t __hash__() const override
   {
      PYBIND11_OVERRIDE_PURE(
         size_t, /* Return type */
         Observation, /* Parent class */
         __hash__ /* Name of function in C++ (must match Python name) */
         /* Argument(s) */
      );
   }

   bool __eq__(const Observation&) const override
   {
      PYBIND11_OVERRIDE_PURE(
         bool, /* Return type */
         Observation, /* Parent class */
         __eq__ /* Name of function in C++ (must match Python name) */
         const Observation& /* Argument(s) */
      );
   }
};

}  // namespace pynor

namespace std {

template <>
struct hash< pynor::Observation > {
   size_t operator()(const pynor::Observation& observation) const { return observation.__hash__(); }
};

}  // namespace std

#endif  // NOR_PYOBSERVATION_HPP