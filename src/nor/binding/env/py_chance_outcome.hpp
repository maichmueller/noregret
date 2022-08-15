#ifndef NOR_PYChanceOutcome_HPP
#define NOR_PYChanceOutcome_HPP

#include <pybind11/pybind11.h>

#include <range/v3/all.hpp>
#include <string>
#include <vector>

#include "nor/fosg_states.hpp"
#include "nor/fosg_traits.hpp"
#include "nor/game_defs.hpp"

namespace py = pybind11;

namespace pynor {

/// the current concept requirement on the c++ side is for the chance outcome to be hashable, so
/// that it can be used in a hash map. This requires us to have a hash function and equality
/// function to check upon collision
struct ChanceOutcome {
   virtual ~ChanceOutcome() = 0;
   virtual size_t __hash__() const = 0;
   virtual bool __eq__(const ChanceOutcome&) const = 0;
};

struct PyChanceOutcome: public ChanceOutcome {
   /* Inherit the constructors */
   using ChanceOutcome::ChanceOutcome;

   /* Trampoline */
   size_t __hash__() const override
   {
      PYBIND11_OVERRIDE_PURE(
         size_t, /* Return type */
         ChanceOutcome, /* Parent class */
         __hash__ /* Name of function in C++ (must match Python name) */
         /* Argument(s) */
      );
   }

   bool __eq__(const ChanceOutcome&) const override
   {
      PYBIND11_OVERRIDE_PURE(
         bool, /* Return type */
         ChanceOutcome, /* Parent class */
         __eq__ /* Name of function in C++ (must match Python name) */
         const ChanceOutcome& /* Argument(s) */
      );
   }
};

}  // namespace pynor

namespace std {

template <>
struct hash< pynor::ChanceOutcome > {
   size_t operator()(const pynor::ChanceOutcome& chance_outcome) const
   {
      return chance_outcome.__hash__();
   }
};

}  // namespace std

#endif  // NOR_PYChanceOutcome_HPP
