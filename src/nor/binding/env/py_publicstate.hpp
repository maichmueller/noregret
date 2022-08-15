#ifndef NOR_PYPUBLICSTATE_HPP
#define NOR_PYPUBLICSTATE_HPP

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

/// the current concept requirements on the c++ side for a publicstate are:
/// 1. hashable.
/// 2. is sized
/// 3. copy constructible
/// 4. equality comparable
/// 5. append method
/// 6. __getitem__ operator
struct Publicstate {
   virtual ~Publicstate() = 0;
   virtual size_t __hash__() const = 0;
   virtual bool __eq__(const Publicstate&) const = 0;
   virtual size_t __len__() const = 0;
   virtual Observation& __getitem__(size_t) = 0;
   virtual Observation& append(const Observation&) = 0;
};

struct PyPublicstate: public Publicstate {
   /* Inherit the constructors */
   using Publicstate::Publicstate;

   /* Trampoline */
   size_t __hash__() const override
   {
      PYBIND11_OVERRIDE_PURE(
         size_t, /* Return type */
         Publicstate, /* Parent class */
         __hash__ /* Name of function in C++ (must match Python name) */
         /* Argument(s) */
      );
   }

   bool __eq__(const Publicstate&) const override
   {
      PYBIND11_OVERRIDE_PURE(
         bool, /* Return type */
         Publicstate, /* Parent class */
         __eq__ /* Name of function in C++ (must match Python name) */
         const Publicstate& /* Argument(s) */
      );
   }

   size_t __len__() const override
   {
      PYBIND11_OVERRIDE_PURE(
         size_t, /* Return type */
         Publicstate, /* Parent class */
         __len__ /* Name of function in C++ (must match Python name) */
      );
   }

   Observation& __getitem__(size_t index) override
   {
      PYBIND11_OVERRIDE_PURE(
         Observation&, /* Return type */
         Publicstate, /* Parent class */
         __getitem__ /* Name of function in C++ (must match Python name) */
         index
      );
   }

   Observation& append(const Observation& obs) override
   {
      PYBIND11_OVERRIDE_PURE(
         Observation&, /* Return type */
         Publicstate, /* Parent class */
         append /* Name of function in C++ (must match Python name) */
         obs
      );
   }
};

}  // namespace pynor

namespace std {

template <>
struct hash< pynor::Publicstate > {
   size_t operator()(const pynor::Publicstate& publicstate) const { return publicstate.__hash__(); }
};

}  // namespace std

#endif  // NOR_PYPUBLICSTATE_HPP
