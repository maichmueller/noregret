#ifndef NOR_PYINFOSTATE_HPP
#define NOR_PYINFOSTATE_HPP

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

/// the current concept requirements on the c++ side for a infostate are:
/// 1. hashable.
/// 2. is sized
/// 3. copy constructible
/// 4. equality comparable
/// 5. append method
/// 6. __getitem__ operator
/// 7. player method
struct Infostate {
   virtual ~Infostate() = 0;
   virtual size_t __hash__() const = 0;
   virtual bool operator==(const Infostate&) const = 0;
   virtual size_t __len__() const = 0;
   virtual Observation& __getitem__(size_t) = 0;
   virtual Observation& append(const Observation&) = 0;
   virtual nor::Player player() const = 0;
};

struct PyInfostate: public Infostate {
   /* Inherit the constructors */
   using Infostate::Infostate;

   /* Trampoline */
   size_t __hash__() const override
   {
      PYBIND11_OVERRIDE_PURE(
         size_t, /* Return type */
         Infostate, /* Parent class */
         __hash__ /* Name of function in C++ (must match Python name) */
      );
   }

   bool operator==(const Infostate& infostate) const override
   {
      PYBIND11_OVERRIDE_PURE_NAME(
         bool, /* Return type */
         Infostate, /* Parent class */
         "__eq__",  // Name of method in Python (name)
         operator==,  // Name of function in C++ (fn)
         infostate /* Argument(s) */
      );
   }

   size_t __len__() const override
   {
      PYBIND11_OVERRIDE_PURE(
         size_t, /* Return type */
         Infostate, /* Parent class */
         __len__ /* Name of function in C++ (must match Python name) */
      );
   }

   Observation& __getitem__(size_t index) override
   {
      PYBIND11_OVERRIDE_PURE(
         Observation&, /* Return type */
         Infostate, /* Parent class */
         __getitem__ /* Name of function in C++ (must match Python name) */
         index
      );
   }

   Observation& append(const Observation& obs) override
   {
      PYBIND11_OVERRIDE_PURE(
         Observation&, /* Return type */
         Infostate, /* Parent class */
         append, /* Name of function in C++ (must match Python name) */
         obs /* Arguments */
      );
   }

   nor::Player player() const override
   {
      PYBIND11_OVERRIDE_PURE(
         nor::Player, /* Return type */
         Infostate, /* Parent class */
         player /* Name of function in C++ (must match Python name) */
      );
   }
};

}  // namespace pynor

namespace std {

template <>
struct hash< pynor::Infostate > {
   size_t operator()(const pynor::Infostate& infostate) const { return infostate.__hash__(); }
};

}  // namespace std

#endif  // NOR_PYINFOSTATE_HPP
