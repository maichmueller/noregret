
#ifndef NOR_PY_TRAMPOLINE_INFOSTATE_HPP
#define NOR_PY_TRAMPOLINE_INFOSTATE_HPP

#include <pybind11/pybind11.h>

#include "nor/env/polymorphic_env.hpp"

namespace nor::py {

/* Trampoline Class */
struct PyInfostate: public nor::env::polymorph::Infostate {
   /* Inherit the constructors */
   using Infostate::Infostate;

   size_t hash() const override
   {
      PYBIND11_OVERRIDE_NAME(
         size_t, /* Return type */
         Infostate, /* Parent class */
         "__hash__", /* Name of function in Python */
         hash, /* Name of function in C++ */
         /* Argument(s) */
      );
   }

   bool operator==(const Infostate& other) const override
   {
      PYBIND11_OVERRIDE_NAME(
         bool, /* Return type */
         Infostate, /* Parent class */
         "__eq__", /* Name of function in Python */
         operator==, /* Name of function in C++ */
         other /* Argument(s) */
      );
   }

   size_t size() const override
   {
      PYBIND11_OVERRIDE_NAME(
         size_t, /* Return type */
         Infostate, /* Parent class */
         "__len__", /* Name of function in Python */
         size, /* Name of function in C++ */
         /* Argument(s) */
      );
   }

   const nor::env::polymorph::Observation& operator[](size_t arg) const override
   {
      PYBIND11_OVERRIDE_NAME(
         const nor::env::polymorph::Observation&, /* Return type */
         Infostate, /* Parent class */
         "__getitem__", /* Name of function in Python */
         operator[], /* Name of function in C++ */
         arg /* Argument(s) */
      );
   }

   Infostate& append(const nor::env::polymorph::Observation& obs) override
   {
      PYBIND11_OVERRIDE_NAME(
         Infostate&, /* Return type */
         Infostate, /* Parent class */
         "append", /* Name of function in Python */
         append, /* Name of function in C++ */
         obs /* Arguments */
      );
   }

   nor::Player player() const override
   {
      PYBIND11_OVERRIDE_NAME(
         nor::Player, /* Return type */
         Infostate, /* Parent class */
         "player", /* Name of function in Python */
         player, /* Name of function in C++ */
      );
   }
};

}  // namespace nor::py

namespace std {

template <>
struct hash< nor::env::polymorph::Infostate > {
   size_t operator()(const nor::env::polymorph::Infostate& infostate) const
   {
      return infostate.hash();
   }
};

}  // namespace std

#endif  // NOR_PY_TRAMPOLINE_INFOSTATE_HPP
