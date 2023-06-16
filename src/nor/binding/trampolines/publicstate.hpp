
#ifndef NOR_PY_TRAMPOLINES_PUBLICSTATE_HPP
#define NOR_PY_TRAMPOLINES_PUBLICSTATE_HPP
#include "nor/env/polymorphic.hpp"

namespace nor::binding {

/* Trampoline Class */
struct PyPublicstate: public nor::games::polymorph::Publicstate {
   /* Inherit the constructors */
   using Publicstate::Publicstate;
   using Publicstate::observation_type;

   size_t hash() const override
   {
      PYBIND11_OVERRIDE_NAME(
         size_t, /* Return type */
         Publicstate, /* Parent class */
         "__hash__", /* Name of function in Python */
         hash, /* Name of function in C++ */
         /* Argument(s) */
      );
   }

   bool operator==(const Publicstate& other) const override
   {
      PYBIND11_OVERRIDE_NAME(
         bool, /* Return type */
         Publicstate, /* Parent class */
         "__eq__", /* Name of function in Python */
         operator==, /* Name of function in C++ */
         other /* Argument(s) */
      );
   }

   size_t size() const override
   {
      PYBIND11_OVERRIDE_NAME(
         size_t, /* Return type */
         Publicstate, /* Parent class */
         "__len__", /* Name of function in Python */
         size, /* Name of function in C++ */
         /* Argument(s) */
      );
   }

   const observation_type& operator[](size_t arg) const override
   {
      PYBIND11_OVERRIDE_NAME(
         const observation_type&, /* Return type */
         Publicstate, /* Parent class */
         "__getitem__", /* Name of function in Python */
         operator[], /* Name of function in C++ */
         arg /* Argument(s) */
      );
   }

   Publicstate& update(const observation_type& obs) override
   {
      PYBIND11_OVERRIDE_NAME(
         Publicstate&, /* Return type */
         Publicstate, /* Parent class */
         "update", /* Name of function in Python */
         update, /* Name of function in C++ */
         obs /* Arguments */
      );
   }
};

}  // namespace nor::binding

namespace std {

template <>
struct hash< nor::games::polymorph::Publicstate > {
   size_t operator()(const nor::games::polymorph::Publicstate& publicstate) const
   {
      return publicstate.hash();
   }
};

}  // namespace std

#endif  // NOR_PY_TRAMPOLINES_PUBLICSTATE_HPP
