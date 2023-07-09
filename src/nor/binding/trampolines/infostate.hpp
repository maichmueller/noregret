
#ifndef NOR_PY_TRAMPOLINE_INFOSTATE_HPP
#define NOR_PY_TRAMPOLINE_INFOSTATE_HPP

#include <pybind11/pybind11.h>

#include "nor/env/polymorphic.hpp"

namespace nor::binding {

/* Trampoline Class */
struct PyInfostate: public nor::games::polymorph::Infostate {
   /* Inherit the constructors */
   using Infostate::Infostate;
   using Infostate::observation_type;

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

   const std::pair< ObservationHolder< observation_type >, ObservationHolder< observation_type > >&
   operator[](size_t arg) const override
   {
      PYBIND11_OVERRIDE_NAME(
         NOR_SINGLE_ARG(const std::pair<
                        ObservationHolder< observation_type >,
                        ObservationHolder< observation_type > >&), /* Return type */
         Infostate, /* Parent class */
         "__getitem__", /* Name of function in Python */
         operator[], /* Name of function in C++ */
         arg /* Argument(s) */
      );
   }

   void update(const observation_type& pub_obs, const observation_type& priv_obs) override
   {
      PYBIND11_OVERRIDE_NAME(
         void, /* Return type */
         Infostate, /* Parent class */
         "update", /* Name of function in Python */
         update, /* Name of function in C++ */
         pub_obs, /* Arguments */
         priv_obs
      );
   }
};

}  // namespace nor::binding

namespace std {

template <>
struct hash< nor::games::polymorph::Infostate > {
   size_t operator()(const nor::games::polymorph::Infostate& infostate) const
   {
      return infostate.hash();
   }
};

}  // namespace std

#endif  // NOR_PY_TRAMPOLINE_INFOSTATE_HPP
