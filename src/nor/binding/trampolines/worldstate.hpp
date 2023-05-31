
#ifndef NOR_PY_TRAMPOLINES_WORLDSTATE_HPP
#define NOR_PY_TRAMPOLINES_WORLDSTATE_HPP

#include "nor/env/polymorphic_env.hpp"
namespace nor::py {

/* Trampoline Class */
struct PyWorldstate: public nor::env::polymorph::Worldstate {
   /* Inherit the constructors */
   using Worldstate::Worldstate;

   sptr< Worldstate > clone() const override
   {
      PYBIND11_OVERRIDE_NAME(
         sptr< Worldstate >, /* Return type */
         Worldstate, /* Parent class */
         "clone", /* Name of function in Python */
         clone, /* Name of function in C++ */
         /* Argument(s) */
      );
   }
};

}  // namespace nor::py

#endif  // NOR_PY_TRAMPOLINES_WORLDSTATE_HPP
