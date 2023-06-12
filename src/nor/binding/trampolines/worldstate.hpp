
#ifndef NOR_PY_TRAMPOLINES_WORLDSTATE_HPP
#define NOR_PY_TRAMPOLINES_WORLDSTATE_HPP

#include "nor/env/polymorphic_env.hpp"
namespace nor::py {

/* Trampoline Class */
struct PyWorldstate: public nor::games::polymorph::Worldstate {
   /* Inherit the constructors */
   using Worldstate::Worldstate;
};

}  // namespace nor::py

#endif  // NOR_PY_TRAMPOLINES_WORLDSTATE_HPP
