
#ifndef NOR_PYBINDING_DECLARATIONS_HPP
#define NOR_PYBINDING_DECLARATIONS_HPP

#include <pybind11/pybind11.h>

#include "declarations.hpp"
#include "nor/fosg_states.hpp"
#include "trampolines/action.hpp"
#include "trampolines/chance_outcome.hpp"
#include "trampolines/infostate.hpp"
#include "trampolines/observation.hpp"
#include "trampolines/publicstate.hpp"
#include "trampolines/worldstate.hpp"

namespace py = pybind11;

// custom holder types
PYBIND11_DECLARE_HOLDER_TYPE(T, nor::ActionHolder< T >);
PYBIND11_DECLARE_HOLDER_TYPE(T, nor::ChanceOutcomeHolder< T >);
PYBIND11_DECLARE_HOLDER_TYPE(T, nor::ObservationHolder< T >);
PYBIND11_DECLARE_HOLDER_TYPE(T, nor::InfostateHolder< T >);
PYBIND11_DECLARE_HOLDER_TYPE(T, nor::PublicstateHolder< T >);
PYBIND11_DECLARE_HOLDER_TYPE(T, nor::WorldstateHolder< T >);

// needed because the type's `.get()` returns a reference to the underlying type, not a pointer.
namespace PYBIND11_NAMESPACE {

namespace detail {

template < typename T >
struct holder_helper< nor::ActionHolder< T > > {
   static const T *get(const nor::ActionHolder< T > &p) { return &(p.get()); }
};
template < typename T >
struct holder_helper< nor::ChanceOutcomeHolder< T > > {
   static const T *get(const nor::ChanceOutcomeHolder< T > &p) { return &(p.get()); }
};
template < typename T >
struct holder_helper< nor::ObservationHolder< T > > {
   static const T *get(const nor::ObservationHolder< T > &p) { return &(p.get()); }
};
template < typename T >
struct holder_helper< nor::InfostateHolder< T > > {
   static const T *get(const nor::InfostateHolder< T > &p) { return &(p.get()); }
};
template < typename T >
struct holder_helper< nor::PublicstateHolder< T > > {
   static const T *get(const nor::PublicstateHolder< T > &p) { return &(p.get()); }
};
template < typename T >
struct holder_helper< nor::WorldstateHolder< T > > {
   static const T *get(const nor::WorldstateHolder< T > &p) { return &(p.get()); }
};

}  // namespace detail

}  // namespace PYBIND11_NAMESPACE

using py_action_type = nor::games::polymorph::Action;
using py_chance_outcome_type = nor::games::polymorph::ChanceOutcome;
using py_observation_type = nor::games::polymorph::Observation;
using py_info_state_type = nor::games::polymorph::Infostate;
using py_public_state_type = nor::games::polymorph::Publicstate;
using py_world_state_type = nor::games::polymorph::Worldstate;
using py_env_type = nor::games::polymorph::Environment;

void init_env_polymorph(py::module_ &m);
void init_enums(py::module_ &m);
void init_policy(py::module_ &m);

#endif  // NOR_PYBINDING_DECLARATIONS_HPP
