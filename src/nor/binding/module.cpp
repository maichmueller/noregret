

#ifndef NOR_PYTHON_MODULE_BINDING_HPP
#define NOR_PYTHON_MODULE_BINDING_HPP

#include <pybind11/pybind11.h>

#include "nor/fosg_states.hpp"
#include "trampolines/action.hpp"
#include "trampolines/chance_outcome.hpp"
#include "trampolines/infostate.hpp"
#include "trampolines/observation.hpp"
#include "trampolines/publicstate.hpp"
#include "trampolines/worldstate.hpp"

namespace py = pybind11;

using namespace nor;

// custom holder types
PYBIND11_DECLARE_HOLDER_TYPE(T, ActionWrapper< T >);
PYBIND11_DECLARE_HOLDER_TYPE(T, ChanceOutcomeWrapper< T >);
PYBIND11_DECLARE_HOLDER_TYPE(T, ObservationWrapper< T >);
PYBIND11_DECLARE_HOLDER_TYPE(T, InfostateWrapper< T >);
PYBIND11_DECLARE_HOLDER_TYPE(T, PublicstateWrapper< T >);
PYBIND11_DECLARE_HOLDER_TYPE(T, WorldstateWrapper< T >);

// needed because the type's `.get()` returns a reference to the underlying type, not a pointer.
namespace PYBIND11_NAMESPACE {

namespace detail {

template < typename T >
struct holder_helper< ActionWrapper< T > > {
   static const T *get(const ActionWrapper< T > &p) { return &(p.get()); }
};
template < typename T >
struct holder_helper< ChanceOutcomeWrapper< T > > {
   static const T *get(const ChanceOutcomeWrapper< T > &p) { return &(p.get()); }
};
template < typename T >
struct holder_helper< ObservationWrapper< T > > {
   static const T *get(const ObservationWrapper< T > &p) { return &(p.get()); }
};
template < typename T >
struct holder_helper< InfostateWrapper< T > > {
   static const T *get(const InfostateWrapper< T > &p) { return &(p.get()); }
};
template < typename T >
struct holder_helper< PublicstateWrapper< T > > {
   static const T *get(const PublicstateWrapper< T > &p) { return &(p.get()); }
};
template < typename T >
struct holder_helper< WorldstateWrapper< T > > {
   static const T *get(const WorldstateWrapper< T > &p) { return &(p.get()); }
};

}  // namespace detail

}  // namespace PYBIND11_NAMESPACE

PYBIND11_MODULE(_noregret, m)
{
   py::class_<
      games::polymorph::Action,
      ActionWrapper< games::polymorph::Action >,
      binding::PyAction >(m, "Action")
      .def(py::init())
      .def("__hash__", &games::polymorph::Action::hash)
      .def("__eq__", &games::polymorph::Action::operator==, py::arg("other"));

   py::class_<
      games::polymorph::ChanceOutcome,
      ChanceOutcomeWrapper< games::polymorph::ChanceOutcome >,
      binding::PyChanceOutcome >(m, "ChanceOutcome")
      .def(py::init())
      .def("__hash__", &games::polymorph::ChanceOutcome::hash)
      .def("__eq__", &games::polymorph::ChanceOutcome::operator==, py::arg("other"));

   py::class_<
      games::polymorph::Observation,
      ObservationWrapper< games::polymorph::Observation >,
      binding::PyObservation >(m, "Observation")
      .def(py::init())
      .def("__hash__", &games::polymorph::Observation::hash)
      .def("__eq__", &games::polymorph::Observation::operator==, py::arg("other"));

   py::class_<
      games::polymorph::Infostate,
      InfostateWrapper< games::polymorph::Infostate >,
      binding::PyInfostate >(m, "Infostate")
      .def(py::init< Player >(), py::arg("player"))
      .def(
         "player",
         &games::polymorph::Infostate::player,
         "The player that this information state belongs to."
      )
      .def(
         "update",
         &games::polymorph::Infostate::update,
         py::arg("public_obs"),
         py::arg("private_obs"),
         "Update the information state with the latest public and private observations."
      )
      .def("__hash__", &games::polymorph::Infostate::hash)
      .def("__eq__", &games::polymorph::Infostate::operator==, py::arg("other"))
      .def("__len__", &games::polymorph::Infostate::size)
      .def("__getitem__", &games::polymorph::Infostate::operator[], py::arg("index"));

   py::class_<
      games::polymorph::Publicstate,
      PublicstateWrapper< games::polymorph::Publicstate >,
      binding::PyPublicstate >(m, "Publicstate")
      .def(py::init())
      .def(
         "update",
         &games::polymorph::Publicstate::update,
         py::arg("obs"),
         "Update the public state with the latest public observation."
      )
      .def("__hash__", &games::polymorph::Publicstate::hash)
      .def("__eq__", &games::polymorph::Publicstate::operator==, py::arg("other"))
      .def("__len__", &games::polymorph::Publicstate::size)
      .def("__getitem__", &games::polymorph::Publicstate::operator[], py::arg("index"));

   py::enum_< Player >(m, "Player")
      .value("unknown", Player::unknown)
      .value("chance", Player::chance)
      .value("alex", Player::alex)
      .value("bob", Player::bob)
      .value("cedric", Player::cedric)
      .value("dexter", Player::dexter)
      .value("emily", Player::emily)
      .value("florence", Player::florence)
      .value("gustavo", Player::gustavo)
      .value("henrick", Player::henrick)
      .value("ian", Player::ian)
      .value("julia", Player::julia)
      .value("kelvin", Player::kelvin)
      .value("lea", Player::lea)
      .value("michael", Player::michael)
      .value("norbert", Player::norbert)
      .value("oscar", Player::oscar)
      .value("pedro", Player::pedro)
      .value("quentin", Player::quentin)
      .value("rosie", Player::rosie)
      .value("sophia", Player::sophia)
      .value("tristan", Player::tristan)
      .value("ulysses", Player::ulysses)
      .value("victoria", Player::victoria)
      .value("william", Player::william)
      .value("xavier", Player::xavier)
      .value("yusuf", Player::yusuf)
      .value("zoey", Player::zoey);
}

#endif  // NOR_PYTHON_MODULE_BINDING_HPP
