#include "declarations.hpp"
#include "nor/env/polymorphic.hpp"
#include "trampolines/env.hpp"

using namespace nor;

void init_env_polymorph(py::module_& m)
{
   py::class_< py_action_type, ActionHolder< py_action_type >, binding::PyAction >(m, "Action")
      .def(py::init())
      .def("__hash__", &py_action_type::hash)
      .def("__eq__", &py_action_type::operator==, py::arg("other"));

   py::class_<
      py_chance_outcome_type,
      ChanceOutcomeHolder< py_chance_outcome_type >,
      binding::PyChanceOutcome >(m, "ChanceOutcome")
      .def(py::init())
      .def("__hash__", &py_chance_outcome_type::hash)
      .def("__eq__", &py_chance_outcome_type::operator==, py::arg("other"));

   //   py::class_<
   //      py_observation_type,
   //      ObservationHolder< py_observation_type >,
   //      binding::PyObservation >(m, "Observation")
   //      .def(py::init())
   //      .def("__hash__", &py_observation_type::hash)
   //      .def("__eq__", &py_observation_type::operator==, py::arg("other"));

   py::class_<
      py_info_state_type,  //
      InfostateHolder< py_info_state_type >,
      binding::PyInfostate >(m, "Infostate")
      .def(py::init< Player >(), py::arg("player"))
      .def(
         "player", &py_info_state_type::player, "The player that this information state belongs to."
      )
      .def(
         "update",
         &py_info_state_type::update,
         py::arg("public_obs"),
         py::arg("private_obs"),
         "Update the information state with the latest public and private observations."
      )
      .def("__hash__", &py_info_state_type::hash)
      .def("__eq__", &py_info_state_type::operator==, py::arg("other"))
      .def("__len__", &py_info_state_type::size)
      .def("__getitem__", &py_info_state_type::operator[], py::arg("index"));

   py::class_<
      py_public_state_type,
      PublicstateHolder< py_public_state_type >,
      binding::PyPublicstate >(m, "Publicstate")
      .def(py::init())
      .def(
         "update",
         &py_public_state_type::update,
         py::arg("obs"),
         "Update the public state with the latest public observation."
      )
      .def("__hash__", &py_public_state_type::hash)
      .def("__eq__", &py_public_state_type::operator==, py::arg("other"))
      .def("__len__", &py_public_state_type::size)
      .def("__getitem__", &py_public_state_type::operator[], py::arg("index"));

   py::
      class_< py_world_state_type, WorldstateHolder< py_world_state_type >, binding::PyWorldstate >(
         m, "Worldstate"
      )
         .def(py::init());

   py::class_< py_env_type, std::shared_ptr< py_env_type >, binding::PyEnvironment >(
      m, "Environment"
   )
      .def(py::init());
}
