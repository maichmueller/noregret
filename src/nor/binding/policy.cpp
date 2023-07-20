
#include "declarations.hpp"
#include "nor/env/polymorphic.hpp"
#include "nor/policy/action_policy.hpp"
#include "nor/policy/tabular_policy.hpp"

using namespace nor;

using hashmap_action_policy = HashmapActionPolicy< py_action_type >;
using tabular_policy = TabularPolicy< py_info_state_type, hashmap_action_policy >;

void init_policy(py::module_& m)
{
   py::class_< hashmap_action_policy >(m, "TabularActionPolicy")
      .def(py::init())
      .def(py::init< std::function< double() > >(), py::arg("default_value"))
      .def(
         py::init<
            std::vector< ActionHolder< py_action_type > >,
            double,
            std::function< double() > >(),
         py::arg("actions"),
         py::arg("value") = 0.,
         py::arg("default_value") = std::function< double() >(&_zero< double >)
      )
      .def(
         py::init<
            std::unordered_map< ActionHolder< py_action_type >, double >,
            std::function< double() > >(),
         py::arg("policy_table"),
         py::arg("default_value") = std::function< double() >(&_zero< double >)
      )
      .def(
         "__getitem__",
         [](const hashmap_action_policy& policy, const py_action_type& action) {
            return policy.at(action);
         },
         py::arg("action")
      )
      .def(
         "__setitem__",
         [](hashmap_action_policy& policy, const py_action_type& action, double value) {
            return policy[action] = value;
         },
         py::arg("action"),
         py::arg("value")
      )
      .def("__len__", &hashmap_action_policy ::size)
      .def("__eq__", &hashmap_action_policy ::operator==, py::arg("other"));

   py::class_< tabular_policy >(m, "TabularPolicy")
      .def(py::init())
      .def(py::init< py::dict >([](const auto& dict) {
         tabular_policy policy;
         for(const auto& [key, value] : dict) {
            policy.emplace(key.cast< py_info_state_type >(), value.cast< hashmap_action_policy >());
         }
         return policy;
      }));
}
