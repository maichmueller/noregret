
#include "declarations.hpp"
#include "nor/type_defs.hpp"

using namespace nor;

void init_enums(py::module_& m)
{
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
