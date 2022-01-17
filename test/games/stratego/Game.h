
#pragma once

#include <aze/aze.h>

#include "Action.h"
#include "Config.hpp"
#include "State.h"
#include "StrategoDefs.hpp"

namespace stratego {

class Game: public aze::Game< State, Logic, Game, 2 > {
  public:
   using base_type = aze::Game< State, Logic, Game, 2 >;
   using base_type::base_type;

   Game(
      const std::array< size_t, 2 > &shape,
      const std::map< Position , int > &setup_0,
      const std::map< Position, int > &setup_1,
      const sptr< aze::Agent< State > > &ag0,
      const sptr< aze::Agent< State > > &ag1);

   Game(
      size_t shape,
      const std::map< Position, int > &setup_0,
      const std::map< Position, int > &setup_1,
      const sptr< aze::Agent< State > > &ag0,
      const sptr< aze::Agent< State > > &ag1);

   Game(
      const std::array< size_t, 2 > &shape,
      const std::map< Position, Token > &setup_0,
      const std::map< Position, Token > &setup_1,
      const sptr< aze::Agent< State > > &ag0,
      const sptr< aze::Agent< State > > &ag1);

   Game(
      size_t shape,
      const std::map< Position, Token > &setup_0,
      const std::map< Position, Token > &setup_1,
      const sptr< aze::Agent< State > > &ag0,
      const sptr< aze::Agent< State > > &ag1);

   std::map< Position, Piece > draw_setup_(aze::Team team);
};
}  // namespace stratego