
#pragma once

#include "State.h"
#include <aze/aze.h>

#include "StrategoDefs.hpp"
#include "Config.hpp"

namespace stratego {

class Game: public aze::Game< State, Logic< Board >, Game > {
  public:
   using base_type = aze::Game< State, Logic< Board >, Game >;
   using base_type::base_type;

   Game(
      const std::array< size_t, 2 > &shape,
      const std::map< position_type, int > &setup_0,
      const std::map< position_type, int > &setup_1,
      const sptr< aze::Agent< state_type > > &ag0,
      const sptr< aze::Agent< state_type > > &ag1);

   Game(
      size_t shape,
      const std::map< position_type, int > &setup_0,
      const std::map< position_type, int > &setup_1,
      const sptr< aze::Agent< state_type > > &ag0,
      const sptr< aze::Agent< state_type > > &ag1);

   Game(
      const std::array< size_t, 2 > &shape,
      const std::map< position_type, token_type > &setup_0,
      const std::map< position_type, token_type > &setup_1,
      const sptr< aze::Agent< state_type > > &ag0,
      const sptr< aze::Agent< state_type > > &ag1);

   Game(
      size_t shape,
      const std::map< position_type, token_type > &setup_0,
      const std::map< position_type, token_type > &setup_1,
      const sptr< aze::Agent< state_type > > &ag0,
      const sptr< aze::Agent< state_type > > &ag1);

   std::map< position_type, sptr_piece_type > draw_setup_(int team);

   aze::Status check_terminal() override;
};
}