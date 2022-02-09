
#pragma once

#include "aze/aze.h"

#include "Action.hpp"
#include "Config.hpp"
#include "State.hpp"
#include "StrategoDefs.hpp"

namespace stratego {

class Game: public aze::Game< State, Logic, 2 > {
  public:
   using base_type = aze::Game< State, Logic, 2 >;
   using base_type::base_type;

   Game(
      state_type &&state,
      const sptr< aze::Agent< State > > &ag0,
      const sptr< aze::Agent< State > > &ag1)
       : base_type(std::make_unique< state_type >(std::move(state)), {ag0, ag1})
   {
   }

   Game(
      uptr< state_type > state,
      const sptr< aze::Agent< State > > &ag0,
      const sptr< aze::Agent< State > > &ag1)
       : base_type(std::move(state), {ag0, ag1})
   {
   }

   aze::Status run_game(const sptr< aze::utils::Plotter< state_type > > &plotter) override;
   aze::Status run_step() override;
   void reset() override;
};

}  // namespace stratego