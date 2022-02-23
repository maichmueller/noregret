#include <gtest/gtest.h>

#include "../games/stratego/fixtures.hpp"
#include "nor/nor.hpp"
#include "stratego/stratego.hpp"

using namespace nor;

TEST_F(MinimalState, vanilla_cfr_basic_usage)
{
   struct plotter: aze::utils::Plotter< State > {
      void plot(const State& state) override { std::cout << state.to_string(Team::BLUE, false); }
   };

   // run the game with a fixed execution list of actions from the two agents. These sequences
   // should lead to a captre of the red flag by a blue scout and thus end the game with a blue win.
   Game game{
      std::move(state),
      std::make_shared< aze::RandomAgent< State > >(Team::BLUE),
      std::make_shared< aze::RandomAgent< State > >(Team::RED)};

   //   rm::VanillaCFR cfr(rm::CFRConfig{}, game, TabularPolicy<std::map<>);
   //   cfr.iterate();
}
