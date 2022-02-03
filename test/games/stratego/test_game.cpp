
#include <gtest/gtest.h>

#include <utility>

#include "fixtures.hpp"
#include "utils.hpp"

TEST_F(MinimalState, run_game)
{
   struct plotter: aze::utils::Plotter< State > {
      void plot(const State& state) override { std::cout << state.to_string(Team::BLUE, false); }
   };

   Game game{
      std::move(state),
      std::make_shared< aze::RandomAgent< State > >(Team::BLUE, 0),
      std::make_shared< aze::RandomAgent< State > >(Team::RED, 1)};
   game.run_game(std::make_shared< plotter >());
}