
#include <gtest/gtest.h>

#include <utility>

#include "fixtures.hpp"
#include "utils.hpp"

using namespace stratego;

TEST_F(MinimalState, run_game_fixed_actions)
{
   struct plotter: aze::utils::Plotter< State > {
      void plot(const State& state) override { std::cout << state.to_string(Team::BLUE, false); }
   };

   // run the game with a fixed execution list of actions from the two agents. These sequences
   // should lead to a captre of the red flag by a blue scout and thus end the game with a blue win.
   Game game{
      std::move(state),
      std::make_shared< aze::FixedAgent< State > >(
         Team::BLUE,
         std::vector{Action{Team::BLUE, {{1, 1}, {2, 1}}}, Action{Team::BLUE, {{1, 4}, {2, 4}}},
                     Action{Team::BLUE, {{2, 1}, {3, 1}}}, Action{Team::BLUE, {{3, 1}, {2, 1}}},
                     Action{Team::BLUE, {{0, 4}, {1, 4}}}, Action{Team::BLUE, {{2, 1}, {2, 0}}},
                     Action{Team::BLUE, {{2, 0}, {3, 0}}}, Action{Team::BLUE, {{1, 4}, {0, 4}}},
                     Action{Team::BLUE, {{3, 0}, {2, 0}}}, Action{Team::BLUE, {{0, 1}, {1, 1}}},
                     Action{Team::BLUE, {{1, 1}, {2, 1}}}, Action{Team::BLUE, {{0, 2}, {0, 1}}},
                     Action{Team::BLUE, {{0, 3}, {0, 2}}}, Action{Team::BLUE, {{0, 4}, {0, 3}}},
                     Action{Team::BLUE, {{0, 1}, {2, 1}}}, Action{Team::BLUE, {{2, 0}, {3, 0}}},
                     Action{Team::BLUE, {{0, 3}, {0, 4}}}, Action{Team::BLUE, {{2, 1}, {4, 1}}},
                     Action{Team::BLUE, {{4, 1}, {0, 1}}}, Action{Team::BLUE, {{1, 2}, {1, 1}}},
                     Action{Team::BLUE, {{1, 1}, {2, 1}}}, Action{Team::BLUE, {{2, 1}, {4, 1}}},
                     Action{Team::BLUE, {{4, 1}, {3, 1}}}, Action{Team::BLUE, {{3, 1}, {2, 1}}},
                     Action{Team::BLUE, {{0, 1}, {1, 1}}}, Action{Team::BLUE, {{0, 2}, {1, 2}}},
                     Action{Team::BLUE, {{1, 1}, {0, 1}}}, Action{Team::BLUE, {{1, 2}, {0, 2}}},
                     Action{Team::BLUE, {{0, 4}, {0, 3}}}, Action{Team::BLUE, {{0, 1}, {1, 1}}},
                     Action{Team::BLUE, {{0, 2}, {0, 1}}}, Action{Team::BLUE, {{2, 1}, {2, 0}}},
                     Action{Team::BLUE, {{0, 3}, {0, 2}}}, Action{Team::BLUE, {{2, 0}, {2, 1}}},
                     Action{Team::BLUE, {{2, 1}, {4, 1}}}, Action{Team::BLUE, {{4, 1}, {4, 2}}},
                     Action{Team::BLUE, {{3, 0}, {4, 0}}}, Action{Team::BLUE, {{1, 1}, {1, 2}}},
                     Action{Team::BLUE, {{0, 1}, {2, 1}}}, Action{Team::BLUE, {{0, 2}, {0, 1}}},
                     Action{Team::BLUE, {{4, 0}, {4, 1}}}, Action{Team::BLUE, {{1, 2}, {0, 2}}},
                     Action{Team::BLUE, {{2, 1}, {1, 1}}}, Action{Team::BLUE, {{1, 1}, {1, 2}}},
                     Action{Team::BLUE, {{0, 2}, {0, 3}}}, Action{Team::BLUE, {{0, 3}, {0, 4}}},
                     Action{Team::BLUE, {{0, 4}, {4, 4}}}}),
      std::make_shared< aze::FixedAgent< State > >(
         Team::RED,
         std::vector{Action{Team::RED, {{3, 0}, {1, 0}}}, Action{Team::RED, {{3, 1}, {2, 1}}},
                     Action{Team::RED, {{3, 4}, {2, 4}}}, Action{Team::RED, {{4, 0}, {3, 0}}},
                     Action{Team::RED, {{3, 0}, {4, 0}}}, Action{Team::RED, {{3, 3}, {2, 3}}},
                     Action{Team::RED, {{2, 4}, {3, 4}}}, Action{Team::RED, {{4, 1}, {3, 1}}},
                     Action{Team::RED, {{4, 0}, {3, 0}}}, Action{Team::RED, {{2, 3}, {2, 4}}},
                     Action{Team::RED, {{2, 4}, {0, 4}}}, Action{Team::RED, {{3, 1}, {4, 1}}},
                     Action{Team::RED, {{4, 1}, {3, 1}}}, Action{Team::RED, {{3, 1}, {2, 1}}},
                     Action{Team::RED, {{3, 4}, {3, 3}}}, Action{Team::RED, {{3, 3}, {2, 3}}},
                     Action{Team::RED, {{2, 3}, {3, 3}}}, Action{Team::RED, {{3, 3}, {3, 4}}},
                     Action{Team::RED, {{3, 4}, {2, 4}}}, Action{Team::RED, {{2, 4}, {3, 4}}},
                     Action{Team::RED, {{3, 4}, {2, 4}}}, Action{Team::RED, {{2, 4}, {3, 4}}},
                     Action{Team::RED, {{4, 3}, {3, 3}}}, Action{Team::RED, {{3, 4}, {2, 4}}},
                     Action{Team::RED, {{2, 4}, {1, 4}}}, Action{Team::RED, {{3, 3}, {3, 4}}},
                     Action{Team::RED, {{1, 4}, {1, 3}}}, Action{Team::RED, {{3, 4}, {2, 4}}},
                     Action{Team::RED, {{2, 4}, {1, 4}}}, Action{Team::RED, {{1, 4}, {0, 4}}},
                     Action{Team::RED, {{0, 4}, {1, 4}}}, Action{Team::RED, {{1, 4}, {0, 4}}},
                     Action{Team::RED, {{0, 4}, {1, 4}}}, Action{Team::RED, {{1, 4}, {0, 4}}},
                     Action{Team::RED, {{0, 4}, {1, 4}}}, Action{Team::RED, {{1, 4}, {0, 4}}},
                     Action{Team::RED, {{0, 4}, {0, 3}}}, Action{Team::RED, {{0, 3}, {0, 4}}},
                     Action{Team::RED, {{0, 4}, {1, 4}}}, Action{Team::RED, {{1, 4}, {2, 4}}},
                     Action{Team::RED, {{2, 4}, {3, 4}}}, Action{Team::RED, {{3, 4}, {3, 3}}},
                     Action{Team::RED, {{3, 3}, {3, 4}}}, Action{Team::RED, {{3, 4}, {2, 4}}},
                     Action{Team::RED, {{2, 4}, {1, 4}}}, Action{Team::RED, {{1, 4}, {1, 3}}}})};

   //   EXPECT_EQ(game.run(std::make_shared< plotter >()), Status::WIN_BLUE);
   EXPECT_EQ(game.run(nullptr), Status::WIN_BLUE);
}