
#include <gtest/gtest.h>

#include <utility>

#include "fixtures.hpp"
#include "utils.hpp"

using namespace stratego;

TEST_F(StrategoState3x3, run_game_fixed_actions)
{
   struct plotter: aze::utils::Plotter< State > {
      void plot(const State& state) override { std::cout << state.to_string(Team::BLUE, false); }
   };
   // run the game with a fixed execution list of actions from the two agents.
   // These sequences should lead to a capture of the blue flag by a red spy and thus end the game
   // with a red win.
   Game game{
      std::move(state),
      std::make_shared< aze::FixedAgent< State > >(
         Team::BLUE, std::vector{Move{{0, 1}, {0, 2}}, Move{{0, 2}, {1, 2}}, Move{{1, 2}, {1, 1}}}),
      std::make_shared< aze::FixedAgent< State > >(
         Team::RED, std::vector{Move{{2, 1}, {2, 0}}, Move{{2, 0}, {1, 0}}, Move{{1, 0}, {0, 0}}})};

   //   EXPECT_EQ(game.run(std::make_shared< plotter >()), Status::WIN_RED);
   EXPECT_EQ(game.run(nullptr), Status::WIN_RED);
}

TEST_F(StrategoState5x5, run_game_fixed_actions)
{
   struct plotter: aze::utils::Plotter< State > {
      void plot(const State& state) override { std::cout << state.to_string(Team::BLUE, false); }
   };
   // run the game with a fixed execution list of actions from the two agents. These sequences
   // should lead to a capture of the red flag by a blue scout and thus end the game with a blue
   // win.
   Game game{
      std::move(state),
      std::make_shared< aze::FixedAgent< State > >(
         Team::BLUE,
         std::vector{
            Move{{1, 1}, {2, 1}}, Move{{1, 4}, {2, 4}}, Move{{2, 1}, {3, 1}}, Move{{3, 1}, {2, 1}},
            Move{{0, 4}, {1, 4}}, Move{{2, 1}, {2, 0}}, Move{{2, 0}, {3, 0}}, Move{{1, 4}, {0, 4}},
            Move{{3, 0}, {2, 0}}, Move{{0, 1}, {1, 1}}, Move{{1, 1}, {2, 1}}, Move{{0, 2}, {0, 1}},
            Move{{0, 3}, {0, 2}}, Move{{0, 4}, {0, 3}}, Move{{0, 1}, {2, 1}}, Move{{2, 0}, {3, 0}},
            Move{{0, 3}, {0, 4}}, Move{{2, 1}, {4, 1}}, Move{{4, 1}, {0, 1}}, Move{{1, 2}, {1, 1}},
            Move{{1, 1}, {2, 1}}, Move{{2, 1}, {4, 1}}, Move{{4, 1}, {3, 1}}, Move{{3, 1}, {2, 1}},
            Move{{0, 1}, {1, 1}}, Move{{0, 2}, {1, 2}}, Move{{1, 1}, {0, 1}}, Move{{1, 2}, {0, 2}},
            Move{{0, 4}, {0, 3}}, Move{{0, 1}, {1, 1}}, Move{{0, 2}, {0, 1}}, Move{{2, 1}, {2, 0}},
            Move{{0, 3}, {0, 2}}, Move{{2, 0}, {2, 1}}, Move{{2, 1}, {4, 1}}, Move{{4, 1}, {4, 2}},
            Move{{3, 0}, {4, 0}}, Move{{1, 1}, {1, 2}}, Move{{0, 1}, {2, 1}}, Move{{0, 2}, {0, 1}},
            Move{{4, 0}, {4, 1}}, Move{{1, 2}, {0, 2}}, Move{{2, 1}, {1, 1}}, Move{{1, 1}, {1, 2}},
            Move{{0, 2}, {0, 3}}, Move{{0, 3}, {0, 4}}, Move{{0, 4}, {4, 4}}}),
      std::make_shared< aze::FixedAgent< State > >(
         Team::RED,
         std::vector{
            Move{{3, 0}, {1, 0}}, Move{{3, 1}, {2, 1}}, Move{{3, 4}, {2, 4}}, Move{{4, 0}, {3, 0}},
            Move{{3, 0}, {4, 0}}, Move{{3, 3}, {2, 3}}, Move{{2, 4}, {3, 4}}, Move{{4, 1}, {3, 1}},
            Move{{4, 0}, {3, 0}}, Move{{2, 3}, {2, 4}}, Move{{2, 4}, {0, 4}}, Move{{3, 1}, {4, 1}},
            Move{{4, 1}, {3, 1}}, Move{{3, 1}, {2, 1}}, Move{{3, 4}, {3, 3}}, Move{{3, 3}, {2, 3}},
            Move{{2, 3}, {3, 3}}, Move{{3, 3}, {3, 4}}, Move{{3, 4}, {2, 4}}, Move{{2, 4}, {3, 4}},
            Move{{3, 4}, {2, 4}}, Move{{2, 4}, {3, 4}}, Move{{4, 3}, {3, 3}}, Move{{3, 4}, {2, 4}},
            Move{{2, 4}, {1, 4}}, Move{{3, 3}, {3, 4}}, Move{{1, 4}, {1, 3}}, Move{{3, 4}, {2, 4}},
            Move{{2, 4}, {1, 4}}, Move{{1, 4}, {0, 4}}, Move{{0, 4}, {1, 4}}, Move{{1, 4}, {0, 4}},
            Move{{0, 4}, {1, 4}}, Move{{1, 4}, {0, 4}}, Move{{0, 4}, {1, 4}}, Move{{1, 4}, {0, 4}},
            Move{{0, 4}, {0, 3}}, Move{{0, 3}, {0, 4}}, Move{{0, 4}, {1, 4}}, Move{{1, 4}, {2, 4}},
            Move{{2, 4}, {3, 4}}, Move{{3, 4}, {3, 3}}, Move{{3, 3}, {3, 4}}, Move{{3, 4}, {2, 4}},
            Move{{2, 4}, {1, 4}}, Move{{1, 4}, {1, 3}}})};

   //   EXPECT_EQ(game.run(std::make_shared< plotter >()), Status::WIN_BLUE);
   EXPECT_EQ(game.run(nullptr), Status::WIN_BLUE);
}