

#include <gtest/gtest.h>

#include <range/v3/all.hpp>

#include "fixtures.hpp"
#include "utils.hpp"

TEST(State, constructor)
{
   std::map< Position, Token > setup0;
   std::map< Position, Token > setup1;

   setup0[{0, 0}] = Token::flag;
   setup0[{1, 1}] = Token::scout;

   setup1[{0, 1}] = Token::miner;
   setup1[{1, 0}] = Token::spy;
   std::vector< Position > hole_pos;

   auto setup = std::map{
      std::pair{Team::BLUE, std::optional{setup0}}, std::pair{Team::RED, std::optional{setup1}}};

   Config config{
      Team::BLUE,
      std::vector{size_t(2), size_t(2)},
      500,
      true,
      setup,
      hole_pos,
   };

   State state{config};

   for(auto [pos, token] : setup0) {
//      std::cout << "Given piece from setup:\n";
//      std::cout << team_name(Team::BLUE) << ", " << pos << ", " << token << "\n";
//      std::cout << "Board piece:\n";
//      std::cout << team_name(state.board()[pos].value().team()) << ", "
//                << state.board()[pos].value().position() << ", "
//                << state.board()[pos].value().token() << "\n";
      EXPECT_EQ((state.board()[pos].value()), Piece(Team::BLUE, pos, token));
   }
   for(auto [pos, token] : setup1) {
//      std::cout << "Given piece from setup:\n";
//      std::cout << team_name(Team::RED) << ", " << pos << ", " << token << "\n";
//      std::cout << "Board piece:\n";
//      std::cout << team_name(state.board()[pos].value().team()) << ", "
//                << state.board()[pos].value().position() << ", "
//                << state.board()[pos].value().token() << "\n";
      EXPECT_EQ((state.board()[pos].value()), Piece(Team::RED, pos, token));
   }
}