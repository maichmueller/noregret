#include "aze/aze.h"

#include <array>
#include <iostream>
#include <memory>
#include <xtensor/xtensor.hpp>

#include "stratego/Game.hpp"
#include "stratego/State.hpp"

int main()
{
   using namespace stratego;

   //
   // build the agents to play.
   //
   auto agent_0 = std::make_shared< aze::RandomAgent< State > >(Team::BLUE);
   auto agent_1 = std::make_shared< aze::RandomAgent< State > >(Team::RED);

   //
   // setup the game
   //
   std::map< Position, Token > setup0;
   std::map< Position, Token > setup1;

   setup0[{0, 0}] = Token::flag;
   setup0[{0, 1}] = Token::spy;
   setup0[{0, 2}] = Token::scout;
   setup0[{0, 3}] = Token::scout;
   setup0[{0, 4}] = Token::miner;
   setup0[{1, 0}] = Token::bomb;
   setup0[{1, 1}] = Token::marshall;
   setup0[{1, 2}] = Token::scout;
   setup0[{1, 3}] = Token::bomb;
   setup0[{1, 4}] = Token::miner;
   setup1[{3, 0}] = Token::scout;
   setup1[{3, 1}] = Token::scout;
   setup1[{3, 2}] = Token::bomb;
   setup1[{3, 3}] = Token::scout;
   setup1[{3, 4}] = Token::marshall;
   setup1[{4, 0}] = Token::miner;
   setup1[{4, 1}] = Token::spy;
   setup1[{4, 2}] = Token::bomb;
   setup1[{4, 3}] = Token::miner;
   setup1[{4, 4}] = Token::flag;

   Config config{
      Team::BLUE,
      size_t(5),
      500,
      true,
      std::map{
         std::pair{Team::BLUE, std::make_optional(setup0)},
         std::pair{Team::RED, std::make_optional(setup1)}}};
   auto game = Game(State(config), agent_0, agent_1);

   //
   // run/train on the game
   //
   game.run_game(nullptr);

   return 0;
}