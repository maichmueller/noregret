
#ifndef NOR_STATE_FIXTURE_HPP
#define NOR_STATE_FIXTURE_HPP

#include <gtest/gtest.h>

#include <stratego/stratego.hpp>


using namespace stratego;

class MinimalState : public ::testing::Test {
  protected:
   State state;

   void SetUp() override {
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
         {false, false},
         std::map{
            std::pair{Team::BLUE, std::make_optional(setup0)},
            std::pair{Team::RED, std::make_optional(setup1)}}};

      state = State(config);

   }
};

#endif  // NOR_STATE_FIXTURE_HPP
