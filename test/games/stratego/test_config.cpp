
#include <gtest/gtest.h>

#include "fixtures.hpp"
#include "utils.hpp"

using namespace stratego;

TEST(Config, constructor_with_setup)
{
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
      std::map{
         std::pair{Team::BLUE, std::optional(setup0)}, std::pair{Team::RED, std::optional(setup1)}},
      Config::nullarg< "holes" >(),
      true,
      true,
      500};

   EXPECT_EQ(config.setups[Team::BLUE].value(), setup0);
   EXPECT_EQ(config.setups[Team::RED].value(), setup1);

   EXPECT_EQ(
      config.token_counters[Team::BLUE],
      (std::map< Token, unsigned int >{
         {Token::flag, 1},
         {Token::spy, 1},
         {Token::scout, 3},
         {Token::miner, 2},
         {Token::marshall, 1},
         {Token::bomb, 2}}));
   EXPECT_EQ(
      config.token_counters[Team::RED],
      (std::map< Token, unsigned int >{
         {Token::flag, 1},
         {Token::spy, 1},
         {Token::scout, 3},
         {Token::miner, 2},
         {Token::marshall, 1},
         {Token::bomb, 2}}));

   EXPECT_EQ(
      config.start_fields[Team::BLUE],
      (std::vector< Position >{
         {0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 0}, {1, 1}, {1, 2}, {1, 3}, {1, 4}}));
   EXPECT_EQ(
      config.start_fields[Team::RED],
      (std::vector< Position >{
         {3, 0}, {3, 1}, {3, 2}, {3, 3}, {3, 4}, {4, 0}, {4, 1}, {4, 2}, {4, 3}, {4, 4}}));
}

TEST_P(BattlematrixParamsF, default_battlematrix_outcomes)
{
   auto [attacker, defender, outcome] = GetParam();
   LOGD2(
      "Observed outcome for [" + common::to_string(attacker) + ", "
         + common::to_string(defender) + "] = ",
      common::to_string(bm[{attacker, defender}]));
   LOGD2(
      "Expected outcome for [" + common::to_string(attacker) + ", "
         + common::to_string(defender) + "] = ",
      common::to_string(outcome));
   EXPECT_EQ((bm[{attacker, defender}]), outcome);
}

INSTANTIATE_TEST_SUITE_P(
   default_battlematrix_tests,
   BattlematrixParamsF,
   ::testing::Values(
      std::tuple(Token::marshall, Token::scout, FightOutcome::kill),
      std::tuple(Token::scout, Token::marshall, FightOutcome::death),
      std::tuple(Token::bomb, Token::marshall, FightOutcome::kill),
      std::tuple(Token::marshall, Token::bomb, FightOutcome::death),
      std::tuple(Token::scout, Token::spy, FightOutcome::kill),
      std::tuple(Token::major, Token::spy, FightOutcome::kill),
      std::tuple(Token::marshall, Token::spy, FightOutcome::kill),
      std::tuple(Token::captain, Token::spy, FightOutcome::kill),
      std::tuple(Token::spy, Token::marshall, FightOutcome::kill),
      std::tuple(Token::spy, Token::captain, FightOutcome::death),
      std::tuple(Token::spy, Token::major, FightOutcome::death),
      std::tuple(Token::spy, Token::colonel, FightOutcome::death),
      std::tuple(Token::spy, Token::lieutenant, FightOutcome::death),
      std::tuple(Token::spy, Token::scout, FightOutcome::death),
      std::tuple(Token::spy, Token::general, FightOutcome::death),
      std::tuple(Token::colonel, Token::major, FightOutcome::kill),
      std::tuple(Token::colonel, Token::captain, FightOutcome::kill),
      std::tuple(Token::lieutenant, Token::captain, FightOutcome::death),
      std::tuple(Token::colonel, Token::general, FightOutcome::death),
      std::tuple(Token::lieutenant, Token::bomb, FightOutcome::death),
      std::tuple(Token::captain, Token::bomb, FightOutcome::death),
      std::tuple(Token::spy, Token::bomb, FightOutcome::death),
      std::tuple(Token::major, Token::bomb, FightOutcome::death),
      std::tuple(Token::marshall, Token::bomb, FightOutcome::death),
      std::tuple(Token::scout, Token::bomb, FightOutcome::death),
      std::tuple(Token::miner, Token::bomb, FightOutcome::kill),
      std::tuple(Token::general, Token::bomb, FightOutcome::death)));

TEST(Config, constructor_custom_dims_with_setup_small)
{
   std::map< Position, Token > setup0;
   std::map< Position, Token > setup1;

   setup0[{0, 0}] = Token::flag;
   setup0[{1, 1}] = Token::scout;

   setup1[{0, 1}] = Token::miner;
   setup1[{1, 0}] = Token::spy;
   std::vector< Position > hole_pos{{1, 1}};

   Config config{
      Team::BLUE,
      std::vector{size_t(2), size_t(2)},
      std::map{
         std::pair{Team::BLUE, std::optional{setup0}}, std::pair{Team::RED, std::optional{setup1}}},
      hole_pos,
      true,
      false,
      500};

   EXPECT_EQ(config.setups[Team::BLUE].value(), setup0);
   EXPECT_EQ(config.setups[Team::RED].value(), setup1);

   EXPECT_EQ(
      config.token_counters[Team::BLUE],
      (std::map< Token, unsigned int >{{Token::flag, 1}, {Token::scout, 1}}));
   EXPECT_EQ(
      config.token_counters[Team::RED],
      (std::map< Token, unsigned int >{{Token::miner, 1}, {Token::spy, 1}}));

   auto pos_comparator = [](const Position& pos1, const Position& pos2) {
      if(pos1[0] == pos2[0]) {
         return pos1[1] <= pos2[1];
      } else {
         return pos1[0] < pos2[0];
      }
   };
   EXPECT_TRUE(cmp_equal_rngs(
      config.start_fields[Team::BLUE],
      std::vector< Position >{{0, 0}, {1, 1}},
      pos_comparator,
      pos_comparator));
   EXPECT_TRUE(cmp_equal_rngs(
      config.start_fields[Team::RED],
      std::vector< Position >{{1, 0}, {0, 1}},
      pos_comparator,
      pos_comparator));
}

TEST(Config, constructor_custom_dims_with_setup_medium)
{
   std::map< Position, Token > setup0;
   std::map< Position, Token > setup1;

   setup0[{0, 0}] = Token::flag;
   setup0[{0, 1}] = Token::spy;
   setup0[{0, 2}] = Token::scout;
   setup0[{1, 3}] = Token::scout;
   setup0[{2, 4}] = Token::miner;

   setup1[{3, 0}] = Token::flag;
   setup1[{2, 1}] = Token::spy;
   setup1[{1, 2}] = Token::spy;
   setup1[{3, 3}] = Token::spy;
   setup1[{3, 4}] = Token::marshall;

   std::vector< Position > hole_pos{{1, 1}};

   Config config{
      Team::BLUE,
      std::vector{size_t(3), size_t(4)},
      std::map{
         std::pair{Team::BLUE, std::optional(setup0)}, std::pair{Team::RED, std::optional(setup1)}},
      hole_pos,
      false,
      true,
      500,
   };

   EXPECT_EQ(config.setups[Team::BLUE].value(), setup0);
   EXPECT_EQ(config.setups[Team::RED].value(), setup1);

   EXPECT_EQ(
      config.token_counters[Team::BLUE],
      (std::map< Token, unsigned int >{
         {Token::flag, 1}, {Token::spy, 1}, {Token::scout, 2}, {Token::miner, 1}}));
   EXPECT_EQ(
      config.token_counters[Team::RED],
      (std::map< Token, unsigned int >{{Token::flag, 1}, {Token::spy, 3}, {Token::marshall, 1}}));

   auto pos_comparator = [](const Position& pos1, const Position& pos2) {
      if(pos1[0] == pos2[0]) {
         return pos1[1] <= pos2[1];
      } else {
         return pos1[0] < pos2[0];
      }
   };
   EXPECT_TRUE(cmp_equal_rngs(
      config.start_fields[Team::BLUE],
      std::vector< Position >{{0, 0}, {0, 1}, {0, 2}, {1, 3}, {2, 4}},
      pos_comparator,
      pos_comparator));
   EXPECT_TRUE(cmp_equal_rngs(
      config.start_fields[Team::RED],
      std::vector< Position >{{3, 0}, {2, 1}, {1, 2}, {3, 3}, {3, 4}},
      pos_comparator,
      pos_comparator));
}

TEST(Config, constructor_custom_dims_no_setup)
{
   std::vector< Position > pos_blue;
   pos_blue.emplace_back(Position{0, 0});
   pos_blue.emplace_back(Position{3, 3});
   pos_blue.emplace_back(Position{1, 3});

   std::vector< Position > pos_red;
   pos_red.emplace_back(Position{1, 2});
   pos_red.emplace_back(Position{3, 4});
   pos_red.emplace_back(Position{1, 4});
   pos_red.emplace_back(Position{3, 1});
   pos_red.emplace_back(Position{2, 4});

   auto start_pos = std::map{
      std::pair{Team::BLUE, std::optional{pos_blue}}, std::pair{Team::RED, std::optional{pos_red}}};

   std::vector< Token > tokens_blue;
   tokens_blue.emplace_back(Token::miner);
   tokens_blue.emplace_back(Token::miner);
   tokens_blue.emplace_back(Token::miner);

   std::vector< Token > tokens_red;
   tokens_red.emplace_back(Token::major);
   tokens_red.emplace_back(Token::lieutenant);
   tokens_red.emplace_back(Token::lieutenant);
   tokens_red.emplace_back(Token::lieutenant);
   tokens_red.emplace_back(Token::lieutenant);

   auto tokens = std::map{
      std::pair{Team::BLUE, std::optional< Config::token_variant_t >{tokens_blue}},
      std::pair{Team::RED, std::optional< Config::token_variant_t >{tokens_red}}};

   std::vector< Position > hole_pos{{1, 1}};

   Config config{
      Team::BLUE,
      std::vector{size_t(3), size_t(4)},
      hole_pos,
      tokens,
      start_pos,
      false,
      false,
      500};

   EXPECT_EQ(
      config.token_counters[Team::BLUE], (std::map< Token, unsigned int >{{Token::miner, 3}}));
   EXPECT_EQ(
      config.token_counters[Team::RED],
      (std::map< Token, unsigned int >{{Token::major, 1}, {Token::lieutenant, 4}}));

   auto pos_comparator = [](const Position& pos1, const Position& pos2) {
      if(pos1[0] == pos2[0]) {
         return pos1[1] <= pos2[1];
      } else {
         return pos1[0] < pos2[0];
      }
   };
   EXPECT_TRUE(
      cmp_equal_rngs(config.start_fields[Team::BLUE], pos_blue, pos_comparator, pos_comparator));
   EXPECT_TRUE(
      cmp_equal_rngs(config.start_fields[Team::RED], pos_red, pos_comparator, pos_comparator));
}