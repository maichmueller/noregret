
#include <gtest/gtest.h>

#include "fixtures.hpp"
#include "utils.hpp"

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
      500,
      {false, false},
      std::map{
         std::pair{Team::BLUE, std::optional(setup0)},
         std::pair{Team::RED, std::optional(setup1)}}};

   ASSERT_EQ(config.setups[Team::BLUE].value(), setup0);
   ASSERT_EQ(config.setups[Team::RED].value(), setup1);

   ASSERT_EQ(
      config.token_counters[Team::BLUE],
      (std::map< Token, unsigned int >{
         {Token::flag, 1},
         {Token::spy, 1},
         {Token::scout, 3},
         {Token::miner, 2},
         {Token::marshall, 1},
         {Token::bomb, 2}}));
   ASSERT_EQ(
      config.token_counters[Team::RED],
      (std::map< Token, unsigned int >{
         {Token::flag, 1},
         {Token::spy, 1},
         {Token::scout, 3},
         {Token::miner, 2},
         {Token::marshall, 1},
         {Token::bomb, 2}}));

   ASSERT_EQ(
      config.start_positions[Team::BLUE],
      (std::vector< Position >{
         {0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 0}, {1, 1}, {1, 2}, {1, 3}, {1, 4}}));
   ASSERT_EQ(
      config.start_positions[Team::RED],
      (std::vector< Position >{
         {3, 0}, {3, 1}, {3, 2}, {3, 3}, {3, 4}, {4, 0}, {4, 1}, {4, 2}, {4, 3}, {4, 4}}));
}

TEST(Config, constructor_custom_dims_with_setup)
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
      500,
      {false, false},
      std::map{
         std::pair{Team::BLUE, std::optional(setup0)}, std::pair{Team::RED, std::optional(setup1)}},
      hole_pos};

   ASSERT_EQ(config.setups[Team::BLUE].value(), setup0);
   ASSERT_EQ(config.setups[Team::RED].value(), setup1);

   ASSERT_EQ(
      config.token_counters[Team::BLUE],
      (std::map< Token, unsigned int >{
         {Token::flag, 1}, {Token::spy, 1}, {Token::scout, 2}, {Token::miner, 1}}));
   ASSERT_EQ(
      config.token_counters[Team::RED],
      (std::map< Token, unsigned int >{{Token::flag, 1}, {Token::spy, 3}, {Token::marshall, 1}}));

   auto pos_comparator = [](const Position& pos1, const Position& pos2) {
      if(pos1[0] == pos2[0]) {
         return pos1[1] <= pos2[1];
      } else {
         return pos1[0] < pos2[0];
      }
   };
   ASSERT_TRUE(cmp_equal_rngs(
      config.start_positions[Team::BLUE],
      std::vector< Position >{{0, 0}, {0, 1}, {0, 2}, {1, 3}, {2, 4}},
      pos_comparator,
      pos_comparator));
   ASSERT_TRUE(cmp_equal_rngs(
      config.start_positions[Team::RED],
      std::vector< Position >{{3, 0}, {2, 1}, {1, 2}, {3, 3}, {3, 4}},
      pos_comparator,
      pos_comparator));
}

TEST(Config, constructor_custom_dims_no_setup)
{
   std::map< Team, std::optional< std::vector< Token > > > tokens;

   std::vector< Position > pos_blue;
   pos_blue.emplace_back(Position{0, 0});
   pos_blue.emplace_back(Position{3, 3});

   std::vector< Position > pos_red;
   pos_red.emplace_back(Position{1, 2});
   pos_red.emplace_back(Position{3, 4});
   pos_red.emplace_back(Position{1, 4});
   pos_red.emplace_back(Position{3, 1});

   auto start_pos = std::map{
      std::pair{Team::BLUE, std::optional{pos_blue}}, std::pair{Team::RED, std::optional{pos_red}}};

   std::vector< Token > tokens_blue;
   tokens_blue.emplace_back(Token::flag);
   tokens_blue.emplace_back(Token::marshall);
   tokens_blue.emplace_back(Token::bomb);
   tokens_blue.emplace_back(Token::bomb);
   tokens_blue.emplace_back(Token::miner);
   tokens_blue.emplace_back(Token::miner);
   tokens_blue.emplace_back(Token::miner);

   std::vector< Token > tokens_red;
   tokens_red.emplace_back(Token::spy);
   tokens_red.emplace_back(Token::marshall);
   tokens_red.emplace_back(Token::major);
   tokens_red.emplace_back(Token::lieutenant);
   tokens_red.emplace_back(Token::lieutenant);
   tokens_red.emplace_back(Token::lieutenant);
   tokens_red.emplace_back(Token::lieutenant);

   std::vector< Position > hole_pos{{1, 1}};

   Config config{
      Team::BLUE,
      std::vector{size_t(3), size_t(4)},
      500,
      {false, false},
      Config::null_arg< Config::setup_t >(),
      hole_pos,
      tokens,
      start_pos,
   };

   ASSERT_EQ(
      config.token_counters[Team::BLUE],
      (std::map< Token, unsigned int >{
         {Token::flag, 1}, {Token::marshall, 1}, {Token::bomb, 2}, {Token::miner, 3}}));
   ASSERT_EQ(
      config.token_counters[Team::RED],
      (std::map< Token, unsigned int >{{Token::spy, 1}, {Token::marshall, 1}, {Token::major, 1}, {Token::lieutenant, 4}}));

   auto pos_comparator = [](const Position& pos1, const Position& pos2) {
      if(pos1[0] == pos2[0]) {
         return pos1[1] <= pos2[1];
      } else {
         return pos1[0] < pos2[0];
      }
   };
   ASSERT_TRUE(cmp_equal_rngs(
      config.start_positions[Team::BLUE],
      pos_blue,
      pos_comparator,
      pos_comparator));
   ASSERT_TRUE(cmp_equal_rngs(
      config.start_positions[Team::RED],
      pos_red,
      pos_comparator,
      pos_comparator));
}