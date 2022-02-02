
#include <gtest/gtest.h>

#include "fixtures.hpp"
#include "utils.hpp"

TEST_F(MinimalState, action_is_valid)
{
   EXPECT_TRUE(state.logic()->is_valid(state, Action{{1, 1}, {2, 1}}, Team::BLUE));
   EXPECT_TRUE(state.logic()->is_valid(state, Action{{1, 4}, {2, 4}}, Team::BLUE));
   EXPECT_TRUE(state.logic()->is_valid(state, Action{{1, 1}, {2, 1}}, Team::BLUE));
   EXPECT_TRUE(state.logic()->is_valid(state, Action{{3, 0}, {2, 0}}, Team::RED));
   EXPECT_TRUE(state.logic()->is_valid(state, Action{{3, 0}, {1, 0}}, Team::RED));

   // cant walk onto own pieces
   EXPECT_FALSE(state.logic()->is_valid(state, Action{{0, 0}, {1, 0}}, Team::BLUE));
   EXPECT_FALSE(state.logic()->is_valid(state, Action{{0, 3}, {1, 3}}, Team::BLUE));
   EXPECT_FALSE(state.logic()->is_valid(state, Action{{1, 1}, {0, 1}}, Team::BLUE));
   EXPECT_FALSE(state.logic()->is_valid(state, Action{{4, 3}, {3, 3}}, Team::RED));
   EXPECT_FALSE(state.logic()->is_valid(state, Action{{4, 2}, {3, 2}}, Team::RED));
   // cant walk diagonally
   EXPECT_FALSE(state.logic()->is_valid(state, Action{{4, 2}, {3, 3}}, Team::RED));
   EXPECT_FALSE(state.logic()->is_valid(state, Action{{1, 2}, {2, 1}}, Team::BLUE));
   // cant walk onto hole
   EXPECT_FALSE(state.logic()->is_valid(state, Action{{1, 2}, {2, 2}}, Team::BLUE));
   EXPECT_FALSE(state.logic()->is_valid(state, Action{{1, 2}, {3, 2}}, Team::BLUE));
   // cant walk too far
   EXPECT_FALSE(state.logic()->is_valid(state, Action{{1, 1}, {3, 1}}, Team::BLUE));
   EXPECT_FALSE(state.logic()->is_valid(state, Action{{1, 4}, {3, 4}}, Team::BLUE));
   // cant step over pieces
   EXPECT_FALSE(state.logic()->is_valid(state, Action{{3, 1}, {0, 1}}, Team::RED));
}

template < typename T >
struct eq {
   T value;
   eq(T val) : value(val) {}
   bool operator==(const eq& actions2) const
   {
      auto pos_sorter = [](const Position& pos1, const Position& pos2) {
         if(pos1[0] == pos2[0]) {
            return pos1[1] <= pos2[1];
         } else {
            return pos1[0] < pos2[0];
         }
      };
      auto sorter = [&](const Action& act1, const Action& act2) {
         if(act1[0] == act2[0]) {
            return pos_sorter(act1[1], act2[1]);
         } else {
            return pos_sorter(act1[0], act2[0]);
         }
      };
      return cmp_equal_rngs(value, actions2.value, sorter, sorter);
   };
};

TEST_F(MinimalState, apply_action)
{
   // move marshall one field up
   state.apply_action({{1, 1}, {2, 1}});
   // previous field should now be empty
   EXPECT_THROW((state.board()[{1, 1}].value()), std::bad_optional_access);

   auto& piece = state.board()[{2, 1}].value();
   EXPECT_EQ(piece.position(), Position(2, 1));
   EXPECT_EQ(piece.token(), Token::marshall);
}

TEST_F(MinimalState, valid_action_list)
{
   //   std::cout << state.to_string();

   EXPECT_EQ(
      eq(state.logic()->valid_actions(state, Team::BLUE)),
      eq(std::vector< Action >{{{1, 1}, {2, 1}}, {{1, 4}, {2, 4}}}));
   EXPECT_EQ(
      eq(state.logic()->valid_actions(state, Team::RED)),
      eq(std::vector< Action >{
         {{3, 0}, {1, 0}},
         {{3, 0}, {2, 0}},
         {{3, 1}, {1, 1}},
         {{3, 1}, {2, 1}},
         {{3, 3}, {1, 3}},
         {{3, 3}, {2, 3}},
         {{3, 4}, {2, 4}}}));

   // move marshall one field up
   state.apply_action({{1, 1}, {2, 1}});

   std::cout << state.to_string();
   LOGD2(
      "Valid actions blue",
      aze::utils::VectorPrinter{state.logic()->valid_actions(state, Team::BLUE)});
   LOGD2(
      "Valid actions expected blue",
      aze::utils::VectorPrinter(std::vector< Action >{
         {{2, 1}, {3, 1}},
         {{2, 1}, {1, 1}},
         {{2, 1}, {2, 0}},
         {{1, 2}, {1, 1}},
         {{0, 1}, {1, 1}},
         {{1, 4}, {2, 4}}}));
   EXPECT_EQ(
      eq(state.logic()->valid_actions(state, Team::BLUE)),
      eq(std::vector< Action >{
         {{2, 1}, {3, 1}},
         {{2, 1}, {1, 1}},
         {{2, 1}, {2, 0}},
         {{0, 1}, {1, 1}},
         {{1, 4}, {2, 4}}}));
   LOGD2(
      "Valid actions red",
      aze::utils::VectorPrinter{state.logic()->valid_actions(state, Team::RED)});
   LOGD2(
      "Valid actions expected red",
      aze::utils::VectorPrinter(std::vector< Action >{
         {{3, 0}, {1, 0}},
         {{3, 0}, {2, 0}},
         {{3, 1}, {2, 1}},
         {{3, 3}, {1, 3}},
         {{3, 3}, {2, 3}},
         {{3, 4}, {2, 4}}}));
   EXPECT_EQ(
      eq(state.logic()->valid_actions(state, Team::RED)),
      eq(std::vector< Action >{
         {{3, 0}, {1, 0}},
         {{3, 0}, {2, 0}},
         {{3, 1}, {2, 1}},
         {{3, 3}, {1, 3}},
         {{3, 3}, {2, 3}},
         {{3, 4}, {2, 4}}}));
}
