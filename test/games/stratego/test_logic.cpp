
#include <gtest/gtest.h>

#include "fixtures.hpp"

TEST_F(MinimalState, action_is_valid)
{
   EXPECT_TRUE(state.logic()->is_valid(state, Action{Position{1, 1}, Position{2, 1}}));
   EXPECT_TRUE(state.logic()->is_valid(state, Action{Position{1, 4}, Position{2, 4}}));
   EXPECT_TRUE(state.logic()->is_valid(state, Action{Position{3, 0}, Position{2, 0}}));
   EXPECT_TRUE(state.logic()->is_valid(state, Action{Position{3, 0}, Position{1, 0}}));
//   EXPECT_TRUE(state.logic()->is_valid(state, Action{Position{1, 1}, Position{2, 1}}));
}