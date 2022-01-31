
#include <gtest/gtest.h>

#include "fixtures.hpp"

TEST_F(MinimalState, action_is_valid)
{
   EXPECT_TRUE(state.logic()->is_valid(state, Action{{1, 1}, {2, 1}}));
   EXPECT_TRUE(state.logic()->is_valid(state, Action{{1, 4}, {2, 4}}));
   EXPECT_TRUE(state.logic()->is_valid(state, Action{{3, 0}, {2, 0}}));
   EXPECT_TRUE(state.logic()->is_valid(state, Action{{3, 0}, {1, 0}}));
   EXPECT_TRUE(state.logic()->is_valid(state, Action{{1, 1}, {2, 1}}));

   // cant walk onto own pieces
   EXPECT_FALSE(state.logic()->is_valid(state, Action{{0, 0}, {1, 0}}));
   EXPECT_FALSE(state.logic()->is_valid(state, Action{{0, 3}, {1, 3}}));
   EXPECT_FALSE(state.logic()->is_valid(state, Action{{1, 1}, {0, 1}}));
   EXPECT_FALSE(state.logic()->is_valid(state, Action{{4, 3}, {3, 3}}));
   EXPECT_FALSE(state.logic()->is_valid(state, Action{{4, 2}, {3, 2}}));
   // cant walk diagonally
   EXPECT_FALSE(state.logic()->is_valid(state, Action{{4, 2}, {3, 3}}));
   EXPECT_FALSE(state.logic()->is_valid(state, Action{{1, 2}, {2, 1}}));
   // cant walk onto hole
   EXPECT_FALSE(state.logic()->is_valid(state, Action{{1, 2}, {2, 2}}));
   EXPECT_FALSE(state.logic()->is_valid(state, Action{{1, 2}, {3, 2}}));
   // cant walk too far
   EXPECT_FALSE(state.logic()->is_valid(state, Action{{1, 1}, {3, 1}}));
   EXPECT_FALSE(state.logic()->is_valid(state, Action{{1, 4}, {3, 4}}));
   // cant step over pieces
   EXPECT_FALSE(state.logic()->is_valid(state, Action{{3, 1}, {0, 1}}));
}