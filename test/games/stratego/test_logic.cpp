
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
auto sorter = [](auto flattable1, auto flattable2) {
   std::array< int, 2 > reduces{0, 0};
   std::array both{flattable1.flatten(), flattable2.flatten()};
   for(unsigned int i = 0; i <= 2; ++i) {
      int factor = 1;
      for(auto val = both[i].rbegin(); val != both[i].rend(); val++) {
         reduces[i] += *val * factor;
         factor *= 10;
      }
   }
   return reduces[0] <= reduces[1];
};
template < typename T, typename Sorter = decltype(sorter) >
struct sorted {
   T value;
   explicit sorted(T val, Sorter sort = Sorter()) : value(val)
   {
      std::sort(value.begin(), value.end(), sort);
   }
};
template < typename T, typename Sorter = decltype(sorter) >
struct eq {
   sorted< T > sorted_elem;
   eq(T val, Sorter sort = Sorter()) : sorted_elem(val, sort) {}
   bool operator==(const eq& other) const
   {
      return cmp_equal_rngs(sorted_elem.value, other.sorted_elem.value);
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

   // move marshall onto enemy scout -> fight and win
   state.apply_action({{2, 1}, {3, 1}});

   piece = state.board()[{3, 1}].value();
   EXPECT_THROW((state.board()[{2, 1}].value()), std::bad_optional_access);
   EXPECT_EQ(piece.position(), Position(3, 1));
   EXPECT_EQ(piece.token(), Token::marshall);

   // move marshall onto enemy bomb -> fight and die
   auto state_copy = state.clone();
   state.apply_action({{3, 1}, {3, 2}});

   EXPECT_THROW((state.board()[{3, 1}].value()), std::bad_optional_access);
   piece = state.board()[{3, 2}].value();
   EXPECT_EQ(piece.position(), Position(3, 2));
   EXPECT_EQ(piece.token(), Token::bomb);
   EXPECT_EQ(piece.team(), Team::RED);

   // move spy onto enemy marshall -> fight and win
   state_copy->apply_action({{4, 1}, {3, 1}});

   EXPECT_THROW((state_copy->board()[{4, 1}].value()), std::bad_optional_access);
   piece = state_copy->board()[{3, 1}].value();
   EXPECT_EQ(piece.position(), Position(3, 1));
   EXPECT_EQ(piece.token(), Token::spy);
   EXPECT_EQ(piece.team(), Team::RED);
}

TEST_F(MinimalState, valid_action_list)
{
   //   std::cout << state.to_string();

   std::map< Team, std::vector< Action > > expected;
   expected[Team::BLUE] = std::vector< Action >{{{1, 1}, {2, 1}}, {{1, 4}, {2, 4}}};
   expected[Team::RED] = std::vector< Action >{
      {{3, 0}, {1, 0}},
      {{3, 0}, {2, 0}},
      {{3, 1}, {1, 1}},
      {{3, 1}, {2, 1}},
      {{3, 3}, {1, 3}},
      {{3, 3}, {2, 3}},
      {{3, 4}, {2, 4}}};

   for(int i = 0; i < 2; ++i) {
//      LOGD2(
//         "Valid actions Team: " + team_name(Team(i)),
//         aze::utils::VectorPrinter{state.logic()->valid_actions(state, Team(i))});
//      LOGD2(
//         "Valid actions expected Team: " + team_name(Team(i)),
//         aze::utils::VectorPrinter(expected[Team(i)]));

      EXPECT_EQ(eq(state.logic()->valid_actions(state, Team(i))), eq(expected[Team(i)]));
   }

   // move marshall one field up
   state.apply_action({{1, 1}, {2, 1}});

//   std::cout << state.to_string();

   expected[Team::BLUE] = std::vector< Action >{
      {{2, 1}, {3, 1}},
      {{2, 1}, {1, 1}},
      {{1, 2}, {1, 1}},
      {{0, 1}, {1, 1}},
      {{2, 1}, {2, 0}},
      {{1, 4}, {2, 4}}};
   expected[Team::RED] = std::vector< Action >{
      {{3, 0}, {1, 0}},
      {{3, 0}, {2, 0}},
      {{3, 1}, {2, 1}},
      {{3, 3}, {1, 3}},
      {{3, 3}, {2, 3}},
      {{3, 4}, {2, 4}}};

   for(int i = 0; i < 2; ++i) {
//      LOGD2(
//         "Valid actions Team: " + team_name(Team(i)),
//         aze::utils::VectorPrinter{state.logic()->valid_actions(state, Team(i))});
//      LOGD2(
//         "Valid actions expected Team: " + team_name(Team(i)),
//         aze::utils::VectorPrinter(expected[Team(i)]));

      EXPECT_EQ(eq(state.logic()->valid_actions(state, Team(i))), eq(expected[Team(i)]));
   }
}
