
#include <gtest/gtest.h>

#include <utility>

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
      //         "Valid actions Team: " + utils::enum_name(Team(i)),
      //         aze::utils::VectorPrinter{state.logic()->valid_actions(state, Team(i))});
      //      LOGD2(
      //         "Valid actions expected Team: " + utils::enum_name(Team(i)),
      //         aze::utils::VectorPrinter(expected[Team(i)]));

      EXPECT_EQ(eq_rng(state.logic()->valid_actions(state, Team(i))), eq_rng(expected[Team(i)]));
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
      //         "Valid actions Team: " + utils::enum_name(Team(i)),
      //         aze::utils::VectorPrinter{state.logic()->valid_actions(state, Team(i))});
      //      LOGD2(
      //         "Valid actions expected Team: " + utils::enum_name(Team(i)),
      //         aze::utils::VectorPrinter(expected[Team(i)]));

      EXPECT_EQ(eq_rng(state.logic()->valid_actions(state, Team(i))), eq_rng(expected[Team(i)]));
   }
}

TEST_P(CheckTerminalParamsF, check_terminal)
{
   auto hole_pos = std::vector< Position >{Position{2, 2}};
   auto [turn_counter, game_dims, setups, status] = GetParam();

   // proxy state to get the defaults easily instantiated
   State s(Config(
      starting_team,
      fixed_starting_team,
      game_dims,
      max_turn_counts,
      fixed_setups,
      setups,
      hole_pos));

   LOGD2("Ref State to test", s.to_string())
   // actual state to test on
   State s_to_test(
      s.config(), s.graveyard(), s.logic(), s.board(), turn_counter, s.history(), s.rng());

   LOGD2("State to test", s_to_test.to_string())
   LOGD2("Observed Outcome", utils::enum_name(s.logic()->check_terminal(s)))
   LOGD2("Expected Outcome", utils::enum_name(status))

   EXPECT_EQ(s.logic()->check_terminal(s), status);
}

INSTANTIATE_TEST_SUITE_P(
   check_terminal_tests,
   CheckTerminalParamsF,
   ::testing::Values(
      std::tuple{// no movable pieces blue -> win red
                 50,
                 std::array< size_t, 2 >{5, 5},
                 std::map< Team, std::optional< Config::setup_t > >{
                    std::pair{
                       Team::BLUE,
                       std::optional(Config::setup_t{
                          std::pair{Position{0, 0}, Token::flag},
                          std::pair{Position{1, 1}, Token::bomb}})},
                    std::pair{
                       Team::RED,
                       std::optional(Config::setup_t{
                          std::pair{Position{3, 3}, Token::flag},
                          std::pair{Position{3, 4}, Token::spy}})}},
                 Status::WIN_BLUE},
      std::tuple{// no movable pieces red -> win blue
                 50,
                 std::array< size_t, 2 >{34, 28},
                 std::map< Team, std::optional< Config::setup_t > >{
                    std::pair{
                       Team::BLUE,
                       std::optional(Config::setup_t{
                          std::pair{Position{0, 0}, Token::flag},
                          std::pair{Position{1, 1}, Token::major}})},
                    std::pair{
                       Team::RED,
                       std::optional(Config::setup_t{
                          std::pair{Position{3, 3}, Token::flag},
                          std::pair{Position{3, 4}, Token::bomb}})}},
                 Status::WIN_BLUE},
      std::tuple{// mutual movable pieces elimination -> tie
                 50,
                 std::array< size_t, 2 >{4, 8},
                 std::map< Team, std::optional< Config::setup_t > >{
                    std::pair{
                       Team::BLUE,
                       std::optional(Config::setup_t{
                          std::pair{Position{2, 1}, Token::bomb},
                          std::pair{Position{0, 2}, Token::flag}})},
                    std::pair{
                       Team::RED,
                       std::optional(Config::setup_t{
                          std::pair{Position{3, 3}, Token::flag},
                          std::pair{Position{3, 2}, Token::bomb}})}},
                 Status::TIE},
      std::tuple{// turn counter too high, but otherwise ongoing -> tie
                 5000000,
                 std::array< size_t, 2 >{8, 5},
                 std::map< Team, std::optional< Config::setup_t > >{
                    std::pair{
                       Team::BLUE,
                       std::optional(Config::setup_t{
                          std::pair{Position{2, 1}, Token::spy},
                          std::pair{Position{0, 4}, Token::flag}})},
                    std::pair{
                       Team::RED,
                       std::optional(Config::setup_t{
                          std::pair{Position{3, 3}, Token::flag},
                          std::pair{Position{3, 4}, Token::spy}})}},
                 Status::TIE},
      std::tuple{// flag blue captured -> win red
                 50,
                 std::array< size_t, 2 >{10, 10},
                 std::map< Team, std::optional< Config::setup_t > >{
                    std::pair{
                       Team::BLUE,
                       std::optional(Config::setup_t{
                          std::pair{Position{2, 1}, Token::scout},
                          std::pair{Position{0, 4}, Token::bomb}})},
                    std::pair{
                       Team::RED,
                       std::optional(Config::setup_t{
                          std::pair{Position{3, 3}, Token::flag},
                          std::pair{Position{3, 4}, Token::spy}})}},
                 Status::WIN_RED},
      std::tuple{// flag red captured -> win blue
                 50,
                 std::array< size_t, 2 >{7, 7},
                 std::map< Team, std::optional< Config::setup_t > >{
                    std::pair{
                       Team::BLUE,
                       std::optional(Config::setup_t{
                          std::pair{Position{2, 1}, Token::scout},
                          std::pair{Position{0, 4}, Token::flag}})},
                    std::pair{
                       Team::RED,
                       std::optional(Config::setup_t{
                          std::pair{Position{3, 3}, Token::marshall},
                          std::pair{Position{3, 4}, Token::spy}})}},
                 Status::WIN_BLUE}));
