

#include <gtest/gtest.h>

#include "fixtures.hpp"
#include "kuhn_poker/kuhn_poker.hpp"
#include "utils.hpp"

using namespace kuhn;

TEST_F(KuhnPokerState, apply_chance_action)
{
   state.apply_action(Card::king);
   state.apply_action(Card::queen);
   EXPECT_EQ(state.card(Player::one).value(), Card::king);
   EXPECT_EQ(state.card(Player::two).value(), Card::queen);
   EXPECT_THROW(state.apply_action(Card::jack), std::logic_error);
}

TEST_F(KuhnPokerState, apply_action)
{
   state.apply_action(Card::king);
   state.apply_action(Card::queen);
   state.apply_action(Action::check);
   EXPECT_EQ(state.history().sequence.size(), 1);
   EXPECT_EQ(state.history().sequence[0], Action::check);
   state.apply_action(Action::bet);
   EXPECT_EQ(state.history().sequence.size(), 2);
   EXPECT_EQ(state.history().sequence[1], Action::bet);
   state.apply_action(Action::bet);
   EXPECT_EQ(state.history().sequence.size(), 3);
   EXPECT_EQ(state.history().sequence[2], Action::bet);
   state.apply_action(Action::check);
   EXPECT_EQ(state.history().sequence.size(), 4);
   EXPECT_EQ(state.history().sequence[3], Action::check);
   state.apply_action(Action::check);
   EXPECT_EQ(state.history().sequence.size(), 5);
   EXPECT_EQ(state.history().sequence[4], Action::check);
}

TEST_F(KuhnPokerState, is_valid_chance_action)
{
   EXPECT_TRUE(state.is_valid(Card::jack));
   EXPECT_TRUE(state.is_valid(Card::queen));
   EXPECT_TRUE(state.is_valid(Card::king));

   state.apply_action(Card::king);
   EXPECT_FALSE(state.is_valid(Card::king));
   EXPECT_TRUE(state.is_valid(Card::jack));
   EXPECT_TRUE(state.is_valid(Card::queen));

   state.apply_action(Card::queen);
   EXPECT_FALSE(state.is_valid(Card::jack));
   EXPECT_FALSE(state.is_valid(Card::queen));
   EXPECT_FALSE(state.is_valid(Card::king));
}

TEST_F(KuhnPokerState, is_valid_action)
{
   EXPECT_FALSE(state.is_valid(Action::check));
   EXPECT_FALSE(state.is_valid(Action::bet));

   state.apply_action(Card::king);
   state.apply_action(Card::queen);

   EXPECT_TRUE(state.is_valid(Action::check));
   EXPECT_TRUE(state.is_valid(Action::bet));

   state.apply_action(Action::check);
   EXPECT_TRUE(state.is_valid(Action::check));
   EXPECT_TRUE(state.is_valid(Action::bet));
   state.apply_action(Action::bet);
   EXPECT_TRUE(state.is_valid(Action::check));
   EXPECT_TRUE(state.is_valid(Action::bet));
}

TEST_F(KuhnPokerState, valid_chance_actions)
{
   EXPECT_TRUE(
      cmp_equal_rngs(state.chance_actions(), std::vector{Card::jack, Card::queen, Card::king}));

   state.apply_action(Card::king);

   EXPECT_TRUE(cmp_equal_rngs(state.chance_actions(), std::vector{Card::jack, Card::queen}));

   state.apply_action(Card::queen);

   EXPECT_TRUE(state.chance_actions().empty());
}

TEST_F(KuhnPokerState, actions)
{
   EXPECT_TRUE(state.actions().empty());
   state.apply_action(Card::king);
   EXPECT_TRUE(state.actions().empty());
   state.apply_action(Card::jack);

   EXPECT_TRUE(cmp_equal_rngs(state.actions(), std::vector{Action::check, Action::bet}));
}

TEST_P(TerminalParamsF, terminal_situations)
{
   auto [cards, actions, expected_terminal] = GetParam();
   for(auto card : cards) {
      state.apply_action(card);
   }
   for(auto action : actions.sequence) {
      state.apply_action(action);
   }
   EXPECT_EQ(state.is_terminal(), expected_terminal);
}

INSTANTIATE_TEST_SUITE_P(
   terminal_situations_tests,
   TerminalParamsF,
   ::testing::Values(
      std::tuple{
         std::array{Card::jack, Card::queen},
         std::vector{Action::check, Action::check},
         true},
      std::tuple{std::array{Card::queen, Card::jack}, std::vector{Action::bet, Action::bet}, true},
      std::tuple{
         std::array{Card::king, Card::jack},
         std::vector{Action::check, Action::bet, Action::bet},
         true},
      std::tuple{
         std::array{Card::queen, Card::king},
         std::vector{Action::check, Action::bet, Action::check},
         true},
      std::tuple{
         std::array{Card::queen, Card::king},
         std::vector{Action::check, Action::bet, Action::check},
         true},
      std::tuple{
         std::array{Card::queen, Card::king},
         std::vector{Action::check, Action::bet},
         false},
      std::tuple{std::array{Card::queen, Card::king}, std::vector{Action::bet}, false},
      std::tuple{
         std::array{Card::king, Card::jack},
         std::vector{Action::check, Action::bet},
         false}));

TEST_P(PayoffParamsF, payoff_combinations)
{
   auto [cards, actions, expected_payoffs] = GetParam();
   for(auto card : cards) {
      state.apply_action(card);
   }
   for(auto action : actions.sequence) {
      state.apply_action(action);
   }
   for(size_t player = 0; player < 2; player++) {
      EXPECT_EQ(state.payoff(Player(player)), expected_payoffs[player]);
   }
}

INSTANTIATE_TEST_SUITE_P(
   payoff_combinations_tests,
   PayoffParamsF,
   ::testing::Values(
      std::tuple{
         std::array{Card::jack, Card::queen},
         std::vector{Action::check, Action::check},
         std::array{-1, 1}},
      std::tuple{
         std::array{Card::queen, Card::jack},
         std::vector{Action::bet, Action::bet},
         std::array{2, -2}},
      std::tuple{
         std::array{Card::king, Card::jack},
         std::vector{Action::check, Action::bet, Action::bet},
         std::array{2, -2}},
      std::tuple{
         std::array{Card::queen, Card::king},
         std::vector{Action::check, Action::bet, Action::check},
         std::array{-1, 1}}));
