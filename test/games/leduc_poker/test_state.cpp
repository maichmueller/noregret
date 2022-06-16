

#include <gtest/gtest.h>

#include "fixtures.hpp"
#include "leduc_poker/leduc_poker.hpp"
#include "utils.hpp"

using namespace leduc;

TEST_F(LeducPokerState, apply_chance_action)
{
   state.apply_action(Card{Rank::king, Suit::diamonds});
   state.apply_action(Card{Rank::jack, Suit::diamonds});
   EXPECT_EQ(state.card(Player::one), Card(Rank::king, Suit::diamonds));
   EXPECT_EQ(state.card(Player::two), Card(Rank::jack, Suit::diamonds));
}

TEST_F(LeducPokerState, apply_action)
{
   state.apply_action(Card{Rank::king, Suit::diamonds});
   state.apply_action(Card{Rank::jack, Suit::diamonds});
   state.apply_action(Action{ActionType::check});
   EXPECT_EQ(state.history().container.size(), 1);
   EXPECT_EQ(state.history().container[0], Action(ActionType::check));
   state.apply_action(Action{ActionType::bet, 2.f});
   EXPECT_EQ(state.history().container.size(), 1);
   EXPECT_EQ(state.history().container[0], Action(ActionType::bet, 2.f));
   state.apply_action(Action{ActionType::bet, 2.f});
   EXPECT_EQ(state.history().container.size(), 2);
   EXPECT_EQ(state.history().container[1], Action(ActionType::bet, 2.f));
   state.apply_action(Card{Rank::queen, Suit::diamonds});
   EXPECT_EQ(state.public_card(), Card(Rank::queen, Suit::diamonds));
   state.apply_action(Action{ActionType::check});
   EXPECT_EQ(state.history().container.size(), 1);
   EXPECT_EQ(state.history().container[1], Action(ActionType::check));
}

//TEST_F(LeducPokerState, is_valid_chance_action)
//{
//   EXPECT_TRUE(state.is_valid(Card{Player::one, Rank::jack}));
//   EXPECT_TRUE(state.is_valid(Card{Player::one, Rank::queen}));
//   EXPECT_TRUE(state.is_valid(Card{Player::one, Card::king}));
//
//   state.apply_action(Card{Player::one, Card::king});
//   EXPECT_FALSE(state.is_valid(Card{Player::two, Card::king}));
//   EXPECT_TRUE(state.is_valid(Card{Player::two, Card::jack}));
//   EXPECT_TRUE(state.is_valid(Card{Player::two, Card::queen}));
//
//   state.apply_action(Card{Player::two, Card::queen});
//   EXPECT_FALSE(state.is_valid(Card{Player::two, Card::jack}));
//   EXPECT_FALSE(state.is_valid(Card{Player::two, Card::queen}));
//   EXPECT_FALSE(state.is_valid(Card{Player::two, Card::king}));
//}
//
//TEST_F(LeducPokerState, is_valid_action)
//{
//   EXPECT_FALSE(state.is_valid(Action::check));
//   EXPECT_FALSE(state.is_valid(Action::bet));
//
//   state.apply_action(Card{Player::one, Card::king});
//   state.apply_action(Card{Player::two, Card::queen});
//
//   EXPECT_TRUE(state.is_valid(Action::check));
//   EXPECT_TRUE(state.is_valid(Action::bet));
//
//   state.apply_action(Action::check);
//   EXPECT_TRUE(state.is_valid(Action::check));
//   EXPECT_TRUE(state.is_valid(Action::bet));
//   state.apply_action(Action::bet);
//   EXPECT_TRUE(state.is_valid(Action::check));
//   EXPECT_TRUE(state.is_valid(Action::bet));
//}
//
//TEST_F(LeducPokerState, valid_chance_actions)
//{
//   EXPECT_TRUE(cmp_equal_rngs(
//      state.chance_actions(),
//      std::vector{
//         Card{Player::one, Card::jack},
//         Card{Player::one, Card::queen},
//         Card{Player::one, Card::king}}));
//
//   state.apply_action(Card{Player::one, Card::king});
//
//   EXPECT_TRUE(cmp_equal_rngs(
//      state.chance_actions(),
//      std::vector{
//         Card{Player::two, Card::jack}, Card{Player::two, Card::queen}}));
//
//   state.apply_action(Card{Player::two, Card::queen});
//
//   EXPECT_TRUE(state.chance_actions().empty());
//}
//
//TEST_F(LeducPokerState, actions)
//{
//   EXPECT_TRUE(state.actions().empty());
//   state.apply_action(Card{Player::one, Card::king});
//   EXPECT_TRUE(state.actions().empty());
//   state.apply_action(Card{Player::two, Card::jack});
//   EXPECT_TRUE(cmp_equal_rngs(state.actions(), std::vector{Action::check, Action::bet}));
//}
//
//TEST_P(TerminalParamsF, terminal_situations)
//{
//   auto [cards, actions, expected_terminal] = GetParam();
//
//   state.apply_action({Player::one, cards[0]});
//   state.apply_action({Player::two, cards[1]});
//
//   for(auto action : actions.container) {
//      state.apply_action(action);
//   }
//   EXPECT_EQ(state.is_terminal(), expected_terminal);
//}
//
//INSTANTIATE_TEST_SUITE_P(
//   terminal_situations_tests,
//   TerminalParamsF,
//   ::testing::Values(
//      std::tuple{
//         std::array{Card::jack, Card::queen},
//         std::vector{Action::check, Action::check},
//         true},
//      std::tuple{std::array{Card::queen, Card::jack}, std::vector{Action::bet, Action::bet}, true},
//      std::tuple{
//         std::array{Card::king, Card::jack},
//         std::vector{Action::check, Action::bet, Action::bet},
//         true},
//      std::tuple{
//         std::array{Card::queen, Card::king},
//         std::vector{Action::check, Action::bet, Action::check},
//         true},
//      std::tuple{
//         std::array{Card::queen, Card::king},
//         std::vector{Action::check, Action::bet, Action::check},
//         true},
//      std::tuple{
//         std::array{Card::queen, Card::king},
//         std::vector{Action::check, Action::bet},
//         false},
//      std::tuple{std::array{Card::queen, Card::king}, std::vector{Action::bet}, false},
//      std::tuple{
//         std::array{Card::king, Card::jack},
//         std::vector{Action::check, Action::bet},
//         false}));
//
//TEST_P(PayoffParamsF, payoff_combinations)
//{
//   auto [cards, actions, expected_payoffs] = GetParam();
//   state.apply_action({Player::one, cards[0]});
//   state.apply_action({Player::two, cards[1]});
//   for(auto action : actions.container) {
//      state.apply_action(action);
//   }
//   for(size_t player = 0; player < 2; player++) {
//      EXPECT_EQ(state.payoff(Player(player)), expected_payoffs[player]);
//   }
//}
//
//INSTANTIATE_TEST_SUITE_P(
//   payoff_combinations_tests,
//   PayoffParamsF,
//   ::testing::Values(
//      std::tuple{
//         std::array{Card::jack, Card::queen},
//         std::vector{Action::check, Action::check},
//         std::array{-1, 1}},
//      std::tuple{
//         std::array{Card::queen, Card::jack},
//         std::vector{Action::bet, Action::bet},
//         std::array{2, -2}},
//      std::tuple{
//         std::array{Card::king, Card::jack},
//         std::vector{Action::check, Action::bet, Action::bet},
//         std::array{2, -2}},
//      std::tuple{
//         std::array{Card::king, Card::jack},
//         std::vector{Action::check, Action::bet, Action::check},
//         std::array{-1, 1}},
//      std::tuple{
//         std::array{Card::king, Card::jack},
//         std::vector{Action::bet, Action::check},
//         std::array{1, -1}},
//      std::tuple{
//         std::array{Card::king, Card::jack},
//         std::vector{Action::bet, Action::bet},
//         std::array{2, -2}},
//      std::tuple{
//         std::array{Card::queen, Card::king},
//         std::vector{Action::check, Action::bet, Action::check},
//         std::array{-1, 1}}));
