

#include <gtest/gtest.h>

#include "fixtures.hpp"
#include "leduc_poker/leduc_poker.hpp"
#include "testing_utils.hpp"

using namespace leduc;

TEST_F(LeducPokerState, apply_chance_action)
{
   state.apply_action(Rank::king, Suit::diamonds);
   state.apply_action(Rank::jack, Suit::diamonds);
   EXPECT_EQ(state.card(Player::one), Card(Rank::king, Suit::diamonds));
   EXPECT_EQ(state.card(Player::two), Card(Rank::jack, Suit::diamonds));
}

TEST_F(LeducPokerState, apply_action)
{
   state.apply_action(Rank::king, Suit::diamonds);
   state.apply_action(Rank::jack, Suit::diamonds);

   // P1 checks
   state.apply_action(ActionType::check);
   EXPECT_EQ(state.cards().size(), 2);
   EXPECT_EQ(state.history_since_bet(Player::one), Action{ActionType::check});
   EXPECT_EQ(state.history_since_bet().container()[0], Action{ActionType::check});
   EXPECT_EQ(state.history_since_bet().container().size(), 2);
   // P2 bets 2
   state.apply_action(ActionType::bet, 2.);
   EXPECT_EQ(state.history_since_bet().container().size(), 2);
   EXPECT_EQ(state.history_since_bet(Player::two), Action(ActionType::bet, 2.));
   EXPECT_EQ(state.history_since_bet(1), Action(ActionType::bet, 2.));
   // P1 raises 2
   state.apply_action(ActionType::bet, 2.);
   EXPECT_EQ(state.history_since_bet().container().size(), 2);
   EXPECT_EQ(state.history_since_bet(0), Action(ActionType::bet, 2.));
   EXPECT_EQ(state.history_since_bet(1), std::nullopt);
   // P2 calls
   state.apply_action(ActionType::check);
   EXPECT_EQ(state.history_since_bet().container().size(), 2);
   EXPECT_EQ(state.history_since_bet(0), Action(ActionType::bet, 2.));
   EXPECT_EQ(state.history_since_bet(1), Action{ActionType::check});
   // now the public card is added
   state.apply_action(Card{Rank::queen, Suit::diamonds});
   EXPECT_EQ(state.public_card(), Card(Rank::queen, Suit::diamonds));
   // P1 checks
   state.apply_action(ActionType::check);
   EXPECT_EQ(state.history_since_bet().container()[0], Action(ActionType::check));
   // P2 checks
   state.apply_action(Action{ActionType::check});
   EXPECT_EQ(state.history_since_bet().container()[1], Action(ActionType::check));
}

TEST_F(LeducPokerState, is_valid_chance_action)
{
   EXPECT_TRUE(state.is_valid(Rank::king, Suit::diamonds));
   EXPECT_TRUE(state.is_valid(Rank::queen, Suit::diamonds));
   EXPECT_TRUE(state.is_valid(Rank::king, Suit::clubs));

   state.apply_action(Rank::king, Suit::diamonds);
   EXPECT_FALSE(state.is_valid(Rank::king, Suit::diamonds));
   EXPECT_TRUE(state.is_valid(Rank::king, Suit::clubs));
   EXPECT_TRUE(state.is_valid(Rank::jack, Suit::diamonds));

   state.apply_action(Rank::jack, Suit::clubs);
   EXPECT_FALSE(state.is_valid(Rank::jack, Suit::clubs));
   EXPECT_FALSE(state.is_valid(Rank::king, Suit::diamonds));
}

TEST_F(LeducPokerState, is_valid_action)
{
   state.apply_action(Rank::king, Suit::diamonds);
   state.apply_action(Rank::jack, Suit::diamonds);
   EXPECT_TRUE(state.is_valid(ActionType::check));
   EXPECT_FALSE(state.is_valid(Rank::ace, Suit::clubs));
   EXPECT_FALSE(state.is_valid(Rank::king, Suit::diamonds));

   EXPECT_TRUE(state.is_valid(ActionType::check));
   EXPECT_TRUE(state.is_valid(ActionType::bet, 2.));
   EXPECT_TRUE(state.is_valid(ActionType::fold));

   state.apply_action(ActionType::bet, 2.);
   EXPECT_TRUE(state.is_valid(ActionType::bet, 2.));
   EXPECT_TRUE(state.is_valid(ActionType::check));
   EXPECT_TRUE(state.is_valid(ActionType::fold));
}

TEST_F(LeducPokerState, valid_chance_actions)
{
   EXPECT_TRUE(cmp_equal_rngs(
      state.chance_actions(),
      std::vector< Card >{
         {Rank::jack, Suit::clubs},
         {Rank::jack, Suit::diamonds},
         {Rank::queen, Suit::clubs},
         {Rank::queen, Suit::diamonds},
         {Rank::king, Suit::clubs},
         {Rank::king, Suit::diamonds}}
   ));

   state.apply_action(Rank::jack, Suit::clubs);

   EXPECT_TRUE(cmp_equal_rngs(
      state.chance_actions(),
      std::vector< Card >{
         {Rank::jack, Suit::diamonds},
         {Rank::queen, Suit::clubs},
         {Rank::queen, Suit::diamonds},
         {Rank::king, Suit::clubs},
         {Rank::king, Suit::diamonds}}
   ));

   state.apply_action(Rank::queen, Suit::diamonds);

   EXPECT_TRUE(cmp_equal_rngs(
      state.chance_actions(),
      std::vector< Card >{
         {Rank::jack, Suit::diamonds},
         {Rank::queen, Suit::clubs},
         {Rank::king, Suit::clubs},
         {Rank::king, Suit::diamonds}}
   ));
}

TEST_F(LeducPokerState, actions)
{
   EXPECT_TRUE(state.actions().empty());
   state.apply_action(Rank::king, Suit::diamonds);
   EXPECT_TRUE(state.actions().empty());
   state.apply_action(Rank::jack, Suit::clubs);
   EXPECT_TRUE(cmp_equal_rngs_unsorted(
      state.actions(),
      std::vector< Action >{{ActionType::check}, {ActionType::bet, 2.}, {ActionType::fold}}
   ));
}

TEST_P(TerminalParamsF, terminal_situations)
{
   auto [cards, actions_r1, actions_r2, expected_terminal] = GetParam();

   for(auto opt_card : ranges::span{cards}.subspan(0, 2)) {
      if(opt_card.has_value())
         state.apply_action(opt_card.value());
   }

   for(auto action : actions_r1) {
      state.apply_action(action);
   }
   if(cards[2].has_value()) {
      state.apply_action(cards[2].value());
   }
   for(auto action : actions_r2) {
      state.apply_action(action);
   }
   EXPECT_EQ(state.is_terminal(), expected_terminal);
}

INSTANTIATE_TEST_SUITE_P(
   terminal_situations_tests,
   TerminalParamsF,
   ::testing::Values(
      std::tuple{
         std::array< std::optional< Card >, 3 >{
            Card{Rank::jack, Suit::clubs},
            Card{Rank::queen, Suit::clubs},
            Card{Rank::king, Suit::diamonds}},
         std::vector< Action >{{ActionType::check}, {ActionType::check}},
         std::vector< Action >{{ActionType::check}, {ActionType::check}},
         true},
      std::tuple{
         std::array< std::optional< Card >, 3 >{
            Card{Rank::king, Suit::clubs},
            Card{Rank::queen, Suit::clubs},
            Card{Rank::king, Suit::diamonds}},
         std::vector< Action >{{ActionType::check}, {ActionType::bet, 2.}, {ActionType::check}},
         std::vector< Action >{{ActionType::check}, {ActionType::check}},
         true},
      std::tuple{
         std::array< std::optional< Card >, 3 >{
            Card{Rank::king, Suit::clubs},
            Card{Rank::queen, Suit::clubs},
            Card{Rank::king, Suit::diamonds}},
         std::vector< Action >{{ActionType::check}, {ActionType::bet, 2.}, {ActionType::fold}},
         std::vector< Action >{},
         true},
      std::tuple{
         std::array< std::optional< Card >, 3 >{
            Card{Rank::king, Suit::clubs},
            Card{Rank::queen, Suit::clubs},
            Card{Rank::jack, Suit::clubs}},
         std::vector< Action >{{ActionType::check}, {ActionType::check}},
         std::vector< Action >{{ActionType::check}},
         false},
      std::tuple{
         std::array< std::optional< Card >, 3 >{
            Card{Rank::king, Suit::clubs},
            Card{Rank::queen, Suit::clubs},
            Card{Rank::king, Suit::diamonds}},
         std::vector< Action >{{ActionType::check}, {ActionType::bet, 2.}, {ActionType::check}},
         std::vector< Action >{},
         false},
      std::tuple{
         std::array< std::optional< Card >, 3 >{
            Card{Rank::king, Suit::clubs},
            Card{Rank::queen, Suit::clubs},
            Card{Rank::queen, Suit::diamonds}},
         std::vector< Action >{{ActionType::check}, {ActionType::bet, 2.}, {ActionType::check}},
         std::vector< Action >{
            {ActionType::bet, 4.},
         },
         false},
      std::tuple{
         std::array< std::optional< Card >, 3 >{
            Card{Rank::king, Suit::clubs},
            Card{Rank::queen, Suit::clubs},
            Card{Rank::king, Suit::diamonds}},
         std::vector< Action >{{ActionType::check}, {ActionType::bet, 2.}, {ActionType::bet, 2.}},
         std::vector< Action >{{ActionType::check}, {ActionType::bet, 2.}},
         false},
      std::tuple{
         std::array< std::optional< Card >, 3 >{Card{Rank::king, Suit::clubs}},
         std::vector< Action >{},
         std::vector< Action >{},
         false},
      std::tuple{
         std::array< std::optional< Card >, 3 >{
            Card{Rank::king, Suit::clubs},
            Card{Rank::queen, Suit::clubs},
            Card{Rank::king, Suit::diamonds}},
         std::vector< Action >{{ActionType::check}, {ActionType::bet, 2.}, {ActionType::check}},
         std::vector< Action >{{ActionType::check}, {ActionType::check}},
         true}
   )
);

TEST_P(PayoffParamsF, payoff_combinations)
{
   auto [cards, actions_r1, actions_r2, expected_payoffs] = GetParam();

   for(auto opt_card : ranges::span{cards}.subspan(0, 2)) {
      if(opt_card.has_value())
         state.apply_action(opt_card.value());
   }

   for(auto action : actions_r1) {
      state.apply_action(action);
   }
   if(cards[2].has_value()) {
      state.apply_action(cards[2].value());
   }
   for(auto action : actions_r2) {
      state.apply_action(action);
   }
   auto payoffs = state.payoff();
   for(size_t player = 0; player < 2; player++) {
      EXPECT_EQ(payoffs[player], expected_payoffs[player]);
   }
}

INSTANTIATE_TEST_SUITE_P(
   payoff_combinations_tests,
   PayoffParamsF,
   ::testing::Values(
      std::tuple{
         std::array< std::optional< Card >, 3 >{
            Card{Rank::jack, Suit::clubs},
            Card{Rank::queen, Suit::clubs},
            Card{Rank::king, Suit::diamonds}},
         std::vector< Action >{{ActionType::check}, {ActionType::check}},
         std::vector< Action >{{ActionType::check}, {ActionType::check}},
         std::array{-1., 1.}},
      std::tuple{
         std::array< std::optional< Card >, 3 >{
            Card{Rank::king, Suit::clubs},
            Card{Rank::queen, Suit::clubs},
            Card{Rank::king, Suit::diamonds}},
         std::vector< Action >{{ActionType::check}, {ActionType::bet, 2.}, {ActionType::check}},
         std::vector< Action >{{ActionType::check}, {ActionType::check}},
         std::array{3., -3.}},
      std::tuple{
         std::array< std::optional< Card >, 3 >{
            Card{Rank::king, Suit::clubs},
            Card{Rank::queen, Suit::clubs},
            Card{Rank::king, Suit::diamonds}},
         std::vector< Action >{{ActionType::check}, {ActionType::bet, 2.}, {ActionType::fold}},
         std::vector< Action >{},
         std::array{-1., 1.}},
      std::tuple{
         std::array< std::optional< Card >, 3 >{
            Card{Rank::king, Suit::clubs},
            Card{Rank::queen, Suit::clubs},
            Card{Rank::jack, Suit::clubs}},
         std::vector< Action >{{ActionType::check}, {ActionType::check}},
         std::vector< Action >{
            {ActionType::check},
            {ActionType::bet, 4.},
            {ActionType::bet, 4.},
            {ActionType::check}},
         std::array{9., -9.}},
      std::tuple{
         std::array< std::optional< Card >, 3 >{
            Card{Rank::king, Suit::clubs},
            Card{Rank::queen, Suit::clubs},
            Card{Rank::king, Suit::diamonds}},
         std::vector< Action >{{ActionType::check}, {ActionType::bet, 2.}, {ActionType::check}},
         std::vector< Action >{{ActionType::bet, 4.}, {ActionType::fold}},
         std::array{3., -3.}},
      std::tuple{
         std::array< std::optional< Card >, 3 >{
            Card{Rank::king, Suit::clubs},
            Card{Rank::queen, Suit::clubs},
            Card{Rank::queen, Suit::diamonds}},
         std::vector< Action >{{ActionType::check}, {ActionType::bet, 2.}, {ActionType::check}},
         std::vector< Action >{{ActionType::bet, 4.}, {ActionType::bet, 4.}, {ActionType::check}},
         std::array{-11., 11.}},
      std::tuple{
         std::array< std::optional< Card >, 3 >{
            Card{Rank::king, Suit::clubs},
            Card{Rank::queen, Suit::clubs},
            Card{Rank::king, Suit::diamonds}},
         std::vector< Action >{
            {ActionType::check},
            {ActionType::bet, 2.},
            {ActionType::bet, 2.},
            {ActionType::check}},
         std::vector< Action >{{ActionType::check}, {ActionType::bet, 2.}, {ActionType::check}},
         std::array{7., -7.}},
      std::tuple{
         std::array< std::optional< Card >, 3 >{
            Card{Rank::jack, Suit::clubs},
            Card{Rank::queen, Suit::clubs},
            Card{Rank::king, Suit::diamonds}},
         std::vector< Action >{{ActionType::check}, {ActionType::bet, 2.}, {ActionType::check}},
         std::vector< Action >{{ActionType::check}, {ActionType::check}},
         std::array{-3., 3.}}
   )
);
