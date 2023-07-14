

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
   EXPECT_EQ(state.history_since_bet().container().size(), 2);
   EXPECT_EQ(state.history_since_bet(Player::one), Action{ActionType::check});
   EXPECT_EQ(state.history_since_bet().container()[0], Action{ActionType::check});
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
   EXPECT_EQ(state.active_player(), Player::chance);
   EXPECT_EQ(state.history_since_bet(0), std::nullopt);
   EXPECT_EQ(state.history_since_bet(1), std::nullopt);
   // now the public card is added
   state.apply_action(Rank::queen, Suit::diamonds);
   EXPECT_EQ(state.public_card(), Card(Rank::queen, Suit::diamonds));
   // all bets are processed --> new betting round
   EXPECT_EQ(state.history_since_bet(0), std::nullopt);
   EXPECT_EQ(state.history_since_bet(1), std::nullopt);
   // P1 checks
   state.apply_action(ActionType::check);
   // assert this is not counted as a bet
   EXPECT_EQ(state.history_since_bet(0), Action{ActionType::check});
   EXPECT_EQ(state.history_since_bet(1), std::nullopt);
   // P2 checks
   state.apply_action(Action{ActionType::check});
   // assert that after finishing a betting round the history is reset
   EXPECT_EQ(state.history_since_bet(0), std::nullopt);
   EXPECT_EQ(state.history_since_bet(1), std::nullopt);
}

TEST(LeducPokerState_3_Players, apply_action_stakes_is_valid)
{
   leduc::State state = leduc::LeducConfig(
      3,
      Player::one,
      2,
      1.,
      {2., 4., 8.},
      {4., 16.},
      std::vector{
         Card{Rank::king, Suit::clubs},
         Card{Rank::queen, Suit::clubs},
         Card{Rank::two, Suit::diamonds},
         Card{Rank::ace, Suit::diamonds},
         Card{Rank::ten, Suit::diamonds},
         Card{Rank::seven, Suit::diamonds}}
   );
   state.apply_action(Rank::king, Suit::clubs);
   state.apply_action(Rank::seven, Suit::diamonds);
   state.apply_action(Rank::ace, Suit::diamonds);

   EXPECT_EQ(state.cards().size(), 3);

   // P1 checks
   auto action = Action{ActionType::check};
   EXPECT_EQ(state.active_player(), Player::one);
   state.apply_action(action);
   EXPECT_EQ(state.history_since_bet().container().size(), 3);
   EXPECT_EQ(state.history_since_bet(Player::one), action);
   EXPECT_EQ(state.history_since_bet().container()[0], action);
   // check certain actions are and some are not valid in this config
   EXPECT_TRUE(state.is_valid(ActionType::bet, 2.));
   EXPECT_TRUE(state.is_valid(ActionType::bet, 4.));
   EXPECT_TRUE(state.is_valid(ActionType::bet, 8.));
   EXPECT_FALSE(state.is_valid(ActionType::bet, 10.));
   EXPECT_FALSE(state.is_valid(ActionType::bet, 11.));
   EXPECT_FALSE(state.is_valid(ActionType::bet, 16.));
   // P2 bets 4
   EXPECT_EQ(state.active_player(), Player::two);
   action = Action{ActionType::bet, 4.};
   state.apply_action(action);
   EXPECT_EQ(state.history_since_bet().container().size(), 3);
   EXPECT_EQ(state.history_since_bet(Player::two), action);
   EXPECT_EQ(state.history_since_bet(Player::one), std::nullopt);
   EXPECT_EQ(state.history_since_bet(Player::three), std::nullopt);
   // P3 raises 8
   EXPECT_EQ(state.active_player(), Player::three);
   action = Action{ActionType::bet, 8.};
   state.apply_action(action);
   EXPECT_EQ(state.history_since_bet().container().size(), 3);
   EXPECT_EQ(state.history_since_bet(Player::three), action);
   EXPECT_EQ(state.history_since_bet(Player::one), std::nullopt);
   EXPECT_EQ(state.history_since_bet(Player::two), std::nullopt);
   // P1 calls
   EXPECT_EQ(state.active_player(), Player::one);
   action = Action{ActionType::check};
   state.apply_action(ActionType::check);
   EXPECT_EQ(state.history_since_bet().container().size(), 3);
   EXPECT_EQ(state.history_since_bet(0), action);
   EXPECT_EQ(state.history_since_bet(1), std::nullopt);
   EXPECT_NE(state.history_since_bet(2), std::nullopt);
   // P2 folds
   EXPECT_EQ(state.active_player(), Player::two);
   action = Action{ActionType::fold};
   state.apply_action(action);
   EXPECT_EQ(state.history_since_bet().container().size(), 3);
   // betting round concluded, all histories wiped
   EXPECT_EQ(state.history_since_bet(0), std::nullopt);
   EXPECT_EQ(state.history_since_bet(1), std::nullopt);
   EXPECT_EQ(state.history_since_bet(2), std::nullopt);
   // now the public card is added
   EXPECT_EQ(state.active_player(), Player::chance);
   auto chance_action = Card{Rank::queen, Suit::clubs};
   state.apply_action(chance_action);
   EXPECT_EQ(state.public_card(), chance_action);
   // all bets are processed --> new betting round
   EXPECT_EQ(state.history_since_bet(0), std::nullopt);
   EXPECT_EQ(state.history_since_bet(1), std::nullopt);
   EXPECT_EQ(state.history_since_bet(2), std::nullopt);
   EXPECT_TRUE(
      cmp_equal_rngs_unsorted(state.remaining_players(), std::vector{Player::one, Player::three})
   );
   // P1 checks
   EXPECT_EQ(state.active_player(), Player::one);
   action = Action{ActionType::check};
   state.apply_action(action);
   // assert this is not counted as a bet
   EXPECT_EQ(state.history_since_bet(Player::one), action);
   EXPECT_EQ(state.history_since_bet(Player::two), std::nullopt);
   EXPECT_EQ(state.history_since_bet(Player::three), std::nullopt);
   // P3 checks
   EXPECT_EQ(state.active_player(), Player::three);
   state.apply_action(ActionType::check);
   // assert that after finishing a betting round the history is reset
   EXPECT_EQ(state.history_since_bet(Player::one), std::nullopt);
   EXPECT_EQ(state.history_since_bet(Player::two), std::nullopt);
   EXPECT_EQ(state.history_since_bet(Player::three), std::nullopt);
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

TEST_F(LeducPokerState, legal_actions)
{
   // chance player is supposed to act
   EXPECT_TRUE(state.actions().empty());
   state.apply_action(Rank::king, Suit::diamonds);
   // chance player is supposed to act
   EXPECT_TRUE(state.actions().empty());
   state.apply_action(Rank::jack, Suit::clubs);
   // now the first player can act
   EXPECT_TRUE(cmp_equal_rngs_unsorted(
      state.actions(),
      std::vector< Action >{{ActionType::check}, {ActionType::bet, 2.}, {ActionType::fold}}
   ));
   state.apply_action(ActionType::check);
   EXPECT_TRUE(cmp_equal_rngs_unsorted(
      state.actions(),
      std::vector< Action >{{ActionType::check}, {ActionType::bet, 2.}, {ActionType::fold}}
   ));
   state.apply_action(ActionType::check);
   EXPECT_TRUE(cmp_equal_rngs_unsorted(state.actions(), std::vector< Action >{}));
}

TEST(LeducPokerState_5_Players, actions_and_stakes)
{
   const size_t n_players = 5;
   auto state = leduc::State(leduc::LeducConfig(
      n_players,
      Player::one,
      2,
      1.,
      {2},
      {4},
      std::vector{
         Card{Rank::ace, Suit::clubs},
         Card{Rank::ace, Suit::hearts},
         Card{Rank::king, Suit::clubs},
         Card{Rank::king, Suit::hearts},
         Card{Rank::queen, Suit::clubs},
         Card{Rank::jack, Suit::diamonds},
         Card{Rank::five, Suit::diamonds},
         Card{Rank::four, Suit::diamonds},
         Card{Rank::three, Suit::clubs},
         Card{Rank::three, Suit::hearts},
         Card{Rank::two, Suit::diamonds},
      }
   ));
   std::vector< Action > first_round{
      {ActionType::check},  // 1-passes
      {ActionType::bet, 2.},  // 2-raises
      {ActionType::check},  // 3-calls
      {ActionType::bet, 2.},  //  4-re-raises
      {ActionType::check},  //  5-calls on the re-raise
      {ActionType::fold},  // 1-folds
      {ActionType::check},  // 2-calls re-raise
      {ActionType::check},  // 3-calls re-raise
   };
   std::vector< Action > second_round{
      {ActionType::bet, 4.},  // 2-raises
      {ActionType::fold},  // 3-folds
      {ActionType::check},  // 4-calls
      {ActionType::fold},  // 5-folds
   };

   state.apply_action(Rank::two, Suit::diamonds);  // P1 - no chance of winning
   state.apply_action(Rank::king, Suit::clubs);  // P2 - good chance
   state.apply_action(Rank::king, Suit::hearts);  // P3 - good chance
   state.apply_action(Rank::ace, Suit::hearts);  // P4 - highest chance
   state.apply_action(Rank::three, Suit::hearts);  // P5 - okay chance
   // first betting round
   for(auto player_action : ranges::views::enumerate(first_round)) {
      auto [player, action] = player_action;
      EXPECT_EQ(state.active_player(), Player(player % n_players));
      EXPECT_TRUE(state.is_valid(action));
      state.apply_action(action);
   }
   for(auto player : {Player::three, Player::four, Player::five}) {
      EXPECT_EQ(state.stake(Player::two), state.stake(player));
   }
   // add the public card
   state.apply_action(Rank::five, Suit::diamonds);
   // second betting round
   for(auto player_action : ranges::views::enumerate(second_round)) {
      auto [player, action] = player_action;
      // player 1 folded so enumerate needs to start at 1
      EXPECT_EQ(state.active_player(), Player((player + 1)));
      EXPECT_TRUE(state.is_valid(action));
      state.apply_action(action);
   }
   EXPECT_EQ(state.stake(Player::two), state.stake(Player::four));
}

TEST_P(TerminalParamsF, terminal_situations)
{
   auto [config, actions_r1, actions_r2, expected_terminal] = GetParam();
   auto state = leduc::State{config};

   for(auto&& card :
       ranges::span{config.available_cards}.subspan(0, static_cast< long >(config.n_players))) {
      state.apply_action(card);
   }

   for(auto action : actions_r1) {
      ASSERT_TRUE(state.is_valid(action));
      state.apply_action(action);
   }
   // public card
   if(not actions_r2.empty() and state.active_player() == Player::chance) {
      ASSERT_TRUE(state.is_valid(config.available_cards.back()));
      state.apply_action(config.available_cards.back());
   }
   for(auto action : actions_r2) {
      ASSERT_TRUE(state.is_valid(action));
      state.apply_action(action);
   }
   EXPECT_EQ(state.is_terminal(), expected_terminal);
}

INSTANTIATE_TEST_SUITE_P(
   terminal_situations_tests,
   TerminalParamsF,
   ::testing::Values(
      std::tuple{
         leduc::LeducConfig(),
         std::vector< Action >{{ActionType::check}, {ActionType::check}},
         std::vector< Action >{{ActionType::check}, {ActionType::check}},
         true},
      std::tuple{
         leduc::LeducConfig(
            5,
            Player::one,
            2,
            1.,
            {2},
            {4},
            std::vector{
               Card{Rank::king, Suit::clubs},
               Card{Rank::queen, Suit::clubs},
               Card{Rank::two, Suit::diamonds},
               Card{Rank::ace, Suit::diamonds},
               Card{Rank::ten, Suit::diamonds},
               Card{Rank::seven, Suit::diamonds}}
         ),
         std::vector< Action >{
            {ActionType::check},
            {ActionType::bet, 2.},
            {ActionType::check},
            {ActionType::check},
            {ActionType::check},
            {ActionType::check}},
         std::vector< Action >{{ActionType::check}, {ActionType::check}, {ActionType::check}},
         false},
      std::tuple{
         leduc::LeducConfig(
            3,
            Player::one,
            2,
            1.,
            {2},
            {4},
            std::vector{
               Card{Rank::king, Suit::clubs},
               Card{Rank::queen, Suit::clubs},
               Card{Rank::two, Suit::diamonds},
               Card{Rank::ace, Suit::diamonds},
               Card{Rank::ten, Suit::diamonds},
               Card{Rank::seven, Suit::diamonds}}
         ),
         std::vector< Action >{
            {ActionType::check},
            {ActionType::bet, 2.},
            {ActionType::fold},
            {ActionType::check}},
         std::vector< Action >{},
         false},
      std::tuple{
         leduc::LeducConfig(
            3,
            Player::one,
            2,
            1.,
            {2},
            {4},
            std::vector{
               Card{Rank::king, Suit::clubs},
               Card{Rank::queen, Suit::clubs},
               Card{Rank::two, Suit::diamonds},
               Card{Rank::ace, Suit::diamonds},
               Card{Rank::ten, Suit::diamonds},
               Card{Rank::seven, Suit::diamonds}}
         ),
         std::vector< Action >{
            {ActionType::check},
            {ActionType::bet, 2.},
            {ActionType::fold},
            {ActionType::check}},
         std::vector< Action >{{ActionType::bet, 4.}, {ActionType::fold}},
         true},
      std::tuple{
         leduc::LeducConfig(
            3,
            Player::one,
            2,
            1.,
            {2},
            {4},
            std::vector{
               Card{Rank::king, Suit::clubs},
               Card{Rank::queen, Suit::clubs},
               Card{Rank::two, Suit::diamonds},
               Card{Rank::ace, Suit::diamonds},
               Card{Rank::ten, Suit::diamonds},
               Card{Rank::seven, Suit::diamonds}}
         ),
         std::vector< Action >{{ActionType::check}, {ActionType::bet, 2.}, {ActionType::fold}},
         std::vector< Action >{},
         false},
      std::tuple{
         leduc::LeducConfig(
            3,
            Player::one,
            2,
            1.,
            {2},
            {4},
            std::vector{
               Card{Rank::king, Suit::clubs},
               Card{Rank::queen, Suit::clubs},
               Card{Rank::two, Suit::diamonds},
               Card{Rank::ace, Suit::diamonds},
               Card{Rank::ten, Suit::diamonds},
               Card{Rank::seven, Suit::diamonds}}
         ),
         std::vector< Action >{{ActionType::check}, {ActionType::check}},
         std::vector< Action >{{ActionType::check}},
         false},
      std::tuple{
         leduc::LeducConfig(),
         std::vector< Action >{{ActionType::check}, {ActionType::bet, 2.}, {ActionType::check}},
         std::vector< Action >{},
         false},
      std::tuple{
         leduc::LeducConfig(),
         std::vector< Action >{{ActionType::check}, {ActionType::bet, 2.}, {ActionType::check}},
         std::vector< Action >{{ActionType::bet, 4.}},
         false},
      std::tuple{
         leduc::LeducConfig(
            4,
            Player::one,
            2,
            1.,
            {2},
            {4},
            std::vector{
               Card{Rank::king, Suit::clubs},
               Card{Rank::queen, Suit::clubs},
               Card{Rank::two, Suit::diamonds},
               Card{Rank::ace, Suit::diamonds},
               Card{Rank::jack, Suit::diamonds},
               Card{Rank::three, Suit::hearts}}
         ),
         std::vector< Action >{
            {ActionType::check},  // 1-passes
            {ActionType::bet, 2.},  // 2-raises
            {ActionType::bet, 2.},  // 3-re-raises
            {ActionType::check},  //  4-calls on the re-raise
            {ActionType::fold},  // 1-folds
            {ActionType::check},  // 2-calls re-raise
         },
         std::vector< Action >{
            {ActionType::bet, 4.},  // 2-raises
            {ActionType::fold},  // 3-folds
            {ActionType::fold},  // 4-folds
         },
         true},
      std::tuple{
         leduc::LeducConfig(
            4,
            Player::one,
            2,
            1.,
            {2},
            {4},
            std::vector{
               Card{Rank::king, Suit::clubs},
               Card{Rank::queen, Suit::clubs},
               Card{Rank::two, Suit::diamonds},
               Card{Rank::ace, Suit::diamonds},
               Card{Rank::ten, Suit::diamonds},
               Card{Rank::jack, Suit::diamonds}}
         ),
         std::vector< Action >{
            {ActionType::check},  // 1-passes
            {ActionType::bet, 2.},  // 2-raises
            {ActionType::bet, 2.},  // 3-re-raises
            {ActionType::check},  //  4-calls on the re-raise
            {ActionType::fold},  // 1-folds
            {ActionType::check},  // 2-calls re-raise
         },
         std::vector< Action >{{ActionType::bet, 4.}},
         false}
   )
);

TEST_P(PayoffParamsF, payoff_combinations)
{
   auto [config, player_cards, public_card, actions_r1, actions_r2, expected_payoffs] = GetParam();
   auto state = leduc::State{config};

   for(auto card : player_cards) {
      state.apply_action(card);
   }

   for(auto action : actions_r1) {
      state.apply_action(action);
   }
   state.apply_action(public_card);

   for(auto action : actions_r2) {
      state.apply_action(action);
   }

   ASSERT_TRUE(state.is_terminal());
   auto payoffs = state.payoff();
   for(size_t player = 0; player < 2; player++) {
      EXPECT_EQ(payoffs[player], expected_payoffs[player])
         << "with player=" << player << ", payoffs=" << common::RangePrinter(payoffs)
         << ", expected_payoffs=" << common::RangePrinter(expected_payoffs);
   }
}

INSTANTIATE_TEST_SUITE_P(
   payoff_combinations_tests,
   PayoffParamsF,
   ::testing::Values(
      std::tuple{
         leduc::LeducConfig(),
         std::vector{Card{Rank::jack, Suit::diamonds}, Card{Rank::jack, Suit::clubs}},
         Card{Rank::queen, Suit::diamonds},
         std::vector< Action >{{ActionType::check}, {ActionType::check}},
         std::vector< Action >{{ActionType::check}, {ActionType::check}},
         std::vector{0., 0.}},
      std::tuple{
         leduc::LeducConfig(),
         std::vector{Card{Rank::king, Suit::diamonds}, Card{Rank::queen, Suit::clubs}},
         Card{Rank::jack, Suit::diamonds},
         std::vector< Action >{{ActionType::check}, {ActionType::bet, 2.}, {ActionType::check}},
         std::vector< Action >{{ActionType::check}, {ActionType::check}},
         std::vector{3., -3.}},
      std::tuple{
         leduc::LeducConfig(),
         std::vector{Card{Rank::queen, Suit::diamonds}, Card{Rank::king, Suit::clubs}},
         Card{Rank::queen, Suit::clubs},
         std::vector< Action >{{ActionType::bet, 2.}, {ActionType::bet, 2.}, {ActionType::check}},
         std::vector< Action >{{ActionType::bet, 4.}, {ActionType::check}},
         std::vector{9., -9.}},
      std::tuple{
         leduc::LeducConfig(),
         std::vector{Card{Rank::jack, Suit::diamonds}, Card{Rank::queen, Suit::clubs}},
         Card{Rank::king, Suit::clubs},
         std::vector< Action >{{ActionType::check}, {ActionType::bet, 2.}, {ActionType::fold}},
         std::vector< Action >{},
         std::vector{-1., 1.}},
      std::tuple{
         leduc::LeducConfig(),
         std::vector{Card{Rank::jack, Suit::diamonds}, Card{Rank::queen, Suit::clubs}},
         Card{Rank::king, Suit::clubs},
         std::vector< Action >{{ActionType::check}, {ActionType::check}},
         std::vector< Action >{
            {ActionType::check},
            {ActionType::bet, 4.},
            {ActionType::bet, 4.},
            {ActionType::check}},
         std::vector{-9., 9.}},
      std::tuple{
         leduc::LeducConfig(),
         std::vector{Card{Rank::jack, Suit::diamonds}, Card{Rank::queen, Suit::clubs}},
         Card{Rank::jack, Suit::clubs},
         std::vector< Action >{{ActionType::check}, {ActionType::check}},
         std::vector< Action >{
            {ActionType::check},
            {ActionType::bet, 4.},
            {ActionType::bet, 4.},
            {ActionType::check}},
         std::vector{9., -9.}},
      std::tuple{
         leduc::LeducConfig(),
         std::vector{Card{Rank::jack, Suit::diamonds}, Card{Rank::queen, Suit::clubs}},
         Card{Rank::jack, Suit::clubs},
         std::vector< Action >{{ActionType::check}, {ActionType::bet, 2.}, {ActionType::check}},
         std::vector< Action >{{ActionType::bet, 4.}, {ActionType::fold}},
         std::vector{3., -3.}},
      std::tuple{
         leduc::LeducConfig(),
         std::vector{Card{Rank::jack, Suit::diamonds}, Card{Rank::queen, Suit::clubs}},
         Card{Rank::jack, Suit::clubs},
         std::vector< Action >{{ActionType::check}, {ActionType::bet, 2.}, {ActionType::check}},
         std::vector< Action >{{ActionType::bet, 4.}, {ActionType::fold}},
         std::vector{3., -3.}},
      std::tuple{
         leduc::LeducConfig(),
         std::vector{Card{Rank::queen, Suit::diamonds}, Card{Rank::queen, Suit::clubs}},
         Card{Rank::jack, Suit::clubs},
         std::vector< Action >{{ActionType::check}, {ActionType::bet, 2.}, {ActionType::check}},
         std::vector< Action >{{ActionType::bet, 4.}, {ActionType::bet, 4.}, {ActionType::check}},
         std::vector{-0., 0.}},
      std::tuple{
         leduc::LeducConfig(
            4,
            Player::one,
            2,
            1.,
            {2},
            {4},
            std::vector< Card >{
               {Rank::ace, Suit::diamonds},
               {Rank::ace, Suit::clubs},
               {Rank::king, Suit::diamonds},
               {Rank::king, Suit::clubs},
               {Rank::queen, Suit::diamonds},
               {Rank::queen, Suit::clubs},
               {Rank::jack, Suit::diamonds},
               {Rank::jack, Suit::clubs},
               {Rank::two, Suit::diamonds},
               {Rank::three, Suit::diamonds},
            }
         ),
         std::vector< Card >{
            {Rank::ace, Suit::diamonds},
            {Rank::ace, Suit::clubs},
            {Rank::king, Suit::diamonds},
            {Rank::king, Suit::clubs}},
         Card{Rank::three, Suit::diamonds},
         std::vector< Action >{
            {ActionType::check},
            {ActionType::bet, 2.},
            {ActionType::check},
            {ActionType::check},
            {ActionType::check}},
         std::vector< Action >{
            {ActionType::bet, 4.},
            {ActionType::bet, 4.},
            {ActionType::check},
            {ActionType::check},
            {ActionType::check}},
         std::vector{11., 11., -11., -11.}},
      std::tuple{
         leduc::LeducConfig(
            4,
            Player::one,
            2,
            1.,
            {2},
            {4},
            std::vector< Card >{
               {Rank::ace, Suit::diamonds},
               {Rank::ace, Suit::clubs},
               {Rank::king, Suit::diamonds},
               {Rank::king, Suit::clubs},
               {Rank::queen, Suit::diamonds},
               {Rank::queen, Suit::clubs},
               {Rank::jack, Suit::diamonds},
               {Rank::jack, Suit::clubs},
               {Rank::two, Suit::diamonds},
               {Rank::three, Suit::diamonds}}
         ),
         std::vector< Card >{
            {Rank::ace, Suit::diamonds},
            {Rank::queen, Suit::clubs},
            {Rank::king, Suit::diamonds},
            {Rank::king, Suit::clubs}},
         Card{Rank::queen, Suit::diamonds},
         std::vector< Action >{
            {ActionType::check},
            {ActionType::bet, 2.},
            {ActionType::check},
            {ActionType::check},
            {ActionType::check}},
         std::vector< Action >{
            {ActionType::bet, 4.},
            {ActionType::bet, 4.},
            {ActionType::check},
            {ActionType::check},
            {ActionType::check}},
         std::vector{-11., 33., -11., -11.}}
   )
);
