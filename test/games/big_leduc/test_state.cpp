

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <array>

#include "fixtures.hpp"
#include "leduc_poker/leduc_poker.hpp"
#include "testing_utils.hpp"

using namespace leduc;

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// Deck & chance dealing //////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

TEST_F(BigLeducState, deck_composition_twelve_ranks_two_suits)
{
   const auto& deck = state.config().available_cards;
   ASSERT_EQ(deck.size(), 24ul);

   std::array< int, 15 > rank_counts{};
   std::array< int, 4 > suit_counts{};
   for(const auto& card : deck) {
      rank_counts[static_cast< int >(card.rank)]++;
      suit_counts[static_cast< int >(card.suit)]++;
   }
   for(int r = static_cast< int >(Rank::two); r <= static_cast< int >(Rank::king); ++r) {
      EXPECT_EQ(rank_counts[r], 2) << "rank " << r;
   }
   EXPECT_EQ(rank_counts[static_cast< int >(Rank::ace)], 0);
   EXPECT_EQ(suit_counts[static_cast< int >(Suit::clubs)], 12);
   EXPECT_EQ(suit_counts[static_cast< int >(Suit::diamonds)], 12);
   EXPECT_EQ(suit_counts[static_cast< int >(Suit::hearts)], 0);
   EXPECT_EQ(suit_counts[static_cast< int >(Suit::spades)], 0);

   // the config reproduces standard Leduc defaults everywhere else
   EXPECT_EQ(state.config().n_players, 2ul);
   EXPECT_EQ(state.config().n_raises_allowed, 6ul);
   EXPECT_EQ(state.config().blind, 1.);
}

TEST_F(BigLeducState, chance_probabilities_and_exhaustion)
{
   // hole-card stage: uniform over all 24 cards
   auto outcomes = state.chance_actions();
   ASSERT_EQ(outcomes.size(), 24ul);
   double stage_sum = 0.;
   for(auto card : outcomes) {
      EXPECT_NEAR(state.chance_probability(card), 1. / 24., 1e-12);
      stage_sum += state.chance_probability(card);
   }
   EXPECT_NEAR(stage_sum, 1., 1e-12);

   state.apply_action(Card{Rank::seven, Suit::clubs});

   // second hole card: uniform over the remaining 23 cards; no repetitions
   outcomes = state.chance_actions();
   ASSERT_EQ(outcomes.size(), 23ul);
   EXPECT_FALSE(state.is_valid(Card{Rank::seven, Suit::clubs}));
   stage_sum = 0.;
   for(auto card : outcomes) {
      EXPECT_NEAR(state.chance_probability(card), 1. / 23., 1e-12);
      stage_sum += state.chance_probability(card);
   }
   EXPECT_NEAR(stage_sum, 1., 1e-12);

   state.apply_action(Card{Rank::king, Suit::diamonds});
   ASSERT_EQ(state.active_player(), Player::one);
   EXPECT_TRUE(state.actions().empty());

   // flop stage: uniform over the remaining 22 cards
   outcomes = state.chance_actions();
   ASSERT_EQ(outcomes.size(), 22ul);
   EXPECT_FALSE(state.is_valid(Card{Rank::seven, Suit::clubs}));
   EXPECT_FALSE(state.is_valid(Card{Rank::king, Suit::diamonds}));
   stage_sum = 0.;
   for(auto card : outcomes) {
      EXPECT_NEAR(state.chance_probability(card), 1. / 22., 1e-12);
      stage_sum += state.chance_probability(card);
   }
   EXPECT_NEAR(stage_sum, 1., 1e-12);

   state.apply_action(Card{Rank::two, Suit::diamonds});

   // exhaustion: after holes + flop there is nothing left to deal
   EXPECT_TRUE(state.chance_actions().empty());
   EXPECT_NEAR(state.chance_probability(Card{Rank::three, Suit::clubs}), 0., 1e-12);
   EXPECT_FALSE(state.is_terminal());
   EXPECT_EQ(state.active_player(), Player::one);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////// Bet ladder sizes ////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

TEST_F(BigLeducState, bet_ladder_matches_transcription)
{
   // round 1 offers exactly one bet size of 2 (fixed-limit small bet)
   state.apply_action(Card{Rank::king, Suit::clubs});
   state.apply_action(Card{Rank::jack, Suit::diamonds});
   EXPECT_TRUE(cmp_equal_rngs_unsorted(
      state.actions(),
      std::vector< Action >{{ActionType::check}, {ActionType::bet, 2.}, {ActionType::fold}}
   ));
   EXPECT_TRUE(state.is_valid(ActionType::bet, 2.));
   EXPECT_FALSE(state.is_valid(ActionType::bet, 4.));
   EXPECT_FALSE(state.is_valid(ActionType::bet, 0.5));
   EXPECT_FALSE(state.is_valid(ActionType::bet, 8.));

   // round 2 offers exactly one bet size of 4 (fixed-limit big bet)
   state.apply_action(ActionType::check);
   state.apply_action(ActionType::check);
   state.apply_action(Card{Rank::queen, Suit::clubs});
   EXPECT_TRUE(cmp_equal_rngs_unsorted(
      state.actions(),
      std::vector< Action >{{ActionType::check}, {ActionType::bet, 4.}, {ActionType::fold}}
   ));
   EXPECT_TRUE(state.is_valid(ActionType::bet, 4.));
   EXPECT_FALSE(state.is_valid(ActionType::bet, 2.));
   EXPECT_FALSE(state.is_valid(ActionType::bet, 8.));
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// Raise cap (max 6 bets) /////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

/// plays `n_bets` alternating opening bets/raises of `size` starting with player one and asserts
/// that the (n+1)-th bet became illegal while check(call)/fold remain available
void play_bets_until_cap(leduc::State& state, double size, size_t n_bets)
{
   for(size_t i = 0; i < n_bets; ++i) {
      ASSERT_TRUE(state.is_valid(ActionType::bet, size)) << "bet #" << i + 1;
      state.apply_action(ActionType::bet, size);
   }
   EXPECT_FALSE(state.is_valid(ActionType::bet, size));
   EXPECT_TRUE(cmp_equal_rngs_unsorted(
      state.actions(), std::vector< Action >{{ActionType::check}, {ActionType::fold}}
   ));
}

}  // namespace

TEST_F(BigLeducState, raise_cap_enforcement_round_one)
{
   state.apply_action(Card{Rank::king, Suit::clubs});
   state.apply_action(Card{Rank::queen, Suit::diamonds});

   play_bets_until_cap(state, 2., 6);

   // opening bet contributes once, each of the five responses matches + raises by 2
   EXPECT_EQ(state.stake(Player::one), 11.);
   EXPECT_EQ(state.stake(Player::two), 13.);

   // calling equalizes the stakes and closes round 1 --> on to the flop
   state.apply_action(ActionType::check);
   EXPECT_EQ(state.stake(Player::one), 13.);
   EXPECT_EQ(state.stake(Player::two), 13.);
   EXPECT_EQ(state.active_player(), Player::chance);
   EXPECT_FALSE(state.is_terminal());
}

TEST_F(BigLeducState, raise_cap_enforcement_round_two)
{
   state.apply_action(Card{Rank::king, Suit::clubs});
   state.apply_action(Card{Rank::queen, Suit::diamonds});
   state.apply_action(ActionType::check);
   state.apply_action(ActionType::check);
   state.apply_action(Card{Rank::nine, Suit::clubs});

   play_bets_until_cap(state, 4., 6);

   state.apply_action(ActionType::check);
   EXPECT_TRUE(state.is_terminal());
   EXPECT_EQ(state.stake(Player::one), 25.);
   EXPECT_EQ(state.stake(Player::two), 25.);
   // showdown: P1's kings pair the flop and beat P2's queen-high
   EXPECT_EQ(state.payoff(Player::one), 25.);
   EXPECT_EQ(state.payoff(Player::two), -25.);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////// Terminal states ////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(BigLeducTerminalParamsF, terminal_situations)
{
   auto [actions_r1, actions_r2, expected_terminal] = GetParam();
   auto local_state = leduc::State{big_leduc_config()};

   local_state.apply_action(Card{Rank::king, Suit::clubs});
   local_state.apply_action(Card{Rank::queen, Suit::diamonds});

   for(auto action : actions_r1) {
      ASSERT_TRUE(local_state.is_valid(action));
      local_state.apply_action(action);
   }
   if(not actions_r2.empty() and local_state.active_player() == Player::chance) {
      auto flop = Card{Rank::two, Suit::clubs};
      ASSERT_TRUE(local_state.is_valid(flop));
      local_state.apply_action(flop);
   }
   for(auto action : actions_r2) {
      ASSERT_TRUE(local_state.is_valid(action));
      local_state.apply_action(action);
   }
   EXPECT_EQ(local_state.is_terminal(), expected_terminal);
}

INSTANTIATE_TEST_SUITE_P(
   terminal_situations_tests,
   BigLeducTerminalParamsF,
   ::testing::Values(
      std::tuple{
         std::vector< Action >{{ActionType::bet, 2.}, {ActionType::fold}},
         std::vector< Action >{},
         true},
      std::tuple{
         std::vector< Action >{{ActionType::check}, {ActionType::check}},
         std::vector< Action >{},
         false},
      std::tuple{
         std::vector< Action >{{ActionType::check}, {ActionType::check}},
         std::vector< Action >{{ActionType::check}},
         false},
      std::tuple{
         std::vector< Action >{{ActionType::check}, {ActionType::check}},
         std::vector< Action >{{ActionType::check}, {ActionType::check}},
         true},
      std::tuple{
         std::vector< Action >{{ActionType::bet, 2.}, {ActionType::check}},
         std::vector< Action >{{ActionType::bet, 4.}, {ActionType::fold}},
         true}
   )
);
