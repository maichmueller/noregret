

#include <fmt/format.h>
#include <gtest/gtest.h>

#include "fixtures.hpp"
#include "leduc_poker/leduc_poker.hpp"
#include "testing_utils.hpp"

using namespace leduc;

TEST_P(BigLeducPayoffParamsF, payoff_combinations)
{
   auto [hole_cards, flop, actions_r1, actions_r2, expected_payoffs] = GetParam();
   auto local_state = leduc::State{big_leduc_config()};

   local_state.apply_action(hole_cards[0]);
   local_state.apply_action(hole_cards[1]);
   for(auto action : actions_r1) {
      ASSERT_TRUE(local_state.is_valid(action));
      local_state.apply_action(action);
   }
   ASSERT_TRUE(local_state.is_valid(flop));
   local_state.apply_action(flop);
   for(auto action : actions_r2) {
      ASSERT_TRUE(local_state.is_valid(action));
      local_state.apply_action(action);
   }

   ASSERT_TRUE(local_state.is_terminal());
   auto payoffs = local_state.payoff();
   EXPECT_EQ(payoffs[0], expected_payoffs.first)
      << fmt::format("payoffs={}", fmt::join(payoffs, ", "));
   EXPECT_EQ(payoffs[1], expected_payoffs.second)
      << fmt::format("payoffs={}", fmt::join(payoffs, ", "));
   // zero-sum by construction
   EXPECT_NEAR(payoffs[0] + payoffs[1], 0., 1e-12);
}

INSTANTIATE_TEST_SUITE_P(
   payoff_combinations_tests,
   BigLeducPayoffParamsF,
   ::testing::Values(
      // a pair of deuces beats a king-high: pairs trump any non-pair regardless of rank
      std::tuple{
         std::array{Card{Rank::two, Suit::clubs}, Card{Rank::king, Suit::clubs}},
         Card{Rank::two, Suit::diamonds},
         std::vector< Action >{{ActionType::bet, 2.}, {ActionType::check}},
         std::vector< Action >{{ActionType::bet, 4.}, {ActionType::check}},
         std::pair{7., -7.}},
      // a paired three beats queen-high
      std::tuple{
         std::array{Card{Rank::king, Suit::clubs}, Card{Rank::three, Suit::diamonds}},
         Card{Rank::three, Suit::clubs},
         std::vector< Action >{{ActionType::check}, {ActionType::check}},
         std::vector< Action >{{ActionType::bet, 4.}, {ActionType::check}},
         std::pair{-5., 5.}},
      // no pair anywhere: the higher hole card wins
      std::tuple{
         std::array{Card{Rank::nine, Suit::clubs}, Card{Rank::seven, Suit::diamonds}},
         Card{Rank::five, Suit::diamonds},
         std::vector< Action >{{ActionType::check}, {ActionType::check}},
         std::vector< Action >{{ActionType::check}, {ActionType::check}},
         std::pair{1., -1.}},
      // no pair, equal-ranked holes: even split (possible since both suits are in play)
      std::tuple{
         std::array{Card{Rank::nine, Suit::clubs}, Card{Rank::nine, Suit::diamonds}},
         Card{Rank::king, Suit::diamonds},
         std::vector< Action >{{ActionType::check}, {ActionType::check}},
         std::vector< Action >{{ActionType::check}, {ActionType::check}},
         std::pair{0., 0.}},
      // equal split of a bigger pot with odd chip count is exact in halves
      std::tuple{
         std::array{Card{Rank::nine, Suit::clubs}, Card{Rank::nine, Suit::diamonds}},
         Card{Rank::king, Suit::diamonds},
         std::vector< Action >{{ActionType::bet, 2.}, {ActionType::check}},
         std::vector< Action >{{ActionType::bet, 4.}, {ActionType::check}},
         std::pair{0., 0.}},
      // fold pre-flop: the opener takes the antes
      std::tuple{
         std::array{Card{Rank::king, Suit::clubs}, Card{Rank::two, Suit::diamonds}},
         Card{Rank::two, Suit::clubs},
         std::vector< Action >{{ActionType::bet, 2.}, {ActionType::fold}},
         std::vector< Action >{},
         std::pair{1., -1.}},
      // fold after a called first round and a second-round bet: winner nets the full pot
      std::tuple{
         std::array{Card{Rank::queen, Suit::clubs}, Card{Rank::jack, Suit::diamonds}},
         Card{Rank::two, Suit::clubs},
         std::vector< Action >{{ActionType::bet, 2.}, {ActionType::check}},
         std::vector< Action >{{ActionType::bet, 4.}, {ActionType::fold}},
         std::pair{3., -3.}}
   )
);

/// pairs beat high card for every one of the 12 ranks: hole rank r paired with the flop must
/// defeat any higher unpaired hole card
TEST(BigLeducShowdownTable, pair_beats_high_across_all_twelve_ranks)
{
   for(int r = static_cast< int >(Rank::two); r <= static_cast< int >(Rank::king); ++r) {
      auto paired_rank = Rank(r);
      auto state = leduc::State{leduc::LeducConfig::big_leduc()};
      state.apply_action(Card{paired_rank, Suit::clubs});
      // opponent holds the highest possible unpaired card (never the same suit twice)
      auto opp_rank = (paired_rank == Rank::king) ? Rank::queen : Rank::king;
      state.apply_action(Card{opp_rank, Suit::diamonds});
      auto flop = Card{paired_rank, Suit::diamonds};
      ASSERT_TRUE(state.is_valid(flop));
      state.apply_action(flop);
      state.apply_action(ActionType::check);
      state.apply_action(ActionType::check);
      ASSERT_TRUE(state.is_terminal());
      auto payoffs = state.payoff();
      EXPECT_EQ(payoffs[0], 1.) << fmt::format(
         "pair-of-{} should beat {}-high",
         static_cast< int >(paired_rank),
         static_cast< int >(opp_rank)
      );
      EXPECT_EQ(payoffs[1], -1.);
   }
}
