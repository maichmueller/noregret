
#ifndef NOR_KUHN_POKER_FIXTURES_HPP
#define NOR_KUHN_POKER_FIXTURES_HPP

#include <gtest/gtest.h>

#include "kuhn_poker/kuhn_poker.hpp"

struct KuhnPokerState: public ::testing::Test {
   kuhn::State state{};
};

class KuhnTerminalParamsF:
    public ::testing::TestWithParam< std::tuple<
       std::array< kuhn::Card, 2 >,  // chance cards
       kuhn::History,  // action sequence
       bool > > {
  protected:
   kuhn::State state;
};

class KuhnPayoffParamsF:
    public ::testing::TestWithParam< std::tuple<
       std::array< kuhn::Card, 2 >,  // chance cards
       kuhn::History,  // action sequence
       std::array< int, 2 >  // payoffs
       > > {
  protected:
   kuhn::State state;
};

struct KuhnThreePlayerState: public ::testing::Test {
   kuhn::State state{
      std::vector< kuhn::Card >{kuhn::Card::jack, kuhn::Card::queen, kuhn::Card::king},
      3};
};

/// deals one card to each of the three seats from the given deck assignment
inline kuhn::State
make_kuhn_three_player_state(kuhn::Card card_seat1, kuhn::Card card_seat2, kuhn::Card card_seat3)
{
   kuhn::State state{{kuhn::Card::jack, kuhn::Card::queen, kuhn::Card::king}, 3};
   state.apply_action(kuhn::ChanceOutcome{kuhn::Player::one, card_seat1});
   state.apply_action(kuhn::ChanceOutcome{kuhn::Player::two, card_seat2});
   state.apply_action(kuhn::ChanceOutcome{kuhn::Player::three, card_seat3});
   return state;
}

class KuhnThreePlayerDealParamsF:
    public ::testing::TestWithParam< std::tuple<
       std::array< kuhn::Card, 3 >,  // dealt cards in seat order
       double  // probability of the full ordered deal
       > > {};

class KuhnThreePlayerPayoffParamsF:
    public ::testing::TestWithParam< std::tuple<
       std::array< kuhn::Card, 3 >,  // chance cards in seat order
       kuhn::History,  // action sequence (seat order per betting rules)
       std::array< int, 3 >  // payoffs in seat order
       > > {};

#endif  // NOR_KUHN_POKER_FIXTURES_HPP
