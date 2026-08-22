
#ifndef NOR_TEXAS_HOLDEM_POKER_FIXTURES_HPP
#define NOR_TEXAS_HOLDEM_POKER_FIXTURES_HPP

#include <gtest/gtest.h>

#include <variant>

#include "texas_holdem_poker/texas_holdem_poker.hpp"

struct TexasHoldemState: public ::testing::Test {
   texholdem::State state{texholdem::PokerConfig{}};
};

/// one scripted transition of a hold'em hand: either a dealt card or a betting action
using HoldemStep = std::variant< texholdem::Card, texholdem::Action >;

/// applies the hole cards in chronological deal order: the first card goes to the small blind,
/// then clockwise around the table; the second hole card again starting at the small blind.
inline void apply_steps(texholdem::State& state, const std::vector< HoldemStep >& steps)
{
   for(const auto& step : steps) {
      if(std::holds_alternative< texholdem::Card >(step)) {
         state.apply_action(std::get< texholdem::Card >(step));
      } else {
         state.apply_action(std::get< texholdem::Action >(step));
      }
   }
}

class TerminalParamsF:
    public ::testing::TestWithParam< std::tuple<
       texholdem::PokerConfig,  // the state config
       std::vector< HoldemStep >,  // the full scripted action/outcome sequence
       bool  // whether the state is terminal afterwards
       > > {};

class PayoffParamsF:
    public ::testing::TestWithParam< std::tuple<
       texholdem::PokerConfig,  // the state config
       std::vector< HoldemStep >,  // the full scripted action/outcome sequence
       std::vector< double >  // expected net payoffs per player
       > > {};

#endif  // NOR_TEXAS_HOLDEM_POKER_FIXTURES_HPP
