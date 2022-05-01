
#ifndef NOR_KUHN_POKER_FIXTURES_HPP
#define NOR_KUHN_POKER_FIXTURES_HPP

#include <gtest/gtest.h>

#include "kuhn_poker/kuhn_poker.hpp"

struct KuhnPokerState: public ::testing::Test {
   kuhn::State state{};
};

class TerminalParamsF:
    public ::testing::TestWithParam< std::tuple<
       std::array< kuhn::Card, 2 >,  // chance cards
       kuhn::History,  // action sequence
       bool > > {
  protected:
   kuhn::State state;
};

class PayoffParamsF:
    public ::testing::TestWithParam< std::tuple<
       std::array< kuhn::Card, 2 >,  // chance cards
       kuhn::History,  // action sequence
       std::array< int, 2 >  // payoffs
       > > {
  protected:
   kuhn::State state;
};

#endif  // NOR_KUHN_POKER_FIXTURES_HPP
