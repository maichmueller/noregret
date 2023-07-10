
#ifndef NOR_LEDUC_POKER_FIXTURES_HPP
#define NOR_LEDUC_POKER_FIXTURES_HPP

#include <gtest/gtest.h>

#include "leduc_poker/leduc_poker.hpp"

struct LeducPokerState: public ::testing::Test {
   leduc::State state{leduc::LeducConfig::make()};
};

struct Leduc5PokerState: public ::testing::Test {
   leduc::State state{leduc::LeducConfig::make_leduc5()};
};

class TerminalParamsF:
    public ::testing::TestWithParam< std::tuple<
       std::array< std::optional< leduc::Card >, 3 >,  // chance cards
       std::vector< leduc::Action >,  // action sequence round 1
       std::vector< leduc::Action >,  // action sequence round 2
       bool  // whether the state is terminal
       > > {
  protected:
   leduc::State state{leduc::LeducConfig::make()};
};

class PayoffParamsF:
    public ::testing::TestWithParam< std::tuple<
       std::array< std::optional< leduc::Card >, 3 >,  // chance cards
       std::vector< leduc::Action >,  // action sequence round 1
       std::vector< leduc::Action >,  // action sequence round 2
       std::array< double, 2 >  // expected payoffs
       > > {
  protected:
   leduc::State state{leduc::LeducConfig::make()};
};

#endif  // NOR_LEDUC_POKER_FIXTURES_HPP
