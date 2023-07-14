
#ifndef NOR_LEDUC_POKER_FIXTURES_HPP
#define NOR_LEDUC_POKER_FIXTURES_HPP

#include <gtest/gtest.h>

#include "leduc_poker/leduc_poker.hpp"

struct LeducPokerState: public ::testing::Test {
   leduc::State state{leduc::LeducConfig{}};
};

struct Leduc5PokerState: public ::testing::Test {
   leduc::State state{leduc::LeducConfig::leduc5()};
};

class TerminalParamsF:
    public ::testing::TestWithParam< std::tuple<
       leduc::LeducConfig,  // the state config
       std::vector< leduc::Action >,  // action sequence round 1
       std::vector< leduc::Action >,  // action sequence round 2
       bool  // whether the state is terminal
       > > {};

class PayoffParamsF:
    public ::testing::TestWithParam< std::tuple<
       leduc::LeducConfig,  // the state config
       std::vector< leduc::Card >,  // player cards
       leduc::Card,  // public card
       std::vector< leduc::Action >,  // action sequence round 1
       std::vector< leduc::Action >,  // action sequence round 2
       std::vector< double >  // expected payoffs
       > > {};

#endif  // NOR_LEDUC_POKER_FIXTURES_HPP
