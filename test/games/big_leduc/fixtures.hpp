
#ifndef NOR_BIG_LEDUC_FIXTURES_HPP
#define NOR_BIG_LEDUC_FIXTURES_HPP

#include <gtest/gtest.h>

#include "leduc_poker/leduc_poker.hpp"

/// the canonical Big-Leduc configuration (see leduc::LeducConfig::big_leduc for the parameter
/// transcription and sources)
inline leduc::LeducConfig big_leduc_config()
{
   return leduc::LeducConfig::big_leduc();
}

struct BigLeducState: public ::testing::Test {
   leduc::State state{leduc::LeducConfig::big_leduc()};
};

class BigLeducTerminalParamsF:
    public ::testing::TestWithParam< std::tuple<
       std::vector< leduc::Action >,  // action sequence round 1
       std::vector< leduc::Action >,  // action sequence round 2
       bool  // whether the state is terminal
       > > {};

class BigLeducPayoffParamsF:
    public ::testing::TestWithParam< std::tuple<
       std::array< leduc::Card, 2 >,  // hole cards (P1, P2)
       leduc::Card,  // flop card
       std::vector< leduc::Action >,  // action sequence round 1
       std::vector< leduc::Action >,  // action sequence round 2
       std::pair< double, double >  // expected payoffs (P1, P2)
       > > {};

#endif  // NOR_BIG_LEDUC_FIXTURES_HPP
