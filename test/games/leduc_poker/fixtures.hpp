
#ifndef NOR_LEDUC_POKER_FIXTURES_HPP
#define NOR_LEDUC_POKER_FIXTURES_HPP

#include <gtest/gtest.h>

#include "leduc_poker/leduc_poker.hpp"

struct LeducPokerState: public ::testing::Test {
   leduc::State state{std::make_shared<leduc::LeducConfig>()};
};

class TerminalParamsF:
    public ::testing::TestWithParam< std::tuple<
       std::array< leduc::Card, 2 >,  // chance cards
       leduc::HistorySinceBet,  // action sequence
       bool > > {
  protected:
   leduc::State state;
};

class PayoffParamsF:
    public ::testing::TestWithParam< std::tuple<
       std::array< leduc::Card, 2 >,  // chance cards
       leduc::HistorySinceBet,  // action sequence
       std::array< int, 2 >  // payoffs
       > > {
  protected:
   leduc::State state;
};

#endif  // NOR_LEDUC_POKER_FIXTURES_HPP
