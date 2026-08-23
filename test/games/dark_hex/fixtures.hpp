
#ifndef NOR_DARK_HEX_FIXTURES_HPP
#define NOR_DARK_HEX_FIXTURES_HPP

#include <gtest/gtest.h>

#include <vector>

#include "dark_hex/dark_hex.hpp"

struct DarkHexState: public ::testing::Test {
   dark_hex::State state{};
};

/**
 * A fully scripted dark hex game as a flat list of (actor, cell) attempts. Applied in order;
 * the actor labels are only assertions against the engine's own turn keeping (the state
 * dictates who acts).
 */
struct DarkHexScript {
   dark_hex::Config config{};
   std::vector< std::pair< dark_hex::Player, uint8_t > > moves{};
};

/// applies the script's attempts in order; stops early once the state turns terminal
inline dark_hex::State play_script(const DarkHexScript& script)
{
   using namespace dark_hex;
   State state{script.config};
   for(const auto& [player, cell] : script.moves) {
      if(state.terminal()) {
         break;
      }
      // the engine dictates who acts; the label is an assertion against our bookkeeping
      EXPECT_EQ(state.active_player(), player);
      state.apply_action(Move{cell});
   }
   return state;
}

class DarkHexPayoffParamsF:
    public ::testing::TestWithParam< std::tuple< DarkHexScript, double, double > > {};

#endif  // NOR_DARK_HEX_FIXTURES_HPP
