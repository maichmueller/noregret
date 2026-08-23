
#ifndef NOR_OSHI_ZUMO_FIXTURES_HPP
#define NOR_OSHI_ZUMO_FIXTURES_HPP

#include <gtest/gtest.h>

#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "nor/nor.hpp"
#include "oshi_zumo/oshi_zumo.hpp"

namespace oz = nor::games::oshi_zumo;
using namespace ::oshi_zumo;

/// one scripted simultaneous round: (player-one bid, player-two bid)
using OZJointStep = std::pair< Bid, Bid >;

struct OZScript {
   Config config{2, 6, 0, 12};
   std::vector< OZJointStep > rounds{};
};

/// applies the script's rounds through the commit-commit-resolve machine; stops early once
/// terminal and checks the phase progression along the way
inline State play_script(const OZScript& script)
{
   State state{script.config};
   for(const auto& [bid_one, bid_two] : script.rounds) {
      if(state.terminal()) {
         break;
      }
      EXPECT_EQ(state.phase(), Phase::commit_p1);
      EXPECT_EQ(state.active_player(), Player::one);
      state.apply_action(bid_one);
      EXPECT_EQ(state.phase(), Phase::commit_p2);
      EXPECT_EQ(state.active_player(), Player::two);
      state.apply_action(bid_two);
   }
   return state;
}

/// small-board default test configuration (CFR-sized coins)
inline Config test_config()
{
   return Config(3, 6, 0, 9);
}

struct OshiZumoState: public ::testing::Test {
   State state{test_config()};
};

struct OshiZumoBidding: public ::testing::Test {
   State state{test_config()};
};

struct OshiZumoPush: public ::testing::Test {
   State state{test_config()};
};

struct OshiZumoCoins: public ::testing::Test {
   State state{test_config()};
};

struct OshiZumoTermination: public ::testing::Test {};

struct OshiZumoRandomPlayouts: public ::testing::Test {};

struct OshiZumoInfo: public ::testing::Test {};

struct OshiZumoTraits: public ::testing::Test {};

struct OshiZumoCFR: public ::testing::Test {};

class OshiZumoPayoffParamsF:
    public ::testing::TestWithParam< std::tuple< OZScript, double, double > > {};

/// plays uniformly random legal actions through both commit phases until terminality; returns the
/// terminal cause
template < typename Rng >
TerminalCause random_oz_playout(State& state, Rng&& rng)
{
   size_t guard = 0;
   while(not state.terminal()) {
      auto legal = state.actions(state.active_player());
      if(legal.empty()) {
         ADD_FAILURE() << "no legal actions in a non-terminal state";
         return TerminalCause::none;
      }
      std::uniform_int_distribution< size_t > dist(0, legal.size() - 1);
      state.apply_action(legal[dist(rng)]);
      if(++guard >= 4 * Config::max_horizon + 8u) {
         ADD_FAILURE() << "playout failed to terminate";
         return TerminalCause::none;
      }
   }
   return state.terminal_cause();
}

#endif  // NOR_OSHI_ZUMO_FIXTURES_HPP
