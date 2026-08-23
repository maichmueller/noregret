
#ifndef NOR_PURSUIT_EVASION_FIXTURES_HPP
#define NOR_PURSUIT_EVASION_FIXTURES_HPP

#include <gtest/gtest.h>

#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "nor/nor.hpp"
#include "pursuit_evasion/pursuit_evasion.hpp"

namespace pe = nor::games::pursuit_evasion;
using namespace ::pursuit_evasion;

/// the attacker move taking the black edge from 'from' to 'to' (throws if the figure has no such
/// directed edge)
inline AttMove att_edge(uint8_t from, uint8_t to)
{
   for(size_t e = 0; e < k_attacker_edges.size(); ++e) {
      if(k_attacker_edges[e].from == from && k_attacker_edges[e].to == to) {
         return AttMove{uint8_t(e)};
      }
   }
   throw std::invalid_argument("no attacker edge in the transcribed graph");
}

inline AttMove att_wait()
{
   return AttMove{AttMove::wait_edge_id};
}

/// the defender compound keeping both patrols on their initial centre nodes
inline DefMove def_stay()
{
   return DefMove{node_C, node_I};
}

inline DefMove def_to(uint8_t p1, uint8_t p2)
{
   return DefMove{p1, p2};
}

/// one scripted simultaneous round: (attacker commitment, defender commitment)
using PEJointStep = std::pair< AttMove, DefMove >;

struct PEScript {
   Config config{4};
   std::vector< PEJointStep > rounds{};
};

/// applies the script's rounds through the commit-commit-resolve machine; stops early once
/// terminal and checks the phase progression along the way
inline State play_script(const PEScript& script)
{
   State state{script.config};
   for(const auto& [att, def] : script.rounds) {
      if(state.terminal()) {
         break;
      }
      EXPECT_EQ(state.phase(), Phase::commit_attacker);
      EXPECT_EQ(state.active_player(), Player::one);
      state.apply_action(att);
      EXPECT_EQ(state.phase(), Phase::commit_defender);
      EXPECT_EQ(state.active_player(), Player::two);
      state.apply_action(def);
   }
   return state;
}

struct PEState: public ::testing::Test {
   State state{Config{4}};
};

struct PESimultaneity: public ::testing::Test {
   State state{Config{4}};
};

struct PEGraph: public ::testing::Test {};

struct PETraces: public ::testing::Test {};

struct PERandomPlayouts: public ::testing::Test {};

struct PETraits: public ::testing::Test {};

struct PECFR: public ::testing::Test {};

class PEPayoffParamsF: public ::testing::TestWithParam< std::tuple< PEScript, double, double > > {};

/// plays uniformly random legal actions through both commit phases until terminality; returns the
/// terminal cause
template < typename Rng >
TerminalCause random_pe_playout(State& state, Rng&& rng)
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
      if(++guard >= 4 * Config::max_rounds + 8u) {
         ADD_FAILURE() << "playout failed to terminate";
         return TerminalCause::none;
      }
   }
   return state.terminal_cause();
}

#endif  // NOR_PURSUIT_EVASION_FIXTURES_HPP
