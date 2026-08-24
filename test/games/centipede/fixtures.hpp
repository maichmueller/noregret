
#ifndef NOR_CENTIPEDE_FIXTURES_HPP
#define NOR_CENTIPEDE_FIXTURES_HPP

#include <gtest/gtest.h>

#include <array>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "centipede/centipede.hpp"
#include "nor/exploitability.hpp"
#include "nor/nor.hpp"

namespace cp = nor::games::centipede;
using namespace ::centipede;

/// default test configuration: the McKelvey-Palfrey-style four-round instance G(4, 4, 1)
inline Config test_config()
{
   return Config(4, 4, 1);
}

/// one scripted round: (mover move); a full game script is just the chronological move list
using CPScript = std::vector< Move >;

struct CentipedeState: public ::testing::Test {
   State state{test_config()};
};

struct CentipedeMoves: public ::testing::Test {
   State state{test_config()};
};

struct CentipedeTermination: public ::testing::Test {};

struct CentipedeTruthTable: public ::testing::Test {};

struct CentipedeDominance: public ::testing::Test {};

struct CentipedeRandomPlayouts: public ::testing::Test {};

struct CentipedeInfo: public ::testing::Test {};

struct CentipedeTraits: public ::testing::Test {};

struct CentipedeCFR: public ::testing::Test {};

/// applies the scripted moves through the alternating machine; stops early once terminal and
/// checks the mover alternation along the way
inline State play_script(const Config& config, const CPScript& moves)
{
   State state{config};
   for(const auto& move : moves) {
      if(state.terminal()) {
         break;
      }
      state.apply_action(move);
   }
   return state;
}

/// plays uniformly random legal moves until terminality; returns the terminal cause
template < typename Rng >
TerminalCause random_playout(State& state, Rng&& rng)
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

/**
 * @brief Per-player best-response gaps of a policy profile (the GENERAL-SUM-safe decomposition of
 * nash_conv).
 *
 * gap_i = u_i(BR_i, pi_-i) - u_i(pi). Mirrors the internal computation of nor::nash_conv (see
 * nor/exploitability.hpp): nash_conv(constant_sum=false) = sum_i gap_i. For a general-sum game
 * each gap is reported individually instead of relying on the zero-sum normalization that
 * exploitability() would apply.
 */
template < typename Env, typename Policy >
std::array< double, 2 > per_player_br_gaps(
   Env& env,
   const typename Env::world_state_type& root_state,
   const nor::player_hashmap< Policy >& player_policies
)
{
   using info_state_type = typename Env::info_state_type;
   using action_type = typename Env::action_type;

   std::array< double, 2 > gaps{};
   auto profile_value = nor::rm::policy_value(env, root_state, player_policies);

   for(const auto& [best_responder, policy] : player_policies) {
      auto best_response = nor::factory::make_best_response_policy< info_state_type, action_type >(
         best_responder
      );
      best_response.allocate(env, root_state, player_policies);
      auto
         policy_map = nor::player_hashmap< nor::StatePolicyView< info_state_type, action_type > >{};
      for(const auto& p : env.players(root_state)) {
         if(p == nor::Player::chance) {
            continue;
         }
         if(p == best_responder) {
            policy_map.emplace(p, nor::StatePolicyView{best_response});
         } else {
            policy_map.emplace(p, nor::StatePolicyView{player_policies.at(p)});
         }
      }
      const double
         br_value = nor::rm::policy_value(env, root_state, policy_map).get().at(best_responder);
      const double pi_value = profile_value.get().at(best_responder);
      gaps.at(size_t(best_responder)) = br_value - pi_value;
   }
   return gaps;
}

#endif  // NOR_CENTIPEDE_FIXTURES_HPP
