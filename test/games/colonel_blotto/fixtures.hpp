
#ifndef NOR_COLONEL_BLOTTO_FIXTURES_HPP
#define NOR_COLONEL_BLOTTO_FIXTURES_HPP

#include <gtest/gtest.h>

#include <array>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "colonel_blotto/colonel_blotto.hpp"
#include "nor/exploitability.hpp"
#include "nor/nor.hpp"

namespace cb = nor::games::colonel_blotto;
using namespace ::colonel_blotto;

/// default test configuration: B = 3 troops on N = 3 battlefields of uniform value 1
inline BlottoConfig test_config()
{
   return BlottoConfig(3);
}

/// one scripted battlefield: (player-one deployment, player-two deployment)
using CBJointField = std::pair< Deploy, Deploy >;

struct CBScript {
   BlottoConfig config{3};
   std::vector< CBJointField > fields{};
};

struct ColonelBlottoState: public ::testing::Test {
   State state{test_config()};
};

struct ColonelBlottoCommitments: public ::testing::Test {
   State state{test_config()};
};

struct ColonelBlottoResolution: public ::testing::Test {};

struct ColonelBlottoTruthTable: public ::testing::Test {};

struct ColonelBlottoTreeSize: public ::testing::Test {};

struct ColonelBlottoRandomPlayouts: public ::testing::Test {};

struct ColonelBlottoInfo: public ::testing::Test {};

struct ColonelBlottoTraits: public ::testing::Test {};

struct ColonelBlottoCFR: public ::testing::Test {};

/// applies the script's battlefields through the per-field commit-commit machine; stops early
/// once terminal and checks the phase/field progression along the way
inline State play_script(const CBScript& script)
{
   State state{script.config};
   for(const auto& [deploy_one, deploy_two] : script.fields) {
      if(state.terminal()) {
         break;
      }
      EXPECT_EQ(state.phase(), Phase::commit_p1);
      EXPECT_EQ(state.active_player(), Player::one);
      state.apply_action(deploy_one);
      EXPECT_EQ(state.phase(), Phase::commit_p2);
      EXPECT_EQ(state.active_player(), Player::two);
      state.apply_action(deploy_two);
   }
   return state;
}

/// plays uniformly random legal deployments until terminality; returns the terminal cause
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
      if(++guard >= 8 * battlefield_count + 16u) {
         ADD_FAILURE() << "playout failed to terminate";
         return TerminalCause::none;
      }
   }
   return state.terminal_cause();
}

/// counts all reachable world states (DFS over the full tree) from a fresh root
inline size_t count_tree_states(const BlottoConfig& config)
{
   std::vector< State > stack{State{config}};
   size_t count = 0;
   while(not stack.empty()) {
      auto s = stack.back();
      stack.pop_back();
      ++count;
      if(s.terminal()) {
         continue;
      }
      for(const auto& action : s.actions(s.active_player())) {
         auto child = s;
         child.apply_action(action);
         stack.push_back(child);
      }
   }
   return count;
}

/**
 * @brief Per-player best-response gaps of a policy profile (the GENERAL-SUM-safe decomposition of
 * nash_conv).
 *
 * gap_i = u_i(BR_i, pi_-i) - u_i(pi). Mirrors the internal computation of nor::nash_conv (see
 * nor/exploitability.hpp): nash_conv = sum_i gap_i (for constant-sum rewards also with
 * constant_sum=true since the centered values already sum to zero).
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

#endif  // NOR_COLONEL_BLOTTO_FIXTURES_HPP
