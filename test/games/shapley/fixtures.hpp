
#ifndef NOR_SHAPLEY_FIXTURES_HPP
#define NOR_SHAPLEY_FIXTURES_HPP

#include <gtest/gtest.h>

#include <array>
#include <utility>
#include <vector>

#include "nor/exploitability.hpp"
#include "nor/nor.hpp"
#include "shapley/shapley.hpp"

namespace sh = nor::games::shapley;
using namespace ::shapley;

struct ShapleyState: public ::testing::Test {
   State state;
};

struct ShapleyBidding: public ::testing::Test {
   State state;
};

struct ShapleyTruthTable: public ::testing::Test {};

struct ShapleyBestResponse: public ::testing::Test {};

struct ShapleyInfo: public ::testing::Test {};

struct ShapleyRandomPlayouts: public ::testing::Test {};

struct ShapleyTraits: public ::testing::Test {};

struct ShapleyCFR: public ::testing::Test {};

/// plays the scripted joint profile (player-one play, player-two play) through the commit-commit
/// machine and checks the phase progression along the way
inline State play_profile(Play one, Play two)
{
   State state;
   EXPECT_EQ(state.phase(), Phase::commit_p1);
   EXPECT_EQ(state.active_player(), Player::one);
   state.apply_action(one);
   EXPECT_EQ(state.phase(), Phase::commit_p2);
   EXPECT_EQ(state.active_player(), Player::two);
   state.apply_action(two);
   EXPECT_TRUE(state.terminal());
   return state;
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

#endif  // NOR_SHAPLEY_FIXTURES_HPP
