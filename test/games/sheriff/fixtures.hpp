
#ifndef NOR_SHERIFF_FIXTURES_HPP
#define NOR_SHERIFF_FIXTURES_HPP

#include <gtest/gtest.h>

#include <array>
#include <utility>
#include <vector>

#include "nor/exploitability.hpp"
#include "nor/nor.hpp"
#include "sheriff/sheriff.hpp"

// global-scope aliases/directives mirroring the shapley test conventions
namespace sh = nor::games::sheriff;
// NOTE: this transitively imports ::sheriff as well (the adapter namespace embeds a
// 'using namespace ::sheriff'); nor's player enum stays explicitly spelled nor::Player
using namespace nor::games::sheriff;

namespace sh_test {
/// the paper's baseline instance (Farina et al. 2019, Section 5.2):
/// v=5, p=1, s=1, n_max=10, b_max=2, r=2
inline sheriff::Config baseline_config()
{
   return sheriff::Config{};
}

/// a small smoke-sized instance used for CFR runs
inline sheriff::Config tiny_config()
{
   return sheriff::Config{/*v=*/5., /*p=*/1., /*s=*/1., /*n_max=*/2, /*b_max=*/1, /*rounds=*/1};
}

}  // namespace sh_test

struct SheriffState: public ::testing::Test {
   sheriff::State state{sh_test::tiny_config()};
};

struct SheriffLegality: public ::testing::Test {
   sheriff::State state{sh_test::baseline_config()};
};

struct SheriffTruthTable: public ::testing::Test {};

struct SheriffInfo: public ::testing::Test {};

struct SheriffRandomPlayouts: public ::testing::Test {};

struct SheriffTraits: public ::testing::Test {};

struct SheriffCFR: public ::testing::Test {};

/// a fully scripted sheriff game: cargo load + one bribe/response pair per bargaining round
struct SheriffScript {
   sheriff::Config config{};
   uint32_t cargo = 0;
   std::vector< uint32_t > bribes;  //< exactly config.rounds entries
   std::vector< bool > accept;  //< exactly config.rounds entries
};

inline sheriff::State play_script(const SheriffScript& script)
{
   using namespace sheriff;
   State state{script.config};
   state.apply_action(Action{Load{script.cargo}});
   for(size_t r : std::views::iota(size_t{0}, script.config.rounds)) {
      state.apply_action(Action{Offer{script.bribes.at(r)}});
      state.apply_action(Action{Respond{script.accept.at(r)}});
   }
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

class SheriffPayoffParamsF:
    public ::testing::TestWithParam< std::tuple< SheriffScript, std::pair< double, double > > > {};

#endif  // NOR_SHERIFF_FIXTURES_HPP
