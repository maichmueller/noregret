
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <ranges>
#include <string>
#include <unordered_map>

#include "nor/env.hpp"
#include "nor/exploitability.hpp"
#include "nor/nor.hpp"
#include "rm_specific_testing_utils.hpp"

/// Tests for the RESTRICTED NASH RESPONSE family (Johanson, Zinkevich & Bowling, "Computing
/// Robust Counter-Strategies to Opponent Models", NeurIPS 2007).
///
/// PARAMETER CONVENTION (paper-faithful; the forcing weight p weights the MODEL):
///   p = 1  ->  pure best response against the model
///   p = 0  ->  plain self-play CFR, i.e. a Nash equilibrium of the unmodified game

namespace {

using namespace nor;
using namespace nor::opponent_aware;

constexpr auto cfr_plus_kernel = rm::CFRConfig{
   .update_mode = rm::UpdateMode::alternating,
   .regret_minimizing_mode = rm::RegretMinimizingMode::regret_matching_plus};

constexpr auto rps_rnr_config = rm::CFRConfig{
   .update_mode = rm::UpdateMode::alternating,
   .regret_minimizing_mode = rm::RegretMinimizingMode::regret_matching_plus};

/// fully mixed {action_prob} entry per every infostate in 'named_istates'
template < typename InfostateType, typename Action >
auto mix_table(const auto& named_istates, double check_prob)
{
   std::unordered_map< InfostateType, HashmapActionPolicy< Action > > table;
   for(const auto& [_, infostate] : named_istates) {
      table.emplace(
         infostate,
         HashmapActionPolicy< Action >{
            std::pair{Action::check, check_prob}, std::pair{Action::bet, 1. - check_prob}}
      );
   }
   return table;
}

}  // namespace

TEST(OpponentAwareRNR, forcing_one_edge_is_pure_best_response_vs_model)
{
   using namespace games::kuhn;
   using tabular_policy_type = TabularPolicy< Infostate, HashmapActionPolicy< Action > >;
   auto env = Environment{};
   auto root = State{};
   auto istates = make_kuhn_named_infostates();

   // skewed model: bob never bets or raises, i.e. a passive calling station
   auto bob_model = mix_table< Infostate, Action >(istates.bob, /*check_prob=*/1.);

   constexpr size_t n_iters = 4000;
   auto result = rnr_response< cfr_plus_kernel >(
      env,
      root,
      nor::Player::alex,
      bob_model,
      /*forcing_probability=*/1.,
      n_iters
   );

   // direct best response against the same model profile, for equality
   auto alex_filler = tabular_policy_type{mix_table< Infostate, Action >(istates.alex, 0.5)};
   auto bob_model_policy = tabular_policy_type{bob_model};
   auto br_alex = factory::make_best_response_policy< Infostate, Action >(nor::Player::alex);
   br_alex.allocate(
      env,
      root,
      player_hashmap{
         std::pair{nor::Player::alex, alex_filler}, std::pair{nor::Player::bob, bob_model_policy}}
   );
   double br_value = rm::policy_value(
                        env,
                        root,
                        player_hashmap{
                           std::pair{nor::Player::alex, StatePolicyView{br_alex}},
                           std::pair{nor::Player::bob, StatePolicyView{bob_model_policy}}}
   )
                        .get()
                        .at(nor::Player::alex);

   double rnr_value = opponent_aware::value_against(
      env, root, nor::Player::alex, result.policy, bob_model
   );

   // the converged restricted response attains the best-response value against the model ...
   EXPECT_NEAR(rnr_value, br_value, 1e-3);
   // ... it strictly beats equilibrium play against this exploitable model ...
   EXPECT_GT(br_value, 1. / 18.);
   // ... and (paper Figure 1) the pure response pays the documented robustness price: the
   // worst-case adversary extracts strictly more from it than from an equilibrium strategist
   double nemesis_br = best_response_value_against(env, root, nor::Player::alex, result.policy);
   EXPECT_GT(nemesis_br, 1. / 18. + 5e-2);
}

TEST(OpponentAwareRNR, forcing_zero_edge_recovers_equilibrium)
{
   using namespace games::kuhn;
   auto env = Environment{};
   auto root = State{};
   auto istates = make_kuhn_named_infostates();

   auto bob_model = mix_table< Infostate, Action >(istates.bob, /*check_prob=*/0.5);

   auto result = rnr_response< cfr_plus_kernel >(
      env,
      root,
      nor::Player::alex,
      bob_model,
      /*forcing_probability=*/0.,
      4000
   );

   // unmodified-game self-play: kuhn's equilibrium value for the first mover up to slack
   EXPECT_NEAR(result.final_root_values.at(nor::Player::alex), -1. / 18., 1e-2);
   // and near-equilibrium exploitability of the returned strategy: the worst-case adversary's
   // extraction stays within the (+1/18 equilibrium) plus small CFR slack
   double nemesis_br = best_response_value_against(env, root, nor::Player::alex, result.policy);
   EXPECT_LE(nemesis_br, 1. / 18. + 2e-2);
}

TEST(OpponentAwareRNR, tradeoff_curve_interpolates_exploitation_vs_safety)
{
   using namespace games::kuhn;
   auto env = Environment{};
   auto root = State{};
   auto istates = make_kuhn_named_infostates();

   // skewed model: passive calling-station bob (never bets/raises)
   auto bob_model = mix_table< Infostate, Action >(istates.bob, /*check_prob=*/1.);

   constexpr std::array< double, 5 > forcing_levels{0., 0.25, 0.5, 0.75, 1.};
   std::array< double, forcing_levels.size() > value_vs_model{};
   std::array< double, forcing_levels.size() > nemesis_value{};

   for(auto [idx, forcing] : std::views::enumerate(forcing_levels)) {
      auto result = rnr_response< cfr_plus_kernel >(
         env, root, nor::Player::alex, bob_model, forcing, 4000
      );
      value_vs_model[idx] = opponent_aware::value_against(
         env, root, nor::Player::alex, result.policy, bob_model
      );
      nemesis_value[idx] = best_response_value_against(env, root, nor::Player::alex, result.policy);
   }

   std::cout << "[          ] kuhn rnr tradeoff (p: value_vs_model | nemesis_value): ";
   for(auto [idx, forcing] : std::views::enumerate(forcing_levels)) {
      std::cout << forcing << ": " << value_vs_model[idx] << "/" << nemesis_value[idx] << "  ";
   }
   std::cout << "\n";
   constexpr double alex_equilibrium_value = -1. / 18.;
   constexpr double nemesis_equilibrium_value = 1. / 18.;
   // directional assertions with margins (paper Figure 1): raising the forcing weight moves
   // the response from Nash-like (safe, non-exploitative) to best-response-like (exploitative,
   // vulnerable)
   EXPECT_LT(value_vs_model[0], value_vs_model[4] - 0.05);
   EXPECT_GT(value_vs_model[3], value_vs_model[1] + 0.01);
   // safety decreases towards the pure-response edge ...
   EXPECT_GT(nemesis_value[4], nemesis_value[0] + 0.05);
   // ... starting from equilibrium-level robustness at p = 0 ...
   EXPECT_LE(nemesis_value[0], nemesis_equilibrium_value + 2e-2);
   // ... while the p = 0 side already exploits better than the equilibrium strategy does
   EXPECT_GT(value_vs_model[0], alex_equilibrium_value);
}

TEST(OpponentAwareRNR, rps_edges)
{
   using namespace nor::games::rps;
   Environment env{};
   auto
      [env_unused,
       avg0,
       avg1,
       cur0,
       cur1,
       infostate_alex,
       infostate_bob,
       state_unused] = setup_rps_test();
   (void) env_unused;
   (void) avg0;
   (void) avg1;
   (void) cur0;
   (void) cur1;
   (void) state_unused;

   // skewed model: bob plays rock-heavy
   std::unordered_map< Infostate, HashmapActionPolicy< Action > > bob_model;
   bob_model.emplace(
      infostate_bob,
      HashmapActionPolicy< Action >{
         std::pair{Action::rock, 0.8},
         std::pair{Action::paper, 0.1},
         std::pair{Action::scissors, 0.1}}
   );

   State root{};

   // p = 1: pure best response -- always paper against the rock-heavy model
   auto br_result = rnr_response< rps_rnr_config >(
      env,
      root,
      nor::Player::alex,
      bob_model,
      /*forcing_probability=*/1.,
      2000
   );
   const auto& br_entry = br_result.policy.at(infostate_alex);
   EXPECT_GT(normalize_action_policy(br_entry).at(Action::paper), 0.97);
   double br_value = opponent_aware::value_against(
      env, root, nor::Player::alex, br_result.policy, bob_model
   );
   EXPECT_NEAR(br_value, 0.7, 3e-3);

   // p = 0: self-play equilibrium -- near-uniform and game value zero
   auto ne_result = rnr_response< rps_rnr_config >(
      env,
      root,
      nor::Player::alex,
      bob_model,
      /*forcing_probability=*/0.,
      3000
   );
   for(const auto& prob :
       normalize_action_policy(ne_result.policy.at(infostate_alex)) | std::views::values) {
      EXPECT_NEAR(prob, 1. / 3., 0.05);
   }
   double ne_value = opponent_aware::value_against(
      env, root, nor::Player::alex, ne_result.policy, bob_model
   );
   EXPECT_LE(ne_value, br_value - 0.05);
}
