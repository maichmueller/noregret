#include <gtest/gtest.h>

#include <cmath>
#include <unordered_map>
#include <utility>
#include <vector>

#include "nor/env.hpp"
#include "nor/nor.hpp"

using namespace nor;

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// kernel unit tests //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

using KuhnAction = games::kuhn::Action;
constexpr KuhnAction k_check = KuhnAction::check;
constexpr KuhnAction k_bet = KuhnAction::bet;

using PredictiveKernel = rm::PredictiveRegretMatchingPlus< KuhnAction >;
using SapPredictiveKernel = rm::SAPPredictiveRegretMatchingPlus< KuhnAction >;
using NodeData = PredictiveKernel::node_data_type;
using TestPolicy = HashmapActionPolicy< KuhnAction >;

NodeData two_action_node_data()
{
   NodeData data{};
   data.register_action(k_check);
   data.register_action(k_bet);
   return data;
}

double& z(NodeData& data, KuhnAction action)
{
   return data.regret[std::cref(action)];
}
double& rho(NodeData& data, KuhnAction action)
{
   return data.instant_regret[std::cref(action)];
}
double& snap(NodeData& data, KuhnAction action)
{
   return data.strategy_snapshot[std::cref(action)];
}
const double& z(const NodeData& data, KuhnAction action)
{
   return data.regret.at(std::cref(action));
}
const double& rho(const NodeData& data, KuhnAction action)
{
   return data.instant_regret.at(std::cref(action));
}
const double& snap(const NodeData& data, KuhnAction action)
{
   return data.strategy_snapshot.at(std::cref(action));
}

}  // namespace

// concept conformance (spec item 7) against the real policy output type
static_assert(
   rm::regret_minimizer_for< PredictiveKernel, KuhnAction, TestPolicy >,
   "PredictiveRegretMatchingPlus must satisfy the regret minimizer protocol"
);
static_assert(
   rm::regret_minimizer_for< SapPredictiveKernel, KuhnAction, TestPolicy >,
   "SAPPredictiveRegretMatchingPlus must satisfy the regret minimizer protocol"
);

/// theta = max(0, clip(z) + rho): the STORED table is clipped first
/// (Algorithm 5, line 7) and the persistence-predicted regret increment is
/// added on top before the final clamp. this test probes the recommendation
/// rule on directly-injected table contents.
TEST(PredictiveRegretMatchingPlusKernel, clamp_after_prediction_shift)
{
   auto data = two_action_node_data();
   z(data, k_check) = -1.;
   z(data, k_bet) = 2.;
   rho(data, k_check) = -0.5;
   rho(data, k_bet) = 0.5;
   snap(data, k_check) = 0.5;
   snap(data, k_bet) = 0.5;

   TestPolicy out;
   PredictiveKernel::recommend(data, out, /*iteration=*/0);

   // stored table is clipped: (-1, 2) -> (0, 2)
   // theta(check) = max(0, 0 + (-0.5)) = 0
   // theta(bet)   = max(0, 2 + 0.5)    = 2.5
   EXPECT_DOUBLE_EQ(out[k_check], 0.);
   EXPECT_DOUBLE_EQ(out[k_bet], 1.);
   EXPECT_DOUBLE_EQ(snap(data, k_check), 0.);
   EXPECT_DOUBLE_EQ(snap(data, k_bet), 1.);
   // the cumulative table ends up clipped (RM+ forgetting)
   EXPECT_DOUBLE_EQ(z(data, k_check), 0.);
   EXPECT_DOUBLE_EQ(z(data, k_bet), 2.);
   // the instantaneous buffer is consumed by the recommendation
   EXPECT_DOUBLE_EQ(rho(data, k_check), 0.);
   EXPECT_DOUBLE_EQ(rho(data, k_bet), 0.);
}

/// observe() folds the RM+ clamp into the cumulative table as soon as the
/// increment arrives (Algorithm 5 line 7 semantics realized at fold-in time)
TEST(PredictiveRegretMatchingPlusKernel, observe_clips_cumulative_regret)
{
   auto data = two_action_node_data();
   PredictiveKernel::observe(data, k_check, -5.);
   EXPECT_DOUBLE_EQ(z(data, k_check), 0.);
   PredictiveKernel::observe(data, k_check, 3.);
   EXPECT_DOUBLE_EQ(z(data, k_check), 3.);
}

/// when all predicted regrets are non-positive the recommendation falls back to
/// the uniform distribution and the snapshot mirrors it
TEST(PredictiveRegretMatchingPlusKernel, uniform_fallback_when_prediction_vanishes)
{
   auto data = two_action_node_data();
   z(data, k_check) = -3.;
   z(data, k_bet) = -3.;

   TestPolicy out;
   PredictiveKernel::recommend(data, out, /*iteration=*/7);

   EXPECT_DOUBLE_EQ(out[k_check], 0.5);
   EXPECT_DOUBLE_EQ(out[k_bet], 0.5);
   EXPECT_DOUBLE_EQ(snap(data, k_check), 0.5);
   EXPECT_DOUBLE_EQ(snap(data, k_bet), 0.5);
}

/// hand-computed two-round trace reproducing the persistence prediction: round
/// one has a zero prediction source on top of the clipped fresh table; round
/// two adds the freshly observed increments as the new prediction term
TEST(PredictiveRegretMatchingPlusKernel, snapshot_and_rho_semantics_over_two_rounds)
{
   auto data = two_action_node_data();

   // ---- round 1 -----------------------------------------------------------------
   PredictiveKernel::observe(data, k_check, 2.);
   PredictiveKernel::observe(data, k_bet, -2.);
   // fold-in clipping: the negative increment never enters storage
   EXPECT_DOUBLE_EQ(z(data, k_check), 2.);
   EXPECT_DOUBLE_EQ(z(data, k_bet), 0.);
   EXPECT_DOUBLE_EQ(rho(data, k_check), 2.);
   EXPECT_DOUBLE_EQ(rho(data, k_bet), -2.);

   TestPolicy out1;
   PredictiveKernel::recommend(data, out1, /*iteration=*/0);
   // theta(check) = max(0, 2 + 2) = 4 ; theta(bet) = max(0, 0 + (-2)) = 0
   EXPECT_DOUBLE_EQ(out1[k_check], 1.);
   EXPECT_DOUBLE_EQ(out1[k_bet], 0.);
   EXPECT_DOUBLE_EQ(snap(data, k_check), 1.);
   EXPECT_DOUBLE_EQ(snap(data, k_bet), 0.);
   EXPECT_DOUBLE_EQ(z(data, k_check), 2.);
   EXPECT_DOUBLE_EQ(z(data, k_bet), 0.);
   EXPECT_DOUBLE_EQ(rho(data, k_check), 0.);
   EXPECT_DOUBLE_EQ(rho(data, k_bet), 0.);

   // ---- round 2 -----------------------------------------------------------------
   PredictiveKernel::observe(data, k_check, -1.);
   PredictiveKernel::observe(data, k_bet, 3.);
   EXPECT_DOUBLE_EQ(z(data, k_check), 1.);
   EXPECT_DOUBLE_EQ(z(data, k_bet), 3.);

   TestPolicy out2;
   PredictiveKernel::recommend(data, out2, /*iteration=*/1);
   // theta(check) = max(0, 1 + (-1)) = 0 ; theta(bet) = max(0, 3 + 3) = 6
   EXPECT_DOUBLE_EQ(out2[k_check], 0.);
   EXPECT_DOUBLE_EQ(out2[k_bet], 1.);
   EXPECT_DOUBLE_EQ(snap(data, k_check), 0.);
   EXPECT_DOUBLE_EQ(snap(data, k_bet), 1.);
}

/// an infostate generally contains many histories; each contributes one observe
/// call per action within an iteration, so rho must accumulate to r^t(I,a)
TEST(PredictiveRegretMatchingPlusKernel, rho_accumulates_within_one_iteration)
{
   auto data = two_action_node_data();

   PredictiveKernel::observe(data, k_check, 1.);
   PredictiveKernel::observe(data, k_check, 2.);
   PredictiveKernel::observe(data, k_bet, -1.5);

   EXPECT_DOUBLE_EQ(z(data, k_check), 3.);
   EXPECT_DOUBLE_EQ(rho(data, k_check), 3.);
   EXPECT_DOUBLE_EQ(rho(data, k_bet), -1.5);

   TestPolicy out;
   PredictiveKernel::recommend(data, out, /*iteration=*/0);
   // consumed after use regardless of accumulation count
   EXPECT_DOUBLE_EQ(rho(data, k_check), 0.);
}

/// SAPCFR+ damps ONLY the prediction shift term by 1/(1 + alpha), alpha = 2
TEST(SAPPredictiveRegretMatchingPlusKernel, prediction_shift_damped_by_one_third)
{
   using SapNodeData = SapPredictiveKernel::node_data_type;

   const auto sap_node_data = [] {
      SapNodeData data{};
      data.register_action(k_check);
      data.register_action(k_bet);
      return data;
   };

   auto data = sap_node_data();
   auto z_sap = [&](SapNodeData& d, KuhnAction a) -> double& { return d.regret[std::cref(a)]; };
   auto snap_sap = [&](const SapNodeData& d, KuhnAction a) -> const double& {
      return d.strategy_snapshot.at(std::cref(a));
   };

   SapPredictiveKernel::observe(data, k_check, 2.);
   SapPredictiveKernel::observe(data, k_bet, -2.);

   TestPolicy out_sap;
   SapPredictiveKernel::recommend(data, out_sap, /*iteration=*/0);
   // stored table clips to (2, 0); scale = 1/3
   // theta(check) = max(0, 2 + 2/3) = 8/3 ; theta(bet) = max(0, 0 - 2/3) = 0
   EXPECT_DOUBLE_EQ(out_sap[k_check], 1.);
   EXPECT_DOUBLE_EQ(out_sap[k_bet], 0.);
   EXPECT_DOUBLE_EQ(snap_sap(data, k_check), 1.);
   EXPECT_DOUBLE_EQ(snap_sap(data, k_bet), 0.);

   SapPredictiveKernel::observe(data, k_check, -1.);
   SapPredictiveKernel::observe(data, k_bet, 3.);
   EXPECT_DOUBLE_EQ(z_sap(data, k_check), 1.);
   EXPECT_DOUBLE_EQ(z_sap(data, k_bet), 3.);
   TestPolicy out_sap2;
   SapPredictiveKernel::recommend(data, out_sap2, /*iteration=*/1);
   // scale = 1/3
   // theta(check) = max(0, 1 + (-1)/3) = 2/3
   // theta(bet)   = max(0, 3 + 3/3)   = 4
   EXPECT_DOUBLE_EQ(out_sap2[k_check], (2. / 3.) / (14. / 3.));
   EXPECT_DOUBLE_EQ(out_sap2[k_bet], (4.) / (14. / 3.));
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// convergence tests ///////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

template < auto config >
double kuhn_exploitability_after(size_t n_iterations)
{
   using namespace nor;
   games::kuhn::Environment env{};
   auto root_state = std::make_unique< games::kuhn::State >();

   auto curr_policy = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{}
   );
   auto avg_policy = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{}
   );

   auto solver = factory::make_cfr< config, true >(
      env, std::move(root_state), std::move(curr_policy), std::move(avg_policy)
   );
   solver.iterate(n_iterations);

   const auto& avg_policies = solver.average_policy();
   if(not ranges::all_of(avg_policies | ranges::views::values, [](const auto& policy) {
         return policy.size() == size_t(6);
      })) {
      return std::numeric_limits< double >::max();
   }
   return exploitability(
      env,
      games::kuhn::State{},
      player_hashmap< std::decay_t< decltype(avg_policies.at(Player::alex)) > >{
         std::pair{Player::alex, normalize_state_policy(avg_policies.at(Player::alex))},
         std::pair{Player::bob, normalize_state_policy(avg_policies.at(Player::bob))}}
   );
}

template < auto config >
double leduc_exploitability_after(size_t n_iterations)
{
   using namespace nor;
   games::leduc::Environment env{};
   auto root_state = std::make_unique< games::leduc::State >();

   auto curr_policy = factory::make_tabular_policy(
      std::unordered_map< games::leduc::Infostate, HashmapActionPolicy< games::leduc::Action > >{}
   );
   auto avg_policy = factory::make_tabular_policy(
      std::unordered_map< games::leduc::Infostate, HashmapActionPolicy< games::leduc::Action > >{}
   );

   auto solver = factory::make_cfr< config, true >(
      env, std::move(root_state), std::move(curr_policy), std::move(avg_policy)
   );
   solver.iterate(n_iterations);

   const auto& avg_policies = solver.average_policy();
   return exploitability(
      env,
      games::leduc::State{},
      player_hashmap< std::decay_t< decltype(avg_policies.at(Player::alex)) > >{
         std::pair{Player::alex, normalize_state_policy(avg_policies.at(Player::alex))},
         std::pair{Player::bob, normalize_state_policy(avg_policies.at(Player::bob))}}
   );
}

constexpr auto k_cfr_plus_config = rm::CFRPlusConfig{};
constexpr auto k_pcfr_plus_config = rm::CFRDiscountedConfig{
   .regret_minimizing_mode = rm::RegretMinimizingMode::predictive_regret_matching_plus};
constexpr auto k_sap_pcfr_plus_config = rm::CFRDiscountedConfig{
   .regret_minimizing_mode = rm::RegretMinimizingMode::sap_predictive_regret_matching_plus};

}  // namespace

/// PCFR+ must reach lower exploitability than classic CFR+ at equal iteration
/// counts on Kuhn poker (deterministic solver; no seeding involved)
TEST(KuhnPoker, PCFR_PLUS_converges_faster_than_CFR_PLUS)
{
   const double expl_pcfr_100 = kuhn_exploitability_after< k_pcfr_plus_config >(100);
   const double expl_pcfr_300 = kuhn_exploitability_after< k_pcfr_plus_config >(300);
   const double expl_cfr_plus_100 = kuhn_exploitability_after< k_cfr_plus_config >(100);
   const double expl_cfr_plus_300 = kuhn_exploitability_after< k_cfr_plus_config >(300);
   const double expl_sap_100 = kuhn_exploitability_after< k_sap_pcfr_plus_config >(100);
   const double expl_sap_300 = kuhn_exploitability_after< k_sap_pcfr_plus_config >(300);

   std::cout << "exploitability @100 iters | PCFR+: " << expl_pcfr_100
             << " | CFR+: " << expl_cfr_plus_100 << "\n";
   std::cout << "exploitability @300 iters | PCFR+: " << expl_pcfr_300
             << " | CFR+: " << expl_cfr_plus_300 << "\n";
   std::cout << "diag @100 | SAP-on-kuhn: " << expl_sap_100 << "\n";
   std::cout << "diag @300 | SAP-on-kuhn: " << expl_sap_300 << "\n";

   EXPECT_LT(expl_pcfr_100, expl_cfr_plus_100);
   EXPECT_LT(expl_pcfr_300, expl_cfr_plus_300);
   // and progress over time for the predictive variant itself
   EXPECT_LT(expl_pcfr_300, expl_pcfr_100);
}

/// smoke test of the robustified variant on Leduc: runs, decreases
/// exploitability, and stays numerically sound
TEST(LeducPoker, SAP_PCFR_PLUS_smoke)
{
   const double expl_early = leduc_exploitability_after< k_sap_pcfr_plus_config >(4);
   const double expl_late = leduc_exploitability_after< k_sap_pcfr_plus_config >(50);

   std::cout << "SAPCFR+ exploitability on leduc | @4 iters: " << expl_early
             << " | @50 iters: " << expl_late << "\n";

   EXPECT_TRUE(std::isfinite(expl_early));
   EXPECT_TRUE(std::isfinite(expl_late));
   // near-equilibrium after 25 cycles per player; the raw value may wobble
   // microscopically below zero through floating point cancellation
   EXPECT_LT(expl_late, 0.1);
   EXPECT_LT(expl_late, expl_early);
}
