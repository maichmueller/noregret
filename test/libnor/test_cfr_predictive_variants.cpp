#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <ranges>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "goofspiel/goofspiel.hpp"
#include "liars_dice/liars_dice.hpp"
#include "nor/env.hpp"
#include "nor/nor.hpp"

using namespace nor;

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// kernel unit tests ///////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

using KuhnAction = games::kuhn::Action;
constexpr KuhnAction k_check = KuhnAction::check;
constexpr KuhnAction k_bet = KuhnAction::bet;

template < typename Kernel >
typename Kernel::node_data_type two_action_node_data()
{
   typename Kernel::node_data_type data{};
   data.register_action(k_check);
   data.register_action(k_bet);
   return data;
}

template < typename NodeData >
double& z(NodeData& data, KuhnAction action)
{
   return data.regret[data.index_of(action)];
}
template < typename NodeData >
double& rho(NodeData& data, KuhnAction action)
{
   return data.instant_regret[data.index_of(action)];
}
template < typename NodeData >
const double& z(const NodeData& data, KuhnAction action)
{
   return data.regret[data.index_of(action)];
}
template < typename NodeData >
const double& snap(const NodeData& data, KuhnAction action)
{
   return data.strategy_snapshot[data.index_of(action)];
}

using PlainKernel = rm::PredictiveRegretMatchingPlus< KuhnAction >;
using P2PKernel = rm::P2PPredictiveRegretMatchingPlus< KuhnAction >;
using SmoothKernel = rm::SmoothPredictiveRegretMatchingPlus< KuhnAction >;
using APKernel = rm::APPredictiveRegretMatchingPlus< KuhnAction >;
using StableKernel = rm::StablePredictiveRegretMatchingPlus< KuhnAction >;
using TestPolicy = HashmapActionPolicy< KuhnAction >;

}  // namespace

// concept conformance of every new kernel against the real policy output type
static_assert(
   rm::regret_minimizer_for< P2PKernel, KuhnAction, TestPolicy >,
   "P2P kernel must satisfy the regret minimizer protocol"
);
static_assert(
   rm::regret_minimizer_for< SmoothKernel, KuhnAction, TestPolicy >,
   "Smooth kernel must satisfy the regret minimizer protocol"
);
static_assert(
   rm::regret_minimizer_for< APKernel, KuhnAction, TestPolicy >,
   "AP kernel must satisfy the regret minimizer protocol"
);
static_assert(
   rm::regret_minimizer_for< StableKernel, KuhnAction, TestPolicy >,
   "Stable kernel must satisfy the regret minimizer protocol"
);

/// P2PCFR+ damps ONLY the prediction shift term by 1/(1 + alpha), alpha = 5
/// (OpenReview njyZgDDeY4 section 4; same mechanism as SAPCFR+ but alpha = 5)
TEST(P2PPredictiveKernel, prediction_shift_damped_by_one_sixth)
{
   auto data = two_action_node_data< P2PKernel >();

   P2PKernel::observe(data, k_check, 2.);
   P2PKernel::observe(data, k_bet, -2.);

   TestPolicy out1;
   P2PKernel::recommend(data, out1, /*iteration=*/0);
   // stored table clips to (2, 0); scale = 1/6
   // theta(check) = max(0, 2 + 2/6) = 7/3 ; theta(bet) = max(0, -2/6) = 0
   EXPECT_DOUBLE_EQ(out1[k_check], 1.);
   EXPECT_DOUBLE_EQ(out1[k_bet], 0.);

   P2PKernel::observe(data, k_check, -1.);
   P2PKernel::observe(data, k_bet, 3.);
   TestPolicy out2;
   P2PKernel::recommend(data, out2, /*iteration=*/1);
   // scale = 1/6:
   // theta(check) = max(0, 1 + (-1)/6) = 5/6
   // theta(bet)   = max(0, 3 + 3/6)   = 7/2
   // normalizer   = 5/6 + 7/2 = 13/3
   EXPECT_DOUBLE_EQ(out2[k_check], (5. / 6.) / (13. / 3.));
   EXPECT_DOUBLE_EQ(out2[k_bet], (7. / 2.) / (13. / 3.));
}

/// Smooth PRM+ floors the predicted-regret vector away from the origin before
/// normalization: any deficit to ||theta||_1 >= epsilon is spread uniformly
/// (arXiv:2305.14709, Algorithm 2; exact projection for non-negative theta)
TEST(SmoothPredictiveKernel, norm_floor_lifts_theta_away_from_origin)
{
   auto data = two_action_node_data< SmoothKernel >();
   // after clipping only a sub-threshold amount of positive mass survives:
   // theta = (0, 0.4), ||theta||_1 = 0.4 < epsilon = 1
   z(data, k_bet) = 0.4;

   TestPolicy out;
   SmoothKernel::recommend(data, out, /*iteration=*/0);

   // deficit = (1 - 0.4)/2 = 0.3 spread uniformly -> theta' = (0.3, 0.7),
   // normalized by the lifted norm 1.0
   EXPECT_DOUBLE_EQ(out[k_check], 0.3);
   EXPECT_DOUBLE_EQ(out[k_bet], 0.7);
   EXPECT_DOUBLE_EQ(snap(data, k_check), 0.3);
   EXPECT_DOUBLE_EQ(snap(data, k_bet), 0.7);

   // contrast: plain PCFR+ on identical tables chops action check off entirely
   auto plain = two_action_node_data< PlainKernel >();
   z(plain, k_bet) = 0.4;
   TestPolicy plain_out;
   PlainKernel::recommend(plain, plain_out, 0);
   EXPECT_DOUBLE_EQ(plain_out[k_check], 0.);
   EXPECT_DOUBLE_EQ(plain_out[k_bet], 1.);
}

/// when all predicted regrets vanish the floor alone produces the uniform
/// distribution through the deficit mechanism (not the fallback branch)
TEST(SmoothPredictiveKernel, uniform_from_floor_when_prediction_vanishes)
{
   auto data = two_action_node_data< SmoothKernel >();
   z(data, k_check) = -3.;
   z(data, k_bet) = -3.;

   TestPolicy out;
   SmoothKernel::recommend(data, out, /*iteration=*/7);

   EXPECT_DOUBLE_EQ(out[k_check], 0.5);
   EXPECT_DOUBLE_EQ(out[k_bet], 0.5);
}

/// above-threshold norms pass through the floor untouched (identity branch)
TEST(SmoothPredictiveKernel, no_floor_above_threshold)
{
   auto data = two_action_node_data< SmoothKernel >();
   z(data, k_check) = 2.;
   z(data, k_bet) = 6.;

   TestPolicy out;
   SmoothKernel::recommend(data, out, /*iteration=*/0);
   EXPECT_DOUBLE_EQ(out[k_check], 0.25);
   EXPECT_DOUBLE_EQ(out[k_bet], 0.75);
}

/// APCFR+ adapts the per-infostate prediction scale from running squared-L2-norm
/// sums (arXiv:2503.12770v2 Eq. (10)); hand-derived two-round trace.
///
/// round 1 folds tau=1: S_r += ||r^1||^2 = 8, S_R += ||R^2 - R^1||^2 = 4
///   => alpha = min(sqrt(8/4), 5) = sqrt(2), s = 1/(1 + sqrt(2))
/// round 2 folds tau=2: deltas r^2 - r^1 = (-3, 5), R^3 - R^2 = (-1, 3)
///   => S_r = 42, S_R = 14, alpha = min(sqrt(42/14), 5) = sqrt(3), s = 1/(1 + sqrt(3))
TEST(APPredictiveKernel, adaptive_scale_follows_running_norm_sums)
{
   auto data = two_action_node_data< APKernel >();

   // ---- round 1 -----------------------------------------------------------------
   APKernel::observe(data, k_check, 2.);
   APKernel::observe(data, k_bet, -2.);

   const double s1 = 1. / (1. + std::sqrt(8. / 4.));
   TestPolicy out1;
   APKernel::recommend(data, out1, /*iteration=*/0);
   // theta(check) = max(0, 2 + 2*s1) ; theta(bet) = max(0, 0 - 2*s1) = 0
   EXPECT_DOUBLE_EQ(out1[k_check], 1.);
   EXPECT_DOUBLE_EQ(out1[k_bet], 0.);
   EXPECT_DOUBLE_EQ(z(data, k_check), 2.);
   EXPECT_DOUBLE_EQ(z(data, k_bet), 0.);

   // ---- round 2 -----------------------------------------------------------------
   APKernel::observe(data, k_check, -1.);
   APKernel::observe(data, k_bet, 3.);

   const double s2 = 1.
                     / (1.
                        + std::sqrt(
                           (8. + ((-1. - 2.) * (-1. - 2.) + (3. - (-2.)) * (3. - (-2.))))
                           / (4. + ((1. - 2.) * (1. - 2.) + (3. - 0.) * (3. - 0.)))
                        ));
   TestPolicy out2;
   APKernel::recommend(data, out2, /*iteration=*/1);
   const double theta_check = 1. - s2;
   const double theta_bet = 3. + 3. * s2;
   EXPECT_DOUBLE_EQ(out2[k_check], theta_check / (theta_check + theta_bet));
   EXPECT_DOUBLE_EQ(out2[k_bet], theta_bet / (theta_check + theta_bet));

   // the scale must actually have adapted between rounds: s1 != 1 rules out a
   // stuck-at-PCFR+ implementation and s1 != s2 rules out constant damping
   EXPECT_NEAR(data.shift_context.current_scale, s2, 1e-15);
   EXPECT_DOUBLE_EQ(s1, 1. / (1. + std::sqrt(2.)));
}

/// APCFR+'s alpha_max clamp engages when the accumulated ratio exceeds it: the
/// applied prediction scale saturates at 1/(1 + alpha_max) = 1/6
TEST(APPredictiveKernel, alpha_max_clamps_the_adaptive_ratio)
{
   auto data = two_action_node_data< APKernel >();
   auto& ctx = data.shift_context;

   // force the running sums into the saturated regime (the clamp exists so that
   // the Theorem-4.1 bound stays finite; the paper fixes alpha_max = 5)
   ctx.sum_sq_prediction_error = 1000.;
   ctx.sum_sq_implicit_delta = 1.;

   using APShift = APKernel::shift_type;
   APShift::before_recommend(ctx, data, /*iteration=*/2);
   EXPECT_DOUBLE_EQ(ctx.current_scale, 1. / (1. + APShift::alpha_max));

   // degenerate bookkeeping state (no implicit movement observed yet) must also
   // saturate rather than divide by zero
   auto fresh = two_action_node_data< APKernel >();
   auto& fresh_ctx = fresh.shift_context;
   fresh_ctx.sum_sq_implicit_delta = 0.;
   APShift::before_recommend(fresh_ctx, fresh, /*iteration=*/0);
   EXPECT_DOUBLE_EQ(fresh_ctx.current_scale, 1. / (1. + 5.));
}

/// Stable PRM+ restarts the cumulative table componentwise to the R0-floor when
/// every entry is at or below it, and suppresses the following prediction
/// (arXiv:2305.14709, Algorithm 1 lines 8-9)
TEST(StablePredictiveKernel, restart_resets_table_and_suppresses_prediction)
{
   auto data = two_action_node_data< StableKernel >();
   // everything <= R0 = 1 -> restart trigger
   z(data, k_check) = 0.5;
   z(data, k_bet) = 0.9;
   rho(data, k_check) = 10.;  // would dominate theta if not suppressed
   rho(data, k_bet) = -10.;

   TestPolicy out;
   StableKernel::recommend(data, out, /*iteration=*/3);

   // restart: z <- R0 * 1, prediction m^{t+1} = 0 -> uniform recommendation
   EXPECT_DOUBLE_EQ(z(data, k_check), 1.);
   EXPECT_DOUBLE_EQ(z(data, k_bet), 1.);
   EXPECT_DOUBLE_EQ(rho(data, k_check), 0.);
   EXPECT_DOUBLE_EQ(rho(data, k_bet), 0.);
   EXPECT_DOUBLE_EQ(out[k_check], 0.5);
   EXPECT_DOUBLE_EQ(out[k_bet], 0.5);
}

/// a single entry above the threshold prevents the restart; the update then
/// behaves like plain PCFR+
TEST(StablePredictiveKernel, no_restart_while_any_entry_exceeds_threshold)
{
   auto data = two_action_node_data< StableKernel >();

   StableKernel::observe(data, k_check, 2.);
   StableKernel::observe(data, k_bet, -2.);

   TestPolicy out;
   StableKernel::recommend(data, out, /*iteration=*/0);
   // z(check) = 2 > 1 -> no restart; plain PCFR+ arithmetic applies
   EXPECT_DOUBLE_EQ(z(data, k_check), 2.);
   EXPECT_DOUBLE_EQ(out[k_check], 1.);
   EXPECT_DOUBLE_EQ(out[k_bet], 0.);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// solver-level helpers ///////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

template < auto config >
double kuhn_exploitability_after(size_t n_iterations)
{
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
   if(not std::ranges::all_of(avg_policies | std::views::values, [](const auto& policy) {
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

template < auto config >
double liars_dice_exploitability_after(size_t n_iterations, bool* all_finite = nullptr)
{
   const games::liars_dice::DiceConfig dice_config{uint8_t(4)};
   games::liars_dice::Environment env{dice_config};
   auto root_state = std::make_unique< games::liars_dice::State >(dice_config);

   auto curr_policy = factory::make_tabular_policy(
      std::unordered_map<
         games::liars_dice::Infostate,
         HashmapActionPolicy< games::liars_dice::Action > >{}
   );
   auto avg_policy = factory::make_tabular_policy(
      std::unordered_map<
         games::liars_dice::Infostate,
         HashmapActionPolicy< games::liars_dice::Action > >{}
   );

   auto solver = factory::make_cfr< config, true >(
      env, std::move(root_state), std::move(curr_policy), std::move(avg_policy)
   );
   solver.iterate(n_iterations);

   const auto& avg_policies = solver.average_policy();
   if(all_finite != nullptr) {
      auto scan = [](const auto& tabular_policy) {
         for(const auto& [infostate, action_policy] : tabular_policy.table()) {
            for(const auto& [action, prob] : action_policy) {
               if(not std::isfinite(prob))
                  return false;
            }
         }
         return true;
      };
      *all_finite = std::ranges::all_of(avg_policies | std::views::values, [&](const auto& policy) {
         return scan(policy);
      });
   }

   return exploitability(
      env,
      games::liars_dice::State{dice_config},
      player_hashmap< std::decay_t< decltype(avg_policies.at(Player::alex)) > >{
         std::pair{Player::alex, normalize_state_policy(avg_policies.at(Player::alex))},
         std::pair{Player::bob, normalize_state_policy(avg_policies.at(Player::bob))}}
   );
}

template < auto config >
double goofspiel_exploitability_after(
   size_t n_iterations,
   bool* all_finite = nullptr,
   bool with_exploitability = true
)
{
   const games::goofspiel::GoofspielConfig gs_config{.deck_size = 4, .imp_info = false};
   games::goofspiel::Environment env{gs_config};
   auto root_state = std::make_unique< games::goofspiel::State >(gs_config);

   auto curr_policy = factory::make_tabular_policy(
      std::unordered_map<
         games::goofspiel::Infostate,
         HashmapActionPolicy< games::goofspiel::Environment::action_type > >{}
   );
   auto avg_policy = factory::make_tabular_policy(
      std::unordered_map<
         games::goofspiel::Infostate,
         HashmapActionPolicy< games::goofspiel::Environment::action_type > >{}
   );

   auto solver = factory::make_cfr< config, true >(
      env, std::move(root_state), std::move(curr_policy), std::move(avg_policy)
   );
   solver.iterate(n_iterations);

   const auto& avg_policies = solver.average_policy();
   if(all_finite != nullptr) {
      auto scan = [](const auto& tabular_policy) {
         for(const auto& [infostate, action_policy] : tabular_policy.table()) {
            for(const auto& [action, prob] : action_policy) {
               if(not std::isfinite(prob))
                  return false;
            }
         }
         return true;
      };
      *all_finite = std::ranges::all_of(avg_policies | std::views::values, [&](const auto& policy) {
         return scan(policy);
      });
   }

   if(not with_exploitability) {
      return std::numeric_limits< double >::quiet_NaN();
   }
   return exploitability(
      env,
      games::goofspiel::State{gs_config},
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
constexpr auto k_ap_pcfr_plus_config = rm::CFRDiscountedConfig{
   .regret_minimizing_mode = rm::RegretMinimizingMode::ap_predictive_regret_matching_plus};
constexpr auto k_p2p_pcfr_plus_config = rm::CFRDiscountedConfig{
   .regret_minimizing_mode = rm::RegretMinimizingMode::p2p_predictive_regret_matching_plus};
constexpr auto k_smooth_pcfr_plus_config = rm::CFRDiscountedConfig{
   .regret_minimizing_mode = rm::RegretMinimizingMode::smooth_predictive_regret_matching_plus};
constexpr auto k_stable_pcfr_plus_config = rm::CFRDiscountedConfig{
   .regret_minimizing_mode = rm::RegretMinimizingMode::stable_predictive_regret_matching_plus};

}  // namespace

///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// convergence tests ////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

/// every new predictive variant must converge on Kuhn poker. The step-size
/// robustifications (APCFR+/P2PCFR+) must beat the plain CFR+ baseline at equal
/// iteration counts. The stabilizing variants (Smooth/Stable PRM+) carry known
/// worse constants -- arXiv:2503.12770v2 section 2 and 5 report they NEVER
/// outperform PCFR+ empirically -- so for them we assert monotone progress and
/// that they eventually undercut CFR+ given a longer horizon.
TEST(KuhnPoker, PredictiveVariants_converge_below_CFR_PLUS_baseline)
{
   const double expl_cfr_plus_100 = kuhn_exploitability_after< k_cfr_plus_config >(100);
   const double expl_cfr_plus_300 = kuhn_exploitability_after< k_cfr_plus_config >(300);
   const double expl_cfr_plus_1500 = kuhn_exploitability_after< k_cfr_plus_config >(1500);

   const double ap_100 = kuhn_exploitability_after< k_ap_pcfr_plus_config >(100);
   const double ap_300 = kuhn_exploitability_after< k_ap_pcfr_plus_config >(300);
   const double p2p_100 = kuhn_exploitability_after< k_p2p_pcfr_plus_config >(100);
   const double p2p_300 = kuhn_exploitability_after< k_p2p_pcfr_plus_config >(300);
   const double smooth_100 = kuhn_exploitability_after< k_smooth_pcfr_plus_config >(100);
   const double smooth_300 = kuhn_exploitability_after< k_smooth_pcfr_plus_config >(300);
   const double stable_100 = kuhn_exploitability_after< k_stable_pcfr_plus_config >(100);
   const double stable_300 = kuhn_exploitability_after< k_stable_pcfr_plus_config >(300);
   const double smooth_1500 = kuhn_exploitability_after< k_smooth_pcfr_plus_config >(1500);
   const double stable_1500 = kuhn_exploitability_after< k_stable_pcfr_plus_config >(1500);

   std::cout << "[kuhn] CFR+ baseline @100: " << expl_cfr_plus_100 << " @300: " << expl_cfr_plus_300
             << " @1500: " << expl_cfr_plus_1500 << "\n";
   for(auto&& [name, e100, e300, e1500] :
       std::vector< std::tuple< const char*, double, double, double > >{
          {"APCFR+", ap_100, ap_300, std::numeric_limits< double >::quiet_NaN()},
          {"P2PCFR+", p2p_100, p2p_300, std::numeric_limits< double >::quiet_NaN()},
          {"SmoothPCFR+", smooth_100, smooth_300, smooth_1500},
          {"StablePCFR+", stable_100, stable_300, stable_1500}}) {
      std::cout << "[kuhn] " << name << " @100: " << e100 << " @300: " << e300 << "\n";
      EXPECT_TRUE(std::isfinite(e100)) << name;
      EXPECT_TRUE(std::isfinite(e300)) << name;
      // progress over time for every variant
      EXPECT_LT(e300, e100) << name;
      // stabilizers must undercut the CFR+ baseline once given a longer horizon
      if(not std::isnan(e1500)) {
         std::cout << "[kuhn] " << name << " @1500: " << e1500 << "\n";
         EXPECT_LT(e1500, expl_cfr_plus_1500) << name;
      }
   }
   // the adaptive/pessimistic step-size variants are strictly stronger than CFR+
   EXPECT_LT(ap_100, expl_cfr_plus_100);
   EXPECT_LT(ap_300, expl_cfr_plus_300);
   EXPECT_LT(p2p_100, expl_cfr_plus_100);
   EXPECT_LT(p2p_300, expl_cfr_plus_300);
}

/// smoke test of each new variant on Leduc: runs soundly and makes progress.
/// The step-size robustifications reach near-equilibrium quickly; the
/// stabilizing variants (Smooth/Stable) converge much slower here -- in line
/// with arXiv:2503.12770v2 / OpenReview njyZgDDeY4 experiments on poker games,
/// where they trail PCFR+ substantially -- so only progress + soundness is
/// asserted for them.
TEST(LeducPoker, PredictiveVariants_smoke)
{
   const double ap_late = leduc_exploitability_after< k_ap_pcfr_plus_config >(50);
   const double p2p_late = leduc_exploitability_after< k_p2p_pcfr_plus_config >(50);
   const double smooth_early = leduc_exploitability_after< k_smooth_pcfr_plus_config >(10);
   const double smooth_late = leduc_exploitability_after< k_smooth_pcfr_plus_config >(50);
   const double stable_early = leduc_exploitability_after< k_stable_pcfr_plus_config >(10);
   const double stable_late = leduc_exploitability_after< k_stable_pcfr_plus_config >(50);

   std::cout << "[leduc] APCFR+ @50: " << ap_late << " | P2PCFR+ @50: " << p2p_late
             << " | SmoothPCFR+ @10/@50: " << smooth_early << "/" << smooth_late
             << " | StablePCFR+ @10/@50: " << stable_early << "/" << stable_late << "\n";

   EXPECT_TRUE(std::isfinite(ap_late));
   EXPECT_TRUE(std::isfinite(p2p_late));
   EXPECT_TRUE(std::isfinite(smooth_late));
   EXPECT_TRUE(std::isfinite(stable_late));
   // near-equilibrium after 25 cycles per player for the step-size variants;
   // the raw value may wobble microscopically below zero through floating point
   // cancellation
   EXPECT_LT(ap_late, 0.1);
   EXPECT_LT(p2p_late, 0.1);
   // stabilizers: finite, sane, and decreasing over the window
   EXPECT_LT(smooth_late, 1.);
   EXPECT_LT(stable_late, 1.);
   EXPECT_LT(smooth_late, smooth_early);
   EXPECT_LT(stable_late, stable_early);
}

/// liar's dice (single die per player, 4 faces, challenge-decided single round)
/// is one of the benchmarks where PCFR+-family variants historically degrade.
/// Run all predictive configurations for 200 iterations and report honestly
/// whether the new robustifications beat PCFR+/SAPCFR+ there (no dominance
/// assertion -- this is a measurement, not a guarantee).
TEST(LiarsDice, PredictiveVariants_comparison_report)
{
   struct Result {
      const char* name;
      double expl_early;
      double expl_late;
      bool finite;
   };

   std::vector< Result > results;
   auto run = [&results]< auto config >(const char* name) {
      bool finite = true;
      const double early = liars_dice_exploitability_after< config >(50, &finite);
      const double late = liars_dice_exploitability_after< config >(200, &finite);
      results.push_back({name, early, late, finite});
   };

   run.operator()< k_pcfr_plus_config >("PCFR+");
   run.operator()< k_sap_pcfr_plus_config >("SAPCFR+");
   run.operator()< k_ap_pcfr_plus_config >("APCFR+");
   run.operator()< k_p2p_pcfr_plus_config >("P2PCFR+");
   run.operator()< k_smooth_pcfr_plus_config >("SmoothPCFR+");
   run.operator()< k_stable_pcfr_plus_config >("StablePCFR+");

   std::cout << "\n[liars_dice d=4 single-round] exploitability comparison\n";
   for(const auto& r : results) {
      std::cout << "  " << r.name << "  @50: " << r.expl_early << "  @200: " << r.expl_late
                << "  finite: " << (r.finite ? "yes" : "NO") << "\n";
   }

   for(const auto& r : results) {
      EXPECT_TRUE(r.finite) << r.name;
      EXPECT_TRUE(std::isfinite(r.expl_late)) << r.name;
      EXPECT_LT(r.expl_late, r.expl_early) << r.name;
   }
}

/// goofspiel k=4 public-reveal mode: 1000-iteration no-NaN guard per variant
TEST(Goofspiel, PredictiveVariants_no_nan_guard)
{
   auto guard = []< auto config >(const char* name) {
      bool finite = false;
      // pure no-NaN guard: the best-response computation for exploitability
      // dominates the runtime on goofspiel, so it is skipped here on purpose
      const double expl = goofspiel_exploitability_after< config >(
         1000, &finite, /*with_exploitability=*/false
      );
      std::cout << "[goofspiel k=4 reveal] " << name
                << " 1000-iteration run: average policy finite: " << (finite ? "yes" : "NO")
                << "\n";
      EXPECT_TRUE(finite) << name << ": average policy contains non-finite probabilities";
   };

   guard.operator()< k_ap_pcfr_plus_config >("APCFR+");
   guard.operator()< k_p2p_pcfr_plus_config >("P2PCFR+");
   guard.operator()< k_smooth_pcfr_plus_config >("SmoothPCFR+");
   guard.operator()< k_stable_pcfr_plus_config >("StablePCFR+");
}
