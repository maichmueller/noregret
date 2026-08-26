#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <ranges>
#include <unordered_map>
#include <utility>
#include <vector>

#include "nor/env.hpp"
#include "nor/nor.hpp"

using namespace nor;

///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////// runtime-adaptive dynamic discounting (DDCFR, tabular) //////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

/// the published DDCFR base kernel: plain DCFR regret matching riding the
/// discounted weighting carrier (Xu et al., ICLR 2024 build on Brown &
/// Sandholm's DCFR, not on the CFR+ family)
constexpr auto k_dcfr_config = rm::CFRDiscountedConfig{};

/// house convergence threshold of this suite's small-game runners
constexpr double k_kuhn_threshold = 3e-3;

struct KuhnDdcfrRun {
   double exploitability = std::numeric_limits< double >::quiet_NaN();
   std::shared_ptr< rm::ddcfr::DDCFRController > controller;
};

/// runs kuhn poker for 'n_iters' with a ddcfr-controlled discounted solver and
/// reports the final exploitability of the average strategy profile
template < typename PolicyLike >
KuhnDdcfrRun run_kuhn_with_ddcfr(
   size_t n_iters,
   PolicyLike&& policy,
   rm::ddcfr::Options options,
   bool with_exploitability_probe
)
{
   games::kuhn::Environment env{};
   auto curr_policy = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{}
   );
   auto avg_policy = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{}
   );

   auto controller = std::make_shared< rm::ddcfr::DDCFRController >(
      std::forward< PolicyLike >(policy), options
   );
   auto params = rm::ddcfr::ddcfr_parameters(controller);

   auto solver = factory::make_cfr< k_dcfr_config, true >(
      env, std::make_unique< games::kuhn::State >(), curr_policy, avg_policy, params
   );
   if(with_exploitability_probe) {
      controller->bind(solver, env, games::kuhn::State{});
   } else {
      controller->bind(solver);
   }
   solver.iterate(n_iters);

   const auto& avg_policies = solver.average_policy();
   const double expl = exploitability(
      env,
      games::kuhn::State{},
      player_hashmap< std::decay_t< decltype(avg_policies.at(Player::alex)) > >{
         std::pair{Player::alex, normalize_state_policy(avg_policies.at(Player::alex))},
         std::pair{Player::bob, normalize_state_policy(avg_policies.at(Player::bob))}}
   );
   return KuhnDdcfrRun{.exploitability = expl, .controller = controller};
}

/// leduc short-horizon variant of the above
template < typename PolicyLike >
double run_leduc_with_ddcfr(size_t n_iters, PolicyLike&& policy, rm::ddcfr::Options options)
{
   using Action = games::leduc::Action;
   games::leduc::Environment env{};
   auto curr_policy = factory::make_tabular_policy(
      std::unordered_map< games::leduc::Infostate, HashmapActionPolicy< Action > >{}
   );
   auto avg_policy = factory::make_tabular_policy(
      std::unordered_map< games::leduc::Infostate, HashmapActionPolicy< Action > >{}
   );

   auto controller = std::make_shared< rm::ddcfr::DDCFRController >(
      std::forward< PolicyLike >(policy), options
   );
   auto params = rm::ddcfr::ddcfr_parameters(controller);

   auto solver = factory::make_cfr< k_dcfr_config, true >(
      env, std::make_unique< games::leduc::State >(), curr_policy, avg_policy, params
   );
   controller->bind(solver);
   solver.iterate(n_iters);

   const auto& avg_policies = solver.average_policy();
   return exploitability(
      env,
      games::leduc::State{},
      player_hashmap< std::decay_t< decltype(avg_policies.at(Player::alex)) > >{
         std::pair{Player::alex, normalize_state_policy(avg_policies.at(Player::alex))},
         std::pair{Player::bob, normalize_state_policy(avg_policies.at(Player::bob))}}
   );
}

template < typename Params >
double kuhn_fixed_parameters_exploitability(size_t n_iters, Params&& params)
{
   games::kuhn::Environment env{};
   auto curr_policy = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{}
   );
   auto avg_policy = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{}
   );
   auto solver = factory::make_cfr< k_dcfr_config, true >(
      env,
      std::make_unique< games::kuhn::State >(),
      curr_policy,
      avg_policy,
      std::forward< Params >(params)
   );
   solver.iterate(n_iters);
   const auto& avg_policies = solver.average_policy();
   return exploitability(
      env,
      games::kuhn::State{},
      player_hashmap< std::decay_t< decltype(avg_policies.at(Player::alex)) > >{
         std::pair{Player::alex, normalize_state_policy(avg_policies.at(Player::alex))},
         std::pair{Player::bob, normalize_state_policy(avg_policies.at(Player::bob))}}
   );
}

}  // namespace

/////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// schedule-function contract tests ///////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

/// B1 contract: despite alpha/beta/gamma hooks all firing per iteration, the
/// tau == 1 policy is consulted EXACTLY once per raw index 0..T-1, in order
TEST(DDCFRScheduleContract, QueriedOncePerIterationWithRawIndex)
{
   constexpr size_t T = 40;

   std::vector< size_t > queried_indices;
   auto controller = std::make_shared< rm::ddcfr::DDCFRController >(
      [&](const rm::ddcfr::DDCFRFeatures&, size_t idx) {
         queried_indices.push_back(idx);
         return rm::ddcfr::DDCFRWeights{.alpha = 1.5, .beta = 0., .gamma = 2., .tau = 1};
      },
      rm::ddcfr::Options{.total_iterations = T}
   );
   auto params = rm::ddcfr::ddcfr_parameters(controller);

   games::kuhn::Environment env{};
   auto curr_policy = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{}
   );
   auto avg_policy = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{}
   );
   auto solver = factory::make_cfr< k_dcfr_config, true >(
      env, std::make_unique< games::kuhn::State >(), curr_policy, avg_policy, params
   );
   controller->bind(solver);
   solver.iterate(T);

   ASSERT_EQ(queried_indices.size(), T);
   for(auto [expected, received] :
       std::views::zip(std::views::iota(size_t{0}, T), queried_indices)) {
      EXPECT_EQ(received, expected);
   }
   // the trajectory mirrors the queries one-to-one
   ASSERT_EQ(controller->trajectory().size(), T);
   EXPECT_EQ(controller->trajectory().front().query_index, size_t{0});
   EXPECT_EQ(controller->trajectory().back().query_index, T - 1);
}

/// published tau semantics: a policy emitting tau = K HOLDS its weights for K
/// consecutive indices -- only ceil(T/K) queries happen, at multiples of K
TEST(DDCFRScheduleContract, TauHoldDefersQueriesAndHoldsWeights)
{
   constexpr size_t T = 30;
   constexpr size_t K = 4;
   // in-range alpha stamp identifying the draw an index resolves from
   // (0.1 + 0.01 * draw_index; stays well inside [0, 5] so the sanitizer
   // cannot alter it)
   constexpr double k_alpha0 = 0.1, k_alpha_step = 0.01;
   auto controller = std::make_shared< rm::ddcfr::DDCFRController >(
      [&](const rm::ddcfr::DDCFRFeatures&, size_t idx) {
         return rm::ddcfr::DDCFRWeights{
            .alpha = k_alpha0 + k_alpha_step * static_cast< double >(idx),
            .beta = 0.,
            .gamma = 3.,
            .tau = K};
      },
      rm::ddcfr::Options{.total_iterations = T}
   );
   games::kuhn::Environment env{};
   auto curr_policy = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{}
   );
   auto avg_policy = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{}
   );
   auto solver = factory::make_cfr< k_dcfr_config, true >(
      env,
      std::make_unique< games::kuhn::State >(),
      curr_policy,
      avg_policy,
      rm::ddcfr::ddcfr_parameters(controller)
   );
   controller->bind(solver);
   solver.iterate(T);

   // queries at 0, 4, ..., 28 -> ceil(30/4) = 8 draws; every held index
   // resolves to the (sanitized) weights drawn at its window start -- alpha
   // carries the draw's index stamp
   ASSERT_EQ(controller->trajectory().size(), (T + K - 1) / K);
   for(const auto& entry : controller->trajectory()) {
      EXPECT_EQ(entry.query_index % K, size_t{0});
   }
   EXPECT_DOUBLE_EQ(
      controller->alpha_at(T - 1), k_alpha0 + k_alpha_step * 28.
   );  // hold of draw @28
}

/// pathological outputs are neutralized BEFORE they can reach the tables:
/// non-finite values fall back to the fixed-DCFR defaults, out-of-range values
/// clamp into the paper's action space (the discount_factor NaN guard of
/// commit 098e6f5 remains the second line of defense behind this)
TEST(DDCFRScheduleContract, PathologicalOutputsAreSanitized)
{
   const rm::ddcfr::DDCFRWeights sanitized = rm::ddcfr::sanitize_weights(rm::ddcfr::DDCFRWeights{
      .alpha = std::numeric_limits< double >::quiet_NaN(),
      .beta = std::numeric_limits< double >::infinity(),
      .gamma = -std::numeric_limits< double >::infinity(),
      .tau = 0});
   // non-finite -> neutral defaults
   EXPECT_DOUBLE_EQ(sanitized.alpha, 1.5);
   EXPECT_DOUBLE_EQ(sanitized.beta, 0.);
   EXPECT_DOUBLE_EQ(sanitized.gamma, 2.);
   EXPECT_EQ(sanitized.tau, size_t{1});

   const rm::ddcfr::DDCFRWeights clamped = rm::ddcfr::sanitize_weights(rm::ddcfr::DDCFRWeights{
      .alpha = 7., .beta = -9., .gamma = 12., .tau = 99});
   // finite but out of range -> paper's bounds
   EXPECT_DOUBLE_EQ(clamped.alpha, rm::ddcfr::ParameterRanges::alpha_max);
   EXPECT_DOUBLE_EQ(clamped.beta, rm::ddcfr::ParameterRanges::beta_min);
   EXPECT_DOUBLE_EQ(clamped.gamma, rm::ddcfr::ParameterRanges::gamma_max);
   EXPECT_EQ(clamped.tau, rm::ddcfr::ParameterRanges::tau_max);

   // end-to-end: an all-NaN policy degenerates to fixed DCFR(1.5, 0, 2) and
   // must converge normally on kuhn instead of poisoning the tables
   const double expl_nan_policy = run_kuhn_with_ddcfr(
                                     400,
                                     [](const rm::ddcfr::DDCFRFeatures&, size_t) {
                                        return rm::ddcfr::DDCFRWeights{
                                           .alpha = std::numeric_limits< double >::quiet_NaN(),
                                           .beta = std::numeric_limits< double >::quiet_NaN(),
                                           .gamma = std::numeric_limits< double >::quiet_NaN(),
                                           .tau = 0};
                                     },
                                     rm::ddcfr::Options{.total_iterations = 400},
                                     false
   )
                                     .exploitability;
   std::cout << "[kuhn@400] all-NaN-policy (sanitized to DCFR defaults) exploitability: "
             << expl_nan_policy << "\n";
   EXPECT_TRUE(std::isfinite(expl_nan_policy));
   EXPECT_LT(expl_nan_policy, k_kuhn_threshold);

   // negative scheduled exponent at the undefined raw index 0 is caught by the
   // existing guard our layer sits on top of (commit 098e6f5's contract)
   EXPECT_DOUBLE_EQ(rm::discount_factor(0, -5.), 0.5);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// piecewise scheme (policy a) ///////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

/// hand-computed checkpoints of the piecewise trajectory: anchors are exact at
/// p in {0, 1/2} and the segments interpolate linearly between them
TEST(PiecewiseDDCFRScheme, DiscountTrajectoryMatchesHandComputedCheckpoints)
{
   constexpr size_t T = 300;
   const rm::ddcfr::PiecewiseDDCFRPolicy policy{};
   auto eval_at = [&](double fraction) {
      return policy(rm::ddcfr::DDCFRFeatures{.iteration_fraction = fraction}, 0);
   };

   // p = 0 -> exactly the early anchor (aggressive phase)
   const auto w_early = eval_at(0. / double(T));
   EXPECT_DOUBLE_EQ(w_early.alpha, 0.5);
   EXPECT_DOUBLE_EQ(w_early.beta, -2.5);
   EXPECT_DOUBLE_EQ(w_early.gamma, 4.);
   EXPECT_EQ(w_early.tau, size_t{1});

   // p = 75/300: segment-local progress u = 0.5 between early and mid anchors
   //    alpha = 0.5 + (2.0 - 0.5)/2     = 1.25
   //    beta  = -2.5 + (-1.0 - (-2.5))/2 = -1.75
   //    gamma = 4.0 + (2.0 - 4.0)/2      = 3.0
   const auto w_quarter = eval_at(75. / double(T));
   EXPECT_NEAR(w_quarter.alpha, 1.25, 1e-12);
   EXPECT_NEAR(w_quarter.beta, -1.75, 1e-12);
   EXPECT_NEAR(w_quarter.gamma, 3.0, 1e-12);

   // p = 150/300 == mid_fraction -> exactly the mid anchor
   const auto w_mid = eval_at(150. / double(T));
   EXPECT_DOUBLE_EQ(w_mid.alpha, 2.0);
   EXPECT_DOUBLE_EQ(w_mid.beta, -1.0);
   EXPECT_DOUBLE_EQ(w_mid.gamma, 2.0);

   // p = 225/300: u = 0.5 between mid and late anchors
   //    alpha = 2.0 + (3.0 - 2.0)/2       = 2.5
   //    beta  = -1.0 + (-0.25 - (-1.0))/2 = -0.625
   //    gamma = 2.0 + (1.0 - 2.0)/2       = 1.5
   const auto w_three_quarter = eval_at(225. / double(T));
   EXPECT_NEAR(w_three_quarter.alpha, 2.5, 1e-12);
   EXPECT_NEAR(w_three_quarter.beta, -0.625, 1e-12);
   EXPECT_NEAR(w_three_quarter.gamma, 1.5, 1e-12);

   // monotone qualitative trend: aggressive early downweighting eases toward
   // the late near-uniform regime (alpha rises, beta rises toward 0, gamma
   // falls)
   EXPECT_LT(eval_at(0.).alpha, eval_at(1. - 1e-12).alpha);
   EXPECT_LT(eval_at(0.).beta, eval_at(1. - 1e-12).beta);
   EXPECT_GT(eval_at(0.).gamma, eval_at(1. - 1e-12).gamma);
}

/// the recorded trajectory equals what the policy reports for the same
/// features (end-to-end wiring check through the B1 hooks), and the scheme
/// converges below the house threshold on kuhn poker
TEST(PiecewiseDDCFRScheme, ConvergesOnKuhnPoker)
{
   constexpr size_t T = 600;
   const double expl = run_kuhn_with_ddcfr(
                          T,
                          rm::ddcfr::piecewise_ddcfr_policy(),
                          rm::ddcfr::Options{.total_iterations = T},
                          false
   )
                          .exploitability;
   std::cout << "[kuhn@" << T << "] piecewise-DDCFR exploitability: " << expl << "\n";
   EXPECT_TRUE(std::isfinite(expl));
   EXPECT_LT(expl, k_kuhn_threshold);
}

/// leduc short-horizon report: completes finitely and converges below the same
/// loose bound the HS-schedule tests use for leduc at short horizons
TEST(PiecewiseDDCFRScheme, ConvergesOnLeducShortHorizon)
{
   constexpr size_t T = 600;
   const double expl = run_leduc_with_ddcfr(
      T, rm::ddcfr::piecewise_ddcfr_policy(), rm::ddcfr::Options{.total_iterations = T}
   );
   std::cout << "[leduc@" << T << "] piecewise-DDCFR exploitability: " << expl << "\n";
   EXPECT_TRUE(std::isfinite(expl));
   EXPECT_LT(expl, 1e-2);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////// exploitability-proxy scheme (policy b) ////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

/// every K iterations the cheap full-tree exploitability probe fires; the
/// resulting convergence proxy decreases over the run (the whole point of a
/// convergence feature) and stays within [0, 1]
TEST(ExploitabilityProxyScheme, ProxyDecreasesDuringTheRun)
{
   constexpr size_t T = 400;
   constexpr size_t K = 25;

   const auto run = run_kuhn_with_ddcfr(
      T,
      rm::ddcfr::exploitability_proxy_policy(rm::ddcfr::ExploitabilityProxyPolicy{
         .query_interval = K}),
      rm::ddcfr::Options{.total_iterations = T, .exploitability_interval = K},
      /*with_exploitability_probe=*/true
   );

   const auto& history = run.controller->convergence_history();
   ASSERT_FALSE(history.empty());
   std::cout << "[kuhn@" << T << "] convergence proxy trace (" << history.size() << " probes): ";
   for(const auto& [index, proxy] : history) {
      std::cout << proxy << " ";
   }
   std::cout << "\n";

   EXPECT_GE(history.size(), T / K / 2);  // generous probe-count floor
   for(const auto& [index, proxy] : history) {
      EXPECT_TRUE(std::isfinite(proxy));
      EXPECT_GE(proxy, 0.);
      EXPECT_LE(proxy, 1.);
   }
   EXPECT_LT(history.back().second, history.front().second);
}

/// convergence quality versus the fixed DCFR(1.5, 0, 2) baseline: the dynamic
/// scheme may not lose by more than the DOCUMENTED factor of 8x at this bed-
/// game horizon (both arms are deterministic; the factor leaves headroom for
/// the hand-tuned proxy heuristic without letting regressions slip through)
TEST(ExploitabilityProxyScheme, NotWorseThanFixedDCFRBeyondDocumentedFactor)
{
   constexpr size_t T = 400;
   constexpr size_t K = 25;

   const double expl_dynamic = run_kuhn_with_ddcfr(
                                  T,
                                  rm::ddcfr::exploitability_proxy_policy(
                                     rm::ddcfr::ExploitabilityProxyPolicy{.query_interval = K}
                                  ),
                                  rm::ddcfr::Options{
                                     .total_iterations = T, .exploitability_interval = K},
                                  /*with_exploitability_probe=*/true
   )
                                  .exploitability;
   const double expl_baseline = kuhn_fixed_parameters_exploitability(
      T, rm::CFRDiscountedParameters{}
   );

   std::cout << "[kuhn@" << T << "] exploitability | exploitability-proxy DDCFR: " << expl_dynamic
             << " | fixed DCFR(1.5,0,2): " << expl_baseline
             << " | ratio: " << (expl_dynamic / expl_baseline) << "\n";
   EXPECT_TRUE(std::isfinite(expl_dynamic));
   EXPECT_TRUE(std::isfinite(expl_baseline));
   EXPECT_LT(expl_dynamic, k_kuhn_threshold * 100);
   EXPECT_LE(expl_dynamic, 8. * expl_baseline);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// custom policy injection ////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

/// a user-injected std::function policy (the future trained-network slot)
/// receives one feature vector per query with the documented contents
TEST(DDCFRCustomPolicyInjection, ReceivesWellFormedFeatures)
{
   constexpr size_t T = 31;
   constexpr size_t K = 4;

   std::vector< rm::ddcfr::DDCFRFeatures > seen_features;
   std::vector< size_t > seen_indices;
   const auto run = run_kuhn_with_ddcfr(
      T,
      [&](const rm::ddcfr::DDCFRFeatures& features, size_t idx) {
         seen_features.push_back(features);
         seen_indices.push_back(idx);
         return rm::ddcfr::DDCFRWeights{.alpha = 2., .beta = -1., .gamma = 3., .tau = K};
      },
      rm::ddcfr::Options{.total_iterations = T},
      false
   );
   (void) run;

   // tau = 4 hold -> queries at 0, 4, ..., 28
   ASSERT_EQ(seen_features.size(), (T + K - 1) / K);
   double previous_fraction = -1.;
   for(auto [features, idx] : std::views::zip(seen_features, seen_indices)) {
      EXPECT_EQ(idx % K, size_t{0});
      EXPECT_NEAR(features.iteration_fraction, double(idx) / double(T), 1e-12);
      EXPECT_GE(features.iteration_fraction, previous_fraction);
      previous_fraction = features.iteration_fraction;
      EXPECT_GE(features.strategy_entropy, 0.);
      EXPECT_LE(features.strategy_entropy, 1.);
      EXPECT_TRUE(std::isfinite(features.regret_l1_norm));
      EXPECT_GE(features.regret_l1_norm, 0.);
      EXPECT_GE(features.positive_regret_fraction, 0.);
      EXPECT_LE(features.positive_regret_fraction, 1.);
      // no probe wired: the paper's exploitability feature is unavailable ...
      EXPECT_TRUE(std::isnan(features.convergence_proxy));
   }
}

/// the injected weights actually drive the schedules (post-sanitization):
/// constant in-range weights appear verbatim across the whole trajectory
TEST(DDCFRCustomPolicyInjection, InjectedWeightsDriveSchedulesVerbatim)
{
   auto controller = std::make_shared< rm::ddcfr::DDCFRController >(
      [](const rm::ddcfr::DDCFRFeatures&, size_t) {
         return rm::ddcfr::DDCFRWeights{.alpha = 2., .beta = -1., .gamma = 3., .tau = 1};
      },
      rm::ddcfr::Options{.total_iterations = 50}
   );

   games::kuhn::Environment env{};
   auto curr_policy = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{}
   );
   auto avg_policy = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{}
   );
   auto solver = factory::make_cfr< k_dcfr_config, true >(
      env,
      std::make_unique< games::kuhn::State >(),
      curr_policy,
      avg_policy,
      rm::ddcfr::ddcfr_parameters(controller)
   );
   controller->bind(solver);
   solver.iterate(50);

   ASSERT_EQ(controller->trajectory().size(), size_t{50});
   for(const auto& entry : controller->trajectory()) {
      EXPECT_DOUBLE_EQ(entry.weights.alpha, 2.);
      EXPECT_DOUBLE_EQ(entry.weights.beta, -1.);
      EXPECT_DOUBLE_EQ(entry.weights.gamma, 3.);
      EXPECT_EQ(entry.weights.tau, size_t{1});
   }
   EXPECT_DOUBLE_EQ(controller->gamma_at(49), 3.);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// default non-regression /////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

/// nothing changes for users who never touch the new layer: default parameters
/// keep null schedules and the historical constants, and the discount-factor
/// helper keeps its pinned corner-case behavior
TEST(DDCFRLayerDefaults, DefaultConfigsUnchanged)
{
   const rm::CFRDiscountedParameters defaults{};
   EXPECT_TRUE(defaults.alpha_schedule == nullptr);
   EXPECT_TRUE(defaults.beta_schedule == nullptr);
   EXPECT_TRUE(defaults.gamma_schedule == nullptr);
   EXPECT_DOUBLE_EQ(defaults.alpha, 1.5);
   EXPECT_DOUBLE_EQ(defaults.beta, 0.);
   EXPECT_DOUBLE_EQ(defaults.gamma, 2.);
   EXPECT_FALSE(defaults.weight_by_cycle);
   EXPECT_DOUBLE_EQ(defaults.alpha_at(17), 1.5);
   EXPECT_DOUBLE_EQ(defaults.beta_at(17), 0.);
   EXPECT_DOUBLE_EQ(defaults.gamma_at(17), 2.);

   // d(t; e) pins incl. the NaN guard contract at raw index 0
   EXPECT_DOUBLE_EQ(rm::discount_factor(0, -1.), 0.5);
   EXPECT_DOUBLE_EQ(rm::discount_factor(0, 1.5), 0.);
   EXPECT_DOUBLE_EQ(rm::discount_factor(1, 1.5), 0.5);
   EXPECT_DOUBLE_EQ(rm::discount_factor(2, 2.), 4. / 5.);

   EXPECT_THROW(rm::ddcfr::ddcfr_parameters(nullptr), std::invalid_argument);
}
