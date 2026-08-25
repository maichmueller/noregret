
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <utility>
#include <vector>

#include "cfr_run_funcs.hpp"
#include "nor/env.hpp"
#include "nor/exploitability.hpp"
#include "nor/nor.hpp"
// NOTE: the shapley game headers must come LAST among the library includes: they inject
// 'using namespace ::shapley' into 'nor::games::shapley', and parsing any sibling env wrapper
// (kuhn/rps/stratego via nor/env.hpp) afterwards makes its unqualified 'State' lookups ambiguous.
#include "shapley/shapley.hpp"

using namespace nor;

namespace sh = nor::games::shapley;
namespace sp = ::shapley;

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////// static conformance of the new family ////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

constexpr auto kInternalCfg = rm::CFRConfig{
   .update_mode = rm::UpdateMode::alternating,
   .regret_minimizing_mode = rm::RegretMinimizingMode::internal_regret_matching};

constexpr auto kInternalThresholdedCfg = rm::CFRConfig{
   .update_mode = rm::UpdateMode::alternating,
   .regret_minimizing_mode = rm::RegretMinimizingMode::internal_regret_matching,
   .weighting_mode = rm::CFRWeightingMode::uniform,
   .pruning_mode = rm::CFRPruningMode::dynamic_thresholding};

}  // namespace

// the append-at-end contract of the selection enum (tests and persisted configs rely on the
// numeric order of the RegretMinimizingMode enumerators)
static_assert(
   rm::RegretMinimizingMode::internal_regret_matching == static_cast< rm::RegretMinimizingMode >(10)
);

static_assert(rm::regret_minimizer_for<
              rm::InternalRegretMatching< sp::Play >,
              sp::Play,
              HashmapActionPolicy< sp::Play > >);

static_assert(std::same_as<
              rm::minimizer_for_t< kInternalCfg, sp::Play >,
              rm::InternalRegretMatching< sp::Play > >);

static_assert(std::same_as<
              rm::minimizer_for_t< kInternalThresholdedCfg, sp::Play >,
              rm::Thresholded< rm::InternalRegretMatching< sp::Play >, 3. > >);

TEST(InternalRegretMatching, config_selection_and_factory_wiring)
{
   using PolicyT = TabularPolicy< sh::Infostate, HashmapActionPolicy< sh::Play > >;
   // the named alias resolves to the vanilla frame carrying the internal-regret kernel ...
   static_assert(std::same_as<
                 rm::InternalRegretCFR< sh::Environment, PolicyT, PolicyT >,
                 rm::VanillaCFR< kInternalCfg, sh::Environment, PolicyT, PolicyT > >);
   // ... and the generic factory routes such configs there (structural config equality)
   static_assert(std::same_as<
                 rm::base_minimizer_for_t< kInternalCfg, sp::Play >,
                 rm::InternalRegretMatching< sp::Play > >);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////// kernel arithmetic: hand-computed two-action fold /////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

TEST(InternalRegretMatching, folds_instant_regret_against_the_recommendation_snapshot)
{
   using Minimizer = rm::InternalRegretMatching< sp::Play >;
   Minimizer::node_data_type data{};
   Minimizer::register_action(data, sp::Play{0});
   Minimizer::register_action(data, sp::Play{1});

   HashmapActionPolicy< sp::Play > policy{};

   // ---- round 1: no snapshot yet -> the fold pairs against the uniform initialization
   Minimizer::observe(data, sp::Play{0}, 2.);
   Minimizer::observe(data, sp::Play{1}, 0.);
   // v = 0.5*2 + 0.5*0 = 1 ; R(phi) += (2 - 1, 0 - 1) = (1, -1)
   Minimizer::recommend(data, policy, /*iteration=*/0);
   EXPECT_EQ(data.regret[0], 1.);
   EXPECT_EQ(data.regret[1], -1.);
   EXPECT_DOUBLE_EQ(policy[sp::Play{0}], 1.);
   EXPECT_DOUBLE_EQ(policy[sp::Play{1}], 0.);
   // the snapshot was refreshed to the recommendation ...
   EXPECT_TRUE(data.snapshot_live);
   EXPECT_EQ(data.policy_snapshot[0], 1.);
   EXPECT_EQ(data.policy_snapshot[1], 0.);
   // ... and the instantaneous buffer was consumed
   EXPECT_DOUBLE_EQ(data.instant_regret[0], 0.);
   EXPECT_DOUBLE_EQ(data.instant_regret[1], 0.);

   // ---- round 2: the fold now pairs against the pure-on-action-0 snapshot
   Minimizer::observe(data, sp::Play{0}, -0.5);
   Minimizer::observe(data, sp::Play{1}, +0.5);
   // v = 1*(-0.5) + 0*(+0.5) = -0.5 ; R(phi) += (-0.5 + 0.5, 0.5 + 0.5) = (0, 1) -> (1, 0)
   Minimizer::recommend(data, policy, /*iteration=*/1);
   EXPECT_EQ(data.regret[0], 1.);
   EXPECT_EQ(data.regret[1], 0.);
   EXPECT_DOUBLE_EQ(policy[sp::Play{0}], 1.);
   EXPECT_DOUBLE_EQ(policy[sp::Play{1}], 0.);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////// unit-level repeated-matrix validation of the empirical regret shrinkage /////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

struct RegretReport {
   /// max_a sum_t [u(a, o_t) - <sigma^t, u(., o_t)>] (best fixed pure deviation)
   double external_regret = 0.;
   /// max_{i != j} sum_t sigma^t(i) [u(j, o_t) - u(i, o_t)] (best fixed swap deviation, i.e.
   /// the full Hart-Mas-Colell pairwise internal-regret matrix in mixed-strategy form)
   double internal_regret = 0.;
   /// sum_t <sigma^t, u(., o_t)> / rounds (average secured value of the trajectory)
   double average_strategy_value = 0.;
   size_t rounds = 0;
};

/// plays the ROW player of the bimatrix 'row_payoff' against a deterministic scripted column
/// opponent ('opponent_play(t)' = column index of round t) for 'rounds' rounds, driving the
/// InternalRegretMatching kernel DIRECTLY through the minimizer protocol (unweighted increments,
/// i.e. cf-reach 1). The empirical deviations are audited OFFLINE over the whole mixed-strategy
/// trajectory (deterministic: no sampling noise enters the measurements).
template < typename OpponentScript >
   requires std::regular_invocable< OpponentScript, size_t >
RegretReport play_repeated_matrix(
   const std::array< std::array< double, sp::strategy_count >, sp::strategy_count >& row_payoff,
   OpponentScript&& opponent_play,
   size_t rounds
)
{
   using Minimizer = rm::InternalRegretMatching< sp::Play >;
   Minimizer::node_data_type data{};
   for(uint8_t s = 0; s < sp::strategy_count; ++s) {
      Minimizer::register_action(data, sp::Play{s});
   }

   HashmapActionPolicy< sp::Play > policy{};
   // the played (mixed) strategy: uniform until the kernel's first recommendation arrives,
   // afterwards exactly the snapshot the kernel folds against -- mirroring the solver frame
   std::array< double, sp::strategy_count > sigma{};
   sigma.fill(1. / double(sp::strategy_count));

   std::array< double, sp::strategy_count > fixed_action_value_sum{};
   double strategy_value_sum{0.};
   // weighted swap-deviation accumulators R(i,j) of the REALIZED mixed trajectory
   std::array< std::array< double, sp::strategy_count >, sp::strategy_count > swaps{};

   for(size_t t : std::views::iota(size_t{0}, rounds)) {
      const auto opp = static_cast< uint8_t >(opponent_play(t));

      // audit bookkeeping of this round's counterfactual value vector r(a) = u(a, o_t)
      double expected_value{0.};
      for(uint8_t a = 0; a < sp::strategy_count; ++a) {
         fixed_action_value_sum[a] += row_payoff[a][opp];
         expected_value += sigma[a] * row_payoff[a][opp];
      }
      strategy_value_sum += expected_value;
      for(uint8_t i = 0; i < sp::strategy_count; ++i) {
         for(uint8_t j = 0; j < sp::strategy_count; ++j) {
            swaps[i][j] += sigma[i] * (row_payoff[j][opp] - row_payoff[i][opp]);
         }
      }

      // counterfactually weighted instantaneous regret increments (cf-reach 1)
      for(uint8_t a = 0; a < sp::strategy_count; ++a) {
         Minimizer::observe(data, sp::Play{a}, row_payoff[a][opp]);
      }
      // end-of-iteration recommendation refresh (the kernel's own protocol step)
      Minimizer::recommend(data, policy, t);
      for(uint8_t a = 0; a < sp::strategy_count; ++a) {
         sigma[a] = policy[sp::Play{a}];
      }
   }

   RegretReport report{};
   report.rounds = rounds;
   report.external_regret = *std::ranges::max_element(fixed_action_value_sum) - strategy_value_sum;
   for(uint8_t i = 0; i < sp::strategy_count; ++i) {
      for(uint8_t j = 0; j < sp::strategy_count; ++j) {
         if(i != j) {
            report.internal_regret = std::max(report.internal_regret, swaps[i][j]);
         }
      }
   }
   report.average_strategy_value = strategy_value_sum / double(rounds);
   return report;
}

}  // namespace

///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// Shapley bimatrix (general-sum) ///////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(InternalRegretShapleyMatrix, empirical_internal_and_external_regret_shrink_polynomially)
{
   // the canonical Shapley row-payoff transcribed from src/games/shapley (see state.hpp)
   constexpr std::array< std::array< double, 3 >, 3 > row_payoff{
      {{{1., 0., 0.}}, {{0., 1., 0.}}, {{0., 0., 1.}}}};
   // deterministic cycling opponent walking the famous best-response cycle columns
   constexpr std::array< uint8_t, 6 > br_cycle_columns{0, 2, 2, 1, 1, 0};

   constexpr std::array< size_t, 3 > horizons{400, 1600, 6400};
   std::array< double, horizons.size() > avg_internal{};
   std::array< double, horizons.size() > avg_external{};

   for(auto [idx, horizon] : std::views::enumerate(horizons)) {
      auto report = play_repeated_matrix(
         row_payoff,
         [&](size_t t) { return br_cycle_columns[t % br_cycle_columns.size()]; },
         horizon
      );
      avg_internal[idx] = report.internal_regret / double(report.rounds);
      avg_external[idx] = report.external_regret / double(report.rounds);
      fmt::print(
         "[internal-regret-shapley-unit] T={} internal={:.6e} external={:.6e}\n",
         horizon,
         report.internal_regret,
         report.external_regret
      );
      // hard blowup guards alongside the rate assertions below
      EXPECT_LT(report.internal_regret, 5. * std::sqrt(double(horizon)));
      EXPECT_LT(report.external_regret, 5. * std::sqrt(double(horizon)));
   }

   // polynomial (O(1/sqrt(T))-consistent) shrinkage of the AVERAGE regrets: quadrupling the
   // horizon at least halves the average internal regret -- twice in a row
   EXPECT_LT(avg_internal[1], 0.5 * avg_internal[0]);
   EXPECT_LT(avg_internal[2], 0.5 * avg_internal[1]);
   EXPECT_LT(avg_external[1], 0.75 * avg_external[0]);
   EXPECT_LT(avg_external[2], 0.75 * avg_external[1]);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// rock-paper-scissors //////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(InternalRegretRPSCycleMatrix, internal_regret_vanishes_and_average_approaches_nash)
{
   // zero-sum RPS: rock=0 beats scissors=2, paper=1 beats rock=0, scissors=2 beats paper=1
   constexpr std::array< std::array< double, 3 >, 3 > row_payoff{
      {{{0., -1., 1.}}, {{1., 0., -1.}}, {{-1., 1., 0.}}}};
   constexpr std::array< uint8_t, 3 > cycle_columns{0, 1, 2};

   constexpr size_t kHorizon = 6000;
   auto report = play_repeated_matrix(
      row_payoff, [&](size_t t) { return cycle_columns[t % cycle_columns.size()]; }, kHorizon
   );
   fmt::print(
      "[internal-regret-rps-unit] T={} internal={:.6e} external={:.6e}\n",
      kHorizon,
      report.internal_regret,
      report.external_regret
   );

   // the empirical internal (swap) regret is driven towards zero ...
   EXPECT_LT(report.internal_regret, 1e-2 * double(kHorizon));
   // ... and the average strategy approaches the Nash equilibrium profile: uniform play secures
   // value 0 against ANY cycling opponent, so a near-zero secured average value bounds the
   // trajectory's distance from the equilibrium
   EXPECT_GT(report.average_strategy_value, -0.05);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////// full-frame integration: Shapley's general-sum testbed ////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

struct InternalRegretConvergenceReport {
   double nash_conv_first_checkpoint = std::numeric_limits< double >::quiet_NaN();
   double nash_conv_final = std::numeric_limits< double >::quiet_NaN();
   size_t iterations = 0;
};

}  // namespace

// METRIC NOTE. Mirrors the baseline characterization of
// test/games/shapley/test_state.cpp: Shapley's game is GENERAL-SUM, so the meaningful metric is
// nash_conv(..., constant_sum=false) = sum_i u_i(BR_i, pi_-i) - u_i(pi) (plus the per-player gaps
// decomposition). The unique Nash equilibrium is the uniform profile, which also lies in the
// correlated-equilibrium set; an average profile whose NashConv vanishes therefore converges to
// the CE set through its Nash point. The internal-regret dynamics additionally enjoy the
// polynomial-swap-regret property validated at unit level above.
TEST(InternalRegretShapleyCFR, average_profile_converges_to_the_equilibrium_set)
{
   sh::Environment env{};
   auto root_state = std::make_unique< sh::State >();

   auto avg_policy = factory::make_tabular_policy(
      std::unordered_map< sh::Infostate, HashmapActionPolicy< sh::Play > >{}
   );
   auto curr_policy = factory::make_tabular_policy(
      std::unordered_map< sh::Infostate, HashmapActionPolicy< sh::Play > >{}
   );

   auto solver = factory::make_cfr< kInternalCfg, true >(
      std::move(env), std::move(root_state), curr_policy, avg_policy
   );
   sh::Environment expl_env{};

   constexpr size_t kIterations = 300;
   constexpr size_t kFirstCheckpoint = 2;
   constexpr size_t kCheckpoint = 50;

   InternalRegretConvergenceReport report{};
   report.iterations = kIterations;
   for(size_t iter : std::views::iota(size_t{1}, kIterations + 1)) {
      solver.iterate(1);
      if(iter != kFirstCheckpoint and (iter < kFirstCheckpoint or iter % kCheckpoint != 0)) {
         continue;
      }
      const auto& avg_policies = solver.average_policy();
      using AvgTablePolicy = std::decay_t< decltype(avg_policies.at(nor::Player::alex)) >;
      auto normalized_profile = player_hashmap< AvgTablePolicy >{
         std::pair{nor::Player::alex, normalize_state_policy(avg_policies.at(nor::Player::alex))},
         std::pair{nor::Player::bob, normalize_state_policy(avg_policies.at(nor::Player::bob))}};
      // constant_sum=false: the general-sum-safe metric
      double nc = nash_conv(expl_env, sh::State{}, normalized_profile, false);
      fmt::print("[internal-regret-shapley-cfr] iter={} nash_conv={:.6e}\n", iter, nc);
      if(iter == kFirstCheckpoint) {
         report.nash_conv_first_checkpoint = nc;
      }
      if(iter == kIterations) {
         report.nash_conv_final = nc;
      }
   }

   fmt::print(
      "[internal-regret-shapley-cfr] summary iterations={} nash_conv_at_iter_{}={:.6e} "
      "nash_conv_final={:.6e}\n",
      report.iterations,
      kFirstCheckpoint,
      report.nash_conv_first_checkpoint,
      report.nash_conv_final
   );

   // the profile never becomes MORE exploitable than at the very first checkpoint ...
   EXPECT_LE(report.nash_conv_final, report.nash_conv_first_checkpoint + 1e-12);
   // ... and it converges to the equilibrium (CE-through-Nash) set
   EXPECT_LT(report.nash_conv_final, 1e-9);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////// full-frame integration: RPS + Kuhn poker smoke /////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(InternalRegretRockPaperScissors, exploitability_vanishes_from_biased_initialization)
{
   // reuses the house RPS runner (which deliberately biases BOTH players' initial policies away
   // from the uniform Nash profile): the average strategy must recover Nash, i.e. the accumulated
   // internal regret of both players must have shrunk to an epsilon-exploitability level
   ASSERT_TRUE((run_cfr_on_rps_checked< kInternalCfg >(20000, 10)));
}

TEST(InternalRegretKuhnPoker, exploitability_decreasing_smoke)
{
   // sanity smoke on the smallest sequential game of the suite: the internal-regret kernel
   // reaches the house exploitability threshold within the vanilla iteration budget
   ASSERT_TRUE((run_cfr_on_kuhn_poker_checked< kInternalCfg >(100000, 25)));
}
