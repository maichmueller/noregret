#include <gtest/gtest.h>

#include <iostream>
#include <unordered_map>
#include <utility>

#include "../games/stratego/fixtures.hpp"
#include "cfr_run_funcs.hpp"
#include "nor/env.hpp"
#include "nor/nor.hpp"
#include "rm_specific_testing_utils.hpp"

using namespace nor;

///////////////////////////////////////////////////////////////////////////////////////
//// static configuration contracts /////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

// 'greedy' is APPENDED at the end of CFRWeightingMode (numeric enumerator order is
// relied upon elsewhere)
static_assert(std::to_underlying(rm::CFRWeightingMode::greedy) == 4);

// regression guard: the default configuration is and stays UNIFORM weighted --
// none of the greedy plumbing may change its behavior
static_assert(rm::CFRConfig{}.weighting_mode == rm::CFRWeightingMode::uniform);

namespace {

constexpr rm::CFRConfig greedy_config{
   .update_mode = rm::UpdateMode::simultaneous,
   .regret_minimizing_mode = rm::RegretMinimizingMode::regret_matching,
   .weighting_mode = rm::CFRWeightingMode::greedy};

// supported combinations: greedy composes with plain RM and RM+ kernels under
// SIMULTANEOUS updates (the published scheme) without pruning
static_assert(rm::detail::sanity_check_cfr_config< greedy_config >());
static_assert(rm::detail::sanity_check_cfr_config< rm::CFRConfig{
                 .update_mode = rm::UpdateMode::simultaneous,
                 .regret_minimizing_mode = rm::RegretMinimizingMode::regret_matching_plus,
                 .weighting_mode = rm::CFRWeightingMode::greedy} >());

// unsupported combinations are rejected at compile time: greedy weights is a
// simultaneous-update scheme (alternating per-player instances break convergence),
// has no defined insertion point into the predictive / DCFR+-style kernels (they
// require the discounted carrier mode) and does not compose with pruning
template < rm::CFRConfig cfg >
constexpr bool rejected_combination = not rm::detail::sanity_check_cfr_config< cfg >();

static_assert(rejected_combination< rm::CFRConfig{
                 .update_mode = rm::UpdateMode::alternating,
                 .regret_minimizing_mode = rm::RegretMinimizingMode::regret_matching,
                 .weighting_mode = rm::CFRWeightingMode::greedy} >);
static_assert(rejected_combination< rm::CFRConfig{
                 .update_mode = rm::UpdateMode::alternating,
                 .regret_minimizing_mode = rm::RegretMinimizingMode::regret_matching_plus,
                 .weighting_mode = rm::CFRWeightingMode::greedy} >);
static_assert(rejected_combination< rm::CFRConfig{
                 .update_mode = rm::UpdateMode::simultaneous,
                 .regret_minimizing_mode =
                    rm::RegretMinimizingMode::predictive_regret_matching_plus,
                 .weighting_mode = rm::CFRWeightingMode::greedy} >);
static_assert(rejected_combination< rm::CFRConfig{
                 .update_mode = rm::UpdateMode::simultaneous,
                 .regret_minimizing_mode =
                    rm::RegretMinimizingMode::discounted_predictive_regret_matching_plus,
                 .weighting_mode = rm::CFRWeightingMode::greedy} >);
static_assert(rejected_combination< rm::CFRConfig{
                 .update_mode = rm::UpdateMode::simultaneous,
                 .regret_minimizing_mode = rm::RegretMinimizingMode::regret_matching,
                 .weighting_mode = rm::CFRWeightingMode::greedy,
                 .pruning_mode = rm::CFRPruningMode::dynamic_thresholding} >);
static_assert(rejected_combination< rm::CFRConfig{
                 .update_mode = rm::UpdateMode::simultaneous,
                 .regret_minimizing_mode = rm::RegretMinimizingMode::regret_matching_plus,
                 .weighting_mode = rm::CFRWeightingMode::greedy,
                 .pruning_mode = rm::CFRPruningMode::regret_based} >);

/// builds an (uninitialized => uniform) tabular policy pair for kuhn poker
auto make_kuhn_policies()
{
   using PolicyTable = std::
      unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >;
   return std::make_pair(
      factory::make_tabular_policy(PolicyTable{}), factory::make_tabular_policy(PolicyTable{})
   );
}

/// normalized average-policy profile of the three-seat kuhn solver (mirrors the
/// value type that 'exploitability' consumes in cfr_run_funcs.hpp: a map of
/// POLICY objects, not of raw action tables)
auto kuhn_average_profile(auto& solver)
{
   return player_hashmap<
      std::decay_t< decltype(normalize_state_policy(solver.average_policy().at(Player::alex))) > >{
      std::pair{Player::alex, normalize_state_policy(solver.average_policy().at(Player::alex))},
      std::pair{Player::bob, normalize_state_policy(solver.average_policy().at(Player::bob))},
      std::pair{
         Player::cedric, normalize_state_policy(solver.average_policy().at(Player::cedric))}};
}

}  // namespace

///////////////////////////////////////////////////////////////////////////////////////
//// two-player sanity: exploitability drops below the standard threshold ////////////
///////////////////////////////////////////////////////////////////////////////////////

TEST(KuhnPoker, CFR_GREEDY_simultaneous)
{
   const bool converged = run_cfr_on_kuhn_poker_checked< rm::CFRGreedyConfig{
      .update_mode = rm::UpdateMode::simultaneous} >();
   EXPECT_TRUE(converged);
}

TEST(KuhnPoker, CFR_GREEDY_RMPLUS_simultaneous)
{
   const bool converged = run_cfr_on_kuhn_poker_checked< rm::CFRGreedyConfig{
      .update_mode = rm::UpdateMode::simultaneous,
      .regret_minimizing_mode = rm::RegretMinimizingMode::regret_matching_plus} >();
   EXPECT_TRUE(converged);
}

///////////////////////////////////////////////////////////////////////////////////////
//// regression guard: default (uniform) mode routes through the shared engine ///////
///////////////////////////////////////////////////////////////////////////////////////

TEST(KuhnPoker, CFR_GREEDY_default_uniform_regression)
{
   const bool converged = run_cfr_on_kuhn_poker_checked< rm::CFRConfig{} >();
   EXPECT_TRUE(converged);
}

///////////////////////////////////////////////////////////////////////////////////////
//// three-player coverage: convergence + demonstrably active greedy weighting ///////
///////////////////////////////////////////////////////////////////////////////////////

namespace {

constexpr std::size_t n_3p_iterations = 6000;
/// generous upper bound on the average exploitability of the average-strategy
/// profile after the fixed iteration budget above (tuned empirically; vanilla CFR
/// descends well below this on 3-seat kuhn poker)
constexpr double expl_threshold_3p = 0.05;

template < auto config >
auto run_kuhn_3p()
{
   games::kuhn::Environment env{};
   auto [current_policy, average_policy] = make_kuhn_policies();
   auto solver = factory::make_cfr< config, true >(
      games::kuhn::Environment{},
      std::make_unique< games::kuhn::State >(
         std::vector< games::kuhn::Card >{
            games::kuhn::Card::jack,
            games::kuhn::Card::queen,
            games::kuhn::Card::king,
            games::kuhn::Card::ace},
         std::size_t{3}
      ),
      std::move(current_policy),
      std::move(average_policy)
   );
   for(auto _ [[maybe_unused]] : std::views::iota(std::size_t{0}, n_3p_iterations)) {
      solver.iterate(1);
   }
   const double expl = exploitability(env, solver.root_state(), kuhn_average_profile(solver));
   return std::pair{expl, solver.greedy_weight_stats()};
}

}  // namespace

TEST(KuhnPokerThreePlayer, CFR_GREEDY_simultaneous_converges_and_weights_active)
{
   auto [expl, stats] = run_kuhn_3p< rm::CFRGreedyConfig{
      .update_mode = rm::UpdateMode::simultaneous} >();

   std::cout << "[  3p greedy ] exploitability after " << n_3p_iterations << " iterations: " << expl
             << " | weights drawn: " << stats.weight_draws
             << " | history discards: " << stats.history_discards
             << " | min/mean/max: " << stats.min_weight << " / "
             << stats.total_weight / double(stats.weight_draws) << " / " << stats.max_weight
             << "\n";

   EXPECT_LT(expl, expl_threshold_3p);
   // simultaneous updates draw a SINGLE JOINT weight per iteration
   ASSERT_GT(stats.weight_draws, std::size_t{0});
   EXPECT_EQ(stats.weight_draws, n_3p_iterations);
   EXPECT_GT(stats.min_weight, 0.);
   // greedy weighting demonstrably active: effective weights deviate from the
   // constant-1 profile uniform averaging would produce
   EXPECT_GT(stats.max_weight, stats.min_weight);
   EXPECT_GT(stats.max_weight, 1. + 1e-9);
}
