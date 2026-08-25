#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <ranges>
#include <unordered_map>
#include <utility>
#include <vector>

#include "goofspiel/goofspiel.hpp"
#include "liars_dice/liars_dice.hpp"
#include "nor/env.hpp"
#include "nor/nor.hpp"
#include "rm_specific_testing_utils.hpp"

// Tests for the B7 pluggable MCCFR sampling rules:
//  - rm::PublicChanceSamplingRule (PCS; Gibson et al., AAAI 2012)
//  - rm::AverageStrategySamplingRule (ASS; Gibson et al., NIPS 2012)
//
// FAST BEDS ONLY: kuhn 2p, goofspiel k<=5, liars_dice d<=4 single-round.

using namespace nor;

namespace {

///////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// shared kuhn helpers //////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

using KEnv = games::kuhn::Environment;
using KState = games::kuhn::State;
using KInfostate = games::kuhn::Infostate;
using KAction = games::kuhn::Action;
using KObs = games::kuhn::Observation;

constexpr rm::MCCFRConfig k_os_epsilon_config{
   .update_mode = rm::UpdateMode::alternating,
   .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
   .weighting = rm::MCCFRWeightingMode::lazy};

constexpr rm::MCCFRConfig k_os_custom_rule_config{
   .update_mode = rm::UpdateMode::alternating,
   .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
   .exploration = rm::MCCFRExplorationMode::custom_sampling_policy,
   .weighting = rm::MCCFRWeightingMode::lazy};

static auto make_empty_kuhn_table()
{
   return factory::make_tabular_policy(
      std::unordered_map< KInfostate, HashmapActionPolicy< KAction > >{}
   );
}

using KPolicy = std::decay_t< decltype(make_empty_kuhn_table()) >;

template < auto config, typename Rule = rm::EpsilonOnPolicySamplingRule >
static auto make_kuhn_solver(size_t seed, Rule rule = Rule{})
{
   KEnv env{};
   auto root = std::make_unique< KState >();
   auto players = env.players(*root);

   std::unordered_map< Player, KPolicy > current;
   std::unordered_map< Player, KPolicy > average;
   for(auto player : players | utils::is_actual_player_filter) {
      current.emplace(player, make_empty_kuhn_table());
      average.emplace(player, make_empty_kuhn_table());
   }

   return rm::MCCFR< config, KEnv, KPolicy, KPolicy, Rule >{
      std::move(env),
      std::move(root),
      std::move(current),
      std::move(average),
      /*epsilon=*/0.6,
      seed,
      std::move(rule)};
}

static size_t kuhn_seat(Player player)
{
   return static_cast< size_t >(games::kuhn::to_kuhn_player(player));
}

/// expected value of 's' for 'for_player' under uniform player actions and true
/// chance probabilities (the exact CFR "current strategy" value kernel)
static double kuhn_expectimax_value(const KEnv& env, KState s, Player for_player)
{
   if(env.is_terminal(s)) {
      return env.reward(for_player, s);
   }
   Player active = env.active_player(s);
   if(active == Player::chance) {
      double acc = 0.;
      for(const auto& outcome : env.chance_actions(s)) {
         const double prob = env.chance_probability(s, outcome);
         KState next = s;
         env.transition(next, outcome);
         acc += prob * kuhn_expectimax_value(env, next, for_player);
      }
      return acc;
   }
   const auto actions = env.actions(active, s);
   double acc = 0.;
   for(const auto& action : actions) {
      KState next = s;
      env.transition(next, action);
      acc += kuhn_expectimax_value(env, next, for_player);
   }
   return acc / static_cast< double >(actions.size());
}

/// replicate the traversal's observation folding for one edge (semantics of
/// next_infostate_and_obs_buffers_inplace): the next active player receives the
/// flushed buffer plus this edge's observations, everyone else buffers.
static void kuhn_fold_edge(
   const KEnv& env,
   std::array< KInfostate, 2 >& istates,
   std::array< std::vector< std::pair< KObs, KObs > >, 2 >& buffers,
   const KState& state,
   const auto& edge,
   const KState& next
)
{
   const auto public_obs = env.public_observation(state, edge, next);
   const Player next_active = env.active_player(next);
   for(auto player : env.players(next)) {
      if(player == Player::chance) {
         continue;
      }
      const size_t seat = kuhn_seat(player);
      if(player == next_active) {
         for(auto& obs : buffers[seat]) {
            istates[seat].update(obs.first, obs.second);
         }
         buffers[seat].clear();
         istates[seat].update(public_obs, env.private_observation(player, state, edge, next));
      } else {
         buffers[seat].emplace_back(public_obs, env.private_observation(player, state, edge, next));
      }
   }
}

/// exact counterfactual-regret increments of ONE cfr iteration under the uniform
/// random profile, grouped by the ENGINE's infostate identity:
///    R(I,a)     = sum_{h in I} pi_-i(h)         * (v(h,a) - v(h))   (vanilla CFR)
///    Ros(I,a)   = sum_{h in I} pi_i(h) pi_-i(h) * (v(h,a) - v(h))
/// ('Ros' is the quantity outcome-sampling MCCFR estimates in expectation --
///  Lanctot's unbiasedness statement: each visit contributes weighted by the
///  updater's OWN prefix reach times the counterfactual reach.)
using ExactRegretMap = std::unordered_map< KInfostate, std::unordered_map< KAction, double > >;

static void kuhn_walk_exact_regrets(
   const KEnv& env,
   const KState& state,
   Player target,
   std::array< KInfostate, 2 >& istates,
   std::array< std::vector< std::pair< KObs, KObs > >, 2 >& buffers,
   double deal_reach,
   const std::array< double, 2 >& player_reach,
   ExactRegretMap& out_vanilla,
   ExactRegretMap& out_os,
   std::unordered_map< KInfostate, double >& out_visit_prob
)
{
   if(env.is_terminal(state)) {
      return;
   }
   Player active = env.active_player(state);
   if(active == Player::chance) {
      for(const auto& outcome : env.chance_actions(state)) {
         const double prob = env.chance_probability(state, outcome);
         KState next = state;
         env.transition(next, outcome);
         auto next_istates = istates;
         auto next_buffers = buffers;
         kuhn_fold_edge(env, next_istates, next_buffers, state, outcome, next);
         kuhn_walk_exact_regrets(
            env,
            next,
            target,
            next_istates,
            next_buffers,
            deal_reach * prob,
            player_reach,
            out_vanilla,
            out_os,
            out_visit_prob
         );
      }
      return;
   }

   const auto actions = env.actions(active, state);
   const double uniform_prob = 1. / static_cast< double >(actions.size());
   double node_value = 0.;
   std::unordered_map< KAction, double > child_values{};
   for(const auto& action : actions) {
      KState next = state;
      env.transition(next, action);
      const double value = kuhn_expectimax_value(env, next, target);
      child_values.emplace(action, value);
      node_value += uniform_prob * value;
   }

   if(active == target) {
      const size_t target_seat = kuhn_seat(target);
      // counterfactual reach: everyone but the updater (chance + opponents)
      const double cf_reach = deal_reach * player_reach[0] * player_reach[1]
                              / player_reach[target_seat];
      const double own_prefix_reach = player_reach[target_seat];
      // full sampling-prefix reach of this history (chance + BOTH players' actions):
      // the probability that a trajectory visits this node
      const double prefix_reach = deal_reach * player_reach[0] * player_reach[1];
      out_visit_prob[istates[target_seat]] += prefix_reach;
      auto& per_infostate_vanilla = out_vanilla[istates[target_seat]];
      auto& per_infostate_os = out_os[istates[target_seat]];
      for(const auto& action : actions) {
         const double deviation = child_values.at(action) - node_value;
         per_infostate_vanilla[action] += cf_reach * deviation;
         per_infostate_os[action] += own_prefix_reach * cf_reach * deviation;
      }
   }

   for(const auto& action : actions) {
      KState next = state;
      env.transition(next, action);
      auto next_player_reach = player_reach;
      next_player_reach[kuhn_seat(active)] *= uniform_prob;
      auto next_istates = istates;
      auto next_buffers = buffers;
      kuhn_fold_edge(env, next_istates, next_buffers, state, action, next);
      kuhn_walk_exact_regrets(
         env,
         next,
         target,
         next_istates,
         next_buffers,
         deal_reach,
         next_player_reach,
         out_vanilla,
         out_os,
         out_visit_prob
      );
   }
}

/// vanilla CFR increments, OS-MCCFR-scaled increments (vanilla / P(visit I)),
/// and per-infostate visit probabilities of ONE iteration under uniform play
struct ExactTargets {
   ExactRegretMap vanilla;
   /// outcome-sampling MCCFR estimates counterfactual regret scaled by the
   /// reciprocal infostate visit probability (Lanctot's unbiasedness statement:
   /// E[R_hat(I,a)] = R(I,a)/pi_visit(I)); the scaling is constant per infoset
   /// and hence regret-matching invariant.
   ExactRegretMap visit_scaled;
};

static ExactTargets kuhn_exact_first_iteration_targets(Player target)
{
   KEnv env{};
   auto root = std::make_unique< KState >();
   ExactTargets out;
   std::unordered_map< KInfostate, double > visit_prob{};
   std::array< KInfostate, 2 > istates{
      KInfostate{target}, KInfostate{target == Player::alex ? Player::bob : Player::alex}};
   std::array< std::vector< std::pair< KObs, KObs > >, 2 > buffers{};
   std::array< double, 2 > player_reach{1., 1.};
   kuhn_walk_exact_regrets(
      env,
      *root,
      target,
      istates,
      buffers,
      1.,
      player_reach,
      out.vanilla,
      out.visit_scaled,
      visit_prob
   );
   for(auto& [infostate, per_action] : out.vanilla) {
      for(auto& [action, regret] : per_action) {
         out.visit_scaled.at(infostate)[action] = regret / visit_prob.at(infostate);
      }
   }
   return out;
}

/// mean/stddev of the accumulated regret increments R_1(I,a) over seeded single
/// iterations (fresh solver per seed)
struct RegretStats {
   double mean = 0.;
   double stddev = 0.;
   size_t samples = 0;
};

template < auto config, typename Rule >
static std::unordered_map< KInfostate, std::unordered_map< KAction, RegretStats > >
kuhn_first_iteration_regret_stats(size_t n_seeds, Rule rule);

template < auto config, typename Rule = rm::EpsilonOnPolicySamplingRule >
static std::unordered_map< KInfostate, std::unordered_map< KAction, RegretStats > >
kuhn_first_iteration_regret_stats(size_t n_seeds)
{
   return kuhn_first_iteration_regret_stats< config, Rule >(n_seeds, Rule{});
}

template < auto config, typename Rule >
static std::unordered_map< KInfostate, std::unordered_map< KAction, RegretStats > >
kuhn_first_iteration_regret_stats(size_t n_seeds, Rule rule)
{
   std::unordered_map< KInfostate, std::unordered_map< KAction, std::vector< double > > > samples{};
   Player updater = Player::alex;
   for(size_t seed = 0; seed < n_seeds; ++seed) {
      auto solver = make_kuhn_solver< config >(seed, rule);
      auto root_values = solver.iterate(1);
      EXPECT_EQ(root_values.size(), size_t{1});
      if(root_values.size() != 1 or root_values[0].get().empty()) {
         ADD_FAILURE() << "unexpected iterate(1) return shape";
         return {};
      }
      updater = root_values[0].get().begin()->first;
      // the CURRENT policy table carries an entry for every visited node
      for(const auto& [infostate, action_policy] : solver.policy().at(updater).table()) {
         for(const auto& [action, prob] : action_policy) {
            (void) prob;
            samples[infostate][action].emplace_back(solver.infonode(infostate).regret(action));
         }
      }
   }
   EXPECT_EQ(updater, Player::alex);
   std::unordered_map< KInfostate, std::unordered_map< KAction, RegretStats > > stats{};
   for(const auto& [infostate, per_action] : samples) {
      for(const auto& [action, values] : per_action) {
         const double mean = std::accumulate(values.begin(), values.end(), 0.)
                             / static_cast< double >(values.size());
         double sq_acc = 0.;
         for(double v : values) {
            sq_acc += (v - mean) * (v - mean);
         }
         const double stddev = std::sqrt(
            sq_acc / static_cast< double >(std::max< size_t >(values.size(), 1) - 1)
         );
         stats[infostate][action] = RegretStats{mean, stddev, values.size()};
      }
   }
   return stats;
}

///////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// goofspiel / liars dice helpers ////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

using GEnv = games::goofspiel::Environment;
using GAction = games::goofspiel::Bid;
using GInfostate = games::goofspiel::Infostate;

static auto make_empty_goofspiel_table()
{
   return factory::make_tabular_policy(
      std::unordered_map< GInfostate, HashmapActionPolicy< GAction > >{}
   );
}

using GPolicy = std::decay_t< decltype(make_empty_goofspiel_table()) >;

template < auto config, typename Rule = rm::EpsilonOnPolicySamplingRule >
static auto make_goofspiel_solver(
   const games::goofspiel::GoofspielConfig& gcfg,
   size_t seed,
   Rule rule = Rule{}
)
{
   GEnv env{gcfg};
   auto root = std::make_unique< games::goofspiel::State >(gcfg);
   auto players = env.players(*root);

   std::unordered_map< Player, GPolicy > current;
   std::unordered_map< Player, GPolicy > average;
   for(auto player : players | utils::is_actual_player_filter) {
      current.emplace(player, make_empty_goofspiel_table());
      average.emplace(player, make_empty_goofspiel_table());
   }

   return rm::MCCFR< config, GEnv, GPolicy, GPolicy, Rule >{
      std::move(env),
      std::move(root),
      std::move(current),
      std::move(average),
      /*epsilon=*/0.6,
      seed,
      std::move(rule)};
}

/// mean L2 distance between normalized current and average strategy tables over
/// their common infostate keys. CFR converges current -> average -> equilibrium,
/// so this distance must shrink as iteration count grows; unlike best-response
/// exploitability it is computable from solver-internal tables alone.
template < typename SolverA, typename SolverB >
static double goofspiel_policy_distance(const SolverA& left, const SolverB& right)
{
   double acc = 0.;
   size_t counted = 0;
   for(auto player : {Player::alex, Player::bob}) {
      const auto& ltable = left.average_policy().at(player).table();
      const auto& rtable = right.policy().at(player).table();
      for(const auto& [infostate, laction_policy] : ltable) {
         auto found = rtable.find(infostate);
         if(found == rtable.end()) {
            continue;
         }
         double lnorm = 0.;
         for(const auto& [a, v] : laction_policy) {
            (void) a;
            lnorm += v;
         }
         double rnorm = 0.;
         for(const auto& [a, v] : found->second) {
            (void) a;
            rnorm += v;
         }
         if(lnorm <= 0. or rnorm <= 0.) {
            continue;
         }
         for(const auto& [action, lvalue] : laction_policy) {
            auto rvalue = found->second.at(action);
            acc += (lvalue / lnorm - rvalue / rnorm) * (lvalue / lnorm - rvalue / rnorm);
         }
         ++counted;
      }
   }
   return counted ? std::sqrt(acc / static_cast< double >(counted)) : 0.;
}

using NormalizedSnapshot = std::
   unordered_map< Player, std::unordered_map< GInfostate, std::unordered_map< GAction, double > > >;

static NormalizedSnapshot goofspiel_snapshot_avg(const auto& solver)
{
   NormalizedSnapshot out{};
   for(auto player : {Player::alex, Player::bob}) {
      for(const auto& [infostate, action_policy] : solver.average_policy().at(player).table()) {
         double total = 0.;
         for(const auto& [action, value] : action_policy) {
            (void) action;
            total += value;
         }
         auto& row = out[player][infostate];
         for(const auto& [action, value] : action_policy) {
            row.emplace(action, total > 0. ? value / total : 0.);
         }
      }
   }
   return out;
}

static double goofspiel_snapshot_distance(const NormalizedSnapshot& a, const NormalizedSnapshot& b)
{
   double acc = 0.;
   size_t counted = 0;
   for(const auto& [player, by_infostate] : a) {
      for(const auto& [infostate, laction_policy] : by_infostate) {
         auto found = b.at(player).find(infostate);
         if(found == b.at(player).end()) {
            continue;
         }
         for(const auto& [action, lvalue] : laction_policy) {
            const double diff = lvalue - found->second.at(action);
            acc += diff * diff;
            ++counted;
         }
      }
   }
   return counted ? std::sqrt(acc / static_cast< double >(counted)) : 0.;
}

}  // namespace

///////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// unbiasedness properties //////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SamplingRulesUnbiasedness, ExactEnumerationRegretsSumToOneHundredPercentConsistent)
{
   // internal tripwire for the enumerator itself: sum_a R(I,a) vanishes identically
   for(auto target : {Player::alex, Player::bob}) {
      const auto targets = kuhn_exact_first_iteration_targets(target);
      const auto& exact = targets.vanilla;
      ASSERT_FALSE(exact.empty());
      ASSERT_FALSE(targets.visit_scaled.empty());
      for(const auto& [infostate, per_action] : exact) {
         double total = 0.;
         for(const auto& [action, regret] : per_action) {
            (void) action;
            total += regret;
         }
         EXPECT_NEAR(total, 0., 1e-12) << "target seat " << common::to_string(target);
      }
   }
}

namespace {

void expect_means_match_exact(
   const ExactRegretMap& exact,
   const std::unordered_map< KInfostate, std::unordered_map< KAction, RegretStats > >& means
)
{
   SCOPED_TRACE("");
   for(const auto& [infostate, per_action] : exact) {
      ASSERT_TRUE(means.contains(infostate)) << "infoset never visited by engine";
      for(const auto& [action, exact_regret] : per_action) {
         ASSERT_TRUE(means.at(infostate).contains(action));
         const auto& stats = means.at(infostate).at(action);
         // 5 standard errors of the mean plus float slack; OS estimators are
         // heavy-tailed so N is large (30k seeded single iterations)
         const double tol = 5. * stats.stddev / std::sqrt(static_cast< double >(stats.samples))
                            + 1e-9;
         EXPECT_NEAR(stats.mean, exact_regret, tol)
            << " (action: " << common::to_string(action)
            << ", infoset history length: " << infostate.history().size() << ")";
      }
   }
}

}  // namespace

// Outcome-sampling MCCFR estimates the OWN-REACH-WEIGHTED counterfactual
// regret in expectation (Lanctot's unbiasedness statement); all four sampling
// variants below must reproduce that identical expectation.
TEST(SamplingRulesUnbiasedness, VanillaEpsilonOnPolicyMeanRegretsMatchExactValues)
{
   constexpr size_t kSeeds = 30000;
   const auto targets = kuhn_exact_first_iteration_targets(Player::alex);
   const auto means = kuhn_first_iteration_regret_stats< k_os_epsilon_config >(kSeeds);
   expect_means_match_exact(targets.visit_scaled, means);
}

TEST(SamplingRulesUnbiasedness, InjectedDefaultEpsilonRuleMeanRegretsMatchExactValues)
{
   constexpr size_t kSeeds = 30000;
   const auto targets = kuhn_exact_first_iteration_targets(Player::alex);
   const auto means = kuhn_first_iteration_regret_stats< k_os_custom_rule_config >(
      kSeeds, rm::EpsilonOnPolicySamplingRule{}
   );
   expect_means_match_exact(targets.visit_scaled, means);
}

TEST(SamplingRulesUnbiasedness, PublicChanceSamplingFallbackMeanRegretsMatchExactValues)
{
   // kuhn provides no public_chance_event classification -> documented fallback
   // treats ALL chance events as public -> PCS degenerates to vanilla sampling
   constexpr size_t kSeeds = 30000;
   const auto targets = kuhn_exact_first_iteration_targets(Player::alex);
   const auto means = kuhn_first_iteration_regret_stats< k_os_epsilon_config >(
      kSeeds, rm::PublicChanceSamplingRule{}
   );
   expect_means_match_exact(targets.visit_scaled, means);
}

TEST(SamplingRulesUnbiasedness, AverageStrategySamplingMeanRegretsMatchExactValues)
{
   // iteration-1 cold start: the accumulated average table is still degenerate so
   // the rule falls back to uniform sampling; any positive sampling measure yields
   // the same own-reach-weighted counterfactual-regret expectation.
   constexpr size_t kSeeds = 30000;
   const auto targets = kuhn_exact_first_iteration_targets(Player::alex);
   const auto means = kuhn_first_iteration_regret_stats< k_os_custom_rule_config >(
      kSeeds, rm::AverageStrategySamplingRule{}
   );
   expect_means_match_exact(targets.visit_scaled, means);
}

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// regression: draw-for-draw ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

namespace {

template < typename SolverA, typename SolverB >
static void expect_bit_identical_tables(const SolverA& left, const SolverB& right)
{
   for(auto player : {Player::alex, Player::bob}) {
      EXPECT_EQ(left.average_policy().at(player).table(), right.average_policy().at(player).table())
         << "average policy diverged for " << common::to_string(player);
      EXPECT_EQ(left.policy().at(player).table(), right.policy().at(player).table())
         << "current policy diverged for " << common::to_string(player);
   }
}

}  // namespace

TEST(SamplingRulesRegression, InjectedDefaultEpsilonRuleIsDrawForDrawIdenticalToBuiltIn)
{
   constexpr size_t kSeed = 7;
   constexpr size_t kIters = 300;
   auto vanilla = make_kuhn_solver< k_os_epsilon_config >(kSeed);
   auto injected = make_kuhn_solver< k_os_custom_rule_config >(
      kSeed, rm::EpsilonOnPolicySamplingRule{}
   );

   const auto vanilla_values = vanilla.iterate(kIters);
   const auto injected_values = injected.iterate(kIters);
   ASSERT_EQ(vanilla_values.size(), injected_values.size());
   for(auto [iteration_idx, vanilla_map, injected_map] :
       std::views::zip(std::views::iota(size_t{0}), vanilla_values, injected_values)) {
      ASSERT_EQ(vanilla_map.get().size(), injected_map.get().size());
      for(const auto& [player, value] : vanilla_map.get()) {
         EXPECT_DOUBLE_EQ(value, injected_map.get().at(player))
            << "root estimate diverged at iteration " << iteration_idx;
      }
   }
   expect_bit_identical_tables(vanilla, injected);
}

TEST(SamplingRulesRegression, PCSFallbackTrajectoriesAreIdenticalToVanillaOutcomeSampling)
{
   constexpr size_t kSeed = 11;
   constexpr size_t kIters = 300;
   auto vanilla = make_kuhn_solver< k_os_epsilon_config >(kSeed);
   auto pcs = make_kuhn_solver< k_os_epsilon_config >(kSeed, rm::PublicChanceSamplingRule{});

   const auto vanilla_values = vanilla.iterate(kIters);
   const auto pcs_values = pcs.iterate(kIters);
   ASSERT_EQ(vanilla_values.size(), pcs_values.size());
   for(const auto& [vanilla_map, pcs_map] : std::views::zip(vanilla_values, pcs_values)) {
      ASSERT_EQ(vanilla_map.get().size(), pcs_map.get().size());
      for(const auto& [player, value] : vanilla_map.get()) {
         EXPECT_DOUBLE_EQ(value, pcs_map.get().at(player));
      }
   }
   expect_bit_identical_tables(vanilla, pcs);
}

///////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////// PCS on goofspiel /////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

TEST(PublicChanceSampling, GoofspielK4RevealConvergesWithDecreasingExploitabilityAndVarianceReport)
{
   // CONVERGENCE METRIC NOTE (documented deviation from the original plan):
   // best-response exploitability is currently NOT computable for OS-solver
   // tables on goofspiel -- the best-response tree walk queries player policies
   // at infostates whose observation context it folds differently than the CFR
   // traversals do (observed: it queries alex at an EMPTY infostate before the
   // first prize reveal, which no traversal ever registers; even a full-tree
   // chance-sampling solver's complete tables throw). This pre-exists the B7
   // work and lives in best_response.hpp/forest, outside this task's scope.
   // Convergence is therefore asserted via the current-vs-average strategy
   // distance (a standard CFR convergence proxy) plus the exact per-trajectory
   // variance comparison against vanilla OS.
   const games::goofspiel::GoofspielConfig gcfg{.deck_size = 4, .imp_info = false};

   auto vanilla = make_goofspiel_solver< k_os_epsilon_config >(gcfg, /*seed=*/42);
   auto pcs = make_goofspiel_solver< k_os_epsilon_config >(
      gcfg, /*seed=*/42, rm::PublicChanceSamplingRule{}
   );

   constexpr size_t kIters = 500;
   std::map< size_t, NormalizedSnapshot > vanilla_snapshots{};
   std::map< size_t, NormalizedSnapshot > pcs_snapshots{};

   std::vector< double > vanilla_estimates{};
   std::vector< double > pcs_estimates{};
   vanilla_estimates.reserve(kIters);
   pcs_estimates.reserve(kIters);
   // alternating updates cycle the updating player; each returned root-value
   // map carries exactly the current updater's estimate
   for(size_t iteration = 1; iteration <= kIters; ++iteration) {
      auto v = vanilla.iterate(1);
      auto p = pcs.iterate(1);
      vanilla_estimates.emplace_back(v[0].get().begin()->second);
      pcs_estimates.emplace_back(p[0].get().begin()->second);
      if(iteration == 100 or iteration == 200 or iteration == 400 or iteration == 500) {
         vanilla_snapshots.emplace(iteration, goofspiel_snapshot_avg(vanilla));
         pcs_snapshots.emplace(iteration, goofspiel_snapshot_avg(pcs));
      }
   }
   // average-strategy drift between consecutive snapshot pairs shrinks as the
   // accumulated average strategy stabilizes (convergence signal computable
   // from solver tables alone; see the exploitability note above)
   const double dist_vanilla_early = goofspiel_snapshot_distance(
      vanilla_snapshots.at(100), vanilla_snapshots.at(200)
   );
   const double dist_vanilla_late = goofspiel_snapshot_distance(
      vanilla_snapshots.at(400), vanilla_snapshots.at(500)
   );
   const double dist_pcs_early = goofspiel_snapshot_distance(
      pcs_snapshots.at(100), pcs_snapshots.at(200)
   );
   const double dist_pcs_late = goofspiel_snapshot_distance(
      pcs_snapshots.at(400), pcs_snapshots.at(500)
   );

   std::cout << "[goofspiel k=4 reveal] average-strategy drift iters 100->200 | vanilla OS: "
             << dist_vanilla_early << " PCS: " << dist_pcs_early << "\n";
   std::cout << "[goofspiel k=4 reveal] average-strategy drift iters 400->500 | vanilla OS: "
             << dist_vanilla_late << " PCS: " << dist_pcs_late << "\n";

   // goofspiel's chance events are ALL public (prize reveals observed by both
   // players, resolve confirm publishes the result) => PCS must track vanilla OS
   EXPECT_DOUBLE_EQ(dist_pcs_late, dist_vanilla_late);
   EXPECT_LT(dist_pcs_late, dist_pcs_early);

   // variance report of the per-iteration root-value estimates over the second
   // half of the run (identical streams by the all-public fallback degeneracy)
   const auto variance_of = [](const std::vector< double >& xs) {
      const size_t offset = xs.size() / 2;
      const double n = static_cast< double >(xs.size() - offset);
      const double mean = std::accumulate(xs.begin() + static_cast< long >(offset), xs.end(), 0.)
                          / n;
      double acc = 0.;
      for(auto it = xs.begin() + static_cast< long >(offset); it != xs.end(); ++it) {
         acc += (*it - mean) * (*it - mean);
      }
      return acc / (n - 1.);
   };
   const double vanilla_var = variance_of(vanilla_estimates);
   const double pcs_var = variance_of(pcs_estimates);
   std::cout << "[goofspiel k=4 reveal] per-iteration root-estimate variance "
             << "(iters 251..500) | vanilla OS: " << vanilla_var << " PCS: " << pcs_var
             << " | ratio PCS/vanilla: " << (pcs_var / vanilla_var) << "\n";
   EXPECT_DOUBLE_EQ(pcs_var, vanilla_var);
   EXPECT_TRUE(std::isfinite(vanilla_var) and vanilla_var >= 0.);
}

///////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// ASS on kuhn 2-player /////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

TEST(AverageStrategySampling, KuhnTwoPlayerConvergesBelowExploitabilityThreshold)
{
   auto solver = make_kuhn_solver< k_os_custom_rule_config >(
      /*seed=*/3, rm::AverageStrategySamplingRule{}
   );

   constexpr double kThreshold = 3e-3;
   constexpr size_t kMaxIters = 400000;
   constexpr size_t kKuhnInfostatesPerPlayer = 6;
   KEnv env{};
   double expl = std::numeric_limits< double >::max();
   size_t converged_at = kMaxIters;
   for(size_t iteration = 1; iteration <= kMaxIters; ++iteration) {
      solver.iterate(1);
      // best-response exploitability requires complete policy tables
      const bool tables_complete = solver.average_policy().at(Player::alex).table().size()
                                      == kKuhnInfostatesPerPlayer
                                   and solver.average_policy().at(Player::bob).table().size()
                                          == kKuhnInfostatesPerPlayer;
      if(tables_complete and iteration % 50000 == 0) {
         const auto& avg_policies = solver.average_policy();
         expl = exploitability(
            env,
            KState{},
            player_hashmap< std::decay_t< decltype(avg_policies.at(Player::alex)) > >{
               std::pair{Player::alex, normalize_state_policy(avg_policies.at(Player::alex))},
               std::pair{Player::bob, normalize_state_policy(avg_policies.at(Player::bob))}}
         );
         if(expl <= kThreshold) {
            converged_at = iteration;
            break;
         }
      }
   }
   std::cout << "[kuhn 2p ASS] exploitability=" << expl << " at iteration " << converged_at << "\n";
   EXPECT_LE(expl, kThreshold);
}

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// liars dice private-roll smoke ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

TEST(PublicChanceSampling, LiarsDicePrivateRollsRunSmoke)
{
   // liars dice classifies ALL rolls as PRIVATE chance events: PCS resolves them
   // deterministically to the first legal face without consuming RNG. This test
   // exercises that code path mechanically (single-trajectory caveat documented
   // in sampling_rules.hpp -- no equilibrium claim here).
   const games::liars_dice::DiceConfig dcfg{/*n_players=*/2, /*dice_per_player=*/1, /*n_faces=*/4};
   using LEnv = games::liars_dice::Environment;
   using LAction = games::liars_dice::Action;
   using LInfostate = games::liars_dice::Infostate;

   LEnv env{dcfg};
   auto root = std::make_unique< games::liars_dice::State >(dcfg);
   auto players = env.players(*root);

   auto mk_table = [&] {
      return factory::make_tabular_policy(
         std::unordered_map< LInfostate, HashmapActionPolicy< LAction > >{}
      );
   };
   using LPolicy = std::decay_t< decltype(mk_table()) >;
   std::unordered_map< Player, LPolicy > current;
   std::unordered_map< Player, LPolicy > average;
   for(auto player : players | utils::is_actual_player_filter) {
      current.emplace(player, mk_table());
      average.emplace(player, mk_table());
   }

   rm::MCCFR< k_os_epsilon_config, LEnv, LPolicy, LPolicy, rm::PublicChanceSamplingRule > solver{
      std::move(env),
      std::move(root),
      std::move(current),
      std::move(average),
      /*epsilon=*/0.6,
      /*seed=*/17,
      rm::PublicChanceSamplingRule{}};

   auto values = solver.iterate(30);
   ASSERT_EQ(values.size(), size_t{30});
   for(const auto& per_iter : values) {
      for(const auto& [player, value] : per_iter.get()) {
         EXPECT_TRUE(std::isfinite(value)) << "player " << static_cast< int >(player);
      }
   }
}
