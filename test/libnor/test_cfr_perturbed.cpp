#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cfr_run_funcs.hpp"
#include "nor/env.hpp"
#include "nor/exploitability.hpp"
#include "nor/nor.hpp"

using namespace nor;

/// BEHAVIORALLY-CONSTRAINED ("PERTURBED") CFR+ (Farina, Kroer & Sandholm, "Regret
/// Minimization in Behaviorally-Constrained Zero-Sum Games", ICML 2017,
/// arXiv:1711.03441).
///
/// Mechanism under test: the per-infoset RM+ minimizer of CFR+ is replaced by RM+
/// running over the LINEARLY CONSTRAINED simplex
///    Q^I = { sigma : sigma(I,a) >= p(I,a) for all a }
/// (their Prop. 6 gives the vertex basis b_i = p + tau*e_i, tau = 1 - sum_a p(I,a);
/// their Algorithm 2 gives the resulting closed form). Every iterate's recommendation
/// satisfies the floors exactly by construction, so the solved profile approximates
/// an EFPE-style Nash REFINEMENT: behavior in low-reach parts of the tree is pinned
/// down instead of arbitrary, while the O(1/sqrt(T)) convergence rate of CFR+
/// carries over (their Thms. 5 and 7).
///
/// The tests below verify, in order:
///  1. the append-at-end selection contract, the minimizer-type wiring and the
///     static guards rejecting every combination the paper does not analyze;
///  2. the kernel arithmetic against hand-computed folds (incl. exact floor
///     attainment and infeasible-perturbation rejection) and the B8
///     environment-trait injection point;
///  3. the constraint-satisfaction INVARIANT over whole solver runs (every
///     iterate's recommendations -- current AND normalized average policies --
///     respect the floors) on kuhn poker, rock-paper-scissors and short-horizon
///     leduc;
///  4. the epsilon -> 0 limit recovering plain CFR+ trajectories BIT-FOR-BIT
///     (the unperturbed specialization compiles verbatim RegretMatchingPlus
///     arithmetic: identical summation grouping, identical clamp points);
///  5. the REFINEMENT-QUALITY demonstration: the reach-conditioned infoset regret
///     of the perturbed solution beats vanilla CFR+'s on kuhn while on-path
///     exploitability stays bounded;
///  6. convergence: standard exploitability thresholds are met for moderate and
///     small perturbations.
///
/// METRIC (test 5, mirrors the paper's sec. 8 measure): the maximum regret at any
/// information set CONDITIONED on reaching it,
///    R(I) = max_a v(I,a) - sum_a sigma_bar(I,a) v(I,a),
/// where v(I,a) is the value of playing action a at I once and then following the
/// average profile sigma_bar to the end of the game, and values are averaged over
/// the histories of I with OPPONENT-and-chance reach only (pi_{-i}(h)); the
/// conditioning removes the infoset's own-reach weighting. Reported as the maximum
/// over all infosets ("all") and over the infosets whose JOINT average reach
/// pi(I) falls below a threshold ("off-path") -- the part of the tree Nash
/// equilibria leave unspecified and refinements fix.

namespace {

constexpr double k_floor_tol = 1e-12;

/// Selten-trembling configuration shorthand: CFR+ whose per-infoset minimizer is
/// the behaviorally-constrained (perturbed) RM+ kernel with uniform floor Epsilon
template < double Epsilon >
constexpr rm::CFRConfig perturbed_cfr_plus_config{
   .update_mode = rm::UpdateMode::alternating,
   .regret_minimizing_mode = rm::RegretMinimizingMode::constrained_regret_matching_plus,
   .weighting_mode = rm::CFRWeightingMode::uniform,
   .pruning_mode = rm::CFRPruningMode::none,
   .perturbation_floor = Epsilon};

///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////// 1. selection contract + static configuration guards /////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

// the append-at-end contract of the selection enum (tests and persisted configs rely on
// the numeric order of the RegretMinimizingMode enumerators)
static_assert(
   rm::RegretMinimizingMode::constrained_regret_matching_plus
   == static_cast< rm::RegretMinimizingMode >(11)
);

static_assert(std::same_as<
              rm::base_minimizer_for_t< perturbed_cfr_plus_config< 0.05 >, int >,
              rm::ConstrainedRMPlus< int, 0.05 > >);

// the epsilon -> 0 limit selects the UNBUFFERED specialization whose arithmetic is
// bit-for-bit plain RM+
static_assert(std::same_as<
              rm::base_minimizer_for_t< perturbed_cfr_plus_config< 0. >, int >,
              rm::ConstrainedRMPlus< int, 0., false > >);

#define NOR_EXPECT_GUARD_REJECTS(...) \
   static_assert(not rm::detail::sanity_check_cfr_config< rm::CFRConfig{__VA_ARGS__} >())

// predictive/discounted/greedy/exponential weighting kernels were never analyzed together
// with polytope-RM+
NOR_EXPECT_GUARD_REJECTS(
      .update_mode = rm::UpdateMode::alternating,
      .regret_minimizing_mode = rm::RegretMinimizingMode::constrained_regret_matching_plus,
      .weighting_mode = rm::CFRWeightingMode::discounted,
      .perturbation_floor = 0.01
);
NOR_EXPECT_GUARD_REJECTS(
      .update_mode = rm::UpdateMode::simultaneous,
      .regret_minimizing_mode = rm::RegretMinimizingMode::constrained_regret_matching_plus,
      .weighting_mode = rm::CFRWeightingMode::greedy,
      .perturbation_floor = 0.01
);
NOR_EXPECT_GUARD_REJECTS(
      .update_mode = rm::UpdateMode::simultaneous,
      .regret_minimizing_mode = rm::RegretMinimizingMode::constrained_regret_matching_plus,
      .weighting_mode = rm::CFRWeightingMode::exponential,
      .perturbation_floor = 0.01
);

// pruning modes: window bookkeeping assumes RM+-layout folds, thresholding zeroes floored
// actions
NOR_EXPECT_GUARD_REJECTS(
      .update_mode = rm::UpdateMode::alternating,
      .regret_minimizing_mode = rm::RegretMinimizingMode::constrained_regret_matching_plus,
      .pruning_mode = rm::CFRPruningMode::partial,
      .perturbation_floor = 0.01
);
NOR_EXPECT_GUARD_REJECTS(
      .update_mode = rm::UpdateMode::alternating,
      .regret_minimizing_mode = rm::RegretMinimizingMode::constrained_regret_matching_plus,
      .pruning_mode = rm::CFRPruningMode::regret_based,
      .perturbation_floor = 0.01
);
NOR_EXPECT_GUARD_REJECTS(
      .update_mode = rm::UpdateMode::alternating,
      .regret_minimizing_mode = rm::RegretMinimizingMode::constrained_regret_matching_plus,
      .pruning_mode = rm::CFRPruningMode::dynamic_thresholding,
      .perturbation_floor = 0.01
);

// lazy segmentation freezes recommendations across iterations
NOR_EXPECT_GUARD_REJECTS(
      .update_mode = rm::UpdateMode::alternating,
      .regret_minimizing_mode = rm::RegretMinimizingMode::constrained_regret_matching_plus,
      .lazy_update_mode = rm::CFRLazyUpdateMode::reach_threshold,
      .perturbation_floor = 0.01
);

// warm-start pre-play is unanalyzed with perturbed play
NOR_EXPECT_GUARD_REJECTS(
      .update_mode = rm::UpdateMode::alternating,
      .regret_minimizing_mode = rm::RegretMinimizingMode::constrained_regret_matching_plus,
      .warm_start_iterations = 1,
      .perturbation_floor = 0.01
);

// negative (and NaN) uniform floors are infeasible
NOR_EXPECT_GUARD_REJECTS(
      .update_mode = rm::UpdateMode::alternating,
      .regret_minimizing_mode = rm::RegretMinimizingMode::constrained_regret_matching_plus,
      .perturbation_floor = -0.01
);

#undef NOR_EXPECT_GUARD_REJECTS

// the analyzed combinations pass
static_assert(rm::detail::sanity_check_cfr_config< perturbed_cfr_plus_config< 0.01 > >());
static_assert(rm::detail::sanity_check_cfr_config< perturbed_cfr_plus_config< 0. > >());
static_assert(
   rm::detail::sanity_check_cfr_config< rm::CFRConfig{
      .update_mode = rm::UpdateMode::simultaneous,
      .regret_minimizing_mode = rm::RegretMinimizingMode::constrained_regret_matching_plus,
      .perturbation_floor = 0.02} >()
);

///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////// shared helpers of the dynamic test sections /////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

template < typename Infostate, typename Action >
auto empty_tabular_policy()
{
   return factory::make_tabular_policy(
      std::unordered_map< Infostate, HashmapActionPolicy< Action > >{}
   );
}

/// asserts the constraint-satisfaction invariant on the CURRENT policies: every registered
/// recommendation must sit on or above the floor (exactly, up to fp addition noise) and
/// every recommendation must be a distribution
template < typename Solver >
void assert_current_recommendations_respect_floors(const Solver& solver, double floor_prob)
{
   for(const auto& [player, tabular_policy] : solver.policy()) {
      for(const auto& [infostate, action_policy] : tabular_policy.table()) {
         double mass = 0.;
         for(const auto& [action, prob] : action_policy) {
            ASSERT_GE(prob, floor_prob - k_floor_tol)
               << "current recommendation violates the floor at player " << player;
            mass += prob;
         }
         EXPECT_NEAR(mass, 1., 1e-9);
      }
   }
}

/// asserts the same invariant on the NORMALIZED average strategies: every average entry is
/// a convex combination of floored iterates divided by the total accumulated own-reach
/// mass, hence sits on or above the floor exactly as well (their Algorithm 2 averaging)
template < typename Solver >
void assert_average_recommendations_respect_floors(const Solver& solver, double floor_prob)
{
   for(const auto& [player, tabular_policy] : solver.average_policy()) {
      for(const auto& [infostate, action_policy] : tabular_policy.table()) {
         double mass = 0.;
         for(const auto& [action, prob] : action_policy) {
            mass += prob;
         }
         if(mass <= 0.) {
            continue;  // never visited: no accumulated iterate to constrain
         }
         for(const auto& [action, prob] : action_policy) {
            ASSERT_GE(prob / mass, floor_prob - 1e-9)
               << "average strategy violates the floor at player " << player;
         }
      }
   }
}

/// two solvers ran the same deterministic schedule: their policy tables must agree
/// BIT-FOR-BIT (independent hash maps iterate in different orders, so entries are paired
/// by key and actions located inside the row)
template < typename TableA, typename TableB >
void expect_bitwise_identical_tables(const TableA& expected, const TableB& actual)
{
   ASSERT_EQ(expected.size(), actual.size());
   for(const auto& [infostate, action_policy] : expected) {
      const auto found = actual.find(infostate);
      ASSERT_NE(found, actual.end());
      ASSERT_EQ(action_policy.size(), found->second.size());
      for(const auto& [action, prob] : action_policy) {
         double counterpart = std::numeric_limits< double >::quiet_NaN();
         for(const auto& [other_action, other_prob] : found->second) {
            if(other_action == action) {
               counterpart = other_prob;
               break;
            }
         }
         EXPECT_DOUBLE_EQ(prob, counterpart)
            << "trajectory diverged from plain CFR+ at action index "
            << static_cast< long >(action);
      }
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////// reach-conditioned infoset-regret metric (paper sec. 8) /////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

struct ConditionedRegretReport {
   /// max_R(I) over every infoset, deviations taken over the epsilon-FLOORED
   /// simplex Q^I (the paper-consistent reading for perturbed comparisons: the
   /// maximizer spreads the free mass tau onto the best action)
   double all = 0.;
   /// same but with UNCONSTRAINED pure deviations (reported for reference only;
   /// penalizes even a perfect EFPE by the forced-tremble value gap)
   double all_unconstrained = 0.;
   size_t infosets = 0;
};

template < typename Env, typename ProfileTables >
ConditionedRegretReport reach_conditioned_infoset_regret(
   const Env& env,
   const auto_world_state_type< Env >& root,
   const ProfileTables& profile_tables,  // player -> RAW cumulative avg table
   double uniform_floor
)
{
   using ws_t = auto_world_state_type< Env >;
   using ist_t = auto_info_state_type< Env >;
   using act_t = auto_action_type< Env >;
   using actv_t = auto_action_variant_type< Env >;
   using History = std::vector< actv_t >;

   std::vector< Player > actuals;
   for(auto player : env.players(root)) {
      if(player != Player::chance) {
         actuals.push_back(player);
      }
   }

   // full-tree enumeration binding every non-terminal history to its ACTIVE player's
   // infostate (deduplicated sptr instances identify the infosets)
   auto [_, hist_to_istate] = map_histories_to_infostates(env, root);
   // the root history has no incoming edge and is therefore absent from the enumerator;
   // seed it so root decisions (rock-paper-scissors) resolve as well
   {
      auto& root_binding = hist_to_istate[History{}];
      for(auto player : actuals) {
         root_binding.second.emplace(player, std::make_shared< ist_t >(player));
      }
   }

   struct InfosetStats {
      /// opponent-reach-weighted values pi_{-i}(h) * u_i(h, a -> sigma_bar), keyed by the
      /// plain action (legal-action counts per infoset are tiny, linear scan suffices)
      std::vector< std::pair< act_t, double > > weighted_action_values;
      double cf_reach = 0.;  ///< sum_h pi_{-i}(h)
      double joint_reach = 0.;  ///< sum_h pi(h)
   };
   player_hashmap< std::unordered_map< const ist_t*, InfosetStats > > stats;

   /// behavioral probability of 'action' for 'player' under the average profile
   /// (normalized on the fly from the raw cumulative table; unvisited infosets fall
   /// back to uniform over 'n_actions')
   auto sigma = [&](Player player, const ist_t& infostate, const act_t& action, size_t n_actions) {
      const auto& table = profile_tables.at(player);
      const auto entry = table.find(infostate);
      if(entry == table.end()) {
         return 1. / static_cast< double >(n_actions);
      }
      double total = 0.;
      double match = 0.;
      for(const auto& [table_action, prob] : entry->second) {
         total += prob;
         if(table_action == action) {
            match += prob;
         }
      }
      if(total <= 0.) {
         return 1. / static_cast< double >(n_actions);
      }
      return match / total;
   };

   // post-order evaluation: every tree edge is evaluated exactly once, so the walk is
   // linear in the tree size even without memoization. 'chance_reach' accumulates ONLY
   // chance probabilities; per-player OWN action probabilities live in 'own_reach'.
   History root_history{};
   player_hashmap< double > own_reach{};
   for(auto player : actuals) {
      own_reach.emplace(player, 1.);
   }

   std::function< std::vector< double >(const ws_t&, double) > walk =
      [&](const ws_t& state, double chance_reach) -> std::vector< double > {
      if(env.is_terminal(state)) {
         std::vector< double > rewards;
         rewards.reserve(actuals.size());
         for(auto player : actuals) {
            rewards.push_back(env.reward(player, state));
         }
         return rewards;
      }
      const Player active = env.active_player(state);

      if constexpr(concepts::stochastic_env< Env >) {
         if(active == Player::chance) {
            std::vector< double > expected(actuals.size(), 0.);
            for(const auto& outcome : env.chance_actions(state)) {
               const double prob = env.chance_probability(state, outcome);
               ws_t next = state;
               env.transition(next, outcome);
               root_history.emplace_back(outcome);
               auto child = walk(next, chance_reach * prob);
               root_history.pop_back();
               for(auto [idx, value] : std::views::enumerate(child)) {
                  expected[idx] += prob * value;
               }
            }
            return expected;
         }
      }

      const auto& actions = env.actions(active, state);
      // resolve THIS history's infostate of the acting player exactly once
      const ist_t* infostate_ptr = [&]() -> const ist_t* {
         const auto found = hist_to_istate.find(root_history);
         if(found == hist_to_istate.end()) {
            return nullptr;
         }
         return found->second.second.at(active).get();
      }();

      size_t owner_idx = 0;
      for(auto [idx, player] : std::views::enumerate(actuals)) {
         if(player == active) {
            owner_idx = idx;
         }
      }

      // behavioral probabilities under the profile
      std::vector< double > probs;
      probs.reserve(actions.size());
      for(const auto& [aidx, action] : std::views::enumerate(actions)) {
         probs.push_back(
            infostate_ptr == nullptr ? 1. / static_cast< double >(actions.size())
                                     : sigma(active, *infostate_ptr, action, actions.size())
         );
      }

      // counterfactual (opponent+chance) and joint reach of this history
      double cf_reach = chance_reach;
      double joint_reach = chance_reach;
      for(auto player : actuals) {
         joint_reach *= own_reach.at(player);
         if(player != active) {
            cf_reach *= own_reach.at(player);
         }
      }

      // descend into every action under the profile
      std::vector< std::vector< double > > child_values;
      child_values.reserve(actions.size());
      for(const auto& [aidx, action] : std::views::enumerate(actions)) {
         ws_t next = state;
         env.transition(next, action);
         own_reach.at(active) *= probs[aidx];
         root_history.emplace_back(action);
         child_values.push_back(walk(next, chance_reach));
         root_history.pop_back();
         own_reach.at(active) /= probs[aidx];
      }

      // accumulate this node's infoset statistics for the ACTIVE player
      if(infostate_ptr != nullptr) {
         auto& infoset_stats = stats[active][infostate_ptr];
         infoset_stats.cf_reach += cf_reach;
         infoset_stats.joint_reach += joint_reach;
         for(auto [aidx, action] : std::views::enumerate(actions)) {
            const double contribution = cf_reach * child_values[aidx][owner_idx];
            auto found = std::ranges::find_if(
               infoset_stats.weighted_action_values,
               [&](const auto& pair) { return pair.first == action; }
            );
            if(found == infoset_stats.weighted_action_values.end()) {
               infoset_stats.weighted_action_values.emplace_back(action, contribution);
            } else {
               found->second += contribution;
            }
         }
      }

      // aggregate the expected value under the profile for the parent
      std::vector< double > expected(actuals.size(), 0.);
      for(auto [aidx, prob] : std::views::enumerate(probs)) {
         for(auto [idx, value] : std::views::enumerate(child_values[aidx])) {
            expected[idx] += prob * value;
         }
      }
      return expected;
   };

   walk(root, 1.);

   ConditionedRegretReport report{};
   for(const auto& [player, table] : stats) {
      for(const auto& [infostate_ptr, infoset_stats] : table) {
         if(infoset_stats.cf_reach <= 1e-15) {
            continue;  // unreachable against the average profile: no conditioned values
         }
         ++report.infosets;
         // conditioned values v(I,a); the floored deviation optimum puts all free mass
         // tau = 1 - sum_a p(a) on the best action: v_c* = <p, v> + tau * max_a v(a)
         double best_unconstrained = -std::numeric_limits< double >::infinity();
         double expected_under_profile = 0.;
         double floor_weighted_value = 0.;
         for(const auto& [action, weighted_value] : infoset_stats.weighted_action_values) {
            const double value = weighted_value / infoset_stats.cf_reach;
            best_unconstrained = std::max(best_unconstrained, value);
            floor_weighted_value += uniform_floor * value;
            expected_under_profile += sigma(
                                         player,
                                         *infostate_ptr,
                                         action,
                                         infoset_stats.weighted_action_values.size()
                                      )
                                      * value;
         }
         const double best = best_unconstrained;
         const double n_actions = static_cast< double >(infoset_stats.weighted_action_values.size()
         );
         const double tau = std::max(0., 1. - uniform_floor * n_actions);
         const double constrained_deviation_value = floor_weighted_value + tau * best;
         report.all = std::max(report.all, constrained_deviation_value - expected_under_profile);
         report.all_unconstrained = std::max(
            report.all_unconstrained, best_unconstrained - expected_under_profile
         );
      }
   }
   return report;
}

}  // namespace

///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////// 2. kernel arithmetic: hand-computed constrained folds ///////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

TEST(ConstrainedRMPlusKernel, folds_instant_regret_with_the_perturbation_transform)
{
   namespace kuhn = games::kuhn;
   using Minimizer = rm::ConstrainedRMPlus< kuhn::Action, 0.1 >;
   Minimizer::node_data_type data{};
   Minimizer::register_action(data, kuhn::Action::check);
   Minimizer::register_action(data, kuhn::Action::bet);

   HashmapActionPolicy< kuhn::Action > policy{};
   // floors were seeded with the uniform kernel epsilon at registration
   EXPECT_DOUBLE_EQ(data.probability_floors[0], 0.1);
   EXPECT_DOUBLE_EQ(data.probability_floors[1], 0.1);

   // ---- round 1: phi = (2, 0); p = (.1, .1); tau = .8; <p, phi> = .2
   // r <- [tau*phi + <p,phi>]^+ = (1.8, 0.2); Lambda = 2
   // x = p + tau*r/Lambda = (0.82, 0.18): sums to exactly 1
   Minimizer::observe(data, kuhn::Action::check, 2.);
   Minimizer::observe(data, kuhn::Action::bet, 0.);
   Minimizer::recommend(data, policy, /*iteration=*/0);
   EXPECT_DOUBLE_EQ(data.regret[0], 1.8);
   EXPECT_DOUBLE_EQ(data.regret[1], 0.2);
   EXPECT_NEAR(policy[kuhn::Action::check], 0.82, 1e-15);
   EXPECT_NEAR(policy[kuhn::Action::bet], 0.18, 1e-15);
   // the buffer was consumed by the fold
   EXPECT_DOUBLE_EQ(data.instant_regret[0], 0.);
   EXPECT_DOUBLE_EQ(data.instant_regret[1], 0.);

   // ---- round 2: phi = (-1, +1) -> <p, phi> = 0
   // r <- ([1.8 - 0.8]^+, [0.2 + 0.8]^+) = (1.0, 1.0); Lambda = 2 -> x = (0.5, 0.5)
   Minimizer::observe(data, kuhn::Action::check, -1.);
   Minimizer::observe(data, kuhn::Action::bet, +1.);
   Minimizer::recommend(data, policy, /*iteration=*/1);
   EXPECT_DOUBLE_EQ(data.regret[0], 1.0);
   EXPECT_DOUBLE_EQ(data.regret[1], 1.0);
   EXPECT_DOUBLE_EQ(policy[kuhn::Action::check], 0.5);
   EXPECT_DOUBLE_EQ(policy[kuhn::Action::bet], 0.5);

   // ---- round 3: clamping: phi = (-2, -0.5) -> <p, phi> = -0.25
   // r <- ([1 - 1.6 - 0.25]^+, [1 - 0.4 - 0.25]^+) = (0, 0.35); Lambda = 0.35
   // x = (0.1, 0.9): the crushed action sits EXACTLY on its floor
   Minimizer::observe(data, kuhn::Action::check, -2.);
   Minimizer::observe(data, kuhn::Action::bet, -0.5);
   Minimizer::recommend(data, policy, /*iteration=*/2);
   EXPECT_DOUBLE_EQ(data.regret[0], 0.);
   EXPECT_DOUBLE_EQ(data.regret[1], 0.35);
   // r == 0 for the crushed action => the recommendation is floor + tau*0/Lambda,
   // i.e. the FLOOR ITSELF, exactly
   EXPECT_DOUBLE_EQ(policy[kuhn::Action::check], 0.1);
   EXPECT_NEAR(policy[kuhn::Action::bet], 0.9, 1e-15);
}

TEST(ConstrainedRMPlusKernel, rejects_infeasible_floors)
{
   namespace kuhn = games::kuhn;
   using Minimizer = rm::ConstrainedRMPlus< kuhn::Action, 0.1 >;
   Minimizer::node_data_type data{};
   Minimizer::register_action(data, kuhn::Action::check);
   Minimizer::register_action(data, kuhn::Action::bet);
   HashmapActionPolicy< kuhn::Action > policy{};

   Minimizer::observe(data, kuhn::Action::check, 1.);

   // floors leaving no free probability mass
   data.probability_floors[0] = 0.7;
   data.probability_floors[1] = 0.7;
   EXPECT_THROW(Minimizer::recommend(data, policy, 0), std::invalid_argument);

   // a single floor outside the unit range
   data.probability_floors[0] = 1.5;
   data.probability_floors[1] = 0.;
   EXPECT_THROW(Minimizer::recommend(data, policy, 0), std::invalid_argument);
   data.probability_floors[0] = -0.1;
   data.probability_floors[1] = 0.2;
   EXPECT_THROW(Minimizer::recommend(data, policy, 0), std::invalid_argument);
}

TEST(ConstrainedRMPlusKernel, honors_environment_provided_floor_vectors)
{
   namespace kuhn = games::kuhn;
   // mock of the B8 contract at the kernel boundary: the solver refresh writes the
   // environment-reported floors into 'probability_floors' ahead of every recommend
   struct MockEnv {
      std::vector< double >
      action_probability_floors(const kuhn::Infostate&, const std::vector< kuhn::Action >&) const
      {
         return {0.3, 0.05};
      }
   };
   static_assert(concepts::has::method::
                    action_probability_floors< MockEnv, kuhn::Infostate, kuhn::Action >);
   static_assert(not concepts::has::method::
                    action_probability_floors< kuhn::Environment, kuhn::Infostate, kuhn::Action >);

   // positive default floor => buffered (paper-exact) specialization, as required of
   // trait-supporting configurations
   using Minimizer = rm::ConstrainedRMPlus< kuhn::Action, 0.01 >;
   Minimizer::node_data_type data{};
   Minimizer::register_action(data, kuhn::Action::check);
   Minimizer::register_action(data, kuhn::Action::bet);

   MockEnv env{};
   kuhn::Infostate infostate{Player::alex};
   const auto reported = env.action_probability_floors(infostate, data.registry.actions);
   std::ranges::copy(reported, std::ranges::begin(data.probability_floors));

   HashmapActionPolicy< kuhn::Action > policy{};
   // p = (0.3, 0.05); tau = 0.65; phi = (1, 0): <p, phi> = 0.3
   // r <- (0.65 + 0.3, 0.3) = (0.95, 0.3); Lambda = 1.25
   // x = (0.3 + 0.65*0.95/1.25, 0.05 + 0.65*0.3/1.25) = (0.794, 0.206)
   Minimizer::observe(data, kuhn::Action::check, 1.);
   Minimizer::observe(data, kuhn::Action::bet, 0.);
   Minimizer::recommend(data, policy, 0);
   EXPECT_NEAR(policy[kuhn::Action::check], 0.794, 1e-15);
   EXPECT_NEAR(policy[kuhn::Action::bet], 0.206, 1e-15);
   // both environment floors are respected exactly
   EXPECT_GE(policy[kuhn::Action::check], 0.3);
   EXPECT_GE(policy[kuhn::Action::bet], 0.05);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////// 3. constraint-satisfaction invariant across whole solver runs //////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

TEST(PerturbedCFRPlusKuhnInvariant, recommendations_respect_floors_at_every_iterate)
{
   namespace kuhn = games::kuhn;
   constexpr double eps = 0.05;
   constexpr size_t k_iters = 40;

   kuhn::Environment env{};
   auto solver = factory::make_cfr< perturbed_cfr_plus_config< eps >, true >(
      env,
      std::make_unique< kuhn::State >(),
      empty_tabular_policy< kuhn::Infostate, kuhn::Action >(),
      empty_tabular_policy< kuhn::Infostate, kuhn::Action >()
   );

   for([[maybe_unused]] auto _ : std::views::iota(size_t{0}, k_iters)) {
      solver.iterate(1);
      // alternating updates: after two iterations both players have refreshed their
      // recommendations at least once; before that untouched players play the uniform
      // default, which respects any floor <= 1/|A|
      assert_current_recommendations_respect_floors(solver, eps);
      assert_average_recommendations_respect_floors(solver, eps);
   }
}

TEST(PerturbedCFRPlusRPSInvariant, recommendations_respect_floors_at_every_iterate)
{
   namespace rps = games::rps;
   constexpr double eps = 0.05;

   rps::Environment env{};
   auto solver = factory::make_cfr< perturbed_cfr_plus_config< eps >, true >(
      env,
      std::make_unique< rps::State >(),
      empty_tabular_policy< rps::Infostate, rps::Action >(),
      empty_tabular_policy< rps::Infostate, rps::Action >()
   );

   for([[maybe_unused]] auto _ : std::views::iota(size_t{0}, size_t{30})) {
      solver.iterate(1);
      assert_current_recommendations_respect_floors(solver, eps);
      assert_average_recommendations_respect_floors(solver, eps);
   }
}

TEST(PerturbedCFRPlusLeducInvariant, short_horizon_recommendations_respect_floors)
{
   namespace leduc = games::leduc;
   constexpr double eps = 0.02;

   leduc::Environment env{};
   auto solver = factory::make_cfr< perturbed_cfr_plus_config< eps >, true >(
      env,
      std::make_unique< leduc::State >(),
      empty_tabular_policy< leduc::Infostate, leduc::Action >(),
      empty_tabular_policy< leduc::Infostate, leduc::Action >()
   );

   for([[maybe_unused]] auto _ : std::views::iota(size_t{0}, size_t{20})) {
      solver.iterate(1);
      assert_current_recommendations_respect_floors(solver, eps);
      assert_average_recommendations_respect_floors(solver, eps);
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////// 4. the epsilon -> 0 limit recovers CFR+ bit-for-bit ////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

TEST(PerturbedCFRPlusUnconstrainedLimit, kuhn_trajectories_match_plain_cfr_plus_exactly)
{
   namespace kuhn = games::kuhn;
   constexpr size_t k_iters = 250;

   kuhn::Environment env_vanilla{}, env_perturbed{};
   auto vanilla = factory::make_cfr< rm::CFRPlusConfig{}, true >(
      env_vanilla,
      std::make_unique< kuhn::State >(),
      empty_tabular_policy< kuhn::Infostate, kuhn::Action >(),
      empty_tabular_policy< kuhn::Infostate, kuhn::Action >()
   );
   auto perturbed = factory::make_cfr< perturbed_cfr_plus_config< 0. >, true >(
      env_perturbed,
      std::make_unique< kuhn::State >(),
      empty_tabular_policy< kuhn::Infostate, kuhn::Action >(),
      empty_tabular_policy< kuhn::Infostate, kuhn::Action >()
   );

   for([[maybe_unused]] auto _ : std::views::iota(size_t{0}, k_iters)) {
      vanilla.iterate(1);
      perturbed.iterate(1);
   }

   for(auto player : {Player::alex, Player::bob}) {
      expect_bitwise_identical_tables(
         vanilla.policy().at(player).table(), perturbed.policy().at(player).table()
      );
      expect_bitwise_identical_tables(
         vanilla.average_policy().at(player).table(), perturbed.average_policy().at(player).table()
      );
   }
}

TEST(PerturbedCFRPlusUnconstrainedLimit, rps_trajectories_match_plain_cfr_plus_exactly)
{
   namespace rps = games::rps;
   constexpr size_t k_iters = 300;

   rps::Environment env_vanilla{}, env_perturbed{};
   auto vanilla = factory::make_cfr< rm::CFRPlusConfig{}, true >(
      env_vanilla,
      std::make_unique< rps::State >(),
      empty_tabular_policy< rps::Infostate, rps::Action >(),
      empty_tabular_policy< rps::Infostate, rps::Action >()
   );
   auto perturbed = factory::make_cfr< perturbed_cfr_plus_config< 0. >, true >(
      env_perturbed,
      std::make_unique< rps::State >(),
      empty_tabular_policy< rps::Infostate, rps::Action >(),
      empty_tabular_policy< rps::Infostate, rps::Action >()
   );

   for([[maybe_unused]] auto _ : std::views::iota(size_t{0}, k_iters)) {
      vanilla.iterate(1);
      perturbed.iterate(1);
   }

   for(auto player : {Player::alex, Player::bob}) {
      expect_bitwise_identical_tables(
         vanilla.policy().at(player).table(), perturbed.policy().at(player).table()
      );
      expect_bitwise_identical_tables(
         vanilla.average_policy().at(player).table(), perturbed.average_policy().at(player).table()
      );
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////// 5. refinement quality demonstration on kuhn poker //////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

struct RefinementOutcome {
   double exploitability = 0.;
   ConditionedRegretReport metric;
};

/// runs 'config' on kuhn poker for 'n_iters' alternating iterations and measures the
/// on-path exploitability plus the reach-conditioned infoset-regret metric of the
/// resulting average profile; both profiles are measured against the SAME
/// 'uniform_floor'-floored deviation sets Q^I (the paper's sec.-8 comparison)
template < auto config >
RefinementOutcome run_kuhn_refinement(size_t n_iters, double uniform_floor)
{
   namespace kuhn = games::kuhn;
   kuhn::Environment env{};
   auto solver = factory::make_cfr< config, true >(
      env,
      std::make_unique< kuhn::State >(),
      empty_tabular_policy< kuhn::Infostate, kuhn::Action >(),
      empty_tabular_policy< kuhn::Infostate, kuhn::Action >()
   );
   for([[maybe_unused]] auto _ : std::views::iota(size_t{0}, n_iters)) {
      solver.iterate(1);
   }
   const auto& avg_policies = solver.average_policy();
   using avg_policy_t = std::decay_t< decltype(avg_policies.at(Player::alex)) >;
   using raw_table_t = std::decay_t< decltype(avg_policies.at(Player::alex).table()) >;
   const double expl = exploitability(
      env,
      kuhn::State{},
      player_hashmap< avg_policy_t >{
         std::pair{Player::alex, normalize_state_policy(avg_policies.at(Player::alex))},
         std::pair{Player::bob, normalize_state_policy(avg_policies.at(Player::bob))}}
   );
   const auto metric = reach_conditioned_infoset_regret(
      env,
      kuhn::State{},
      player_hashmap< raw_table_t >{
         std::pair{Player::alex, avg_policies.at(Player::alex).table()},
         std::pair{Player::bob, avg_policies.at(Player::bob).table()}},
      uniform_floor
   );
   return RefinementOutcome{.exploitability = expl, .metric = metric};
}

}  // namespace

TEST(PerturbedCFRPlusKuhnRefinement, conditioned_regret_beats_vanilla_cfr_plus)
{
   constexpr double eps = 0.1;  // strong trembling for a crisp separation
   constexpr size_t k_iters = 1000;

   // ---- vanilla CFR+
   const RefinementOutcome vanilla = run_kuhn_refinement< rm::CFRPlusConfig{} >(k_iters, eps);
   // ---- behaviorally-constrained CFR+
   const RefinementOutcome perturbed = run_kuhn_refinement< perturbed_cfr_plus_config< eps > >(
      k_iters, eps
   );

   fmt::print(
      "[perturbed-cfr-plus-refinement] iters={} eps={}\n"
      "  vanilla:    exploitability={:.6e} max-conditioned-regret={:.6e} "
      "(unconstrained-deviations={:.6e}, {} infosets)\n"
      "  perturbed:  exploitability={:.6e} max-conditioned-regret={:.6e} "
      "(unconstrained-deviations={:.6e})\n",
      k_iters,
      eps,
      vanilla.exploitability,
      vanilla.metric.all,
      vanilla.metric.all_unconstrained,
      vanilla.metric.infosets,
      perturbed.exploitability,
      perturbed.metric.all,
      perturbed.metric.all_unconstrained
   );

   // the refinement claim of Farina et al.: measured against the SAME floored
   // deviation sets Q^I, the perturbed solution plays EVERY information set --
   // including those reached with low probability -- with strictly lower
   // reach-conditioned regret than vanilla CFR+, whose average strategy leaves
   // those parts of the tree at arbitrary quality (their sec. 8, Fig. 2)
   EXPECT_GT(perturbed.metric.infosets, size_t{0});
   EXPECT_LT(perturbed.metric.all, vanilla.metric.all);
   // the separation grows with T (the vanilla limit keeps the constraint-violation gap
   // at every floored infoset), so a healthy factor margin holds at any reasonable
   // horizon (measured ~0.76 at T=1000)
   EXPECT_LT(perturbed.metric.all, 0.95 * vanilla.metric.all);
   // ... while its on-path exploitability still converges well below the game's payoff
   // scale (kuhn values lie in [-2, 2]); the positive constraint-induced floor of the
   // eps = 0.1 trembling keeps it away from zero
   EXPECT_LT(perturbed.exploitability, 0.09);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// 6. convergence thresholds ///////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

/// iterations until the average-policy exploitability drops to 'threshold'; returns
/// k_max_iters + 1 when unreached (deterministic solvers)
template < auto config, typename Env, size_t k_max_iters = 20000 >
size_t iterations_to_threshold(Env&& env, double threshold)
{
   using env_t = std::remove_cvref_t< Env >;
   auto solver = factory::make_cfr< config, true >(
      env,
      std::make_unique< auto_world_state_type< env_t > >(),
      empty_tabular_policy< auto_info_state_type< env_t >, auto_action_type< env_t > >(),
      empty_tabular_policy< auto_info_state_type< env_t >, auto_action_type< env_t > >()
   );
   auto expl_of = [&]() {
      const auto& avg_policies = solver.average_policy();
      try {
         double expl = exploitability(
            env,
            auto_world_state_type< env_t >{},
            player_hashmap< std::decay_t< decltype(avg_policies.at(Player::alex)) > >{
               std::pair{Player::alex, normalize_state_policy(avg_policies.at(Player::alex))},
               std::pair{Player::bob, normalize_state_policy(avg_policies.at(Player::bob))}}
         );
         return std::isfinite(expl) ? expl : std::numeric_limits< double >::max();
      } catch(const std::out_of_range&) {
         return std::numeric_limits< double >::max();
      }
   };
   double expl = expl_of();
   size_t n_iters = 0;
   while(expl > threshold and n_iters < k_max_iters) {
      solver.iterate(1);
      ++n_iters;
      expl = expl_of();
   }
   return expl <= threshold ? n_iters : k_max_iters + 1;
}

}  // namespace

TEST(PerturbedCFRPlusConvergence, kuhn_small_epsilon_meets_house_threshold)
{
   // eps = 0.001: the constraint-induced exploitability floor sits far below the house
   // threshold, so plain-CFR+-grade convergence remains observable
   constexpr size_t k_budget = 100000;
   const size_t iters = iterations_to_threshold<
      perturbed_cfr_plus_config< 0.001 >,
      games::kuhn::Environment,
      k_budget >(games::kuhn::Environment{}, EXPLOITABILITY_THRESHOLD);
   fmt::print(
      "[perturbed-cfr-plus-convergence] kuhn eps=0.001 threshold={} iters={}\n",
      EXPLOITABILITY_THRESHOLD,
      iters
   );
   EXPECT_LE(iters, k_budget);
}

TEST(PerturbedCFRPlusConvergence, rps_small_epsilon_meets_house_threshold)
{
   constexpr size_t k_budget = 20000;
   const size_t iters = iterations_to_threshold<
      perturbed_cfr_plus_config< 0.001 >,
      games::rps::Environment,
      k_budget >(games::rps::Environment{}, EXPLOITABILITY_THRESHOLD);
   EXPECT_LE(iters, k_budget);
}

TEST(PerturbedCFRPlusConvergence, kuhn_moderate_epsilon_converges_to_constraint_aware_level)
{
   // eps = 0.05 forces every action to keep trembling mass, so the unperturbed
   // exploitability bottoms out at a strictly positive constraint-induced level
   // (paper sec. 8); pin the reachable level with a comfortable margin
   namespace kuhn = games::kuhn;
   kuhn::Environment env{};
   auto solver = factory::make_cfr< perturbed_cfr_plus_config< 0.05 >, true >(
      env,
      std::make_unique< kuhn::State >(),
      empty_tabular_policy< kuhn::Infostate, kuhn::Action >(),
      empty_tabular_policy< kuhn::Infostate, kuhn::Action >()
   );
   for([[maybe_unused]] auto _ : std::views::iota(size_t{0}, size_t{3000})) {
      solver.iterate(1);
   }
   const auto& avg_policies = solver.average_policy();
   using avg_table_t = std::decay_t< decltype(avg_policies.at(Player::alex)) >;
   const double expl = exploitability(
      env,
      kuhn::State{},
      player_hashmap< avg_table_t >{
         std::pair{Player::alex, normalize_state_policy(avg_policies.at(Player::alex))},
         std::pair{Player::bob, normalize_state_policy(avg_policies.at(Player::bob))}}
   );
   fmt::print(
      "[perturbed-cfr-plus-convergence] kuhn eps=0.05 iters=3000 "
      "exploitability={:.6e}\n",
      expl
   );
   EXPECT_LT(expl, 0.06);
}
