#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <unordered_map>
#include <vector>

#include "cfr_run_funcs.hpp"
#include "dark_hex/dark_hex.hpp"
#include "nor/env.hpp"
#include "nor/exploitability.hpp"
#include "nor/nor.hpp"

using namespace nor;

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////// ESCHER / predictive-baseline extensions ///////////////////////////
//
// Tests for the ESCHER-style history-value baselines (McAleer, Farina, Lanctot,
// Sandholm; "ESCHER: Eschewing Importance Sampling in Games by Computing a
// History Value Function to Estimate Regret", ICLR 2023, arXiv:2206.04122) and
// the predictive baseline (Davis, Schmid, Bowling; "Low-Variance and Zero-
// Variance Baselines for Extensive-Form Games", ICML 2020, arXiv:1907.09633)
// layered onto the outcome-sampling MCCFR engine next to VR-MCCFR
// (Schmid et al., AAAI 2019).
//
// NOTE on bed sizes: dark hex cdh keeps ALL cells playable every turn (failed
// attempts retry), so the n x n move_limit=L instance branches |cells| at
// EVERY depth -- 3x3/ml9 spans ~4e8 world states. A full-tree best-response
// sweep (required by nor's exact exploitability oracle) plus complete infostate
// coverage are therefore infeasible in-process at that size; the
// exploitability-based convergence test runs on 2x2/ml6 (~5.5k world states,
// densely seedable), while the 3x3/ml9 bed contributes estimator-variance and
// storage evidence over long trajectories instead.
//////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

using kuhn_env = games::kuhn::Environment;
using kuhn_state = games::kuhn::State;
using kuhn_infostate = games::kuhn::Infostate;
using kuhn_action = games::kuhn::Action;
using kuhn_chance_outcome = games::kuhn::ChanceOutcome;

using ActionProbs = std::unordered_map< kuhn_action, double >;
using StateProbs = std::unordered_map< kuhn_infostate, ActionProbs >;
using PolicyProfile = player_hashmap< StateProbs >;

template < typename Table >
StateProbs normalize_table(const Table& table)
{
   StateProbs out;
   for(const auto& [istate, action_policy] : table) {
      double mass = 0.;
      size_t n_actions = 0;
      for(const auto& entry : action_policy) {
         mass += entry.second;
         ++n_actions;
      }
      ActionProbs probs;
      for(const auto& [action, prob] : action_policy) {
         probs[action] = mass > 0. ? prob / mass : 1. / static_cast< double >(n_actions);
      }
      out.emplace(istate, std::move(probs));
   }
   return out;
}

PolicyProfile normalize_profile(const auto& solver_policies)
{
   PolicyProfile profile{};
   profile.emplace(Player::alex, normalize_table(solver_policies.at(Player::alex).table()));
   profile.emplace(Player::bob, normalize_table(solver_policies.at(Player::bob).table()));
   return profile;
}

/// exact expected game values of a policy profile computed by full-tree
/// enumeration of Kuhn poker (6 ordered deals, 2 actions per decision)
class ExactKuhnEvaluator {
  public:
   explicit ExactKuhnEvaluator(kuhn_env env) : m_env(std::move(env)) {}

   std::unordered_map< Player, double > values(const PolicyProfile& profile) const
   {
      std::unordered_map< Player, double > acc{{Player::alex, 0.}, {Player::bob, 0.}};
      constexpr std::array< games::kuhn::Card, 3 > cards{
         games::kuhn::Card::jack, games::kuhn::Card::queen, games::kuhn::Card::king};
      for(size_t first = 0; first < 3; ++first) {
         for(size_t second = 0; second < 3; ++second) {
            if(first == second) {
               continue;
            }
            kuhn_state state0{};
            kuhn_state state{};
            m_env.transition(state, kuhn_chance_outcome{games::kuhn::Player::one, cards[first]});
            m_env.transition(state, kuhn_chance_outcome{games::kuhn::Player::two, cards[second]});
            player_hashmap< kuhn_infostate > infostates{};
            infostates.emplace(Player::alex, kuhn_infostate{Player::alex});
            infostates.emplace(Player::bob, kuhn_infostate{Player::bob});
            kuhn_state pre_deal_state{};
            auto deal1 = kuhn_chance_outcome{games::kuhn::Player::one, cards[first]};
            m_env.transition(pre_deal_state, deal1);
            for(auto player : {Player::alex, Player::bob}) {
               infostates.at(player).update(
                  m_env.public_observation(state0, deal1, pre_deal_state),
                  m_env.private_observation(player, state0, deal1, pre_deal_state)
               );
            }
            auto deal2 = kuhn_chance_outcome{games::kuhn::Player::two, cards[second]};
            for(auto player : {Player::alex, Player::bob}) {
               infostates.at(player).update(
                  m_env.public_observation(pre_deal_state, deal2, state),
                  m_env.private_observation(player, pre_deal_state, deal2, state)
               );
            }
            _traverse(state, infostates, 1. / 6., profile, acc);
         }
      }
      return acc;
   }

  private:
   void _traverse(
      kuhn_state& state,
      player_hashmap< kuhn_infostate >& infostates,
      double prob,
      const PolicyProfile& profile,
      std::unordered_map< Player, double >& acc
   ) const
   {
      if(m_env.is_terminal(state)) {
         acc[Player::alex] += prob * m_env.reward(Player::alex, state);
         acc[Player::bob] += prob * m_env.reward(Player::bob, state);
         return;
      }
      Player active = m_env.active_player(state);
      const auto& actions = m_env.actions(active, state);
      const auto& player_profile = profile.at(active);
      auto policy_it = player_profile.find(infostates.at(active));
      const ActionProbs* state_policy = policy_it != player_profile.end() ? &policy_it->second
                                                                          : nullptr;
      for(const auto& action : actions) {
         double action_prob = 1. / static_cast< double >(actions.size());
         if(state_policy != nullptr) {
            auto found = state_policy->find(action);
            if(found != state_policy->end()) {
               action_prob = found->second;
            }
         }
         if(action_prob <= 0.) {
            continue;
         }
         kuhn_state next_state = state;
         m_env.transition(next_state, action);
         auto next_infostates = infostates;
         for(auto player : {Player::alex, Player::bob}) {
            next_infostates.at(player).update(
               m_env.public_observation(state, action, next_state),
               m_env.private_observation(player, state, action, next_state)
            );
         }
         _traverse(next_state, next_infostates, prob * action_prob, profile, acc);
      }
   }

   kuhn_env m_env;
};

using dh_env = games::dark_hex::Environment;
using dh_state = games::dark_hex::State;
using dh_infostate = games::dark_hex::Infostate;
using dh_action = games::dark_hex::Move;

inline dh_env make_dh_env(size_t board_size, size_t move_limit)
{
   games::dark_hex::Config cfg{};
   cfg.board_size = board_size;
   cfg.rules_mode = games::dark_hex::RulesMode::cdh;
   cfg.move_limit = move_limit;
   return dh_env(cfg);
}

using DHActionProbs = std::unordered_map< dh_action, double >;
using DHStateProbs = std::unordered_map< dh_infostate, DHActionProbs >;
using DHPolicyProfile = player_hashmap< DHStateProbs >;

template < typename Table >
DHStateProbs normalize_dh_table(const Table& table)
{
   DHStateProbs out;
   for(const auto& [istate, action_policy] : table) {
      double mass = 0.;
      size_t n_actions = 0;
      for(const auto& entry : action_policy) {
         mass += entry.second;
         ++n_actions;
      }
      DHActionProbs probs;
      for(const auto& [action, prob] : action_policy) {
         probs[action] = mass > 0. ? prob / mass : 1. / static_cast< double >(n_actions);
      }
      out.emplace(istate, std::move(probs));
   }
   return out;
}

DHPolicyProfile normalize_dh_profile(const auto& solver_policies)
{
   DHPolicyProfile profile{};
   profile.emplace(Player::alex, normalize_dh_table(solver_policies.at(Player::alex).table()));
   profile.emplace(Player::bob, normalize_dh_table(solver_policies.at(Player::bob).table()));
   return profile;
}

/// exact expected game values of a policy profile computed by full-tree
/// enumeration of a (small) deterministic dark hex instance
class ExactDarkHexEvaluator {
  public:
   explicit ExactDarkHexEvaluator(dh_env env) : m_env(std::move(env)) {}

   std::unordered_map< Player, double > values(const DHPolicyProfile& profile) const
   {
      std::unordered_map< Player, double > acc{{Player::alex, 0.}, {Player::bob, 0.}};
      dh_state root{m_env.config()};
      player_hashmap< dh_infostate > infostates{};
      infostates.emplace(Player::alex, dh_infostate{Player::alex});
      infostates.emplace(Player::bob, dh_infostate{Player::bob});
      _traverse(root, infostates, 1., profile, acc);
      return acc;
   }

  private:
   void _traverse(
      dh_state& state,
      player_hashmap< dh_infostate >& infostates,
      double prob,
      const DHPolicyProfile& profile,
      std::unordered_map< Player, double >& acc
   ) const
   {
      if(m_env.is_terminal(state)) {
         acc[Player::alex] += prob * m_env.reward(Player::alex, state);
         acc[Player::bob] += prob * m_env.reward(Player::bob, state);
         return;
      }
      Player active = m_env.active_player(state);
      const auto& actions = m_env.actions(active, state);
      const auto& player_profile = profile.at(active);
      auto policy_it = player_profile.find(infostates.at(active));
      const DHActionProbs* state_policy = policy_it != player_profile.end() ? &policy_it->second
                                                                            : nullptr;
      for(const auto& action : actions) {
         double action_prob = 1. / static_cast< double >(actions.size());
         if(state_policy != nullptr) {
            auto found = state_policy->find(action);
            if(found != state_policy->end()) {
               action_prob = found->second;
            }
         }
         if(action_prob <= 0.) {
            continue;
         }
         dh_state next_state = state;
         m_env.transition(next_state, action);
         auto next_infostates = infostates;
         for(auto player : {Player::alex, Player::bob}) {
            next_infostates.at(player).update(
               m_env.public_observation(state, action, next_state),
               m_env.private_observation(player, state, action, next_state)
            );
         }
         _traverse(next_state, next_infostates, prob * action_prob, profile, acc);
      }
   }

   dh_env m_env;
};

/// enumerates the FULL game tree of a (small) dark hex instance and returns a
/// uniform action-policy table covering every reachable infostate of both
/// players. Used to pre-seed solvers so exact exploitability evaluations are
/// well-defined from iteration 0 onward.
std::unordered_map< dh_infostate, HashmapActionPolicy< dh_action > > dense_dh_uniform_table(
   const dh_env& env
)
{
   std::unordered_map< dh_infostate, HashmapActionPolicy< dh_action > > table{};
   player_hashmap< dh_infostate > infostates{};
   infostates.emplace(Player::alex, dh_infostate{Player::alex});
   infostates.emplace(Player::bob, dh_infostate{Player::bob});

   std::function< void(dh_state&, player_hashmap< dh_infostate >&) > recurse =
      [&](dh_state& state, player_hashmap< dh_infostate >& istates) {
         if(env.is_terminal(state)) {
            return;
         }
         Player active = env.active_player(state);
         const auto& actions = env.actions(active, state);
         table.emplace(
            istates.at(active),
            HashmapActionPolicy< dh_action >(actions, 1. / static_cast< double >(actions.size()))
         );
         for(const auto& action : actions) {
            dh_state next_state = state;
            env.transition(next_state, action);
            auto next_infostates = istates;
            for(auto player : {Player::alex, Player::bob}) {
               next_infostates.at(player).update(
                  env.public_observation(state, action, next_state),
                  env.private_observation(player, state, action, next_state)
               );
            }
            recurse(next_state, next_infostates);
         }
      };
   dh_state root{env.config()};
   recurse(root, infostates);
   return table;
}

struct EstimatorSeries {
   /// root value stream estimates of the solver, keyed by updating player
   std::unordered_map< Player, std::vector< double > > estimates;
   /// exact values of the respective current policies
   std::unordered_map< Player, std::vector< double > > exact_values;
   /// number of materialized history-value entries at the end of the run
   size_t history_entries = 0;
};

template < rm::MCCFRConfig config >
EstimatorSeries collect_kuhn_estimator_series(size_t n_iters, double epsilon, size_t seed)
{
   kuhn_env env{};
   auto root_state = std::make_unique< kuhn_state >();

   auto avg_tabular_policy = factory::make_tabular_policy(
      std::unordered_map< kuhn_infostate, HashmapActionPolicy< kuhn_action > >{}
   );
   auto tabular_policy = factory::make_tabular_policy(
      std::unordered_map< kuhn_infostate, HashmapActionPolicy< kuhn_action > >{}
   );

   auto solver = factory::make_cfr< config, true >(
      env, std::move(root_state), tabular_policy, avg_tabular_policy, epsilon, seed
   );

   ExactKuhnEvaluator evaluator{kuhn_env{}};
   EstimatorSeries series;
   for(size_t iter = 0; iter < n_iters; ++iter) {
      auto result = solver.iterate(1);
      auto exact = evaluator.values(normalize_profile(solver.policy()));
      for(const auto& [player, value] : result.back().get()) {
         series.estimates[player].push_back(value);
         series.exact_values[player].push_back(exact.at(player));
      }
   }
   series.history_entries = solver.history_value_entry_count();
   return series;
}

template < rm::MCCFRConfig config >
EstimatorSeries
collect_dh_estimator_series(size_t n_iters, double epsilon, size_t seed, const dh_env& env)
{
   auto root_state = std::make_unique< dh_state >(env.config());

   auto avg_tabular_policy = factory::make_tabular_policy(
      std::unordered_map< dh_infostate, HashmapActionPolicy< dh_action > >{}
   );
   auto tabular_policy = factory::make_tabular_policy(
      std::unordered_map< dh_infostate, HashmapActionPolicy< dh_action > >{}
   );

   auto solver = factory::make_cfr< config, true >(
      env, std::move(root_state), tabular_policy, avg_tabular_policy, epsilon, seed
   );

   ExactDarkHexEvaluator evaluator{dh_env(env.config())};
   EstimatorSeries series;
   for(size_t iter = 0; iter < n_iters; ++iter) {
      auto result = solver.iterate(1);
      auto exact = evaluator.values(normalize_dh_profile(solver.policy()));
      for(const auto& [player, value] : result.back().get()) {
         series.estimates[player].push_back(value);
         series.exact_values[player].push_back(exact.at(player));
      }
   }
   series.history_entries = solver.history_value_entry_count();
   return series;
}

double mean(const std::vector< double >& xs)
{
   double acc = 0.;
   for(auto x : xs) {
      acc += x;
   }
   return acc / static_cast< double >(xs.size());
}

double stddev(const std::vector< double >& xs)
{
   const double mu = mean(xs);
   double acc = 0.;
   for(auto x : xs) {
      acc += (x - mu) * (x - mu);
   }
   return std::sqrt(acc / static_cast< double >(xs.size() - 1));
}

std::vector< double >
residuals(const std::vector< double >& est, const std::vector< double >& exact)
{
   EXPECT_EQ(est.size(), exact.size());
   std::vector< double > out;
   out.reserve(est.size());
   for(size_t i = 0; i < est.size(); ++i) {
      out.push_back(est[i] - exact[i]);
   }
   return out;
}

/// exploitability trajectory on a small dark hex bed whose average policy was
/// pre-seeded densely (see dense_dh_uniform_table): safe to evaluate the exact
/// oracle at every checkpoint
template < rm::MCCFRConfig config >
std::vector< double >
dh_exploitability_trajectory(dh_env env, size_t n_iters, size_t stride, size_t seed, double epsilon)
{
   auto root_state = std::make_unique< dh_state >(env.config());
   auto dense = dense_dh_uniform_table(env);
   auto avg_tabular_policy = factory::make_tabular_policy(std::move(dense));
   auto tabular_policy = factory::make_tabular_policy(
      std::unordered_map< dh_infostate, HashmapActionPolicy< dh_action > >{}
   );

   auto solver = factory::make_cfr< config, true >(
      env, std::move(root_state), tabular_policy, avg_tabular_policy, epsilon, seed
   );

   std::vector< double > trajectory;
   for(size_t iter = 1; iter <= n_iters; ++iter) {
      solver.iterate(1);
      if(iter % stride != 0) {
         continue;
      }
      const auto& avg_policies = solver.average_policy();
      trajectory.push_back(exploitability(
         env,
         solver.root_state(),
         player_hashmap< std::decay_t< decltype(avg_policies.at(Player::alex)) > >{
            std::pair{Player::alex, normalize_state_policy(avg_policies.at(Player::alex))},
            std::pair{Player::bob, normalize_state_policy(avg_policies.at(Player::bob))}}
      ));
   }
   return trajectory;
}

}  // namespace

// legacy boolean shim resolution: old designated-initializer configs must map
// onto the tri-state exactly as before
TEST(ESCHERShim, legacy_flag_resolves_to_equivalent_mode)
{
   constexpr rm::MCCFRConfig legacy_off{};
   static_assert(rm::effective_variance_reduction(legacy_off) == rm::VarianceReductionMode::none);
   constexpr rm::MCCFRConfig legacy_on{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::lazy,
      .variance_reduced_baselines = true};
   static_assert(
      rm::effective_variance_reduction(legacy_on) == rm::VarianceReductionMode::action_baseline
   );
   // an explicitly chosen enum wins over the legacy bool
   constexpr rm::MCCFRConfig enum_wins{
      .variance_reduced_baselines = true,
      .baseline_update_rate = 1.,
      .variance_reduction = rm::VarianceReductionMode::history_value};
   static_assert(
      rm::effective_variance_reduction(enum_wins) == rm::VarianceReductionMode::history_value
   );
   SUCCEED();
}

// unbiasedness (paper Lemma 2 analog): the history-value-corrected root value
// stream is an unbiased estimate of the traversing player's expected game value
// under the current policy -- verified on-policy (epsilon = 0) against exact
// full-tree enumeration of Kuhn poker, same protocol as the action-baseline mode
TEST(ESCHER, history_value_estimator_unbiased_on_kuhn)
{
   constexpr rm::MCCFRConfig hist_config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::lazy,
      .variance_reduction = rm::VarianceReductionMode::history_value};

   constexpr size_t n_iters = 2000;
   auto series = collect_kuhn_estimator_series< hist_config >(n_iters, /*epsilon=*/0., size_t{42});

   ASSERT_GT(series.estimates.size(), size_t{0});
   std::cout << "[kuhn] history-value entries materialized: " << series.history_entries
             << std::endl;
   for(auto& [player, est] : series.estimates) {
      auto resid = residuals(est, series.exact_values.at(player));
      const double mean_resid = mean(resid);
      std::cout << "history-value unbiasedness player " << player << ": n=" << resid.size()
                << " mean(resid)=" << mean_resid << " sd(resid)=" << stddev(resid) << std::endl;
      EXPECT_NEAR(mean_resid, 0., 0.15);
   }
}

// the predictive baseline (Davis et al., ICML 2020, eq (8)) replaces the
// regression target with the next-strategy re-evaluation; the VALUE ESTIMATOR
// fed to the regrets is unchanged, so unbiasedness must be preserved for both
// baseline backends
TEST(ESCHER, predictive_baseline_unbiased_on_both_backends)
{
   {
      constexpr rm::MCCFRConfig pred_action_cfg{
         .update_mode = rm::UpdateMode::alternating,
         .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
         .weighting = rm::MCCFRWeightingMode::lazy,
         .variance_reduced_baselines = true,
         .baseline_update_rule = rm::BaselineUpdateRule::predictive};
      auto series = collect_kuhn_estimator_series< pred_action_cfg >(
         2000, /*epsilon=*/0., size_t{42}
      );
      for(auto& [player, est] : series.estimates) {
         auto resid = residuals(est, series.exact_values.at(player));
         std::cout << "predictive/action-baseline player " << player
                   << ": mean(resid)=" << mean(resid) << " sd=" << stddev(resid) << std::endl;
         EXPECT_NEAR(mean(resid), 0., 0.15);
      }
   }
   {
      constexpr rm::MCCFRConfig pred_hist_cfg{
         .update_mode = rm::UpdateMode::alternating,
         .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
         .weighting = rm::MCCFRWeightingMode::lazy,
         .variance_reduction = rm::VarianceReductionMode::history_value,
         .baseline_update_rule = rm::BaselineUpdateRule::predictive};
      auto series = collect_kuhn_estimator_series< pred_hist_cfg >(
         2000, /*epsilon=*/0., size_t{42}
      );
      for(auto& [player, est] : series.estimates) {
         auto resid = residuals(est, series.exact_values.at(player));
         std::cout << "predictive/history-value player " << player
                   << ": mean(resid)=" << mean(resid) << " sd=" << stddev(resid) << std::endl;
         EXPECT_NEAR(mean(resid), 0., 0.15);
      }
   }
}

// variance ladder on Kuhn: stddev(vanilla) vs stddev(action_baseline) vs
// stddev(history_value) of the root-value residual stream, identical seeds.
// Ordering verdicts are printed; assertions kept conservative.
TEST(ESCHER, variance_ladder_kuhn)
{
   constexpr rm::MCCFRConfig vanilla_config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::lazy};
   constexpr rm::MCCFRConfig action_baseline_config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::lazy,
      .variance_reduction = rm::VarianceReductionMode::action_baseline};
   constexpr rm::MCCFRConfig history_value_config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::lazy,
      .variance_reduction = rm::VarianceReductionMode::history_value};

   constexpr size_t n_iters = 400;
   auto vanilla = collect_kuhn_estimator_series< vanilla_config >(n_iters, 0.6, size_t{7});
   auto action = collect_kuhn_estimator_series< action_baseline_config >(n_iters, 0.6, size_t{7});
   auto hist = collect_kuhn_estimator_series< history_value_config >(n_iters, 0.6, size_t{7});

   for(auto player : {Player::alex, Player::bob}) {
      const double sd_vanilla = stddev(
         residuals(vanilla.estimates.at(player), vanilla.exact_values.at(player))
      );
      const double sd_action = stddev(
         residuals(action.estimates.at(player), action.exact_values.at(player))
      );
      const double sd_hist = stddev(
         residuals(hist.estimates.at(player), hist.exact_values.at(player))
      );
      std::cout << "kuhn variance ladder player " << player << ": sd(vanilla)=" << sd_vanilla
                << " sd(action_baseline)=" << sd_action << " sd(history_value)=" << sd_hist
                << " ratios: action/vanilla=" << (sd_action / sd_vanilla)
                << " history/vanilla=" << (sd_hist / sd_vanilla)
                << " history/action=" << (sd_hist / sd_action) << std::endl;
      // measured ordering (seeds above): history_value < action_baseline <<
      // vanilla by ~20x on this bed; asserted with a small cross-platform
      // float-noise margin. On dark hex 2x2 ml6 (see below) history_value does
      // NOT beat action_baseline -- reported there, not asserted.
      EXPECT_LT(sd_action, sd_vanilla);
      EXPECT_LT(sd_hist, sd_vanilla);
      EXPECT_LE(sd_hist, sd_action * 1.05 + 1e-9);
   }
}

// deep-tree variance ladder (dark hex cdh 2x2, move limit 6): importance
// weights compound along longer trajectories here
TEST(ESCHER, variance_ladder_dark_hex_2x2)
{
   const dh_env env = make_dh_env(/*board_size=*/2, /*move_limit=*/6);

   constexpr rm::MCCFRConfig vanilla_config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::lazy};
   constexpr rm::MCCFRConfig action_baseline_config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::lazy,
      .variance_reduction = rm::VarianceReductionMode::action_baseline};
   constexpr rm::MCCFRConfig history_value_config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::lazy,
      .variance_reduction = rm::VarianceReductionMode::history_value};

   constexpr size_t n_iters = 600;
   auto vanilla = collect_dh_estimator_series< vanilla_config >(n_iters, 0.6, size_t{11}, env);
   auto action = collect_dh_estimator_series< action_baseline_config >(
      n_iters, 0.6, size_t{11}, env
   );
   auto hist = collect_dh_estimator_series< history_value_config >(n_iters, 0.6, size_t{11}, env);

   std::cout << "[dark_hex 2x2 ml6] history entries materialized: " << hist.history_entries
             << std::endl;

   for(auto player : {Player::alex, Player::bob}) {
      const double sd_vanilla = stddev(
         residuals(vanilla.estimates.at(player), vanilla.exact_values.at(player))
      );
      const double sd_action = stddev(
         residuals(action.estimates.at(player), action.exact_values.at(player))
      );
      const double sd_hist = stddev(
         residuals(hist.estimates.at(player), hist.exact_values.at(player))
      );
      std::cout << "dark_hex 2x2 ml6 variance ladder player " << player
                << ": sd(vanilla)=" << sd_vanilla << " sd(action_baseline)=" << sd_action
                << " sd(history_value)=" << sd_hist
                << " ratios: action/vanilla=" << (sd_vanilla > 0. ? sd_action / sd_vanilla : 0.)
                << " history/vanilla=" << (sd_vanilla > 0. ? sd_hist / sd_vanilla : 0.)
                << " history/action=" << (sd_action > 0. ? sd_hist / sd_action : 0.) << std::endl;
   }
   SUCCEED();
}

// ESCHER-style convergence on the tractable deep bed (dark hex cdh 2x2,
// move limit 6): history-value baselines sampled from the FIXED uniform updater
// policy (no importance sampling anywhere; ESCHER sec. 3 + Theorem 1).
// Exploitability of the average strategy must decrease; iterations-to-half-
// initial-exploitability compared against the action-baseline mode sampling
// from the current policy (single-seed comparison => reported, not asserted).
TEST(ESCHER, escher_fixed_sampling_convergence_dark_hex_2x2)
{
   const dh_env env = make_dh_env(/*board_size=*/2, /*move_limit=*/6);

   constexpr rm::MCCFRConfig escher_config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::lazy,
      .variance_reduction = rm::VarianceReductionMode::history_value,
      .updater_sampling = rm::UpdaterSamplingMode::fixed_uniform};
   constexpr rm::MCCFRConfig action_baseline_config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::lazy,
      .variance_reduction = rm::VarianceReductionMode::action_baseline};

   constexpr size_t n_iters = 2000;
   constexpr size_t stride = 100;

   auto escher_traj = dh_exploitability_trajectory< escher_config >(
      env, n_iters, stride, size_t{0}, 0.6
   );
   ASSERT_GT(escher_traj.size(), size_t{2});
   std::cout << "escher(fixed-uniform) dark_hex 2x2 ml6 exploitability:";
   for(auto e : escher_traj) {
      std::printf(" %.4f", e);
   }
   std::cout << std::endl;
   EXPECT_LT(escher_traj.back(), escher_traj.front());
   EXPECT_LT(escher_traj.back(), 0.5);

   auto ab_traj = dh_exploitability_trajectory< action_baseline_config >(
      env, n_iters, stride, size_t{0}, 0.6
   );
   ASSERT_GT(ab_traj.size(), size_t{2});
   std::cout << "action-baseline       dark_hex 2x2 ml6 exploitability:";
   for(auto e : ab_traj) {
      std::printf(" %.4f", e);
   }
   std::cout << std::endl;
   EXPECT_LT(ab_traj.back(), ab_traj.front());

   auto iters_to_half = [&](const std::vector< double >& traj) {
      const double threshold = traj.front() / 2.;
      for(size_t i = 0; i < traj.size(); ++i) {
         if(traj[i] <= threshold) {
            return (i + 1) * stride;
         }
      }
      return size_t{0};  // never reached within budget
   };
   std::cout << "iterations-to-half-initial-exploitability: escher(fixed-uniform)="
             << iters_to_half(escher_traj) << " action_baseline=" << iters_to_half(ab_traj)
             << std::endl;
}

// deep-tree evidence at the prescribed 3x3/ml9 size (~4e8 world states: exact
// exploitability infeasible in-process, see file header). Report-only: root
// value stream spread over long trajectories plus history-store size.
TEST(ESCHER, deep_tree_evidence_dark_hex_3x3_report_only)
{
   const dh_env env = make_dh_env(/*board_size=*/3, /*move_limit=*/9);

   constexpr rm::MCCFRConfig vanilla_config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::lazy};
   constexpr rm::MCCFRConfig action_baseline_config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::lazy,
      .variance_reduction = rm::VarianceReductionMode::action_baseline};
   constexpr rm::MCCFRConfig escher_config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::lazy,
      .variance_reduction = rm::VarianceReductionMode::history_value,
      .updater_sampling = rm::UpdaterSamplingMode::fixed_uniform};

   constexpr size_t n_iters = 2000;
   auto collect_stream_sd = [&env](auto config_tag) {
      constexpr rm::MCCFRConfig cfg = decltype(config_tag)::value;
      auto root_state = std::make_unique< dh_state >(env.config());
      auto avg_tabular_policy = factory::make_tabular_policy(
         std::unordered_map< dh_infostate, HashmapActionPolicy< dh_action > >{}
      );
      auto tabular_policy = factory::make_tabular_policy(
         std::unordered_map< dh_infostate, HashmapActionPolicy< dh_action > >{}
      );
      auto solver = factory::make_cfr< cfg, true >(
         env, std::move(root_state), tabular_policy, avg_tabular_policy, 0.6, size_t{5}
      );
      std::unordered_map< Player, std::vector< double > > streams;
      for(size_t iter = 0; iter < n_iters; ++iter) {
         auto result = solver.iterate(1);
         for(const auto& [player, value] : result.back().get()) {
            streams[player].push_back(value);
         }
      }
      return std::pair{std::move(streams), solver.history_value_entry_count()};
   };

   auto vanilla = collect_stream_sd(std::integral_constant< rm::MCCFRConfig, vanilla_config >{});
   auto action = collect_stream_sd(
      std::integral_constant< rm::MCCFRConfig, action_baseline_config >{}
   );
   auto escher = collect_stream_sd(std::integral_constant< rm::MCCFRConfig, escher_config >{});

   std::cout << "[dark_hex 3x3 ml9] history entries materialized: " << escher.second << " after "
             << n_iters << " iterations" << std::endl;
   for(auto player : {Player::alex, Player::bob}) {
      const double sd_vanilla = stddev(vanilla.first.at(player));
      const double sd_action = stddev(action.first.at(player));
      const double sd_escher = stddev(escher.first.at(player));
      std::cout << "dark_hex 3x3 ml9 stream-sd player " << player << ": sd(vanilla)=" << sd_vanilla
                << " sd(action_baseline)=" << sd_action << " sd(escher_fixed_uniform)=" << sd_escher
                << std::endl;
   }
   SUCCEED();
}
