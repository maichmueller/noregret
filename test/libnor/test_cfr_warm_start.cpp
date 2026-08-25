#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <ranges>
#include <unordered_map>

#include "nor/env.hpp"
#include "nor/exploitability.hpp"
#include "nor/nor.hpp"

using namespace nor;

/// WARM START pre-play initialization for tabular CFR (rm::CFRConfig::warm_start_iterations).
///
/// Mechanism under test: during the first K global iterations every player's PLAYED strategy
/// is forced to the fixed warm-start policy (uniform by default) at the traversal
/// policy-fetch point, while counterfactual regret accumulation runs unmodified -- seeding
/// the regret tables away from zero against a stationary opposition. The pre-play rounds
/// contribute NOTHING to the average strategy ('before play'), so regular CFR resumes from
/// regret-seeded recommendations without uniform-round pollution.
///
/// EMPIRICAL CHARACTERIZATION (measured on this suite; all solvers deterministic, so every
/// iteration count below is exact and reproducible run-to-run):
///  - rock-paper-scissors with SKEWED initial policies: order-of-magnitude speedup
///    (iters-to-5e-3 drops from 99 to 2 for both RM and RM+): the fixed uniform profile
///    coincides with the game's Nash equilibrium, and the phase prevents the skewed start
///    from polluting regrets or average;
///  - kuhn poker / short-horizon leduc from DEFAULT (uniform) starts: only VERY short
///    phases pay off -- measured optima are K=1 (kuhn RM: 39 -> 32 iters-to-0.05) and K=2
///    (leduc RM+: 130 -> 123 iters-to-0.05 / 66 -> 62 iters-to-0.1); every longer phase
///    measured COSTS more than it saves there, because excluded pre-play rounds delay
///    average-strategy accumulation. The assertions below pin exactly these measured
///    operating points, plus a phase-length overhead guard for the neutral case.
///
/// Terminology note: this is the pre-play / fixed-opposition regime (cf. DeepStack's
/// warm-start form, Moravčík et al., Science 2017), NOT Brown & Sandholm's AAAI 2016
/// "Strategy-Based Warm Starting" which substitutes regret-table values directly.

namespace {

constexpr size_t k_max_iters = 20000;

template < typename Infostate, typename Action >
auto empty_tabular_policy()
{
   return factory::make_tabular_policy(
      std::unordered_map< Infostate, HashmapActionPolicy< Action > >{}
   );
}

template < typename Env, typename Solver >
double average_policy_exploitability(Env& env, const Solver& solver)
{
   const auto& avg_policies = solver.average_policy();
   return exploitability(
      env,
      auto_world_state_type< Env >{},
      player_hashmap< std::decay_t< decltype(avg_policies.at(Player::alex)) > >{
         std::pair{Player::alex, normalize_state_policy(avg_policies.at(Player::alex))},
         std::pair{Player::bob, normalize_state_policy(avg_policies.at(Player::bob))}}
   );
}

/// exploitability that treats not-yet-well-defined measurements as "not converged yet":
/// std::out_of_range while the average-policy tables have not materialized every reachable
/// infostate, NaN while entries are degenerate
template < typename Env, typename Solver >
double average_policy_exploitability_when_ready(Env& env, const Solver& solver)
{
   double expl = std::numeric_limits< double >::max();
   try {
      expl = average_policy_exploitability(env, solver);
   } catch(const std::out_of_range&) {
      return std::numeric_limits< double >::max();
   }
   return std::isfinite(expl) ? expl : std::numeric_limits< double >::max();
}

/// iterations until the kuhn average-policy exploitability drops to 'threshold';
/// k_max_iters + 1 when the threshold is not reached in time (deterministic solver, so the
/// count is reproducible run-to-run)
template < auto config, typename... ExtraFactoryArgs >
size_t kuhn_iterations_to_threshold(double threshold, ExtraFactoryArgs&&... extra_args)
{
   games::kuhn::Environment env{};
   auto root_state = std::make_unique< games::kuhn::State >();

   auto curr_policy = empty_tabular_policy< games::kuhn::Infostate, games::kuhn::Action >();
   auto avg_policy = empty_tabular_policy< games::kuhn::Infostate, games::kuhn::Action >();

   auto solver = factory::make_cfr< config, true >(
      env,
      std::move(root_state),
      std::move(curr_policy),
      std::move(avg_policy),
      std::forward< ExtraFactoryArgs >(extra_args)...
   );

   double expl = std::numeric_limits< double >::max();
   size_t n_iters = 0;
   while(expl > threshold and n_iters < k_max_iters) {
      solver.iterate(1);
      ++n_iters;
      expl = average_policy_exploitability_when_ready(env, solver);
   }
   return expl <= threshold ? n_iters : k_max_iters + 1;
}

/// iterations until the short-horizon leduc average-policy exploitability drops to
/// 'threshold'; k_max_iters + 1 when unreached (deterministic solver)
template < auto config >
size_t leduc_iterations_to_threshold(double threshold)
{
   games::leduc::Environment env{};
   auto root_state = std::make_unique< games::leduc::State >();

   auto curr_policy = empty_tabular_policy< games::leduc::Infostate, games::leduc::Action >();
   auto avg_policy = empty_tabular_policy< games::leduc::Infostate, games::leduc::Action >();

   auto solver = factory::make_cfr< config, true >(
      env, std::move(root_state), std::move(curr_policy), std::move(avg_policy)
   );

   double expl = std::numeric_limits< double >::max();
   size_t n_iters = 0;
   while(expl > threshold and n_iters < k_max_iters) {
      solver.iterate(1);
      ++n_iters;
      expl = average_policy_exploitability_when_ready(env, solver);
   }
   return expl <= threshold ? n_iters : k_max_iters + 1;
}

/// same as above for rock-paper-scissors, but with a deliberately skewed INITIAL current
/// policy (uniform is RPS's exact Nash, so an unskewed start converges instantly and could
/// never discriminate warm vs cold start)
template < auto config, typename... ExtraFactoryArgs >
size_t rps_skewed_iterations_to_threshold(double threshold, ExtraFactoryArgs&&... extra_args)
{
   using Action = games::rps::Action;
   games::rps::Environment env{};
   auto root_state = std::make_unique< games::rps::State >();

   auto curr_policy = factory::make_tabular_policy(
      std::unordered_map< games::rps::Infostate, HashmapActionPolicy< Action > >{}
   );
   // off-set the initial policy like setup_rps_test() does so recovery takes iterations
   curr_policy.emplace(
      games::rps::Infostate{Player::alex},
      std::pair{Action::rock, 1. / 10.},
      std::pair{Action::paper, 2. / 10.},
      std::pair{Action::scissors, 7. / 10.}
   );
   curr_policy.emplace(
      games::rps::Infostate{Player::bob},
      std::pair{Action::rock, 9. / 10.},
      std::pair{Action::paper, .5 / 10.},
      std::pair{Action::scissors, .5 / 10.}
   );
   auto avg_policy = empty_tabular_policy< games::rps::Infostate, games::rps::Action >();

   auto solver = factory::make_cfr< config, true >(
      env, std::move(root_state), std::move(curr_policy), std::move(avg_policy), extra_args...
   );

   double expl = std::numeric_limits< double >::max();
   size_t n_iters = 0;
   while(expl > threshold and n_iters < k_max_iters) {
      solver.iterate(1);
      ++n_iters;
      expl = average_policy_exploitability_when_ready(env, solver);
   }
   return expl <= threshold ? n_iters : k_max_iters + 1;
}

}  // namespace

/////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// speedup test ///////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

TEST(RockPaperScissors, WARM_START_speeds_up_RM_alternating_from_skewed_start)
{
   constexpr double threshold = 5e-3;
   constexpr size_t warm_k = 6;

   const size_t cold_iters = rps_skewed_iterations_to_threshold< rm::CFRConfig{
      .update_mode = rm::UpdateMode::alternating} >(threshold);
   const size_t warm_iters = rps_skewed_iterations_to_threshold< rm::CFRConfig{
      .update_mode = rm::UpdateMode::alternating,
      .regret_minimizing_mode = rm::RegretMinimizingMode::regret_matching,
      .weighting_mode = rm::CFRWeightingMode::uniform,
      .pruning_mode = rm::CFRPruningMode::none,
      .warm_start_iterations = warm_k} >(threshold);

   std::cout << "[          ] rps(skewed) RM iters-to-" << threshold << ": cold=" << cold_iters
             << " warm(K=" << warm_k << ")=" << warm_iters << "\n";

   EXPECT_LE(cold_iters, k_max_iters);
   // the warm-started run must reach the threshold in strictly fewer total iterations ...
   EXPECT_LT(warm_iters, cold_iters);
   // ... by a wide margin (deterministic solver; measured gap is an order of magnitude)
   EXPECT_LE(2 * warm_iters, cold_iters);
}

TEST(RockPaperScissors, WARM_START_speeds_up_RM_PLUS_alternating_from_skewed_start)
{
   constexpr double threshold = 5e-3;
   constexpr size_t warm_k = 4;

   const size_t cold_iters = rps_skewed_iterations_to_threshold< rm::CFRConfig{
      .update_mode = rm::UpdateMode::alternating,
      .regret_minimizing_mode = rm::RegretMinimizingMode::regret_matching_plus} >(threshold);
   const size_t warm_iters = rps_skewed_iterations_to_threshold< rm::CFRConfig{
      .update_mode = rm::UpdateMode::alternating,
      .regret_minimizing_mode = rm::RegretMinimizingMode::regret_matching_plus,
      .weighting_mode = rm::CFRWeightingMode::uniform,
      .pruning_mode = rm::CFRPruningMode::none,
      .warm_start_iterations = warm_k} >(threshold);

   std::cout << "[          ] rps(skewed) RM+ iters-to-" << threshold << ": cold=" << cold_iters
             << " warm(K=" << warm_k << ")=" << warm_iters << "\n";

   EXPECT_LT(warm_iters, cold_iters);
   EXPECT_LE(2 * warm_iters, cold_iters);
}

/////////////////////////////////////////////////////////////////////////////////////////////
//////////////////// bounded-overhead guards (kuhn / leduc) /////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

TEST(KuhnPoker, WARM_START_speeds_up_RM_alternating_with_single_preplay_round)
{
   // measured operating point of the seeding effect on kuhn from a default start: ONE
   // forced-uniform pre-play round (K=1) reaches exploitability 0.05 in 32 instead of 39
   // iterations with plain RM (deterministic counts). Longer phases monotonically lose
   // here (K=2 already measures 48 > 39): every excluded pre-play round delays average-
   // strategy accumulation, so only the shortest seeding pays off on this game.
   constexpr double threshold = 0.05;
   constexpr size_t warm_k = 1;

   const size_t cold_iters = kuhn_iterations_to_threshold< rm::CFRConfig{
      .update_mode = rm::UpdateMode::alternating} >(threshold);
   const size_t warm_iters = kuhn_iterations_to_threshold< rm::CFRConfig{
      .update_mode = rm::UpdateMode::alternating, .warm_start_iterations = warm_k} >(threshold);

   std::cout << "[          ] kuhn RM iters-to-" << threshold << ": cold=" << cold_iters
             << " warm(K=" << warm_k << ")=" << warm_iters << "\n";

   EXPECT_LE(cold_iters, k_max_iters);
   EXPECT_LT(warm_iters, cold_iters);
}

TEST(KuhnPoker, WARM_START_short_phase_never_costs_more_than_its_length)
{
   // RM+ on kuhn is the NEUTRAL case: the measured optimum K=1 lands at 24 vs cold 25 (a
   // marginal win), but the robust contract worth pinning for regression safety is the
   // phase-length bound -- enabling the pre-play phase must never cost more iterations
   // than its own length K.
   constexpr double threshold = 0.05;
   constexpr size_t warm_k = 1;

   const size_t cold_iters = kuhn_iterations_to_threshold< rm::CFRConfig{
      .update_mode = rm::UpdateMode::alternating,
      .regret_minimizing_mode = rm::RegretMinimizingMode::regret_matching_plus} >(threshold);
   const size_t warm_iters = kuhn_iterations_to_threshold< rm::CFRConfig{
      .update_mode = rm::UpdateMode::alternating,
      .regret_minimizing_mode = rm::RegretMinimizingMode::regret_matching_plus,
      .warm_start_iterations = warm_k} >(threshold);

   std::cout << "[          ] kuhn RM+ iters-to-" << threshold << ": cold=" << cold_iters
             << " warm(K=" << warm_k << ")=" << warm_iters << "\n";

   EXPECT_LE(cold_iters, k_max_iters);
   EXPECT_LE(warm_iters, cold_iters + warm_k);
}

TEST(LeducPoker, WARM_START_speeds_up_RM_PLUS_alternating_short_horizon)
{
   // short-horizon leduc, RM+ carrier (rides the rm::CFRPlus alias composition of
   // CFRPlusConfig::warm_start_iterations): a SHORT two-round pre-play phase seeds the much
   // larger regret table with best-response information about stationary opposition while
   // regular CFR is still playing near-uniform; measured operating point K=2 reaches
   // exploitability 0.1 in 62 instead of 66 iterations (0.05: 123 instead of 130). Longer
   // phases lose here for the same average-strategy-deferral reason as on kuhn.
   constexpr double threshold = 0.1;
   constexpr size_t warm_k = 2;

   const size_t cold_iters = leduc_iterations_to_threshold< rm::CFRPlusConfig{} >(threshold);
   const size_t warm_iters = leduc_iterations_to_threshold< rm::CFRPlusConfig{
      .warm_start_iterations = warm_k} >(threshold);

   std::cout << "[          ] leduc RM+ iters-to-" << threshold << ": cold=" << cold_iters
             << " warm(K=" << warm_k << ")=" << warm_iters << "\n";

   EXPECT_LE(cold_iters, k_max_iters);
   EXPECT_LT(warm_iters, cold_iters);
}

TEST(LeducPoker, WARM_START_short_horizon_soundness)
{
   // short-horizon leduc run with the pre-play phase enabled: the run stays numerically
   // sound, exits the phase exactly at K and descends monotonically afterwards
   constexpr size_t warm_k = 16;
   constexpr auto config = rm::CFRConfig{
      .update_mode = rm::UpdateMode::alternating,
      .regret_minimizing_mode = rm::RegretMinimizingMode::regret_matching,
      .weighting_mode = rm::CFRWeightingMode::uniform,
      .pruning_mode = rm::CFRPruningMode::none,
      .warm_start_iterations = warm_k};

   games::leduc::Environment env{};
   auto root_state = std::make_unique< games::leduc::State >();

   auto curr_policy = empty_tabular_policy< games::leduc::Infostate, games::leduc::Action >();
   auto avg_policy = empty_tabular_policy< games::leduc::Infostate, games::leduc::Action >();

   auto solver = factory::make_cfr< config, true >(
      env, std::move(root_state), std::move(curr_policy), std::move(avg_policy)
   );

   solver.iterate(warm_k / 2);
   EXPECT_TRUE(solver.in_warm_start());
   solver.iterate(warm_k - warm_k / 2);
   EXPECT_FALSE(solver.in_warm_start());

   const double expl_mid = average_policy_exploitability_when_ready(env, solver);
   solver.iterate(34);
   EXPECT_FALSE(solver.in_warm_start());
   const double expl_late = average_policy_exploitability_when_ready(env, solver);

   std::cout << "[          ] leduc warm(K=" << warm_k << ") expl@50=" << expl_late << "\n";

   EXPECT_LE(expl_late, expl_mid);
   EXPECT_TRUE(std::isfinite(expl_late));
}

/////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////// off-by-default regression guards //////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

TEST(KuhnPoker, WARM_START_off_converges_like_baseline)
{
   // warm_start_iterations == 0 must reproduce plain CFR behavior end to end: convergence to
   // the known game value and a near-optimal average policy
   constexpr auto config = rm::CFRConfig{
      .update_mode = rm::UpdateMode::alternating,
      .regret_minimizing_mode = rm::RegretMinimizingMode::regret_matching,
      .weighting_mode = rm::CFRWeightingMode::uniform,
      .pruning_mode = rm::CFRPruningMode::none,
      .rbp_min_skip_iterations = 1.,
      .rbp_br_refresh_period = 16,
      .dynamic_threshold_c = 3.,
      .warm_start_iterations = 0};

   games::kuhn::Environment env{};
   auto root_state = std::make_unique< games::kuhn::State >();

   auto curr_policy = empty_tabular_policy< games::kuhn::Infostate, games::kuhn::Action >();
   auto avg_policy = empty_tabular_policy< games::kuhn::Infostate, games::kuhn::Action >();

   auto solver = factory::make_cfr< config, true >(
      env, std::move(root_state), std::move(curr_policy), std::move(avg_policy)
   );

   EXPECT_FALSE(solver.in_warm_start());
   solver.iterate(3000);
   EXPECT_FALSE(solver.in_warm_start());

   const double expl = average_policy_exploitability(env, solver);
   EXPECT_LE(expl, 1e-3);

   // the kuhn game value of any Nash profile is -1/18 for player one
   const auto values = solver.game_value().get().to_hashmap();
   EXPECT_NEAR(values.at(Player::alex), -1. / 18., 5e-3);
}

TEST(KuhnPoker, WARM_START_off_is_deterministic_across_runs)
{
   // two identically-configured off runs must produce identical per-iteration root values
   constexpr auto config = rm::CFRConfig{.update_mode = rm::UpdateMode::alternating};
   constexpr size_t n_iters = 60;

   auto run_once = [&] {
      games::kuhn::Environment env{};
      auto root_state = std::make_unique< games::kuhn::State >();
      auto solver = factory::make_cfr< config, true >(
         std::move(env),
         std::move(root_state),
         empty_tabular_policy< games::kuhn::Infostate, games::kuhn::Action >(),
         empty_tabular_policy< games::kuhn::Infostate, games::kuhn::Action >()
      );
      return solver.iterate(n_iters);
   };

   const auto run = run_once();
   const auto run_again = run_once();

   ASSERT_EQ(run.size(), run_again.size());
   for(auto [iter_values_a, iter_values_b] : std::views::zip(run, run_again)) {
      for(auto p : {Player::alex, Player::bob}) {
         ASSERT_EQ(iter_values_a.get().at(p), iter_values_b.get().at(p));
      }
   }
}

TEST(RockPaperScissors, WARM_START_off_matches_legacy_config_spelling)
{
   // the legacy pre-feature config spellings compose warm_start_iterations == 0 implicitly;
   // their trajectories must be bit-for-bit unaffected by the new field. Uses CFRPlusConfig
   // so both the implicit default AND the explicit-zero spelling of the passthrough field
   // ride the rm::CFRPlus alias composition path.
   const auto legacy_trajectory = [&] {
      games::rps::Environment env{};
      auto root_state = std::make_unique< games::rps::State >();
      auto solver = factory::
         make_cfr< rm::CFRPlusConfig{.update_mode = rm::UpdateMode::simultaneous}, false >(
            std::move(env),
            std::move(root_state),
            empty_tabular_policy< games::rps::Infostate, games::rps::Action >(),
            empty_tabular_policy< games::rps::Infostate, games::rps::Action >()
         );
      return solver.iterate(40);
   }();

   const auto explicit_zero_trajectory = [&] {
      games::rps::Environment env{};
      auto root_state = std::make_unique< games::rps::State >();
      auto solver = factory::make_cfr<
         rm::CFRPlusConfig{.update_mode = rm::UpdateMode::simultaneous, .warm_start_iterations = 0},
         false >(
         std::move(env),
         std::move(root_state),
         empty_tabular_policy< games::rps::Infostate, games::rps::Action >(),
         empty_tabular_policy< games::rps::Infostate, games::rps::Action >()
      );
      return solver.iterate(40);
   }();

   ASSERT_EQ(legacy_trajectory.size(), explicit_zero_trajectory.size());
   for(auto [a, b] : std::views::zip(legacy_trajectory, explicit_zero_trajectory)) {
      for(auto p : {Player::alex, Player::bob}) {
         ASSERT_EQ(a.get().at(p), b.get().at(p));
      }
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// edge cases /////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

TEST(KuhnPoker, WARM_START_larger_than_total_iterations_stays_sane)
{
   // K far beyond the planned iteration budget: the whole run stays inside the pre-play
   // phase; regrets keep seeding while nothing may blow up
   constexpr auto config = rm::CFRConfig{
      .update_mode = rm::UpdateMode::alternating,
      .regret_minimizing_mode = rm::RegretMinimizingMode::regret_matching,
      .weighting_mode = rm::CFRWeightingMode::uniform,
      .pruning_mode = rm::CFRPruningMode::none,
      .warm_start_iterations = 1000000};

   games::kuhn::Environment env{};
   auto root_state = std::make_unique< games::kuhn::State >();

   auto curr_policy = empty_tabular_policy< games::kuhn::Infostate, games::kuhn::Action >();
   auto avg_policy = empty_tabular_policy< games::kuhn::Infostate, games::kuhn::Action >();

   auto solver = factory::make_cfr< config, true >(
      env, std::move(root_state), std::move(curr_policy), std::move(avg_policy)
   );

   solver.iterate(30);
   EXPECT_TRUE(solver.in_warm_start());

   // the pre-play phase contributes NOTHING to the average strategy: every entry stays at
   // its pristine INITIAL value. NOTE: average-policy tables initialize uniformly (the
   // base's 'zero' default policy is an alias of UniformPolicy), so untouched entries hold
   // exactly 1/|A|.
   for(auto player : {Player::alex, Player::bob}) {
      for(const auto& [infostate, action_policy] : solver.average_policy().at(player).table()) {
         (void) infostate;
         for(const auto& [action, prob] : action_policy) {
            (void) action;
            EXPECT_NEAR(prob, 0.5, 1e-15);
         }
      }
   }

   // the untouched (uniform) average is perfectly well-defined; its exploitability is
   // finite and non-negative
   const double expl = average_policy_exploitability(env, solver);
   EXPECT_TRUE(std::isfinite(expl));
   EXPECT_GE(expl, 0.);
}

TEST(KuhnPoker, WARM_START_exactly_total_iterations_boundary)
{
   // running exactly K iterations leaves the solver at the phase boundary without issues;
   // continuing past it hands control back to regular regret matching
   constexpr size_t warm_k = 10;
   constexpr auto config = rm::CFRConfig{
      .update_mode = rm::UpdateMode::alternating, .warm_start_iterations = warm_k};

   games::kuhn::Environment env{};
   auto root_state = std::make_unique< games::kuhn::State >();

   auto solver = factory::make_cfr< config, true >(
      env,
      std::move(root_state),
      empty_tabular_policy< games::kuhn::Infostate, games::kuhn::Action >(),
      empty_tabular_policy< games::kuhn::Infostate, games::kuhn::Action >()
   );

   solver.iterate(warm_k / 2);
   EXPECT_TRUE(solver.in_warm_start());

   // complete the phase exactly: iteration counter now sits ON the boundary, so the NEXT
   // iteration is the first regular CFR one
   solver.iterate(warm_k - warm_k / 2);
   EXPECT_FALSE(solver.in_warm_start());

   solver.iterate(1);
   EXPECT_FALSE(solver.in_warm_start());

   // regular-CFR iterations have begun accumulating average mass; the readiness-guarded
   // probe must not blow up on partially materialized tables
   const double expl = average_policy_exploitability_when_ready(env, solver);
   EXPECT_TRUE(std::isfinite(expl));
}

/////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////// custom fixed-policy selector ///////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

TEST(RockPaperScissors, WARM_START_custom_fixed_policy_selector_plumbing)
{
   // a provided fixed profile must be genuinely consulted: during the pre-play phase EVERY
   // player (including the traversal's updating player) plays whatever the selector reports,
   // so two runs whose selectors fix BOB to different pure actions must traverse differently
   // and produce different per-iteration root values (ALEX stays uniform-fixed in both so
   // the comparison isolates bob's contribution)
   using Action = games::rps::Action;

   auto fixed_bob_selector = [](Action bob_action) {
      rm::WarmStartPolicy< games::rps::Infostate, Action > selector{};
      selector.distribution =
         [bob_action](const games::rps::Infostate& istate, const std::vector< Action >& actions) {
            std::unordered_map< Action, double > dist{};
            for(const auto& action : actions) {
               if(istate.player() == Player::bob) {
                  dist.emplace(action, action == bob_action ? 1. : 0.);
               } else {
                  dist.emplace(action, 1. / static_cast< double >(actions.size()));
               }
            }
            return dist;
         };
      return selector;
   };

   constexpr size_t warm_k = 6;

   auto build_solver = [&](auto&& extra_arg) {
      games::rps::Environment env{};
      auto root_state = std::make_unique< games::rps::State >();
      auto curr_policy = factory::make_tabular_policy(
         std::unordered_map< games::rps::Infostate, HashmapActionPolicy< Action > >{}
      );
      auto avg_policy = empty_tabular_policy< games::rps::Infostate, games::rps::Action >();
      return factory::make_cfr<
         rm::CFRConfig{.update_mode = rm::UpdateMode::alternating, .warm_start_iterations = warm_k},
         true >(
         env,
         std::move(root_state),
         std::move(curr_policy),
         std::move(avg_policy),
         std::forward< decltype(extra_arg) >(extra_arg)
      );
   };

   auto rock_solver = build_solver(fixed_bob_selector(Action::rock));
   auto paper_solver = build_solver(fixed_bob_selector(Action::paper));

   // NOTE: per-iteration ROOT VALUES cannot discriminate here (a uniform-fixed player's
   // expected value against ANY pure opposition is exactly 0 in RPS). Instead run ONE
   // pre-play iteration -- enough to visit and force every decision node -- and compare
   // BOB's current-policy table: it holds the FORCED distribution because bob is not the
   // first updating player, i.e. no recommendation has overwritten it yet.
   rock_solver.iterate(1);
   paper_solver.iterate(1);

   auto action_mass = [](const auto& solver, Player player) {
      std::unordered_map< Action, double > mass{};
      for(const auto& [infostate, action_policy] :
          normalize_state_policy(solver.policy().at(player).table())) {
         (void) infostate;
         for(const auto& [action, prob] : action_policy) {
            mass[action] += prob;
         }
      }
      return mass;
   };

   // bob's forced profile differs between the two solvers (rock-fixed vs paper-fixed)
   const auto rock_mass = action_mass(rock_solver, Player::bob);
   const auto paper_mass = action_mass(paper_solver, Player::bob);
   EXPECT_NE(rock_mass, paper_mass);
   EXPECT_NEAR(rock_mass.at(Action::rock), 1., 1e-12);
   EXPECT_NEAR(paper_mass.at(Action::paper), 1., 1e-12);
}
