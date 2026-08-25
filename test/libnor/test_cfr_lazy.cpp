#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <ranges>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cfr_run_funcs.hpp"
#include "dark_hex/dark_hex.hpp"
#include "nor/env.hpp"
#include "nor/nor.hpp"
#include "rm_specific_testing_utils.hpp"

using namespace nor;

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// configuration helpers ///////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

constexpr rm::CFRConfig
make_config(rm::RegretMinimizingMode rm_mode, rm::CFRLazyUpdateMode lazy_mode, double b = 1.)
{
   return rm::CFRConfig{
      .update_mode = rm::UpdateMode::alternating,
      .regret_minimizing_mode = rm_mode,
      .weighting_mode = rm::CFRWeightingMode::uniform,
      .pruning_mode = rm::CFRPruningMode::none,
      .lazy_update_mode = lazy_mode,
      .lazy_update_threshold_b = b};
}

constexpr rm::CFRConfig plain_cfr = make_config(
   rm::RegretMinimizingMode::regret_matching,
   rm::CFRLazyUpdateMode::off
);
constexpr rm::CFRConfig plain_cfr_plus = make_config(
   rm::RegretMinimizingMode::regret_matching_plus,
   rm::CFRLazyUpdateMode::off
);
constexpr rm::CFRConfig lazy_cfr = make_config(
   rm::RegretMinimizingMode::regret_matching,
   rm::CFRLazyUpdateMode::reach_threshold
);
constexpr rm::CFRConfig lazy_cfr_plus = make_config(
   rm::RegretMinimizingMode::regret_matching_plus,
   rm::CFRLazyUpdateMode::reach_threshold
);
constexpr rm::CFRConfig lazy_cfr_plus_b10 = make_config(
   rm::RegretMinimizingMode::regret_matching_plus,
   rm::CFRLazyUpdateMode::reach_threshold,
   10.
);

/// pruning-engine-independent copy of the lazy activity counters
struct LazyCounters {
   size_t segment_refreshes = 0;
   size_t skipped_refreshes = 0;
};

/// exploitability trajectory (one probe every 'expl_stride' iterations; NaN while the
/// average-policy tables are incomplete -- under lazy updates their folds materialize late)
/// together with the lazy engine's activity counters
template < auto config, typename Env, typename State >
std::pair< std::vector< double >, LazyCounters > exploitability_trajectory(
   Env env,
   std::unique_ptr< State > root_state,
   size_t n_iters,
   size_t expl_stride
)
{
   using namespace nor;
   auto expl_root_state = std::make_unique< State >(*root_state);

   auto curr_policy = factory::make_tabular_policy(
      std::unordered_map<
         auto_info_state_type< Env >,
         HashmapActionPolicy< auto_action_type< Env > > >{}
   );
   auto avg_policy = factory::make_tabular_policy(std::unordered_map<
                                                  auto_info_state_type< Env >,
                                                  HashmapActionPolicy< auto_action_type< Env > > >{}
   );

   auto solver = factory::make_cfr< config, true >(
      Env{env}, std::move(root_state), std::move(curr_policy), std::move(avg_policy)
   );

   std::vector< double > traj;
   traj.reserve(n_iters / expl_stride + 1);
   for(size_t iter : std::views::iota(size_t{1}, n_iters + 1)) {
      solver.iterate(1);
      if(iter % expl_stride != 0u) {
         continue;
      }
      const auto& avg_policies = solver.average_policy();
      const auto& curr_policies = solver.policy();
      // completeness guard: lazily deferred folds populate the average tables late, so a
      // probe is only valid once BOTH players' average tables cover exactly the same
      // infostates as their current-policy tables (which the traversal fills eagerly)
      const bool complete = std::ranges::all_of(
         std::vector{Player::alex, Player::bob},
         [&](Player p) {
            return avg_policies.at(p).size() > 0
                   and avg_policies.at(p).size() == curr_policies.at(p).size();
         }
      );
      if(not complete) {
         traj.push_back(std::numeric_limits< double >::quiet_NaN());
         continue;
      }
      traj.push_back(exploitability(
         env,
         *expl_root_state,
         player_hashmap< std::decay_t< decltype(avg_policies.at(Player::alex)) > >{
            std::pair{Player::alex, normalize_state_policy(avg_policies.at(Player::alex))},
            std::pair{Player::bob, normalize_state_policy(avg_policies.at(Player::bob))}}
      ));
   }
   const auto stats = solver.lazy_stats();
   return {
      std::move(traj),
      LazyCounters{
         .segment_refreshes = stats.segment_refreshes,
         .skipped_refreshes = stats.skipped_refreshes}};
}

/// runs until the exploitability drops below 'threshold' or 'max_iters' are exhausted;
/// returns {converged, final exploitability, iterations performed, lazy counters}
template < auto config, typename Env, typename State >
std::tuple< bool, double, size_t, LazyCounters > run_until_below(
   Env env,
   std::unique_ptr< State > root_state,
   double threshold,
   size_t max_iters,
   size_t update_freq = 10
)
{
   using namespace nor;
   auto expl_root_state = std::make_unique< State >(*root_state);

   auto curr_policy = factory::make_tabular_policy(
      std::unordered_map<
         auto_info_state_type< Env >,
         HashmapActionPolicy< auto_action_type< Env > > >{}
   );
   auto avg_policy = factory::make_tabular_policy(std::unordered_map<
                                                  auto_info_state_type< Env >,
                                                  HashmapActionPolicy< auto_action_type< Env > > >{}
   );

   auto solver = factory::make_cfr< config, true >(
      Env{env}, std::move(root_state), std::move(curr_policy), std::move(avg_policy)
   );

   double expl = std::numeric_limits< double >::max();
   size_t n_iters = 0;
   for(; n_iters < max_iters; ++n_iters) {
      solver.iterate(1);
      if((n_iters + 1) % update_freq != 0u) {
         continue;
      }
      const auto& avg_policies = solver.average_policy();
      const auto& curr_policies = solver.policy();
      const bool complete = std::ranges::all_of(
         std::vector{Player::alex, Player::bob},
         [&](Player p) {
            return avg_policies.at(p).size() > 0
                   and avg_policies.at(p).size() == curr_policies.at(p).size();
         }
      );
      if(not complete) {
         continue;
      }
      expl = exploitability(
         env,
         *expl_root_state,
         player_hashmap< std::decay_t< decltype(avg_policies.at(Player::alex)) > >{
            std::pair{Player::alex, normalize_state_policy(avg_policies.at(Player::alex))},
            std::pair{Player::bob, normalize_state_policy(avg_policies.at(Player::bob))}}
      );
      if(expl <= threshold) {
         ++n_iters;
         break;
      }
   }
   const auto stats = solver.lazy_stats();
   return {
      expl <= threshold,
      expl,
      n_iters,
      LazyCounters{
         .segment_refreshes = stats.segment_refreshes,
         .skipped_refreshes = stats.skipped_refreshes}};
}

double last_finite(const std::vector< double >& traj)
{
   double v = std::numeric_limits< double >::quiet_NaN();
   for(double e : traj) {
      if(std::isfinite(e)) {
         v = e;
      }
   }
   return v;
}

double first_finite(const std::vector< double >& traj)
{
   for(double e : traj) {
      if(std::isfinite(e)) {
         return e;
      }
   }
   return std::numeric_limits< double >::quiet_NaN();
}

void expect_lazy_engaged(const LazyCounters& counters)
{
   EXPECT_GT(counters.skipped_refreshes, 0u)
      << "lazy mode never skipped an end-of-iteration recommendation";
   EXPECT_GT(counters.segment_refreshes, 0u) << "lazy mode never closed a reach-budget segment";
}

}  // namespace

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////// static configuration sanity: illegal combos must fail ///////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

template < auto config >
inline constexpr bool is_legal_lazy_config = rm::detail::sanity_check_cfr_config< config >();

/// fully designated config builder for the illegal-combination static assertions
constexpr rm::CFRConfig variant(
   rm::RegretMinimizingMode rm_mode,
   rm::CFRWeightingMode weighting,
   rm::CFRPruningMode pruning,
   double b
)
{
   return rm::CFRConfig{
      .update_mode = rm::UpdateMode::alternating,
      .regret_minimizing_mode = rm_mode,
      .weighting_mode = weighting,
      .pruning_mode = pruning,
      .lazy_update_mode = rm::CFRLazyUpdateMode::reach_threshold,
      .lazy_update_threshold_b = b};
}

}  // namespace

TEST(CFRLazyConfigSanity, PredictiveDiscountedWeightedAndPrunedCombosRejected)
{
   using RMM = rm::RegretMinimizingMode;
   using WM = rm::CFRWeightingMode;
   using PM = rm::CFRPruningMode;
   // predictive kernels pair one recommendation with exactly one observed iteration -> frozen
   // recommendations break the rho/sigma_snap correspondence
   static_assert(not is_legal_lazy_config<
                 variant(RMM::predictive_regret_matching_plus, WM::uniform, PM::none, 1.) >);
   static_assert(not is_legal_lazy_config<
                 variant(RMM::discounted_regret_matching_plus, WM::uniform, PM::none, 1.) >);
   static_assert(not is_legal_lazy_config< variant(
                    RMM::discounted_predictive_regret_matching_plus, WM::uniform, PM::none, 1.
                 ) >);
   // discounted/exponential weighting rescales buffered increments -> unanalyzed
   static_assert(not is_legal_lazy_config<
                 variant(RMM::regret_matching, WM::discounted, PM::none, 1.) >);
   static_assert(not is_legal_lazy_config<
                 variant(RMM::regret_matching, WM::exponential, PM::none, 1.) >);
   // pruning windows assume a per-iteration recommend cadence
   static_assert(not is_legal_lazy_config<
                 variant(RMM::regret_matching_plus, WM::uniform, PM::regret_based, 1.) >);
   static_assert(not is_legal_lazy_config<
                 variant(RMM::regret_matching_plus, WM::uniform, PM::dynamic_thresholding, 1.) >);
   // non-positive budgets close segments before anything is buffered
   static_assert(not is_legal_lazy_config<
                 variant(RMM::regret_matching, WM::uniform, PM::none, 0.) >);
   static_assert(not is_legal_lazy_config<
                 variant(RMM::regret_matching, WM::uniform, PM::none, -1.) >);
   // the legal forms: Lazy-CFR and Lazy-CFR+ over plain RM/RM+ with uniform weighting
   static_assert(is_legal_lazy_config< plain_cfr >);
   static_assert(is_legal_lazy_config< lazy_cfr >);
   static_assert(is_legal_lazy_config< lazy_cfr_plus >);
   SUCCEED();
}

//////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////// kuhn poker: convergence + engagement ///////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

TEST(KuhnPoker, LAZY_CFR_ConvergesAndEngages)
{
   // plain regret matching converges markedly slower than its RM+ counterpart under lazy
   // updates (no clamping + frozen segments), hence the generous horizon
   auto [converged, expl, iters, counters] = run_until_below< lazy_cfr >(
      games::kuhn::Environment{},
      std::make_unique< games::kuhn::State >(),
      EXPLOITABILITY_THRESHOLD,
      150000
   );
   std::cout << "[          ] lazy-cfr kuhn: iters=" << iters << " expl=" << expl
             << " refreshes=" << counters.segment_refreshes
             << " skipped=" << counters.skipped_refreshes << "\n";
   EXPECT_TRUE(converged);
   expect_lazy_engaged(counters);
}

TEST(KuhnPoker, LAZY_CFR_PLUS_ConvergesAndEngages)
{
   auto [converged, expl, iters, counters] = run_until_below< lazy_cfr_plus >(
      games::kuhn::Environment{},
      std::make_unique< games::kuhn::State >(),
      EXPLOITABILITY_THRESHOLD,
      20000
   );
   std::cout << "[          ] lazy-cfr+ kuhn: iters=" << iters << " expl=" << expl
             << " refreshes=" << counters.segment_refreshes
             << " skipped=" << counters.skipped_refreshes << "\n";
   EXPECT_TRUE(converged);
   expect_lazy_engaged(counters);
}

TEST(KuhnPoker, LAZY_CFR_PLUS_LargeBudgetB10_ConvergesLoosely)
{
   auto [converged, expl, iters, counters] = run_until_below< lazy_cfr_plus_b10 >(
      games::kuhn::Environment{}, std::make_unique< games::kuhn::State >(), 1e-2, 50000
   );
   std::cout << "[          ] lazy-cfr+(B=10) kuhn: iters=" << iters << " expl=" << expl
             << " refreshes=" << counters.segment_refreshes
             << " skipped=" << counters.skipped_refreshes << "\n";
   EXPECT_TRUE(converged);
   expect_lazy_engaged(counters);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
////////////// kuhn poker: loose draw-for-draw sanity against unpruned CFR+ //////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

TEST(KuhnPoker, LAZY_CFR_PLUS_ApproachesPlainCFRPlusLoosely)
{
   constexpr size_t n_iters = 800;
   constexpr size_t stride = 5;
   auto [plain_traj, plain_counters] = exploitability_trajectory< plain_cfr_plus >(
      games::kuhn::Environment{}, std::make_unique< games::kuhn::State >(), n_iters, stride
   );
   auto [lazy_traj, lazy_counters] = exploitability_trajectory< lazy_cfr_plus >(
      games::kuhn::Environment{}, std::make_unique< games::kuhn::State >(), n_iters, stride
   );

   ASSERT_EQ(plain_traj.size(), lazy_traj.size());
   double max_abs_diff = 0.;
   size_t compared = 0;
   for(auto [plain, lazy] : std::views::zip(plain_traj, lazy_traj)) {
      if(std::isnan(plain) or std::isnan(lazy)) {
         continue;
      }
      ASSERT_TRUE(std::isfinite(plain));
      ASSERT_TRUE(std::isfinite(lazy));
      max_abs_diff = std::max(max_abs_diff, std::abs(plain - lazy));
      ++compared;
   }

   const double plain_final = last_finite(plain_traj);
   const double lazy_final = last_finite(lazy_traj);
   std::cout << "[          ] kuhn draw-for-draw: compared=" << compared
             << " plain_final=" << plain_final << " lazy_final=" << lazy_final
             << " max|dExpl|=" << max_abs_diff
             << " | lazy refreshes=" << lazy_counters.segment_refreshes
             << " skipped=" << lazy_counters.skipped_refreshes << "\n";

   // both runs must actually approach the equilibrium ...
   EXPECT_LT(plain_final, 3e-3);
   EXPECT_LT(lazy_final, 3e-2);
   // ... and stay loosely close to each other (lazy changes the dynamics -- NOT identical):
   // mid-trajectory deviations may be substantial while one variant's average tables are
   // still dominated by its uniform warmup phase, so only the ENDPOINTS are held tight
   EXPECT_LT(max_abs_diff, 0.5);
   EXPECT_LT(std::abs(lazy_final - plain_final), 3e-2);
   expect_lazy_engaged(lazy_counters);
   // the eager control run must report zero lazy activity
   EXPECT_EQ(plain_counters.segment_refreshes, 0u);
   EXPECT_EQ(plain_counters.skipped_refreshes, 0u);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////// rock paper scissors: convergence via the named factories ////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

TEST(RockPaperScissors, LAZY_CFR_FactoryEntry_BudgetB4_ConvergesToUniformMix)
{
   // exercise the dedicated rm::CFRLazyConfig carrier + factory entry (Lazy-CFR)
   constexpr rm::CFRLazyConfig lazy_carrier_b4{
      .update_mode = rm::UpdateMode::alternating,
      .regret_minimizing_mode = rm::RegretMinimizingMode::regret_matching,
      .threshold_b = 4.};

   auto avg_policy = factory::make_tabular_policy(
      std::unordered_map< games::rps::Infostate, HashmapActionPolicy< games::rps::Action > >{}
   );
   auto curr_policy = factory::make_tabular_policy(
      std::unordered_map< games::rps::Infostate, HashmapActionPolicy< games::rps::Action > >{}
   );
   auto solver = factory::make_cfr_lazy< lazy_carrier_b4, true >(
      games::rps::Environment{}, std::make_unique< games::rps::State >(), curr_policy, avg_policy
   );

   games::rps::Environment expl_env{};
   double expl = std::numeric_limits< double >::max();
   for(size_t i = 0; i < 20000 and expl > EXPLOITABILITY_THRESHOLD; ++i) {
      solver.iterate(1);
      const auto& avg_policies = solver.average_policy();
      if(i % 10 != 9u) {
         continue;
      }
      if(std::ranges::any_of(avg_policies | std::views::values, [](const auto& p) {
            return p.size() != size_t(1);
         })) {
         continue;
      }
      expl = exploitability(
         expl_env,
         games::rps::State{},
         player_hashmap< std::decay_t< decltype(avg_policies.at(Player::alex)) > >{
            std::pair{Player::alex, normalize_state_policy(avg_policies.at(Player::alex))},
            std::pair{Player::bob, normalize_state_policy(avg_policies.at(Player::bob))}}
      );
   }
   const auto stats = solver.lazy_stats();
   std::cout << "[          ] factory lazy-cfr(B=4) rps: expl=" << expl
             << " refreshes=" << stats.segment_refreshes << " skipped=" << stats.skipped_refreshes
             << "\n";
   EXPECT_LT(expl, EXPLOITABILITY_THRESHOLD);
   assert_optimal_policy_rps(solver, 1e-2);
   EXPECT_GT(stats.skipped_refreshes, 0u);
   EXPECT_GT(stats.segment_refreshes, 0u);
}

TEST(RockPaperScissors, LAZY_CFR_PLUS_FactoryEntry_DefaultBudget_ConvergesToUniformMix)
{
   // exercise the dedicated factory entry of Lazy-CFR+ with its default B = 1
   constexpr rm::CFRLazyConfig lazy_plus_carrier{
      .update_mode = rm::UpdateMode::alternating,
      .regret_minimizing_mode = rm::RegretMinimizingMode::regret_matching_plus,
      .threshold_b = 1.};

   auto avg_policy = factory::make_tabular_policy(
      std::unordered_map< games::rps::Infostate, HashmapActionPolicy< games::rps::Action > >{}
   );
   auto curr_policy = factory::make_tabular_policy(
      std::unordered_map< games::rps::Infostate, HashmapActionPolicy< games::rps::Action > >{}
   );
   auto solver = factory::make_cfr_lazy_plus< lazy_plus_carrier, true >(
      games::rps::Environment{}, std::make_unique< games::rps::State >(), curr_policy, avg_policy
   );

   games::rps::Environment expl_env{};
   double expl = std::numeric_limits< double >::max();
   for(size_t i = 0; i < 20000 and expl > EXPLOITABILITY_THRESHOLD; ++i) {
      solver.iterate(1);
      const auto& avg_policies = solver.average_policy();
      if(i % 10 != 9u) {
         continue;
      }
      if(std::ranges::any_of(avg_policies | std::views::values, [](const auto& p) {
            return p.size() != size_t(1);
         })) {
         continue;
      }
      expl = exploitability(
         expl_env,
         games::rps::State{},
         player_hashmap< std::decay_t< decltype(avg_policies.at(Player::alex)) > >{
            std::pair{Player::alex, normalize_state_policy(avg_policies.at(Player::alex))},
            std::pair{Player::bob, normalize_state_policy(avg_policies.at(Player::bob))}}
      );
   }
   const auto stats = solver.lazy_stats();
   std::cout << "[          ] factory lazy-cfr+ rps: expl=" << expl
             << " refreshes=" << stats.segment_refreshes << " skipped=" << stats.skipped_refreshes
             << "\n";
   EXPECT_LT(expl, EXPLOITABILITY_THRESHOLD);
   assert_optimal_policy_rps(solver, 1e-2);
   EXPECT_GT(stats.skipped_refreshes, 0u);
   EXPECT_GT(stats.segment_refreshes, 0u);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////// dark hex 2x2/ml6 (cdh): sanity + engagement /////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

inline games::dark_hex::Environment make_small_dh_env()
{
   games::dark_hex::Config cfg{};
   cfg.board_size = 2;
   cfg.rules_mode = games::dark_hex::RulesMode::cdh;
   cfg.move_limit = 6;
   return games::dark_hex::Environment(cfg);
}

}  // namespace

TEST(DarkHex, LAZY_CFR_PLUS_2x2_ml6_DecreasesEngagesAndTracksPlainLoosely)
{
   constexpr size_t n_iters = 250;
   constexpr size_t stride = 25;

   auto [plain_traj, plain_counters] = exploitability_trajectory< plain_cfr_plus >(
      make_small_dh_env(),
      std::make_unique< games::dark_hex::State >(make_small_dh_env().config()),
      n_iters,
      stride
   );
   auto [lazy_traj, lazy_counters] = exploitability_trajectory< lazy_cfr_plus >(
      make_small_dh_env(),
      std::make_unique< games::dark_hex::State >(make_small_dh_env().config()),
      n_iters,
      stride
   );

   const double plain_first = first_finite(plain_traj);
   const double plain_final = last_finite(plain_traj);
   const double lazy_first = first_finite(lazy_traj);
   const double lazy_final = last_finite(lazy_traj);
   std::cout << "[          ] dark-hex 2x2/ml6: plain " << plain_first << " -> " << plain_final
             << " | lazy " << lazy_first << " -> " << lazy_final
             << " | lazy refreshes=" << lazy_counters.segment_refreshes
             << " skipped=" << lazy_counters.skipped_refreshes << "\n";

   ASSERT_TRUE(std::isfinite(lazy_first)) << "lazy average tables never completed";
   ASSERT_TRUE(std::isfinite(plain_first));
   ASSERT_TRUE(std::isfinite(lazy_final));
   ASSERT_TRUE(std::isfinite(plain_final));
   // the averaged strategy must improve over the horizon ...
   EXPECT_LT(lazy_final, lazy_first);
   EXPECT_LT(plain_final, plain_first);
   // ... and stay loosely close to plain CFR+ (NOT identical: lazy changes dynamics)
   EXPECT_LT(std::abs(lazy_final - plain_final), 0.35);
   expect_lazy_engaged(lazy_counters);
   EXPECT_EQ(plain_counters.segment_refreshes, 0u);
   EXPECT_EQ(plain_counters.skipped_refreshes, 0u);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
////////////////// leduc poker (short horizon): decrease + engagement + tracking //////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

TEST(LeducPoker, LAZY_CFR_PLUS_ShortHorizon_DecreasesEngagesAndTracksPlainLoosely)
{
   constexpr size_t n_iters = 300;
   constexpr size_t stride = 50;

   auto [plain_traj, plain_counters] = exploitability_trajectory< plain_cfr_plus >(
      games::leduc::Environment{}, std::make_unique< games::leduc::State >(), n_iters, stride
   );
   auto [lazy_traj, lazy_counters] = exploitability_trajectory< lazy_cfr_plus >(
      games::leduc::Environment{}, std::make_unique< games::leduc::State >(), n_iters, stride
   );

   const double plain_first = first_finite(plain_traj);
   const double plain_final = last_finite(plain_traj);
   const double lazy_first = first_finite(lazy_traj);
   const double lazy_final = last_finite(lazy_traj);
   std::cout << "[          ] leduc 300 iters: plain " << plain_first << " -> " << plain_final
             << " | lazy " << lazy_first << " -> " << lazy_final
             << " | lazy refreshes=" << lazy_counters.segment_refreshes
             << " skipped=" << lazy_counters.skipped_refreshes << "\n";

   ASSERT_TRUE(std::isfinite(lazy_first)) << "lazy average tables never completed";
   ASSERT_TRUE(std::isfinite(lazy_final));
   ASSERT_TRUE(std::isfinite(plain_final));
   // the averaged strategy must improve over the horizon ...
   EXPECT_LT(lazy_final, lazy_first);
   // ... and remain loosely in the neighborhood of plain CFR+ at the same iteration
   EXPECT_LT(std::abs(lazy_final - plain_final), 2.0);
   expect_lazy_engaged(lazy_counters);
}
