#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <random>
#include <vector>

#include "goofspiel/environment.hpp"
#include "nor/env.hpp"
#include "nor/nor.hpp"
// NOTE: the shapley game headers must come LAST among the library includes: they inject
// 'using namespace ::shapley' into 'nor::games::shapley', and parsing any sibling env wrapper
// (kuhn/rps/stratego via nor/env.hpp) afterwards makes its unqualified 'State' look ambiguous.
#include "shapley/shapley.hpp"

using namespace nor;

/////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// static conformance /////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(IcfrStatic, type_mappings_and_default_kernels)
{
   using Env = games::kuhn::Environment;
   using Solver = rm::ICFR< Env >;
   static_assert(std::is_same_v< Solver::action_type, games::kuhn::Action >);
   static_assert(std::is_same_v<
                 Solver::internal_rm_type,
                 rm::BlumMansourInternalRegretMatching< games::kuhn::Action > >);
   static_assert(std::is_same_v<
                 Solver::external_rm_type,
                 rm::RegretMatchingPlus< games::kuhn::Action > >);
   static_assert(
      std::is_same_v< Solver::action_variant_type, Env::action_variant_type >,
      "the solver must reuse the environment's action/chance variant"
   );
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////// Blum-Mansour kernel: played-sequence swap-regret decay //////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

/// plays one BM internal-regret unit on the canonical Shapley row-payoff against a deterministic
/// best-response-cycling opponent (same harness spirit as test_internal_regret.cpp) and reports
/// the empirical swap regret of the PLAYED pure sequence
template < typename OpponentScript >
   requires std::regular_invocable< OpponentScript, size_t >
double played_swap_regret_shapley_script(size_t rounds, OpponentScript&& opponent_play)
{
   constexpr size_t n = ::shapley::strategy_count;
   const std::array< std::array< double, n >, n > row_payoff{
      {{{1., 0., 0.}}, {{0., 1., 0.}}, {{0., 0., 1.}}}};
   using K = rm::BlumMansourInternalRegretMatching< uint8_t >;
   K::node_data_type data{};
   for(uint8_t s = 0; s < n; ++s) {
      data.register_action(s);
   }
   HashmapActionPolicy< uint8_t > policy{};
   std::mt19937_64 rng{42};
   auto sample = [&]() -> uint8_t {
      std::uniform_real_distribution< double > uni(0., 1.);
      const double r = uni(rng);
      double acc = 0.;
      uint8_t chosen = n - 1;
      for(const auto& [act, prob] : policy) {
         acc += prob;
         if(r < acc) {
            chosen = act;
            break;
         }
      }
      return chosen;
   };
   size_t played = 0;
   std::vector< std::vector< double > > swaps(n, std::vector< double >(n, 0.));
   for(size_t t = 0; t < rounds; ++t) {
      const auto opp = opponent_play(t);
      // audit BEFORE this round's play uses the strategy from the previous recommend
      for(size_t i = 0; i < n; ++i)
         for(size_t j = 0; j < n; ++j)
            swaps[i][j] += (i == played ? 1. : 0.) * (row_payoff[j][opp] - row_payoff[i][opp]);
      K::recommend(data, policy, t);
      played = sample();
      std::vector< double > utilities(n);
      for(uint8_t a = 0; a < n; ++a) {
         utilities[a] = row_payoff[a][opp];
      }
      K::observe_utilities(data, utilities);
   }
   double mx = 0.;
   for(size_t i = 0; i < n; ++i)
      for(size_t j = 0; j < n; ++j)
         if(i != j)
            mx = std::max(mx, swaps[i][j]);
   return mx;
}

}  // namespace

TEST(IcfrBlumMansourKernel, played_sequence_swap_regret_is_sublinear)
{
   // horizons quadruple; a linearly-growing quantity would quadruple too, while
   // an O(sqrt(T)) quantity at most doubles. Assert both absolute smallness and decay.
   const double r1 = played_swap_regret_shapley_script(25000, [](size_t t) {
      return uint8_t((t / 7) % 3);  // scripted non-degenerate cycling opponent
   });
   const double r2 = played_swap_regret_shapley_script(100000, [](size_t t) {
      return uint8_t((t / 7) % 3);
   });
   fmt::print("[icfr-bm-kernel] swap@25k={:.1f} swap@100k={:.1f}\\n", r1, r2);
   EXPECT_LT(r1, 0.02 * 25000);
   EXPECT_LT(r2, 0.02 * 100000);
   EXPECT_LT(r2, 2.5 * r1 + 10.);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////// plan sampler: recommendations marginal-match their units ////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(IcfrPlanSampler, sampled_actions_track_the_average_recommendation_distribution)
{
   // On shapley every player has exactly ONE always-reachable infoset whose
   // recommendation IS the plan's action, so the sampler's empirical action
   // frequency must match the round-average of the recommendation
   // distributions -- both measured directly:
   //   sampled[a]  := relative frequency of action a appearing as the player's
   //                  commitment along realized trajectories,
   //   avg_rec[a]  := (1/T) sum_t sigma^t(I, a).
   games::shapley::Environment env{};
   rm::ICFR< games::shapley::Environment > solver(
      env, std::make_unique< games::shapley::State >(), /*seed=*/777
   );
   constexpr size_t kRounds = 8000;
   std::vector< double > avg_rec_alex(3, 0.);
   std::vector< double > avg_rec_bob(3, 0.);
   for(size_t t = 0; t < kRounds; ++t) {
      solver.iterate(1);
      const auto& ra = solver.last_recommendation_distribution(Player::alex, 0);
      const auto& rb = solver.last_recommendation_distribution(Player::bob, 0);
      for(size_t a2 = 0; a2 < 3; ++a2) {
         avg_rec_alex[a2] += ra[a2] / double(kRounds);
         avg_rec_bob[a2] += rb[a2] / double(kRounds);
      }
   }
   // cumulative sampled frequencies over the whole run: decode the terminal paths
   std::vector< double > cum_alex(3, 0.), cum_bob(3, 0.);
   double total = 0.;
   for(const auto& [path, count] : solver.empirical_frequency_counts()) {
      const auto& p0 = std::get< games::shapley::Play >(path.at(0));
      const auto& p1 = std::get< games::shapley::Play >(path.at(1));
      cum_alex[p0.strategy] += count;
      cum_bob[p1.strategy] += count;
      total += count;
   }
   ASSERT_DOUBLE_EQ(total, double(kRounds));
   for(size_t a2 = 0; a2 < 3; ++a2) {
      cum_alex[a2] /= double(kRounds);
      cum_bob[a2] /= double(kRounds);
   }
   double max_dev = 0.;
   for(size_t a2 = 0; a2 < 3; ++a2) {
      max_dev = std::max(max_dev, std::abs(avg_rec_alex[a2] - cum_alex[a2]));
      max_dev = std::max(max_dev, std::abs(avg_rec_bob[a2] - cum_bob[a2]));
   }
   fmt::print("[icfr-sampler] max |sampled - avg_rec| = {:.5f}\n", max_dev);
   // both estimators converge to the same limit; their difference decays like O(1/sqrt(T))
   EXPECT_LT(max_dev, 0.02);
}
/////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////// external-unit bookkeeping invariants (Sigma^c counting) ////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(IcfrBookkeeping, external_units_match_co_trigger_sequences)
{
   games::kuhn::Environment env{};
   rm::ICFR< games::kuhn::Environment > solver(env, std::make_unique< games::kuhn::State >());
   solver.iterate(50);

   for(auto player : {Player::alex, Player::bob}) {
      const size_t n_infosets = solver.num_infosets(player);
      ASSERT_GT(n_infosets, size_t{0});
      for(size_t id = 0; id < n_infosets; ++id) {
         const auto chain = solver.infoset_chain(player, id);
         size_t expected_units = 0;
         for(const auto& [ancestor_id, required] : chain) {
            const size_t ancestor_actions = solver.infoset_action_count(player, ancestor_id);
            expected_units += ancestor_actions - 1;  // every deviating action at the ancestor
         }
         EXPECT_EQ(solver.external_unit_count(player, id), expected_units)
            << "player=" << int(player) << " infoset=" << id;
         if(expected_units > 0) {
            // spot-check descriptors: each unit's deviating action differs from the
            // chain requirement at its position, and positions are valid chain indices
            for(size_t u = 0; u < solver.external_unit_count(player, id); ++u) {
               const auto [pos, dev_action] = solver.external_unit_descriptor(player, id, u);
               ASSERT_LT(pos, chain.size());
               EXPECT_NE(dev_action, chain[pos].second);
            }
         } else {
            EXPECT_EQ(chain.size(), size_t{0});
         }
      }
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// EFCE convergence //////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

struct GapTrace {
   std::vector< double > gaps;
   std::vector< double > bounds;
};

template < typename Env >
GapTrace drive_icfr(
   Env env,
   std::unique_ptr< typename Env::world_state_type > root,
   const std::vector< size_t >& checkpoints,
   uint64_t seed = 42
)
{
   rm::ICFR< Env > solver(std::move(env), std::move(root), seed);
   GapTrace trace;
   size_t prev = 0;
   for(size_t target : checkpoints) {
      solver.iterate(target - prev);
      prev = target;
      const auto report = solver.evaluate_efce_gap();
      trace.gaps.push_back(report.efce_gap);
      trace.bounds.push_back(report.trigger_regret_bound);
      fmt::print(
         "[icfr] T={} gap={:.6f} bound={:.6f}\\n",
         target,
         report.efce_gap,
         report.trigger_regret_bound
      );
   }
   return trace;
}

}  // namespace

TEST(IcfrConvergence, shapley_general_sum_gap_descends_and_certificate_holds)
{
   games::shapley::Environment env{};
   rm::ICFR< games::shapley::Environment > solver(
      env, std::make_unique< games::shapley::State >(), /*seed=*/12345
   );
   constexpr size_t kFinal = 50000;
   double first_gap = 0., last_gap = 0.;
   for(const size_t target : {size_t{1000}, size_t{10000}, kFinal}) {
      solver.iterate(target - solver.iteration());
      const auto report = solver.evaluate_efce_gap();
      if(target == 1000)
         first_gap = report.efce_gap;
      if(target == kFinal)
         last_gap = report.efce_gap;
      fmt::print(
         "[icfr-shapley] T={} gap={:.6f} bound={:.6f}\n",
         target,
         report.efce_gap,
         report.trigger_regret_bound
      );
      if(target >= 10000) {
         EXPECT_LT(report.efce_gap, 0.02);
      }
      // Theorem 1: delta(mu_bar^T) <= max_sigma R^T_sigma / T at all times
      EXPECT_LE(report.efce_gap, report.trigger_regret_bound + 1e-9);
   }
   EXPECT_LT(last_gap, first_gap);

   // the correlation: shapley realizes exactly ONE terminal trajectory per
   // iteration (deterministic resolve), so the terminal counts must sum to T --
   // a structural sanity check of the mu_bar accumulator
   double total = 0.;
   for(const auto& [path, count] : solver.empirical_frequency_counts()) {
      (void) path;
      total += count;
   }
   EXPECT_DOUBLE_EQ(total, double(kFinal));
}

TEST(IcfrConvergence, kuhn_poker_three_player_gap_vanishes)
{
   const auto trace = drive_icfr(
      games::kuhn::Environment{},
      std::make_unique< games::kuhn::State >(
         std::vector< games::kuhn::Card >{
            games::kuhn::Card::queen, games::kuhn::Card::king, games::kuhn::Card::ace},
         3
      ),
      /*checkpoints=*/{5000, 20000, 50000}
   );
   EXPECT_LT(trace.gaps.back(), 0.01);
   EXPECT_LT(trace.gaps[2], trace.gaps[0]);
   for(size_t k = 0; k < trace.gaps.size(); ++k) {
      EXPECT_LE(trace.gaps[k], trace.bounds[k] + 1e-9);
   }
}

TEST(IcfrConvergence, kuhn_poker_two_player_smoke)
{
   const auto trace = drive_icfr(
      games::kuhn::Environment{},
      std::make_unique< games::kuhn::State >(),
      /*checkpoints=*/{10000, 50000}
   );
   EXPECT_LT(trace.gaps.back(), 0.05);
   for(size_t k = 0; k < trace.gaps.size(); ++k) {
      EXPECT_LE(trace.gaps[k], trace.bounds[k] + 1e-9);
   }
}

TEST(IcfrConvergence, goofspiel_chance_game_smoke)
{
   using Env = games::goofspiel::Environment;
   const auto cfg = games::goofspiel::GoofspielConfig{.deck_size = 3};
   const auto trace = drive_icfr(
      Env{cfg},
      std::make_unique< games::goofspiel::State >(cfg),
      /*checkpoints=*/{2000, 20000}
   );
   EXPECT_LT(trace.gaps.back(), 0.08);
   EXPECT_LT(trace.gaps[1], trace.gaps[0]);
}
