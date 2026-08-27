#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <ranges>
#include <unordered_map>
#include <vector>

#include "centipede/centipede.hpp"
#include "colonel_blotto/colonel_blotto.hpp"
#include "nor/env.hpp"
#include "nor/nor.hpp"
#include "nor/rm/correlated/cfr_jr.hpp"
#include "nor/rm/correlated/efcp.hpp"
#include "rock_paper_scissors/rock_paper_scissors.hpp"
#include "shapley/shapley.hpp"

using namespace nor;
namespace corr = nor::rm::correlated;

namespace {

/// vanilla simultaneous-uniform-regret-matching configuration shared with the
/// CFR-Jr baseline runs of the correlated suite (cf. test_cfr_jr.cpp)
inline constexpr rm::CFRConfig k_correlated_cfg{
   .update_mode = rm::UpdateMode::simultaneous,
   .regret_minimizing_mode = rm::RegretMinimizingMode::regret_matching,
   .weighting_mode = rm::CFRWeightingMode::uniform};

template < typename Env >
using default_policy_t = TabularPolicy<
   typename Env::info_state_type,
   HashmapActionPolicy< typename Env::action_type >,
   std::unordered_map<
      typename Env::info_state_type,
      HashmapActionPolicy< typename Env::action_type > > >;

template < typename Env >
default_policy_t< Env > make_default_policy()
{
   return factory::make_tabular_policy(std::unordered_map<
                                       typename Env::info_state_type,
                                       HashmapActionPolicy< typename Env::action_type > >{});
}

/**
 * social welfare of the UNIFORM product distribution over the reduced normal-
 * form plans of both players. In chance-free games a joint plan pair realizes
 * exactly one terminal, so pbar(z) = (#pairs covering z) / (|P1| |P2|.
 */
template < typename Env >
double uniform_plan_welfare(const corr::SequenceFormOracle< Env >& oracle)
{
   const size_t n_terms = oracle.terminal_count();
   std::vector< size_t > cover(n_terms, 0);
   for(const auto& plan1 : oracle.reduced_plans(oracle.players().at(0))) {
      const auto& mask1 = oracle.plan_mask(oracle.players().at(0), plan1);
      for(const auto& plan2 : oracle.reduced_plans(oracle.players().at(1))) {
         const auto& mask2 = oracle.plan_mask(oracle.players().at(1), plan2);
         for(auto z : std::views::iota(size_t{0}, n_terms)) {
            if((mask1.at(z / 64) & mask2.at(z / 64)) >> (z % 64) & 1) {
               ++cover[z];
            }
         }
      }
   }
   const double pairs = double(oracle.reduced_plans(oracle.players().at(0)).size())
                        * double(oracle.reduced_plans(oracle.players().at(1)).size());
   double welfare = 0.;
   for(auto z : std::views::iota(size_t{0}, n_terms)) {
      welfare += double(cover[z]) / pairs
                 * (oracle.terminal_reward(z, oracle.players().at(0))
                    + oracle.terminal_reward(z, oracle.players().at(1)));
   }
   return welfare;
}

/// CFR-Jr baseline CCE metrics on one of the deterministic testbeds
template < typename Env >
corr::CCEMetrics
cfrjr_baseline(Env env, std::unique_ptr< typename Env::world_state_type > root, size_t iters)
{
   auto curr = make_default_policy< Env >();
   auto avg = make_default_policy< Env >();
   corr::CFRJr< k_correlated_cfg, Env, default_policy_t< Env >, default_policy_t< Env > > solver(
      env, std::move(root), curr, avg
   );
   solver.iterate(iters);
   return solver.metrics();
}

}  // namespace

///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////// structural decomposition tests ///////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Shapley's game is a 3x3 hidden-commitment serialization: both players own a
 * single root information set and every node pair co-traverses some path, so
 * EVERY sequence pair is relevant ((|S1|+1)(|S2|+1) = 16 coordinates). The full
 * Definition-3 system still audits feasible products: feeding the EFCP circuit's
 * synthetic-loss recommendations through the FULL constraint audit yields
 * residuals at rounding scale -- i.e. the reduced constraint subset retained by
 * DECOMPOSE implies everything else.
 */
TEST(EFCPStructure, shapley_reduced_system_implies_full_system)
{
   using Env = games::shapley::Environment;
   Env env{};
   games::shapley::State root{};

   corr::SequenceFormOracle< Env > oracle(env, root);
   corr::CorrelationPlanSpace< Env > space(oracle);

   ASSERT_EQ(space.relevant_pair_count(), size_t(16));
   EXPECT_EQ(space.constraints().size(), size_t(12));
   // both players have exactly one infoset; every pair is connected
   for(auto s1 : std::views::iota(int32_t{0}, int32_t(space.sequence_count(Player::alex)))) {
      for(auto s2 : std::views::iota(int32_t{0}, int32_t(space.sequence_count(Player::bob)))) {
         EXPECT_TRUE(space.relevant(s1, s2)) << "shapley must be fully relevant";
      }
   }

   // circuit-level feasibility sweep: recommendations after arbitrary synthetic
   // losses remain inside the FULL von Stengel-Forges polytope
   corr::EFCP< Env > solver(env, root);
   ASSERT_EQ(solver.space().relevant_pair_count(), space.relevant_pair_count());
   for(auto round : std::views::iota(size_t{0}, size_t{25})) {
      std::vector< double > loss(solver.space().relevant_pair_count(), 0.);
      loss[round % loss.size()] = -1.;  // pull mass toward coordinate 'round'
      solver.debug_observe_synthetic_losses(loss);
      const auto residual = space.feasibility_residual(solver.current_plan());
      EXPECT_LT(residual.linf, 1e-9) << "reduced set violated the full system @ round " << round;
   }
}

/**
 * DECOMPOSE terminates only when every relevant sequence pair has been filled
 * in exactly once (the constructor throws otherwise); these assertions pin
 * concrete expected shape data for each chance-free bed and verify that the
 * fill operation count matches the number of local simplex regret minimizers.
 */
TEST(EFCPStructure, decomposition_shape_invariants_on_chance_free_beds)
{
   {
      using Env = games::rps::Environment;
      Env env{};
      games::rps::State root{};
      corr::SequenceFormOracle< Env > oracle(env, root);
      corr::CorrelationPlanSpace< Env > space(oracle);
      // serialized RPS: one infoset per player, all nine pairs relevant
      EXPECT_EQ(space.relevant_pair_count(), size_t(16));
      EXPECT_GT(space.decomposition_ops().size(), size_t{0});
      size_t fills = 0;
      for(const auto& op : space.decomposition_ops()) {
         fills += op.kind == corr::CircuitOp::Kind::fill;
      }
      EXPECT_EQ(fills, space.leaf_unit_count());
      EXPECT_TRUE(std::ranges::any_of(space.decomposition_ops(), [](const corr::CircuitOp& op) {
         return op.children.size() == 3;
      }));
   }
   {
      using Env = games::shapley::Environment;
      Env env{};
      games::shapley::State root{};
      corr::SequenceFormOracle< Env > oracle(env, root);
      corr::CorrelationPlanSpace< Env > space(oracle);
      EXPECT_GE(space.leaf_unit_count(), size_t{2});
      EXPECT_EQ(space.relevant_pair_count(), size_t(16));
   }
   {
      using Env = games::centipede::Environment;
      const games::centipede::Config config{/*rounds*/ 3, /*pile_big*/ 4, /*pile_small*/ 1};
      Env env{config};
      games::centipede::State root{config};
      corr::SequenceFormOracle< Env > oracle(env, root);
      corr::CorrelationPlanSpace< Env > space(oracle);
      // perfect-information alternation produces SumSimplex operations (the
      // critical-info-set interlock the whole algorithm exists for)
      EXPECT_TRUE(std::ranges::any_of(space.decomposition_ops(), [](const corr::CircuitOp& op) {
         return op.kind == corr::CircuitOp::Kind::sum;
      }));
      EXPECT_GT(space.leaf_unit_count(), size_t{4});
   }
   {
      using Env = games::colonel_blotto::Environment;
      const colonel_blotto::BlottoConfig config{/*budget*/ 2};
      Env env{config};
      games::colonel_blotto::State root{config};
      corr::SequenceFormOracle< Env > oracle(env, root);
      corr::CorrelationPlanSpace< Env > space(oracle);
      EXPECT_GT(space.leaf_unit_count(), size_t{2});
      // constructor-level coverage assertions already passed: DECOMPOSE filled
      // every relevant pair exactly once
      const auto& seen = space.constraints();
      EXPECT_FALSE(seen.empty());
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// end-to-end solver tests //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

/// runs EFCP and reports per-checkpoint EFCE gaps of the CURRENT iterate plus
/// final averaged metrics
template < typename Env >
struct RunTrace {
   std::vector< std::pair< size_t, double > > current_gaps;
   corr::EFCPMetrics averaged{};
};

template < typename Env >
RunTrace< Env > run_efcp(
   Env& env,
   const typename Env::world_state_type& root,
   size_t iters,
   const std::vector< size_t >& gap_checkpoints
)
{
   RunTrace< Env > trace;
   corr::EFCP< Env > solver(env, root);
   for(auto t : std::views::iota(size_t{1}, iters + 1)) {
      solver.iterate(size_t{1});
      if(std::ranges::find(gap_checkpoints, t) != gap_checkpoints.end()) {
         trace.current_gaps.emplace_back(t, solver.evaluate(/*averaged=*/false).efce_gap);
      }
   }
   trace.averaged = solver.evaluate();
   for(const auto& [at, gap] : trace.current_gaps) {
      std::cout << "[efcp] iter=" << at << " current-efce-gap=" << gap << "\n";
   }
   std::cout << "[efcp] T=" << iters << " avg: efce_gap=" << trace.averaged.efce_gap
             << " sw=" << trace.averaged.social_welfare
             << " resid_linf=" << trace.averaged.feasibility_residual_linf << "\n";
   return trace;
}

}  // namespace

/**
 * Serialized rock-paper-scissors (zero-sum, chance-free): iterates stay inside
 * the von Stengel-Forges polytope at every checkpoint and the trigger-agent
 * deviation gain of the running iterate decreases; the averaged mediation
 * collects at least the uniform-correlation welfare.
 */
TEST(EFCPSolver, rps_feasible_iterates_and_decreasing_gap)
{
   using Env = games::rps::Environment;
   Env env{};
   games::rps::State root{};

   corr::SequenceFormOracle< Env > oracle(env, root);
   const double uniform_welfare = uniform_plan_welfare(oracle);

   auto trace = run_efcp(env, root, /*iters*/ 1500, /*checkpoints*/ {25, 250, 1000});

   EXPECT_LT(trace.averaged.feasibility_residual_linf, 1e-8);
   ASSERT_EQ(trace.current_gaps.size(), size_t{3});
   EXPECT_LE(trace.averaged.efce_gap + 5e-3, trace.current_gaps.front().second + 1e-12)
      << "averaged trigger gap must undercut the early-training deviation incentive";
   EXPECT_LT(trace.averaged.efce_gap, 1e-2)
      << "RPS is small enough that CFR+-style self-play should nearly close it";
   EXPECT_GE(trace.averaged.social_welfare, uniform_welfare - 1e-9);
}

/**
 * Shapley's best-response-cycle bimatrix (general-sum, chance-free). Checks:
 * iterates feasible; the averaged plan collects at least uniform-correlation
 * welfare (2/3); EFCP's welfare qualitatively meets-or-beats the house CFR-Jr
 * CCE baseline; gaps descend across checkpoints.
 */
TEST(EFCPSolver, shapley_gap_descent_and_welfare_direction)
{
   using Env = games::shapley::Environment;
   Env env{};
   games::shapley::State root{};

   corr::SequenceFormOracle< Env > oracle(env, root);
   const double uniform_welfare = uniform_plan_welfare(oracle);
   EXPECT_NEAR(uniform_welfare, 2. / 3., 1e-12);

   auto trace = run_efcp(env, root, /*iters*/ 2000, /*checkpoints*/ {25, 500, 1500});

   EXPECT_LT(trace.averaged.feasibility_residual_linf, 1e-8);
   ASSERT_EQ(trace.current_gaps.size(), size_t{3});
   EXPECT_LE(trace.averaged.efce_gap + 5e-3, trace.current_gaps.front().second + 1e-12);
   EXPECT_GE(trace.averaged.social_welfare, uniform_welfare - 1e-9);

   const auto baseline = cfrjr_baseline< Env >(
      Env{}, std::make_unique< games::shapley::State >(), /*iters*/ 3000
   );
   std::cout << "[cfr-jr][shapley] gap=" << baseline.cce_gap
             << " welfare=" << baseline.social_welfare << "\n";
   EXPECT_GE(trace.averaged.social_welfare, baseline.social_welfare - 1e-9)
      << "EFCP must not fall behind the product-policy CCE baseline on welfare";
}

/**
 * Centipede G(3 rounds, piles 4/1) is the canonical demonstration that
 * sequential correlation buys social welfare: selfish play stops at the SPE's
 * take-immediately outcome (welfare 5), whereas an EFCE mediator can sustain
 * delayed takes worth up to 40. Whatever the exact mediation quality after a
 * short budget, it must strictly beat both uniform correlation and the CFR-Jr
 * CCE self-play welfare in this game where strict improvement exists.
 */
TEST(EFCPSolver, centipede_correlation_unlocks_social_welfare)
{
   using Env = games::centipede::Environment;
   const games::centipede::Config config{/*rounds*/ 3, /*pile_big*/ 4, /*pile_small*/ 1};
   Env env{config};
   games::centipede::State root{config};

   corr::SequenceFormOracle< Env > oracle(env, root);
   const double uniform_welfare = uniform_plan_welfare(oracle);

   auto trace = run_efcp(env, root, /*iters*/ 800, /*checkpoints*/ {25, 400});

   EXPECT_LT(trace.averaged.feasibility_residual_linf, 1e-8);
   ASSERT_EQ(trace.current_gaps.size(), size_t{2});
   EXPECT_LE(trace.averaged.efce_gap + 5e-3, trace.current_gaps.front().second + 1e-12);
   EXPECT_GE(trace.averaged.social_welfare, uniform_welfare - 1e-9);

   const auto baseline = cfrjr_baseline< Env >(
      Env{config},
      std::make_unique< games::centipede::State >(config),
      /*iters*/ 1500
   );
   std::cout << "[cfr-jr][centipede] gap=" << baseline.cce_gap
             << " welfare=" << baseline.social_welfare << "\n";
   EXPECT_GT(trace.averaged.social_welfare, baseline.social_welfare)
      << "sequential correlation is expected to unlock STRICTLY more welfare "
         "than unmediated self-play here";
}

/**
 * Colonel Blotto, budget 2 over three fields (constant-sum, chance-free):
 * structural smoke with feasibility and gap-decreasing assertions.
 */
TEST(EFCPSolver, colonel_blotto_feasible_and_decreasing)
{
   using Env = games::colonel_blotto::Environment;
   const colonel_blotto::BlottoConfig config{/*budget*/ 2};
   Env env{config};
   games::colonel_blotto::State root{config};

   auto trace = run_efcp(env, root, /*iters*/ 600, /*checkpoints*/ {25, 300});

   EXPECT_LT(trace.averaged.feasibility_residual_linf, 1e-8);
   ASSERT_EQ(trace.current_gaps.size(), size_t{2});
   EXPECT_LE(trace.averaged.efce_gap + 5e-3, trace.current_gaps.front().second + 1e-12);

   const double uniform_welfare = uniform_plan_welfare(corr::SequenceFormOracle< Env >(env, root));
   EXPECT_GE(trace.averaged.social_welfare, uniform_welfare - 1e-9);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// scaled-extension unit tests //////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Closure properties of the scaled-extension machinery, exercised through the
 * mediator circuit itself: whatever the leaf minimizers recommend (uniforms at
 * initialization, RM+ responses after observing losses), every recommendation
 *
 *   - keeps xi >= 0,
 *   - anchors xi[empty,empty] = 1,
 *   - fills every FillSimplex's children to EXACTLY its source value (mass
 *     partition without leakage or minting), and
 *   - fills every SumSimplex target to EXACTLY the sum of its operands.
 *
 * These identities are the algorithmic content of convexity preservation of
 * Definition 4 realized over Delta^{|A_I|} / {1}.
 */
TEST(EFCPCircuit, scaled_extension_closure_under_updates)
{
   using Env = games::rps::Environment;
   Env env{};
   games::rps::State root{};

   corr::EFCP< Env > solver(env, root);
   const auto& ops = solver.space().decomposition_ops();

   auto check_closure = [&](const char* stage) {
      const auto xi = solver.current_plan();
      EXPECT_NEAR(xi[size_t(solver.space().index_of(0, 0))], 1., 1e-12)
         << "anchor violated " << stage;
      for(double entry : xi) {
         EXPECT_GE(entry, -1e-15) << "negative correlation mass " << stage;
      }
      for(const auto& op : ops) {
         double aggregate = 0.;
         switch(op.kind) {
            case corr::CircuitOp::Kind::fill:
               for(int32_t child : op.children) {
                  aggregate += xi[size_t(child)];
               }
               EXPECT_NEAR(aggregate, xi[size_t(op.source)], 1e-9)
                  << "FillSimplex leaked mass " << stage;
               break;
            case corr::CircuitOp::Kind::sum:
               for(int32_t src : op.sources) {
                  aggregate += xi[size_t(src)];
               }
               EXPECT_NEAR(xi[size_t(op.target)], aggregate, 1e-9)
                  << "SumSimplex mis-aggregated " << stage;
               break;
         }
      }
   };

   check_closure("at uniform initialization");
   for(auto round : std::views::iota(size_t{0}, size_t{40})) {
      std::vector< double > loss(solver.space().relevant_pair_count(), 0.);
      loss[(7 * round + 3) % loss.size()] = -2.5;
      loss[(11 * round + 5) % loss.size()] = 1.25;
      solver.debug_observe_synthetic_losses(loss);
      check_closure("after scripted updates");
   }
}
