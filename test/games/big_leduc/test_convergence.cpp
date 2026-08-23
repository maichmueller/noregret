

#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

#include "fixtures.hpp"
#include "leduc_poker/leduc_poker.hpp"
#include "nor/env/leduc.hpp"
#include "nor/exploitability.hpp"
#include "nor/nor.hpp"

// the big-leduc-configured leduc environment must satisfy the stochastic fosg framework contract
static_assert(nor::concepts::stochastic_fosg< nor::games::leduc::Environment >);

namespace {

using namespace nor;
using namespace nor::games::leduc;

template < typename Solver >
double exploitability_of_average_policies(Solver& solver)
{
   const auto& avg_policies = solver.average_policy();
   return nor::exploitability(
      Environment{},
      leduc::State{leduc::LeducConfig::big_leduc()},
      nor::player_hashmap< std::decay_t< decltype(avg_policies.at(nor::Player::alex)) > >{
         std::pair{
            nor::Player::alex, nor::normalize_state_policy(avg_policies.at(nor::Player::alex))},
         std::pair{
            nor::Player::bob, nor::normalize_state_policy(avg_policies.at(nor::Player::bob))}}
   );
}

auto make_tabular_policies()
{
   return std::pair{
      factory::make_tabular_policy(std::unordered_map< Infostate, HashmapActionPolicy< Action > >{}
      ),
      factory::make_tabular_policy(std::unordered_map< Infostate, HashmapActionPolicy< Action > >{}
      )};
}

}  // namespace

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// Vanilla CFR (alternating) ///////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

TEST(BigLeducCFR, vanilla_alternating_convergence_smoke)
{
   auto [curr_policy, avg_policy] = make_tabular_policies();

   Environment env{};
   auto root_state = std::make_unique< leduc::State >(leduc::LeducConfig::big_leduc());

   auto solver = factory::
      make_cfr< rm::CFRDiscountedConfig{.update_mode = rm::UpdateMode::alternating}, true >(
         env, std::move(root_state), curr_policy, avg_policy
      );

   constexpr size_t max_iterations = 100;
   constexpr auto wall_budget = std::chrono::seconds{55};
   const auto start_time = std::chrono::steady_clock::now();

   // anchor the trend after 10 full-tree iterations and re-evaluate at the very end
   double expl_early = 0.;
   size_t n_done = 0;
   for(size_t iter = 1; iter <= max_iterations; ++iter) {
      solver.iterate(1);
      n_done = iter;
      if(iter == 10) {
         expl_early = exploitability_of_average_policies(solver);
      }
      const auto elapsed = std::chrono::steady_clock::now() - start_time;
      if(elapsed >= wall_budget) {
         break;
      }
   }
   const double expl_end = exploitability_of_average_policies(solver);
   const auto wall_used = std::chrono::duration_cast< std::chrono::milliseconds >(
                             std::chrono::steady_clock::now() - start_time
   )
                             .count();

   std::cout << "BigLeduc VanillaCFR alternating: " << n_done << " iterations in " << wall_used
             << "ms | exploitability @10: " << expl_early << " | @" << n_done << ": " << expl_end
             << "\n";

   ASSERT_TRUE(std::isfinite(expl_early));
   ASSERT_TRUE(std::isfinite(expl_end));
   EXPECT_LT(expl_end, 0.8 * expl_early + 1e-12);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// MCCFR (external sampling) ////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

TEST(BigLeducCFR, mccfr_external_sampling_smoke)
{
   auto [curr_policy, avg_policy] = make_tabular_policies();

   Environment env{};
   auto root_state = std::make_unique< leduc::State >(leduc::LeducConfig::big_leduc());

   auto solver = factory::make_mccfr<
      rm::MCCFRConfig{
         .update_mode = rm::UpdateMode::alternating,
         .algorithm = rm::MCCFRAlgorithmMode::external_sampling,
         .weighting = rm::MCCFRWeightingMode::stochastic},
      true >(env, std::move(root_state), curr_policy, avg_policy, /*epsilon=*/0., 12345);

   constexpr size_t n_iterations = 200;

   solver.iterate(1);
   // after a single sampled pass the average policy is still near-uniform: use it as baseline
   const double expl_baseline = exploitability_of_average_policies(solver);

   const auto start_time = std::chrono::steady_clock::now();
   solver.iterate(n_iterations - 1);
   const double expl_end = exploitability_of_average_policies(solver);
   const auto wall_used = std::chrono::duration_cast< std::chrono::milliseconds >(
                             std::chrono::steady_clock::now() - start_time
   )
                             .count();

   std::cout << "BigLeduc MCCFR external-sampling: " << n_iterations << " iterations in "
             << wall_used << "ms | exploitability @1: " << expl_baseline << " | @" << n_iterations
             << ": " << expl_end << "\n";

   ASSERT_TRUE(std::isfinite(expl_baseline));
   ASSERT_TRUE(std::isfinite(expl_end));
   EXPECT_LT(expl_end, 0.95 * expl_baseline + 1e-12);
}
