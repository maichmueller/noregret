

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <functional>
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

using avg_table_type = typename TabularPolicy< Infostate, HashmapActionPolicy< Action > >::
   table_type;

/// A read-only view on an average-policy table that falls back to an equal-probability answer
/// for any infostate the (sampling-based) trainer has never visited. Best-response computation
/// walks the whole game tree and would otherwise throw on the sparse tables of MCCFR runs.
/// NOTE: the fallback answers every queried action with 0.5, i.e. it is only approximately
/// uniform whenever more than two actions are legal. Absolute exploitability values under heavy
/// undersampling must therefore be read as indicative; only the trend is asserted.
struct SparseAveragePolicy {
   using action_policy_type = HashmapActionPolicy< Action >;
   using action_type = Action;
   using info_state_type = Infostate;

   avg_table_type table;

   explicit SparseAveragePolicy(const avg_table_type& t) : table(t) {}

   [[nodiscard]] action_policy_type at(const Infostate& infostate) const
   {
      if(auto found = table.find(infostate); found != table.end()) {
         return normalize_action_policy(found->second);
      }
      static const std::function< double() > half = [] { return 0.5; };
      return action_policy_type{half};
   }
   [[nodiscard]] action_policy_type operator()(const Infostate& infostate) const
   {
      return at(infostate);
   }
   [[nodiscard]] size_t size() const { return table.size(); }
};

template < typename Solver >
double exploitability_of_average_policies(Solver& solver)
{
   const auto& avg_policies = solver.average_policy();
   return nor::exploitability(
      Environment{},
      leduc::State{leduc::LeducConfig::big_leduc()},
      nor::player_hashmap< SparseAveragePolicy >{
         std::pair{
            nor::Player::alex, SparseAveragePolicy(avg_policies.at(nor::Player::alex).table())},
         std::pair{
            nor::Player::bob, SparseAveragePolicy(avg_policies.at(nor::Player::bob).table())}}
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

auto elapsed_ms(const std::chrono::steady_clock::time_point& start)
{
   return std::chrono::duration_cast< std::chrono::milliseconds >(
             std::chrono::steady_clock::now() - start
   )
      .count();
}

}  // namespace

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// Vanilla CFR (alternating) ///////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

/// One full-tree vanilla CFR iteration costs tens of seconds on the ~6.5M-history Big-Leduc
/// tree, hence the hard 60s wall-clock budget: run as many complete iterations as fit and
/// require the exploitability of the average strategy to strictly decrease over them.
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

   // exploitability evaluations are cheap relative to an iteration; record every iteration
   std::vector< double > expl_trace;
   size_t n_done = 0;
   for(size_t iter = 1; iter <= max_iterations; ++iter) {
      solver.iterate(1);
      n_done = iter;
      expl_trace.emplace_back(exploitability_of_average_policies(solver));
      if(elapsed_ms(start_time) >= wall_budget.count() * 1000ll) {
         break;
      }
   }

   std::cout << "BigLeduc VanillaCFR alternating: " << n_done << " iterations in "
             << elapsed_ms(start_time) << "ms | exploitability trace:";
   for(size_t i = 0; i < expl_trace.size(); ++i) {
      std::cout << " [" << i + 1 << "]=" << expl_trace[i];
   }
   std::cout << "\n";

   ASSERT_GE(expl_trace.size(), 3ul) << "wall-clock budget allowed too few iterations";
   ASSERT_TRUE(std::isfinite(expl_trace.front()));
   ASSERT_TRUE(std::isfinite(expl_trace.back()));
   EXPECT_LT(expl_trace.back(), expl_trace.front());
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

   auto table_coverage = [&solver] {
      const auto& avg_policies = solver.average_policy();
      return avg_policies.at(nor::Player::alex).table().size()
             + avg_policies.at(nor::Player::bob).table().size();
   };

   solver.iterate(1);
   // after a single sampled pass the average policy is essentially untrained: baseline
   const double expl_baseline = exploitability_of_average_policies(solver);
   const size_t coverage_baseline = table_coverage();

   const auto start_time = std::chrono::steady_clock::now();
   solver.iterate(n_iterations - 1);
   const double expl_end = exploitability_of_average_policies(solver);
   const size_t coverage_end = table_coverage();

   std::cout << "BigLeduc MCCFR external-sampling: " << n_iterations << " iterations in "
             << elapsed_ms(start_time) << "ms | exploitability @1: " << expl_baseline << " @"
             << n_iterations << ": " << expl_end << " | infosets covered " << coverage_baseline
             << " -> " << coverage_end << "\n";

   ASSERT_TRUE(std::isfinite(expl_baseline));
   ASSERT_TRUE(std::isfinite(expl_end));
   EXPECT_LT(expl_end, 0.95 * expl_baseline + 1e-12);
}
