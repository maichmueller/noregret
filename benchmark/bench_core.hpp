
#ifndef NOR_BENCH_CORE_HPP
#define NOR_BENCH_CORE_HPP

#include <chrono>
#include <cstdint>
#include <memory>
#include <type_traits>

#include "nor/env.hpp"
#include "nor/factory.hpp"
#include "nor/nor.hpp"

namespace benchmarks {

using namespace nor;

/// PCFR+ gate configuration: alternating updates, predictive regret matching
/// plus, discounted weighting carrier (quadratic average-policy accumulation)
inline constexpr rm::CFRDiscountedConfig k_pcfr_plus_config{
   .update_mode = rm::UpdateMode::alternating,
   .regret_minimizing_mode = rm::RegretMinimizingMode::predictive_regret_matching_plus};

/// environment-agnostic CFR timing core.
///
/// Constructs a solver from 'config' via the factory (identical construction
/// path as production code), performs 'warmup_iters' iterations to trigger all
/// lazy allocations and then measures 'measured_iters' further iterations.
/// Returns the average wall-clock nanoseconds per iteration.
///
/// This header is deliberately free of any google-benchmark dependency so it
/// can also be compiled by the standalone benchmark harness.
template < auto config, typename Env >
double cfr_ns_per_iter(size_t warmup_iters, size_t measured_iters)
{
   using env_type = std::remove_cvref_t< Env >;

   auto root_state = std::make_unique< auto_world_state_type< env_type > >();

   auto avg_tabular_policy = factory::make_tabular_policy(
      std::unordered_map<
         auto_info_state_type< env_type >,
         HashmapActionPolicy< auto_action_type< env_type > > >{}
   );
   auto tabular_policy = factory::make_tabular_policy(
      std::unordered_map<
         auto_info_state_type< env_type >,
         HashmapActionPolicy< auto_action_type< env_type > > >{}
   );

   auto solver = factory::make_cfr< config, true >(
      env_type{}, std::move(root_state), tabular_policy, avg_tabular_policy
   );

   if(warmup_iters > 0) {
      // iterate a few rounds to assure all necessary allocations have been made
      solver.iterate(warmup_iters);
   }

   const auto start = std::chrono::steady_clock::now();
   for(size_t i : std::views::iota(size_t(0), measured_iters)) {
      (void) i;
      solver.iterate(1);
   }
   const auto stop = std::chrono::steady_clock::now();
   return double(std::chrono::duration_cast< std::chrono::nanoseconds >(stop - start).count())
          / double(measured_iters);
}

}  // namespace benchmarks

#endif  // NOR_BENCH_CORE_HPP
