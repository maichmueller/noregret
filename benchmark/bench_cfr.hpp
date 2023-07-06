

#ifndef NOR_BENCH_CFR_HPP
#define NOR_BENCH_CFR_HPP

#include <benchmark/benchmark.h>

#include "nor/env.hpp"
#include "nor/nor.hpp"

namespace benchmarks {

using namespace nor;

template < auto config, typename Env, size_t nr_warmup_iters = 10 >
void cfr_bench(benchmark::State& state)
{
   using env = std::remove_cvref_t< Env >;

   auto root_state = WorldstateHolder< auto_world_state_type< env > >{};

   auto avg_tabular_policy = factory::make_tabular_policy<
      auto_info_state_type< env >,
      HashmapActionPolicy< auto_action_type< env > > >();

   auto tabular_policy = factory::make_tabular_policy<
      auto_info_state_type< env >,
      HashmapActionPolicy< auto_action_type< env > > >();

   auto solver = factory::make_cfr< config, true >(
      env{}, std::move(root_state), tabular_policy, avg_tabular_policy
   );
   if constexpr(nr_warmup_iters > 0) {
      // iterate a few rounds to assure all necessary allocations have been made
      solver.iterate(nr_warmup_iters);
   }

   for(auto _ : state) {
      solver.iterate(1);
   }
}

inline void (&CFR_VANILLA_alternating)(benchmark::State& state) = cfr_bench<
   rm::CFRConfig{.update_mode = rm::UpdateMode::alternating},
   games::kuhn::Environment >;  // function reference

inline void (&CFR_VANILLA_simultaneous)(benchmark::State& state) = cfr_bench<
   rm::CFRConfig{.update_mode = rm::UpdateMode::simultaneous},
   games::kuhn::Environment >;  // function reference

inline void (&CFR_LINEAR_alternating)(benchmark::State& state) = cfr_bench<
   rm::CFRLinearConfig{.update_mode = rm::UpdateMode::alternating},
   games::kuhn::Environment >;  // function reference

inline void (&CFR_LINEAR_simultaneous)(benchmark::State& state) = cfr_bench<
   rm::CFRLinearConfig{.update_mode = rm::UpdateMode::simultaneous},
   games::kuhn::Environment >;  // function reference

inline void (&CFR_DISCOUNTED_alternating)(benchmark::State& state) = cfr_bench<
   rm::CFRDiscountedConfig{.update_mode = rm::UpdateMode::alternating},
   games::kuhn::Environment >;  // function reference

inline void (&CFR_DISCOUNTED_simultaneous)(benchmark::State& state) = cfr_bench<
   rm::CFRDiscountedConfig{.update_mode = rm::UpdateMode::simultaneous},
   games::kuhn::Environment >;  // function reference

inline void (&CFR_EXPONENTIAL_alternating)(benchmark::State& state) = cfr_bench<
   rm::CFRExponentialConfig{.update_mode = rm::UpdateMode::alternating},
   games::kuhn::Environment >;  // function reference

inline void (&CFR_EXPONENTIAL_simultaneous)(benchmark::State& state) = cfr_bench<
   rm::CFRExponentialConfig{.update_mode = rm::UpdateMode::simultaneous},
   games::kuhn::Environment >;  // function reference

}  // namespace benchmarks

#endif  // NOR_BENCH_CFR_HPP
