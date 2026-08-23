
#ifndef NOR_BENCH_CFR_HPP
#define NOR_BENCH_CFR_HPP

#include <benchmark/benchmark.h>

#include "bench_core.hpp"
#include "nor/env.hpp"
#include "nor/nor.hpp"

namespace benchmarks {

using namespace nor;

template < auto config, typename Env, size_t nr_warmup_iters = 10 >
void cfr_bench(benchmark::State& state)
{
   for(auto _ : state) {
      state.SetIterationTime(cfr_ns_per_iter< config, Env >(nr_warmup_iters, 1) * 1e-9);
   }
   state.SetItemsProcessed(int64_t(state.iterations()));
}

/// the canonical benchmark gate configurations: full-tree vanilla CFR and
/// PCFR+ with alternating updates on kuhn and leduc poker
inline void (&CFR_VANILLA_alternating)(benchmark::State& state) = cfr_bench<
   rm::CFRConfig{.update_mode = rm::UpdateMode::alternating},
   games::kuhn::Environment >;  // function reference

inline void (&CFR_VANILLA_simultaneous)(benchmark::State& state) = cfr_bench<
   rm::CFRConfig{.update_mode = rm::UpdateMode::simultaneous},
   games::kuhn::Environment >;  // function reference

inline void (&CFR_VANILLA_LEDUC)(benchmark::State& state) = cfr_bench<
   rm::CFRConfig{.update_mode = rm::UpdateMode::alternating},
   games::leduc::Environment >;  // function reference

inline void (&PCFR_PLUS_KUHN)(benchmark::State& state
) = cfr_bench< k_pcfr_plus_config, games::kuhn::Environment >;

inline void (&PCFR_PLUS_LEDUC)(benchmark::State& state
) = cfr_bench< k_pcfr_plus_config, games::leduc::Environment >;

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
