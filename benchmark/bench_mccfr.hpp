

#ifndef NOR_BENCH_MCCFR_HPP
#define NOR_BENCH_MCCFR_HPP

#include "nor/nor.hpp"

namespace benchmarks {

using namespace nor;

template < auto config, typename Env, size_t nr_warmup_iters = 1000 >
void mccfr_bench(benchmark::State& state)
{
   using env = std::remove_cvref_t< Env >;

   auto root_state = WorldstateHolder< auto_world_state_type< env > >();

   auto avg_tabular_policy = factory::make_tabular_policy<
      auto_info_state_type< env >,
      HashmapActionPolicy< auto_action_type< env > > >();

   auto tabular_policy = factory::make_tabular_policy<
      auto_info_state_type< env >,
      HashmapActionPolicy< auto_action_type< env > > >();

   auto solver = factory::make_mccfr< config, true >(
      env{}, root_state, tabular_policy, avg_tabular_policy, 0.5
   );
   if constexpr(nr_warmup_iters > 0) {
      // iterate a few rounds to assure all necessary allocations have been made
      solver.iterate(nr_warmup_iters);
   }

   for(auto _ : state) {
      solver.iterate(1);
   }
}

inline void (&MCCFR_OS_optimistic_alternating)(benchmark::State& state) = mccfr_bench<
   rm::MCCFRConfig{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::optimistic},
   games::kuhn::Environment >;  // function reference

inline void (&MCCFR_OS_optimistic_simultaneous)(benchmark::State& state) = mccfr_bench<
   rm::MCCFRConfig{
      .update_mode = rm::UpdateMode::simultaneous,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::optimistic},
   games::kuhn::Environment >;  // function reference

inline void (&MCCFR_OS_lazy_alternating)(benchmark::State& state) = mccfr_bench<
   rm::MCCFRConfig{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::lazy},
   games::kuhn::Environment >;  // function reference

inline void (&MCCFR_OS_lazy_simultaneous)(benchmark::State& state) = mccfr_bench<
   rm::MCCFRConfig{
      .update_mode = rm::UpdateMode::simultaneous,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::lazy},
   games::kuhn::Environment >;  // function reference

inline void (&MCCFR_OS_stochastic_alternating)(benchmark::State& state) = mccfr_bench<
   rm::MCCFRConfig{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::stochastic},
   games::kuhn::Environment >;  // function reference

inline void (&MCCFR_OS_stochastic_simultaneous)(benchmark::State& state) = mccfr_bench<
   rm::MCCFRConfig{
      .update_mode = rm::UpdateMode::simultaneous,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::stochastic},
   games::kuhn::Environment >;  // function reference

inline void (&MCCFR_ES_stochastic)(benchmark::State& state) = mccfr_bench<
   rm::MCCFRConfig{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::external_sampling,
      .weighting = rm::MCCFRWeightingMode::stochastic},
   games::kuhn::Environment >;  // function reference

inline void (&MCCFR_CS_alternating)(benchmark::State& state) = mccfr_bench<
   rm::MCCFRConfig{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::chance_sampling,
      .weighting = rm::MCCFRWeightingMode::none},
   games::kuhn::Environment >;  // function reference

inline void (&MCCFR_CS_simultaneous)(benchmark::State& state) = mccfr_bench<
   rm::MCCFRConfig{
      .update_mode = rm::UpdateMode::simultaneous,
      .algorithm = rm::MCCFRAlgorithmMode::chance_sampling,
      .weighting = rm::MCCFRWeightingMode::none},
   games::kuhn::Environment >;  // function reference

inline void (&CFR_PURE_alternating)(benchmark::State& state) = mccfr_bench<
   rm::MCCFRConfig{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::pure_cfr,
      .weighting = rm::MCCFRWeightingMode::none},
   games::kuhn::Environment >;  // function reference

inline void (&CFR_PURE_simultaneous)(benchmark::State& state) = mccfr_bench<
   rm::MCCFRConfig{
      .update_mode = rm::UpdateMode::simultaneous,
      .algorithm = rm::MCCFRAlgorithmMode::pure_cfr,
      .weighting = rm::MCCFRWeightingMode::none},
   games::kuhn::Environment >;  // function reference

}  // namespace benchmarks

#endif  // NOR_BENCH_MCCFR_HPP
