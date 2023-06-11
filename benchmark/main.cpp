#include <benchmark/benchmark.h>

#include "bench_cfr.hpp"
#include "bench_mccfr.hpp"

BENCHMARK(benchmarks::CFR_VANILLA_alternating);
BENCHMARK(benchmarks::CFR_VANILLA_simultaneous);
BENCHMARK(benchmarks::CFR_LINEAR_alternating);
BENCHMARK(benchmarks::CFR_LINEAR_simultaneous);
BENCHMARK(benchmarks::CFR_DISCOUNTED_alternating);
BENCHMARK(benchmarks::CFR_DISCOUNTED_simultaneous);
BENCHMARK(benchmarks::CFR_EXPONENTIAL_alternating);
BENCHMARK(benchmarks::CFR_EXPONENTIAL_simultaneous);

BENCHMARK(benchmarks::CFR_PURE_alternating);
BENCHMARK(benchmarks::CFR_PURE_simultaneous);

BENCHMARK(benchmarks::MCCFR_OS_optimistic_alternating);
BENCHMARK(benchmarks::MCCFR_OS_optimistic_simultaneous);
BENCHMARK(benchmarks::MCCFR_OS_lazy_alternating);
BENCHMARK(benchmarks::MCCFR_OS_lazy_simultaneous);
BENCHMARK(benchmarks::MCCFR_OS_stochastic_alternating);
BENCHMARK(benchmarks::MCCFR_OS_stochastic_simultaneous);
BENCHMARK(benchmarks::MCCFR_ES_stochastic);
BENCHMARK(benchmarks::MCCFR_CS_alternating);
BENCHMARK(benchmarks::MCCFR_CS_simultaneous);

BENCHMARK_MAIN();
