
/// standalone (no google-benchmark, no conan) iteration-rate harness.
///
/// Measures iterations/sec of the benchmark-gate configurations
///    VanillaCFR / PCFR+ on kuhn poker and leduc poker
/// via benchmark/bench_core.hpp — the exact same construction and timing core
/// the google-benchmark target uses. Build remotely with the nor-build.sh
/// toolchain harness:
///    nor-build.sh -o bench_standalone -O3 benchmark/standalone_main.cpp

#include <fmt/core.h>

#include <cstdint>
#include <string_view>
#include <vector>

#include "bench_core.hpp"
#include "nor/env.hpp"
#include "nor/nor.hpp"

namespace {

using namespace nor;

struct bench_result {
   std::string_view name;
   double ns_per_iter;
   size_t iters_measured;
};

template < auto config, typename Env >
bench_result run(std::string_view name, size_t warmup_iters, size_t measured_iters)
{
   return {
      name,
      benchmarks::cfr_ns_per_iter< config, Env >(warmup_iters, measured_iters),
      measured_iters};
}

}  // namespace

int main()
{
   fmt::print("{:<28} {:>15} {:>15}\n", "benchmark", "ns/iter", "iter/sec");
   std::vector< bench_result > results;

   constexpr size_t kuhn_iters = 20000;  // tiny tree: many iterations needed
   results.push_back(
      run< rm::CFRConfig{.update_mode = rm::UpdateMode::alternating}, games::kuhn::Environment >(
         "kuhn/vanilla-cfr", 1000, kuhn_iters
      )
   );
   results.push_back(run< benchmarks::k_pcfr_plus_config, games::kuhn::Environment >(
      "kuhn/pcfr-plus", 1000, kuhn_iters
   ));

   constexpr size_t leduc_iters = 400;  // larger tree
   results.push_back(
      run< rm::CFRConfig{.update_mode = rm::UpdateMode::alternating}, games::leduc::Environment >(
         "leduc/vanilla-cfr", 20, leduc_iters
      )
   );
   results.push_back(run< benchmarks::k_pcfr_plus_config, games::leduc::Environment >(
      "leduc/pcfr-plus", 20, leduc_iters
   ));

   for(const auto& result : results) {
      fmt::print(
         "{:<28} {:>15.1f} {:>15.0f}\n", result.name, result.ns_per_iter, 1e9 / result.ns_per_iter
      );
   }
   return 0;
}
