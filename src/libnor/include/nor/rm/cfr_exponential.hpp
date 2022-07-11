
#ifndef NOR_CFR_EXPONENTIAL_HPP
#define NOR_CFR_EXPONENTIAL_HPP

#include "cfr_config.hpp"
#include "cfr_vanilla.hpp"

namespace nor::rm {

template < CFRExponentialConfig config, typename Env, typename Policy, typename AveragePolicy >
using CFRExponential = VanillaCFR<
   CFRConfig{
      .update_mode = config.update_mode,
      .regret_minimizing_mode = config.regret_minimizing_mode,
      .weighting_mode = CFRWeightingMode::exponential},
   Env,
   Policy,
   AveragePolicy >;

}

#endif  // NOR_CFR_EXPONENTIAL_HPP
