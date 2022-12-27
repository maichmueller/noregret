
#ifndef NOR_CFR_DISCOUNTED_HPP
#define NOR_CFR_DISCOUNTED_HPP

#include "cfr_config.hpp"
#include "cfr_vanilla.hpp"

namespace nor::rm {

template < CFRDiscountedConfig config, typename Env, typename Policy, typename AveragePolicy >
using CFRDiscounted = VanillaCFR<
   CFRConfig{
      .update_mode = config.update_mode,
      .regret_minimizing_mode = config.regret_minimizing_mode,
      .weighting_mode = CFRWeightingMode::discounted},
   Env,
   Policy,
   AveragePolicy >;

template < CFRLinearConfig config, typename Env, typename Policy, typename AveragePolicy >
using CFRLinear = VanillaCFR<
   CFRConfig{
      .update_mode = config.update_mode,
      .regret_minimizing_mode = config.regret_minimizing_mode,
      .weighting_mode = CFRWeightingMode::discounted},
   Env,
   Policy,
   AveragePolicy >;

}

#endif  // NOR_CFR_DISCOUNTED_HPP
