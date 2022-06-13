
#ifndef NOR_CFR_PLUS_HPP
#define NOR_CFR_PLUS_HPP

#include "cfr_config.hpp"
#include "cfr_vanilla.hpp"

namespace nor::rm {

template < typename Env, typename Policy, typename AveragePolicy >
using CFRPlus = VanillaCFR<
   CFRConfig{
      .update_mode = UpdateMode::alternating,
      .regret_minimizing_mode = RegretMinimizingMode::regret_matching_plus,
      .weighting_mode = CFRWeightingMode::uniform},
   Env,
   Policy,
   AveragePolicy >;

}

#endif  // NOR_CFR_PLUS_HPP
