
#ifndef NOR_PLUSCFR_HPP
#define NOR_PLUSCFR_HPP

#include "cfr_config.hpp"
#include "vcfr.hpp"

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

#endif  // NOR_PLUSCFR_HPP
