
#ifndef NOR_CFR_CONFIG_HPP
#define NOR_CFR_CONFIG_HPP

namespace nor::rm {

enum class RegretMinimizingMode { regret_matching = 0, regret_matching_plus = 1 };

enum class UpdateMode { simultaneous = 0, alternating = 1 };

enum class CFRWeightingMode { uniform = 0, linear = 1 };

struct CFRConfig {
   UpdateMode update_mode = UpdateMode::alternating;
   RegretMinimizingMode regret_minimizing_mode = RegretMinimizingMode::regret_matching;
   CFRWeightingMode weighting_mode = CFRWeightingMode::uniform;
};

struct CFRPlusConfig {
   UpdateMode update_mode = UpdateMode::alternating;
};

enum class MCCFRAlgorithmMode { outcome_sampling = 0, external_sampling = 1 };

enum class MCCFRWeightingMode { lazy = 0, optimistic = 1, stochastic = 2 };

enum class MCCFRExplorationMode { epsilon_on_policy = 0, sampling_policy = 1 };

struct MCCFRConfig {
   UpdateMode update_mode = UpdateMode::alternating;
   MCCFRAlgorithmMode algorithm = MCCFRAlgorithmMode::outcome_sampling;
   MCCFRExplorationMode exploration = MCCFRExplorationMode::epsilon_on_policy;
   MCCFRWeightingMode weighting = MCCFRWeightingMode::lazy;
};

}  // namespace nor::rm

#endif  // NOR_CFR_CONFIG_HPP
