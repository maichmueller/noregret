
#ifndef NOR_CFR_CONFIG_HPP
#define NOR_CFR_CONFIG_HPP

namespace nor::rm {

enum class RegretMinimizingMode { regret_matching = 0, regret_matching_plus = 1 };

enum class UpdateMode { simultaneous = 0, alternating = 1 };

enum class CFRWeightingMode {
   // no particular weighting scheme applied to updates of regret or average policy.
   // Both are incremented by unweighted increments
   uniform = 0,
   // The average policy is being incremented by the weight 't' in iteration 't'
   linear = 1,
   // Both the regret and average policy are updated by the weights
   // t^alpha / (t^alpha +1), t^beta / (t^beta + 1), (t / t+1)^gamma
   discounted = 2,
   // The regret and average policy are weighted by an L1 factor: L1(I, a) = r(I,a) - E[v(I)]
   // where r(I,a) is the instantaneous regret and E[v(I)] is the expected value of the infostate
   exponential = 3
};

struct CFRConfig {
   UpdateMode update_mode = UpdateMode::alternating;
   RegretMinimizingMode regret_minimizing_mode = RegretMinimizingMode::regret_matching;
   CFRWeightingMode weighting_mode = CFRWeightingMode::uniform;
};

struct CFRPlusConfig {
   UpdateMode update_mode = UpdateMode::alternating;
};

struct CFRDiscountedConfig {
   UpdateMode update_mode = UpdateMode::alternating;
   RegretMinimizingMode regret_minimizing_mode = RegretMinimizingMode::regret_matching;
};

struct CFRExponentialConfig {
   UpdateMode update_mode = UpdateMode::alternating;
   RegretMinimizingMode regret_minimizing_mode = RegretMinimizingMode::regret_matching;
};

enum class MCCFRAlgorithmMode {
   // sample only the chance players action according to the chance distribution
   chance_sampling = 0,
   // sample only a single trajectory of the game tree and update the policies
   outcome_sampling = 1,
   // traverse each action of a traversing player, but sample only a single action of each opponent
   // and chance player
   external_sampling = 2
};

enum class MCCFRWeightingMode {
   // the lazy update scheme is the correct average policy update scheme which maintains a table of
   // unsampled action policy values that are pushed alongside once such an action is sampled
   lazy = 0,
   // optimistic updates the average action policy by weighting the current increment with the delay
   // in number of iterations (t-c) this action has not been sampled and updated last
   optimistic = 1,
   // stochastic updates the average policy by weighting the current increment with the reciprocal
   // sampled action sample-probability
   stochastic = 2
};

enum class MCCFRExplorationMode {
   // explore the action space by drawing a legal action uniformly with probability epsilon and
   // otherwise sampling according to the current action policy
   epsilon_on_policy = 0,
   // sample via a custom sampling policy
   // TODO: implement this method
   custom_sampling_policy = 1
};

struct MCCFRConfig {
   UpdateMode update_mode = UpdateMode::alternating;
   MCCFRAlgorithmMode algorithm = MCCFRAlgorithmMode::outcome_sampling;
   MCCFRExplorationMode exploration = MCCFRExplorationMode::epsilon_on_policy;
   MCCFRWeightingMode weighting = MCCFRWeightingMode::lazy;
   RegretMinimizingMode regret_minimizing_mode = RegretMinimizingMode::regret_matching;
};

}  // namespace nor::rm

#endif  // NOR_CFR_CONFIG_HPP
