
#ifndef NOR_RM_SAMPLING_RULES_HPP
#define NOR_RM_SAMPLING_RULES_HPP

#include <cmath>
#include <random>
#include <utility>
#include <vector>

#include "common/common.hpp"

namespace nor::rm {

/**
 * @brief B6 (sampling-rule hook): injectable action-sampling rules for MCCFR.
 *
 * A sampling rule draws one action from 'actions' and reports the PROBABILITY
 * with which it was sampled (importance weight denominator). Rules are
 * injected as an extra template parameter / constructor argument of rm::MCCFR;
 * they take effect when the config selects
 * MCCFRExplorationMode::custom_sampling_policy. Consumers: ESCHER / bandit
 * agents.
 */
template < typename Action >
struct SampledChoice {
   Action action;
   double sample_prob;
};

template < typename Rule, typename Action, typename WeightOf >
concept sampling_rule_for = std::copy_constructible< Rule >
                            and requires(
                               const Rule rule,
                               common::RNG& rng,
                               const std::vector< Action >& actions,
                               const WeightOf& weight_of
                            ) {
                                   {
                                      rule(rng, actions, weight_of)
                                   } -> std::convertible_to< SampledChoice< Action > >;
                                };

/// the DEFAULT rule: epsilon-on-policy mixture. With probability epsilon a
/// uniformly random action is taken, otherwise an action is drawn from the
/// current policy; the reported sampling probability is always the mixture
///    epsilon * uniform(A(I)) + (1 - epsilon) * policy(I, a).
/// This reproduces the historical hard-coded behavior draw-for-draw.
struct EpsilonOnPolicySamplingRule {
   double epsilon = 0.6;

   template < typename Action, typename WeightOf >
   SampledChoice< Action > operator()(
      common::RNG& rng,
      const std::vector< Action >& actions,
      const WeightOf& policy_prob_of
   ) const
   {
      const double uniform_prob = 1. / static_cast< double >(actions.size());
      std::uniform_real_distribution< double > uniform01{0., 1.};
      if(uniform01(rng) < epsilon) {
         const Action& chosen = common::choose(actions, rng);
         return {chosen, epsilon * uniform_prob + (1. - epsilon) * policy_prob_of(chosen)};
      }
      const Action& chosen = common::choose(
         actions, [&](const Action& act) { return policy_prob_of(act); }, rng
      );
      return {chosen, epsilon * uniform_prob + (1. - epsilon) * policy_prob_of(chosen)};
   }
};

}  // namespace nor::rm

#endif  // NOR_RM_SAMPLING_RULES_HPP
