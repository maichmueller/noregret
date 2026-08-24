
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

///////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// B7: PCS / AS rules ///////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief B7 tag-concept: rules steering CHANCE-outcome resolution (PCS).
 *
 * A rule satisfying this concept is recognized by the MCCFR outcome-sampling
 * traversal and replaces the vanilla chance draw at every chance node; the
 * ACTION-side exploration remains governed by 'config.exploration'.
 */
template < typename Rule >
concept public_chance_sampling_rule = std::copy_constructible< Rule > and requires {
   typename Rule::is_public_chance_sampling_rule;
};

/**
 * @brief B7 tag-concept: action-sampling rules drawing from the ACCUMULATED
 * average strategy.
 *
 * Rules satisfying this concept are invoked with an average-strategy probe
 * functor instead of a current-policy probe functor (protocol extension of
 * 'sampling_rule_for'; see AverageStrategySamplingRule).
 */
template < typename Rule >
concept average_strategy_sampling_rule = std::copy_constructible< Rule > and requires {
   typename Rule::is_average_strategy_sampling_rule;
};

template < typename Rule, typename Action, typename AvgWeightOf >
concept average_strategy_sampling_rule_for = average_strategy_sampling_rule< Rule >
                                             and requires(
                                                const Rule rule,
                                                common::RNG& rng,
                                                const std::vector< Action >& actions,
                                                const AvgWeightOf& avg_weight_of
                                             ) {
                                                    {
                                                       rule(rng, actions, avg_weight_of)
                                                    } -> std::convertible_to<
                                                       SampledChoice< Action > >;
                                                 };

/**
 * @brief B7: Public Chance Sampling chance-node marker rule.
 *
 * Gibson, Lanctot, Burch, Szafron, Bowling (AAAI 2012, DOI 10.1609/aaai.v26i1.8241):
 * sample chance outcomes ONLY at public chance events; private deals are resolved
 * deterministically. Injection: pass as the trailing SamplingRule template/ctor
 * parameter of rm::MCCFR together with
 * MCCFRAlgorithmMode::outcome_sampling (enforced by a sanity guard). The
 * game-side classification is provided per environment via
 * nor::concepts::has::public_chance_event(env, wstate, outcome); environments
 * without the trait fall back to "all chance public", which makes PCS
 * degenerate exactly to vanilla outcome sampling (bit-identical draws).
 *
 * ENGINE DEVIATION NOTE (single- vs multi-view): Gibson's PCS traverses one
 * tree PER PLAYER VIEW in which private deals are treated as certain for the
 * observing player and re-drawn from the prior across iterations. noregret's
 * MCCFR follows ONE world-state trajectory, so this implementation resolves a
 * private chance event deterministically to its FIRST legal outcome and threads
 * the outcome's TRUE chance probability into the sample-probability accumulator
 * (the importance-weight denominator), while reach bookkeeping is unchanged.
 * Consequences, documented honestly:
 *  - On games whose chance events are all public (or deterministic support-1
 *    events like goofspiel's resolve confirmation) this IS vanilla OS:
 *    unbiased, bit-identical trajectories.
 *  - On genuinely stochastic private deals the single trajectory freezes the
 *    deal to the first outcome: the estimator keeps zero per-deal variance but
 *    no longer averages over deals, so regret unbiasedness holds ONLY under
 *    the restricted claim above (validated empirically on goofspiel k=4;
 *    liars dice runs are smoke-tested for mechanical soundness only).
 */
struct PublicChanceSamplingRule {
   using is_public_chance_sampling_rule = std::true_type;
};

/**
 * @brief B7: Average Strategy Sampling rule (Gibson, Burch, Bowling, "Efficient
 * Monte Carlo CFR in Games with Many Player Actions", NIPS 2012).
 *
 * The updating player samples own actions proportional to the ACCUMULATED
 * average strategy sigma_bar(I, .) instead of the current recommendation.
 * Injection: pass as the trailing SamplingRule ctor/template parameter of
 * rm::MCCFR with config.exploration ==
 * MCCFRExplorationMode::custom_sampling_policy and
 * config.algorithm == MCCFRAlgorithmMode::outcome_sampling (both enforced by
 * sanity guards). Placement mirrors the epsilon-on-policy scheme: the rule acts
 * at the updating player's infosets (at every actual player's infosets under
 * simultaneous updates); opponent infosets stay on-policy over the CURRENT
 * strategy, exactly as in the paper's ASS formulation for OS-MCCFR.
 *
 * PROTOCOL CHANGE vs 'sampling_rule_for' (B6): rules tagged with
 * 'is_average_strategy_sampling_rule' receive a probe functor evaluating the
 * AVERAGE-policy table (solver-side fetch_policy<PolicyLabel::average>) rather
 * than the current-policy lookup. Untagged rules keep the original protocol.
 *
 * Deviation note: the paper derives tighter per-iteration guarantees from
 * weighting updates with AVERAGE-strategy reaches; this implementation adopts
 * only ASS's sampling distribution and retains the engine's current-reach
 * bookkeeping, so the estimator stays unbiased (any positive sampling measure q
 * yields E[pi_-i(h)u(z)/q(z)] = counterfactual value) while the tightened bound
 * is NOT claimed here.
 */
struct AverageStrategySamplingRule {
   using is_average_strategy_sampling_rule = std::true_type;

   template < typename Action, typename AvgWeightOf >
   SampledChoice< Action > operator()(
      common::RNG& rng,
      const std::vector< Action >& actions,
      const AvgWeightOf& average_strategy_prob_of
   ) const
   {
      std::vector< double > weights{};
      weights.reserve(actions.size());
      double total = 0.;
      for(const auto& action : actions) {
         const double weight = std::max(0., average_strategy_prob_of(action));
         weights.emplace_back(weight);
         total += weight;
      }
      if(not (total > 0.)) {
         // cold start: the accumulated average table is still empty -> uniform
         std::fill(weights.begin(), weights.end(), 1.);
         total = static_cast< double >(actions.size());
      }
      std::discrete_distribution< size_t > draw(weights.begin(), weights.end());
      const size_t index = draw(rng);
      return {actions[index], weights[index] / total};
   }
};

static_assert(
   average_strategy_sampling_rule_for< AverageStrategySamplingRule, int, decltype([](const int&) {
                                          return 1.;
                                       }) >,
   "AverageStrategySamplingRule must satisfy the average-strategy rule protocol"
);
static_assert(
   sampling_rule_for< AverageStrategySamplingRule, int, decltype([](const int&) { return 1.; }) >,
   "AverageStrategySamplingRule must also satisfy the base sampling-rule protocol "
   "(same call arity, different probe semantics)"
);
static_assert(public_chance_sampling_rule< PublicChanceSamplingRule >);
static_assert(average_strategy_sampling_rule< AverageStrategySamplingRule >);

}  // namespace nor::rm

#endif  // NOR_RM_SAMPLING_RULES_HPP
