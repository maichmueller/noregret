
#ifndef NOR_OPPONENT_MODEL_HPP
#define NOR_OPPONENT_MODEL_HPP

#include <algorithm>
#include <cmath>
#include <concepts>
#include <functional>
#include <ranges>
#include <stdexcept>
#include <unordered_map>

#include "common/common.hpp"
#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"

namespace nor::opponent_aware {

/**
 * Raw observation data of an opponent model: per-infostate action counts as recorded from
 * played hands (frequencies are derived lazily by normalization).
 */
template < typename InfoState, typename Action >
using FrequencyTable = std::unordered_map<
   InfoState,
   std::unordered_map< Action, double, common::value_hasher< Action > >,
   common::value_hasher< InfoState >,
   common::value_comparator< InfoState > >;

/// total number of recorded actions at 'infostate' (0 when the infostate was never observed)
template < typename InfoState, typename Action >
[[nodiscard]] inline double
observation_count(const FrequencyTable< InfoState, Action >& counts, const InfoState& infostate)
{
   const auto found = counts.find(infostate);
   return found == counts.end()
             ? 0.
             : std::ranges::fold_left(found->second | std::views::values, double(0.), std::plus{});
}

/// normalized empirical action distribution at 'infostate'; throws when nothing was observed or
/// the distribution does not sum to a positive mass. The result covers exactly the RECORDED
/// actions -- use the actions-aware overload below for full legal-action coverage.
template < typename InfoState, typename Action >
[[nodiscard]] inline std::unordered_map< Action, double, common::value_hasher< Action > >
normalized_frequencies(
   const FrequencyTable< InfoState, Action >& counts,
   const InfoState& infostate
)
{
   const auto found = counts.find(infostate);
   if(found == counts.end()) {
      throw std::invalid_argument("opponent_aware: no observation data for the requested infostate."
      );
   }
   const double total = observation_count(counts, infostate);
   if(not (total > 0.)) {
      throw std::invalid_argument(
         "opponent_aware: observation data at the requested infostate sums to zero."
      );
   }
   auto distribution = found->second;
   for(double& prob : distribution | std::views::values) {
      prob /= total;
   }
   return distribution;
}

/**
 * Normalized empirical distribution over the LEGAL actions of 'infostate': every unobserved
 * action receives probability 0 (the maximum-likelihood estimate subject to the support
 * constraints), and 'actions' must cover everything that was ever recorded there.
 */
template < typename InfoState, typename Action >
[[nodiscard]] inline std::unordered_map< Action, double, common::value_hasher< Action > >
normalized_frequencies(
   const FrequencyTable< InfoState, Action >& counts,
   const InfoState& infostate,
   const std::vector< Action >& actions
)
{
   auto raw = normalized_frequencies(counts, infostate);
   std::unordered_map< Action, double, common::value_hasher< Action > > distribution{};
   for(const auto& action : actions) {
      if(auto found = raw.find(action); found != raw.end()) {
         distribution.emplace(action, found->second);
      } else {
         // unobserved action: part of the modeled strategy's support constraint, just never
         // taken in the data
         distribution.emplace(action, 0.);
      }
   }
   return distribution;
}

/**
 * An OPPONENT MODEL in the sense of Johanson, Zinkevich & Bowling (NeurIPS 2007) and Johanson &
 * Bowling (AISTATS 2009): a behavioral policy guess 'policy' for the modeled player plus a
 * per-infostate CONFIDENCE weight 'confidence' in [0, 1] expressing how much the model is
 * trusted at that infostate.
 *
 * Both members are callables so that models can be backed by fixed tables, frequency tables or
 * fully procedural constructions alike:
 *   - policy(infostate, actions) must return the FULL model distribution over the given legal
 *     actions; it is consulted only where confidence > 0.
 *   - confidence(infostate) returns Pconf(I) (RNR: constant p; DBR: an observation-count-driven
 *     function).
 */
template < typename InfoState, typename Action >
struct OpponentModel {
   using info_state_type = InfoState;
   using action_type = Action;
   using distribution_type = std::unordered_map< Action, double, common::value_hasher< Action > >;

   std::function< distribution_type(const InfoState&, const std::vector< Action >&) > policy{};
   std::function< double(const InfoState&) > confidence{};
};

/**
 * A fixed-table opponent model trusted uniformly: the RNR setting of a fully specified model
 * sigma_fix blended with weight p everywhere ('confidence' == p). The table must provide a
 * distribution for EVERY infostate of the modeled player that can carry positive confidence;
 * lookups go through .at() so missing entries fail loudly.
 *
 * 'model_policy' may be any type exposing .at(infostate) whose mapped type exposes
 * .at(action) -> double (e.g. nor::TabularPolicy over HashmapActionPolicy, or a raw
 * map-of-maps).
 */
template < typename InfoState, typename Action, typename ModelPolicyTable >
[[nodiscard]] auto make_fixed_opponent_model(const ModelPolicyTable& model_policy, double p)
{
   if(p < 0. or p > 1.) {
      throw std::invalid_argument("make_fixed_opponent_model: p must lie in [0, 1].");
   }
   OpponentModel< InfoState, Action > model{};
   model.policy = [&model_policy](
                     const InfoState& infostate, const std::vector< Action >& actions
                  ) -> std::unordered_map< Action, double, common::value_hasher< Action > > {
      auto&& action_policy = model_policy.at(infostate);
      std::unordered_map< Action, double, common::value_hasher< Action > > distribution;
      double sum = 0.;
      for(const auto& action : actions) {
         const double prob = action_policy.at(action);
         distribution.emplace(action, prob);
         sum += prob;
      }
      if(std::abs(sum - 1.) > 1e-6) {
         throw std::invalid_argument(
            "make_fixed_opponent_model: the model distribution does not sum to one at the "
            "requested infostate."
         );
      }
      return distribution;
   };
   model.confidence = [p](const InfoState&) { return p; };
   return model;
}

/**
 * A frequency-table-backed DBR opponent model (Johanson & Bowling, AISTATS 2009, sec. 3/5):
 * at every infostate the empirical action frequencies stand in for the model distribution
 * (maximum-likelihood given the observations; unobserved legal actions get mass 0). Wherever
 * nothing has been observed the confidence functions below report 0, so the DEFAULT policy of
 * the paper never has to fill in here -- the modeled player is simply left free.
 *
 * 'confidence_fn' is forwarded as-is; use one of the factories below to build it from an
 * observation-count-driven shape.
 */
template < typename InfoState, typename Action, typename ConfidenceFn >
[[nodiscard]] auto make_frequency_opponent_model(
   const FrequencyTable< InfoState, Action >& counts,
   ConfidenceFn&& confidence_fn
)
{
   OpponentModel< InfoState, Action > model{};
   model.policy = [&counts](
                     const InfoState& infostate, const std::vector< Action >& actions
                  ) -> std::unordered_map< Action, double, common::value_hasher< Action > > {
      if(not counts.contains(infostate)) {
         throw std::invalid_argument(
            "make_frequency_opponent_model: consulted the model at an unobserved infostate; "
            "the entry-point short-circuits zero-confidence infostates precisely to avoid "
            "this."
         );
      }
      return normalized_frequencies(counts, infostate, actions);
   };
   model.confidence = std::forward< ConfidenceFn >(confidence_fn);
   return model;
}

/**
 * Confidence functions Pconf(I) of the DBR paper (Johanson & Bowling, AISTATS 2009, sec. 5):
 * each maps the number n_I of observations at infostate I into [0, Pmax]. All of them return 0
 * for unobserved infostates, which frees the modeled player entirely there (the modified game
 * plays towards a Nash equilibrium in unobserved territory).
 */

/// "n-Step": Pconf(I) = Pmax iff n_I >= min_observations (the paper's 1-Step and 10-Step
/// variants -- the same tendency with different thresholds)
[[nodiscard]] inline auto step_confidence(double p_max, double min_observations = 1.)
{
   if(p_max < 0. or p_max > 1.) {
      throw std::invalid_argument("step_confidence: p_max must lie in [0, 1].");
   }
   return [p_max, min_observations](double n_observations) {
      return n_observations >= min_observations ? p_max : 0.;
   };
}

/// "0-10 Linear": Pconf(I) ramps linearly from 0 at n_I = 0 to Pmax at n_I = ramp (the paper's
/// 0-10 Linear uses ramp = 10)
[[nodiscard]] inline auto linear_confidence(double p_max, double ramp)
{
   if(ramp <= 0.) {
      throw std::invalid_argument("linear_confidence: the ramp length must be positive.");
   }
   if(p_max < 0. or p_max > 1.) {
      throw std::invalid_argument("linear_confidence: p_max must lie in [0, 1].");
   }
   return
      [p_max, ramp](double n_observations) { return std::min(n_observations / ramp, 1.) * p_max; };
}

/// "s-Curve": Pconf(I) = Pmax * n_I / (s + n_I); the Bayesian-interpretation choice whose
/// posterior reading is exact in the paper's sec. 7 (their experiments use s = 1)
[[nodiscard]] inline auto scurve_confidence(double p_max, double s)
{
   if(s <= 0.) {
      throw std::invalid_argument("scurve_confidence: the prior strength s must be positive.");
   }
   if(p_max < 0. or p_max > 1.) {
      throw std::invalid_argument("scurve_confidence: p_max must lie in [0, 1].");
   }
   return
      [p_max, s](double n_observations) { return p_max * n_observations / (s + n_observations); };
}

}  // namespace nor::opponent_aware

#endif  // NOR_OPPONENT_MODEL_HPP
