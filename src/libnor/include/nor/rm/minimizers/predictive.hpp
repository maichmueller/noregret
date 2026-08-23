
#ifndef NOR_RM_MINIMIZERS_PREDICTIVE_HPP
#define NOR_RM_MINIMIZERS_PREDICTIVE_HPP

// NOTE: this header relies on 'per_action_map' which is defined in
// nor/rm/minimizers/minimizers.hpp BEFORE it includes this file. Include the
// former instead of this file directly.

#include <concepts>
#include <cstddef>

namespace nor::rm {

/**
 * @brief predictive regret matching plus (PCFR+) regret minimizer.
 *
 * Implements the per-infostate regret matching step of PCFR+ (Farina, Kroer,
 * Sandholm — "Faster Game Solving via Predictive Blackwell Approachability",
 * AAAI 2021, arXiv:2007.14358, Algorithm 5) and its SAPCFR+ robustification
 * (arXiv:2503.12770).
 *
 * In addition to the cumulative regret table z(I,a) the minimizer keeps two
 * auxiliary per-action tables:
 *
 *    instant_regret     rho(I,a): buffer of the instantaneous counterfactual
 *                                 regrets of the CURRENT iteration. observe()
 *                                 accumulates into it (an infostate generally
 *                                 contains many histories, each contributing
 *                                 one observe() call per action), and
 *                                 recommend() consumes (= resets) it after
 *                                 use, so that at every recommendation rho
 *                                 holds exactly the last fully observed
 *                                 instantaneous regret vector r^{t}(I, . ).
 *    strategy_snapshot  sigma_snap(I,a): copy of the distribution computed by
 *                                 the LAST recommend() call, i.e. the strategy
 *                                 that generated the regret increments now
 *                                 buffered in rho. Maintained per the PRM+
 *                                 protocol (it parameterizes the predicted
 *                                 Blackwell payoff v^t); the persistence
 *                                 predictor used here cancels it analytically,
 *                                 but keeping it preserves the protocol for
 *                                 future predictor variants.
 *
 * The recommendation implements PRM+ (Algorithm 5 of arXiv:2007.14358):
 *    theta(a) = max(0, clip(z)(a) + s * rho(a))
 *    sigma    = theta / sum(theta)
 * with the STORED cumulative table clipped to non-negative values each
 * iteration (RM+ forgetting, Algorithm 5 line 7) and the clamp of theta
 * applied after adding the prediction term. The prediction source is the
 * persistence estimate m^t = l^{t-1}; the corresponding predicted Blackwell
 * payoff v^t = <m^t, x^{t-1}>1 - m^t collapses analytically to the last
 * observed instantaneous regret vector rho (this is the principled version of
 * 'counting the last regret vector twice', cf. the remark on Brown &
 * Sandholm's heuristic in section 7 of the paper). For the SAP variant only
 * the prediction term is damped, s = 1/(1 + alpha) with alpha = 2
 * (arXiv:2503.12770); otherwise s = 1.
 *
 * Integration contract with rm::VanillaCFR: since recommend() copies its
 * output distribution into sigma_snap BEFORE returning and the solver calls
 * observe() (traversal) strictly before recommend() (end-of-iteration regret
 * minimization) within each iteration, the pairing
 *    (sigma_snap, rho) = (sigma^{t-1}, r^{t-1})
 * required by Algorithm 5 is reproduced without any solver-side changes.
 * The very first recommendation degenerates gracefully: all tables are zero
 * initialized, hence theta vanishes identically and sigma^1 falls back to the
 * uniform distribution (the m^1 = 0 requirement).
 */
template < concepts::action Action, bool SapRobustified = false >
struct PredictiveRegretMatchingPlus {
   /// robustification factor alpha of SAPCFR+ (arXiv:2503.12770). Only the
   /// prediction shift (<rho, sigma_snap> - rho) is damped by 1/(1 + alpha)
   /// while the raw instantaneous regret contribution stays unscaled.
   static constexpr double sap_alpha = 2.;

   struct node_data_type {
      /// cumulative counterfactual regret z(I,a)
      per_action_map< Action > regret;
      /// instantaneous counterfactual regret buffer of the current iteration
      per_action_map< Action > instant_regret;
      /// strategy snapshot written by the previous recommend() call
      per_action_map< Action > strategy_snapshot;

      void register_action(const Action& action)
      {
         regret.emplace(std::cref(action), 0.);
         instant_regret.emplace(std::cref(action), 0.);
         strategy_snapshot.emplace(std::cref(action), 0.);
      }
   };

   /// protocol entry point: zero-initialize all per-action tables for 'action'
   static void register_action(node_data_type& data, const Action& action)
   {
      data.register_action(action);
   }

   static void observe(node_data_type& data, const Action& action, double increment)
   {
      // all legal actions are pre-registered via register_action, so both
      // lookups hit entries whose keys reference storage owned by this node
      // Algorithm 5 line 7: the cumulative table is clipped at fold-in time
      data.regret[std::cref(action)] = std::max(0., data.regret[std::cref(action)] + increment);
      data.instant_regret[std::cref(action)] += increment;
   }

   template < typename PolicyOut >
   static void recommend(node_data_type& data, PolicyOut& policy_out, size_t /*iteration*/)
   {
      // defensive re-clip of the stored table (no-op under regular observe()
      // flow; guards externally injected table states)
      for(auto& [_, cumul_regret] : data.regret) {
         (void) _;
         cumul_regret = std::max(0., cumul_regret);
      }

      const double shift_scale = SapRobustified ? 1. / (1. + sap_alpha) : 1.;

      // theta(a) = clip(z)(a) + s * rho(a), clamped from below at 0
      const auto predicted_regret = [&](const Action& action, double cumul_regret) {
         return cumul_regret + shift_scale * data.instant_regret.at(std::cref(action));
      };

      // first pass: normalizer over the positive parts of the predicted regret.
      // note that the cumulative table itself is NOT clamped in place.
      double pos_sum{0.};
      for(const auto& [action_ref, cumul_regret] : data.regret) {
         pos_sum += std::max(0., predicted_regret(action_ref.get(), cumul_regret));
      }

      // second pass: derive the policy and mirror it into the snapshot such
      // that the NEXT round's recommend pairs it against the then-observed
      // instantaneous regret (predictive x^{t-1} semantics)
      if(pos_sum > 0.) {
         for(const auto& [action_ref, cumul_regret] : data.regret) {
            const double prob = std::max(0., predicted_regret(action_ref.get(), cumul_regret))
                                / pos_sum;
            policy_out[action_ref.get()] = prob;
            data.strategy_snapshot.at(action_ref) = prob;
         }
      } else {
         const double uniform_prob = 1. / static_cast< double >(data.regret.size());
         for(const auto& [action_ref, _] : data.regret) {
            policy_out[action_ref.get()] = uniform_prob;
            data.strategy_snapshot.at(action_ref) = uniform_prob;
         }
      }

      // consume the instantaneous buffer: the next accumulation phase starts
      // from scratch so that rho always reflects exactly one full iteration
      for(auto& [_, instant_regret] : data.instant_regret) {
         (void) _;
         instant_regret = 0.;
      }
   }

   /// averaging weights are supplied externally through the quadratic average
   /// (gamma = 2 side of the DiscountedCFR decorator); nothing to add here
   static constexpr double policy_weight(size_t /*iteration*/) { return 1.; }
};

/// alias for the SAPCFR+-robustified flavor of the predictive minimizer
template < concepts::action Action >
using SAPPredictiveRegretMatchingPlus = PredictiveRegretMatchingPlus< Action, true >;

///////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////// concept conformance checks ////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

/// minimal writable policy sink used solely to probe concept conformance below
template < concepts::action Action >
struct minimal_policy_out_stub {
   double& operator[](const Action&) { return value; }
   double value = 0.;
};

}  // namespace detail

static_assert(
   regret_minimizer_for<
      PredictiveRegretMatchingPlus< int >,
      int,
      detail::minimal_policy_out_stub< int > >,
   "PredictiveRegretMatchingPlus does not satisfy the regret minimizer protocol."
);
static_assert(
   regret_minimizer_for<
      SAPPredictiveRegretMatchingPlus< int >,
      int,
      detail::minimal_policy_out_stub< int > >,
   "SAPPredictiveRegretMatchingPlus does not satisfy the regret minimizer protocol."
);
static_assert(
   regret_minimizer_for<
      DiscountedCFR< PredictiveRegretMatchingPlus< int >, false >,
      int,
      detail::minimal_policy_out_stub< int > >,
   "DiscountedCFR-wrapped predictive minimizer does not satisfy the regret minimizer protocol."
);

}  // namespace nor::rm

#endif  // NOR_RM_MINIMIZERS_PREDICTIVE_HPP
