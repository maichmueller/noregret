
#ifndef NOR_RM_MINIMIZERS_OMWU_HPP
#define NOR_RM_MINIMIZERS_OMWU_HPP

// NOTE: this header relies on 'per_action_table' and the minimizer node data
// protocol which are defined in nor/rm/minimizers/minimizers.hpp BEFORE it
// includes this file. Include the former instead of this file directly.

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <vector>

namespace nor::rm {

/**
 * @brief optimistic multiplicative weights (OMWU): entropic OFTRL on the simplex.
 *
 * This is the optimistic learner that instantiates the EC 2022 upgrade of the
 * trigger-regret dynamics (Anagnostides, Farina, Kroer, Celli & Sandholm,
 * "Faster No-Regret Learning Dynamics for Extensive-Form Correlated and Coarse
 * Correlated Equilibria", EC 2022, arXiv:2202.05446):
 *
 *    x^(t) = argmax_{x in Delta} { <x, m^(t) + sum_{tau<t} l^(tau)> - d(x)/eta }
 *
 * with d = negative entropy, whose closed form is x^(t) proportional to
 * exp(eta * (L^(t-1) + m^(t))) elementwise, where L is the cumulative observed
 * utility vector. The predictor is PERSISTENCE: m^(t) = l^(t-1) (with the
 * conventional l^(0) = 0), i.e. the last fully observed utility vector, which
 * the paper's stability analysis (their Lemmas 4.6/4.7 and Corollary 4.15)
 * requires. Under small stepsizes the resulting iterate sequence is
 * multiplicatively stable, and OFTRL is (Omega_d / eta, eta)-predictive
 * (their Lemma 2.3): Reg^T <= Omega_d / eta + eta * sum_t ||l^(t) - l^(t-1)||^2.
 * Composed inside the trigger-regret framework with the paper's stepsize
 * regime eta = tau * t^{-1/4} (their Section 6.1), this upgrades the O(T^-1/2)
 * EFCE rate of classic ICFR to O(T^-3/4) -- provided the stability precondition
 * transfers (see the DEVIATIONS note in the ICFR class documentation, which
 * pins exactly what composes and what deviates in this adaptation).
 *
 * IMPLEMENTATION. Direct exponentiation of the cumulative tables with an
 * overflow-safe max-shift per recommendation; the normalized distribution is
 * recomputed from scratch every round (no multiplicative drift). The stepsize
 * schedule keys off the kernel's OWN recommend counter (1-based logical round
 * t): eta(t) = StepSizeTau * t^{-1/4}, matching the paper's experimental
 * regime; self-counting makes the kernel insensitive to the iteration index
 * the caller passes (ICFR's Blum-Mansour wrapper hardcodes 0 there).
 *
 * PROTOCOL BOOKKEEPING (prediction-maintenance invariants). Between two
 * consecutive recommendations at most ONE observation round may arrive; its
 * increments accumulate in 'instant_buffer' ('observe' marks the batch open),
 * and the next 'recommend' folds the batch into the cumulative table AND the
 * persisted prediction exactly once. Counters expose this contract:
 *    recommend_calls == observe_folds + 1 and observe_rounds == observe_folds
 *    + has_pending
 * whenever the unit is consulted from its very first round onwards, and more
 * generally |recommend_calls - (observe_folds + has_pending)| <= 1.
 */
template < concepts::action Action, double StepSizeTau = 1. >
struct OptimisticMultiplicativeWeights {
   static_assert(StepSizeTau > 0., "OMWU requires a positive stepsize scale tau");

   struct node_data_type {
      detail::action_registry< Action > registry;
      /// cumulative observed utility L^(t)(a) = sum_{tau<=t} l^(tau)(a)
      per_action_table< Action > cumulative_utility;
      /// persistence prediction m^(t) = last folded utility vector (l^(0) = 0)
      per_action_table< Action > prediction;
      /// instantaneous utility buffer of the current observation batch
      per_action_table< Action > instant_buffer;
      /// whether an unfolded observation batch is buffered
      bool has_pending = false;
      // ---- prediction-maintenance counters ----
      /// number of recommend() invocations (1-based logical rounds)
      std::uint64_t recommend_calls = 0;
      /// number of observation batches folded (= number of prediction refreshes)
      std::uint64_t observe_folds = 0;
      /// number of observation batches opened (first observe of each batch)
      std::uint64_t observe_rounds = 0;

      void register_action(const Action& action)
      {
         registry.register_action(action);
         cumulative_utility.emplace_back(0.);
         prediction.emplace_back(0.);
         instant_buffer.emplace_back(0.);
      }

      [[nodiscard]] size_t index_of(const Action& action) const
      {
         return registry.index_of(action);
      }
   };

   /// the paper's stepsize regime: eta(t) = tau * t^{-1/4} for the 1-based
   /// logical round 'round_index' >= 1 (EC 2022 paper, Section 6.1; the theory
   /// pins the constant-regime counterpart eta = O(1/(T^{1/4} * D * |A| * ||Q||_1)))
   [[nodiscard]] static double stepsize(std::uint64_t round_index)
   {
      assert(round_index >= 1 && "OMWU rounds are 1-based");
      return StepSizeTau * std::pow(static_cast< double >(round_index), -0.25);
   }

   /// protocol entry point: append 'action' to the registry and zero-initialize
   /// all per-action tables for it
   static void register_action(node_data_type& data, const Action& action)
   {
      data.register_action(action);
   }

   static void observe(node_data_type& data, const Action& action, double increment)
   {
      if(not data.has_pending) {
         data.has_pending = true;
         ++data.observe_rounds;
      }
      data.instant_buffer[data.registry.index_of(action)] += increment;
   }

   template < typename PolicyOut >
   static void recommend(node_data_type& data, PolicyOut& policy_out, size_t /*iteration*/)
   {
      // fold the pending observation batch: refresh the persistence prediction
      // and advance the cumulative table -- exactly once per batch
      if(data.has_pending) {
         for(auto idx : std::views::iota(size_t{0}, data.instant_buffer.size())) {
            data.cumulative_utility[idx] += data.instant_buffer[idx];
            data.prediction[idx] = data.instant_buffer[idx];
            data.instant_buffer[idx] = 0.;
         }
         data.has_pending = false;
         ++data.observe_folds;
      }
      ++data.recommend_calls;

      // x^(t) proportional to exp(eta * (L + m)); max-shifted exponentials keep
      // the arithmetic in range while leaving the normalized distribution exact
      const double eta = stepsize(data.recommend_calls);
      double shift = -std::numeric_limits< double >::infinity();
      for(auto idx : std::views::iota(size_t{0}, data.cumulative_utility.size())) {
         shift = std::max(shift, eta * (data.cumulative_utility[idx] + data.prediction[idx]));
      }
      double mass = 0.;
      for(auto idx : std::views::iota(size_t{0}, data.cumulative_utility.size())) {
         mass += std::exp(eta * (data.cumulative_utility[idx] + data.prediction[idx]) - shift);
      }
      if(mass > 0.) {
         for(auto idx : std::views::iota(size_t{0}, data.cumulative_utility.size())) {
            policy_out[data.registry.actions[idx]] = std::exp(
                                                        eta
                                                           * (data.cumulative_utility[idx]
                                                              + data.prediction[idx])
                                                        - shift
                                                     )
                                                     / mass;
         }
      } else {
         const double uniform_prob = 1. / static_cast< double >(data.cumulative_utility.size());
         for(const auto& action : data.registry.actions) {
            policy_out[action] = uniform_prob;
         }
      }
   }

   static constexpr double policy_weight(size_t /*iteration*/) { return 1.; }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// concept conformance checks ///////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template < concepts::action Action >
struct omwu_policy_out_stub {
   double& operator[](const Action&) { return value; }
   double value = 0.;
};

}  // namespace detail

static_assert(
   regret_minimizer_for<
      OptimisticMultiplicativeWeights< int >,
      int,
      detail::omwu_policy_out_stub< int > >,
   "OptimisticMultiplicativeWeights does not satisfy the regret minimizer protocol."
);

}  // namespace nor::rm

#endif  // NOR_RM_MINIMIZERS_OMWU_HPP
