
#ifndef NOR_RM_MINIMIZERS_DISCOUNTED_PREDICTIVE_HPP
#define NOR_RM_MINIMIZERS_DISCOUNTED_PREDICTIVE_HPP

// NOTE: this header relies on 'per_action_table', 'CFRDiscountedParameters' and
// the minimizer node data protocol which are defined in
// nor/rm/minimizers/minimizers.hpp BEFORE it includes this file. Include the
// former instead of this file directly.

#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <ranges>
#include <utility>

namespace nor::rm {

/**
 * @brief transcribed paper-default parameter sets (arXiv:2404.13891, sec. 5.2:
 * "for DCFR+, we set alpha = 1.5 and gamma = 4 following Hang et al. 2022 [...] for
 * PDCFR+, [...] the best one, i.e., alpha = 2.3 and gamma = 5, is then used across all
 * games"). beta stays at its neutral value 0 (vacuous for these kernels anyway).
 */
[[nodiscard]] inline CFRDiscountedParameters dcfrplus_default_parameters()
{
   return CFRDiscountedParameters{.alpha = 1.5, .beta = 0., .gamma = 4.};
}

[[nodiscard]] inline CFRDiscountedParameters pcfrplus_default_parameters()
{
   return CFRDiscountedParameters{.alpha = 2.3, .beta = 0., .gamma = 5.};
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////// hyperparameter schedules (HS) ///////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief named schedule factories of "Faster Game Solving via Hyperparameter
 * Schedules" (Zhang, McAleer, Sandholm; AAAI 2026, arXiv:2404.09097, eq. (4)).
 *
 * The paper defines four linear schedules over the DCFR hyperparameters, where
 * 't' is the current (raw, 0-based) iteration index and 'n' the TOTAL number of
 * iterations the solver will run:
 *    HS_alpha :  alpha(t) = 1 + 3*t/n
 *    HS_beta  :  beta(t)  = -1 - 2*t/n
 *    HS_gamma30: gamma(t) = 30 - 5*t/n
 *    HS_gamma15: gamma(t) = 15 - 5*t/n
 * Each factory below binds 'n' AT CONSTRUCTION TIME and returns a
 * 'std::function<double(size_t)>' that can be plugged into the corresponding
 * '*_schedule' hook of 'CFRDiscountedParameters'. The solver's iterate() never
 * needs to know the horizon; callers simply pass the iteration count they are
 * planning to run (the schedules are linear in t/n, so a mismatched horizon
 * merely rescales the transition point rather than invalidating the scheme).
 */

/// HS_alpha: alpha(t) = 1 + 3*t/n (arXiv:2404.09097, eq. (4))
[[nodiscard]] inline std::function< double(size_t) > hs_alpha(size_t n_iters)
{
   return [n_iters](size_t t) { return 1. + 3. * static_cast< double >(t) / double(n_iters); };
}

/// HS_beta: beta(t) = -1 - 2*t/n (arXiv:2404.09097, eq. (4))
[[nodiscard]] inline std::function< double(size_t) > hs_beta(size_t n_iters)
{
   return [n_iters](size_t t) { return -1. - 2. * static_cast< double >(t) / double(n_iters); };
}

/// HS_gamma30: gamma(t) = 30 - 5*t/n -- the aggressive variant (arXiv:2404.09097, eq. (4));
/// the paper's recommended default when only one schedule can be used (sec. 4.2)
[[nodiscard]] inline std::function< double(size_t) > hs_gamma30(size_t n_iters)
{
   return [n_iters](size_t t) { return 30. - 5. * static_cast< double >(t) / double(n_iters); };
}

/// HS_gamma15: gamma(t) = 15 - 5*t/n -- the moderate variant (arXiv:2404.09097, eq. (4))
[[nodiscard]] inline std::function< double(size_t) > hs_gamma15(size_t n_iters)
{
   return [n_iters](size_t t) { return 15. - 5. * static_cast< double >(t) / double(n_iters); };
}

/// dispatch tag selecting between the two published gamma schedules
enum class HSVariant { gamma30, gamma15 };

[[nodiscard]] inline std::function< double(size_t) > hs_gamma(size_t n_iters, HSVariant variant)
{
   return variant == HSVariant::gamma30 ? hs_gamma30(n_iters) : hs_gamma15(n_iters);
}

/**
 * @brief assembles ready-to-use 'CFRDiscountedParameters' bundles for the
 * HS-powered algorithm variants.
 *
 * - hs_dcfr_parameters        -> plain DCFR regret minimizer (RegretMinimizingMode::
 *                                regret_matching under CFRWeightingMode::discounted)
 *                                with all three hyperparameters scheduled
 *                                (HS-DCFR of arXiv:2404.09097).
 * - hs_dcfrplus_parameters    -> DCFR+ kernel (RegretMinimizingMode::
 *                                discounted_regret_matching_plus) whose alpha side
 *                                consumes the same HS_alpha schedule and whose gamma
 *                                side consumes HS_gamma{30|15}. NOTE: this combination is
 *                                OUR composition of two published methods; the HS paper only
 *                                schedules plain DCFR and PCFR+.
 * - hs_pcfrplus_parameters    -> PCFR+ (predictive_regret_matching_plus) with ONLY the
 *                                gamma schedule applied ("Given that gamma is the only
 *                                adjustable hyperparameter in PCFR+, HSs for alpha and beta
 *                                are not used in this setting", arXiv:2404.09097 sec. 3.2).
 *                                The alpha/beta hooks stay null (the PCFR+ arm compiles the
 *                                regret discounts out entirely).
 * - hs_pdcfrplus_parameters   -> PDCFR+ kernel with scheduled gamma and the PDCFR+ paper
 *                                default alpha = 2.3 kept CONSTANT (mirroring the HS paper's
 *                                policy of scheduling only what the base algorithm exposes;
 *                                extension beyond both papers, documented deviation).
 */
[[nodiscard]] inline CFRDiscountedParameters
hs_dcfr_parameters(size_t n_iters, HSVariant variant = HSVariant::gamma30)
{
   auto params = CFRDiscountedParameters{};
   params.alpha_schedule = hs_alpha(n_iters);
   params.beta_schedule = hs_beta(n_iters);
   params.gamma_schedule = hs_gamma(n_iters, variant);
   return params;
}

[[nodiscard]] inline CFRDiscountedParameters
hs_dcfrplus_parameters(size_t n_iters, HSVariant variant = HSVariant::gamma30)
{
   auto params = dcfrplus_default_parameters();
   params.alpha_schedule = hs_alpha(n_iters);
   params.gamma_schedule = hs_gamma(n_iters, variant);
   return params;
}

[[nodiscard]] inline CFRDiscountedParameters
hs_pcfrplus_parameters(size_t n_iters, HSVariant variant = HSVariant::gamma30)
{
   // PCFR+ rides the discounted machinery purely for its quadratic averaging; the
   // default CFRDiscountedParameters already carry gamma = 2 which the schedule replaces
   auto params = CFRDiscountedParameters{};
   params.gamma_schedule = hs_gamma(n_iters, variant);
   return params;
}

[[nodiscard]] inline CFRDiscountedParameters
hs_pdcfrplus_parameters(size_t n_iters, HSVariant variant = HSVariant::gamma30)
{
   auto params = pcfrplus_default_parameters();
   params.gamma_schedule = hs_gamma(n_iters, variant);
   return params;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////// DCFR+ / PDCFR+ regret minimizer kernel ////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief the shared DCFR+/PDCFR+ regret minimizer (Xu, Li, Liu, Fu, Fu, Xing,
 * Cheng — "Minimizing Weighted Counterfactual Regret with Optimistic Online
 * Mirror Descent", IJCAI 2024, arXiv:2404.13891, Table 1 & sec. 4).
 *
 * Transcribed equations (paper notation, logical iterations t = 1..T):
 *
 *   fold (both variants):
 *      R^t     = [ R^{t-1} * (t-1)^alpha / ((t-1)^alpha + 1) + r^t ]^+
 *   recommend (DCFR+, predictive = false):
 *      x^{t+1} = R^t / ||R^t||_1
 *   recommend (PDCFR+, predictive = true):
 *      R~^{t+1}= [ R^t * t^alpha / (t^alpha + 1) + v^{t+1} ]^+
 *      x^{t+1} = R~^{t+1} / ||R~^{t+1}||_1
 *   average strategy (carried by this class's policy_weight, see below):
 *      X^t     = X^{t-1} * ((t-1)/t)^gamma + xdot^t
 *
 * FOLD ORDER — pinned semantics: the alpha discount multiplies the PREVIOUS
 * cumulative regret BEFORE the instantaneous regret r^t is added, and the clip
 * [.]^+ is applied to the SUM afterwards ("discount applied to the positive part
 * before adding, then clip"). This ordering is exactly what distinguishes DCFR+
 * from composing the historical DiscountedCFR decorator (recommend-time scaling)
 * with RegretMatchingPlus (observe-time clipping): e.g. for R^{t-1} = -5,
 * r^t = +10, d = 0.5 the paper order yields [-5*0.5 + 10]^+ = 7.5 whereas the
 * decorator composition would first clamp (-5 + 10) -> 5 and then scale -> 3.75.
 * For non-negative prior regrets both orders agree; they diverge precisely when
 * a negative stored regret is followed by a positive increment.
 *
 * BETA VACUITY: plain DCFR conditions its discount factor on the SIGN of the
 * stored cumulative regret (positive part -> alpha, otherwise -> beta). In DCFR+
 * the stored table is clamped at every fold, hence R^{t-1} >= 0 always holds and
 * the beta branch can never trigger; the equation above therefore only contains
 * the alpha factor (cf. arXiv:2404.13891 Table 1, DCFR+ row). The beta field of
 * CFRDiscountedParameters is deliberately IGNORED by this kernel.
 *
 * DEFERRED FOLDING: an infostate generally spans many histories, each producing
 * one observe() call per action within an iteration while the paper folds r^t(I,.)
 * ONCE per iteration. observe() therefore buffers the increments into
 * 'instant_regret' (rho), leaving the stored 'regret' table at exactly
 * R^{t-1}, and recommend() completes the update in a single sweep:
 *    regret <- max(regret * d_{t-1} + rho, 0)
 * mirroring the deferred-update pattern of ExponentialCFR. Under the solver's
 * alternating-update regime each infostate observes one full phase between two
 * recommendations, so rho always holds exactly one fully observed instantaneous
 * regret vector when recommend() runs.
 *
 * PERSISTENCE PREDICTION: identical to PredictiveRegretMatchingPlus, the
 * predicted Blackwell payoff v^{t+1} collapses analytically to the last observed
 * instantaneous regret vector rho (Farina, Kroer, Sandholm, AAAI 2021,
 * arXiv:2007.14358, sec. 7), so the prediction source needs no extra state and
 * no strategy snapshot.
 *
 * INDEXING: the solver passes raw 0-based iteration indices; the logical paper
 * iteration is t = iteration + 1. Consequently
 *    d_{t-1} uses exponent alpha evaluated at raw index 'iteration'
 *       (at iteration = 0, (t-1) = 0 and 0^alpha = 0 for alpha > 0, so the very
 *        first fold degenerates gracefully to R^1 = [r^1]^+),
 *    d_t     uses exponent alpha evaluated at raw index 'iteration + 1'.
 * With constant alpha (null schedule) both lookups coincide. Negative scheduled
 * exponents at raw index 0 -- undefined in the papers and NaN-producing under
 * naive pow evaluation -- are neutralized by 'discount_factor' (CFRDiscounted-
 * Parameters-adjacent helper) to the no-discount factor 1/2.
 *
 * AVERAGING SIDE: this class ALSO owns the gamma-side average-policy weighting
 * (it plays the role the DiscountedCFR<Inner, false> decorator plays for PCFR+),
 * exposing 'policy_weight' and 'discounted_parameters()' with the same
 * conventions as that decorator ((i+1)/(i+2))^gamma at raw index i, plus the
 * solver-facing 'weight_by_cycle' switch). It must therefore be selected
 * DIRECTLY as the minimizer type — wrapping it in DiscountedCFR would apply the
 * gamma weight twice.
 */
template < concepts::action Action, bool predictive >
class DiscountedPlusRegretMatching {
  public:
   constexpr DiscountedPlusRegretMatching() = default;
   explicit constexpr DiscountedPlusRegretMatching(CFRDiscountedParameters params)
       : m_params(std::move(params))
   {
   }

   struct node_data_type {
      detail::action_registry< Action > registry;
      /// cumulative counterfactual regret z(I,a); equals R^{t-1} during traversal
      /// (all increments of the running iteration live in 'instant_regret') and
      /// the clamped R^t after each recommend()
      per_action_table< Action > regret;
      /// instantaneous counterfactual regret buffer r^t(I,a) of the current
      /// iteration; consumed (= reset) by recommend()
      per_action_table< Action > instant_regret;

      void register_action(const Action& action)
      {
         registry.register_action(action);
         regret.emplace_back(0.);
         instant_regret.emplace_back(0.);
      }

      [[nodiscard]] size_t index_of(const Action& action) const
      {
         return registry.index_of(action);
      }
   };

   /// protocol entry point: append 'action' to the registry and zero-initialize
   /// all per-action tables for it
   static void register_action(node_data_type& data, const Action& action)
   {
      data.register_action(action);
   }

   /// buffer the counterfactually weighted instantaneous regret increment; the
   /// cumulative fold happens once per iteration in recommend()
   static void observe(node_data_type& data, const Action& action, double increment)
   {
      data.instant_regret[data.registry.index_of(action)] += increment;
   }

   /**
    * @brief completes the iteration's fold and derives the next recommendation.
    *
    * Pass 1 folds every entry in place, consumes rho and parks the recommendation
    * source theta(a) (= R^t(a) for DCFR+, R~^{t+1}(a) for PDCFR+) back into the
    * freed 'instant_regret' scratch slots while accumulating the normalizer over
    * its positive parts. Pass 2 writes the normalized policy and clears the
    * scratch. All-vanishing sources fall back to the uniform distribution
    * (RM+'s 0/0 := uniform convention).
    */
   template < typename PolicyOut >
   void recommend(node_data_type& data, PolicyOut& policy_out, size_t iteration) const
   {
      const auto n_actions = data.regret.size();

      // d_{t-1}: exponent evaluated at the raw index; at raw index 0 the fold
      // degenerates to [r^1]^+ for every alpha > 0 (0^a = 0). 'discount_factor'
      // additionally guards exotic negative scheduled exponents (NaN safety)
      const double disc_prev = discount_factor(iteration, m_params.alpha_at(iteration));
      const double disc_curr = [&] {
         if constexpr(predictive) {
            // LIMITATION: recommend() runs mid-sweep (per infostate, during the
            // post-order unwind of an in-flight iteration), yet PDCFR+'s
            // persistence prediction evaluates alpha at raw index t+1 HERE --
            // i.e. before the iteration has completed. A scheduled alpha
            // therefore observes an iterated-one-step-ahead input for every
            // infoset swept within the current iteration; with a CONSTANT alpha
            // (the published PDCFR+ default and all null-schedule parameter
            // sets) this is numerically irrelevant.
            return discount_factor(iteration + 1, m_params.alpha_at(iteration + 1));
         } else {
            return 0.;  // unused
         }
      }();

      double pos_sum{0.};
      for(auto idx : std::views::iota(size_t{0}, n_actions)) {
         const double rho = data.instant_regret[idx];
         // fold: R^t = max(R^{t-1} * d_{t-1} + r^t, 0) -- discount BEFORE add,
         // clip on the sum (pinned ordering, see class docs)
         const double folded = std::max(data.regret[idx] * disc_prev + rho, 0.);
         data.regret[idx] = folded;
         if constexpr(predictive) {
            // R~^{t+1} = max(R^t * d_t + v^{t+1}, 0) with persistence v^{t+1} = r^t;
            // parked in the consumed rho slot as recommendation scratch
            const double predicted = std::max(folded * disc_curr + rho, 0.);
            data.instant_regret[idx] = predicted;
            pos_sum += predicted;
         } else {
            data.instant_regret[idx] = folded;
            pos_sum += folded;
         }
      }

      if(pos_sum > 0.) {
         for(auto idx : std::views::iota(size_t{0}, n_actions)) {
            policy_out[data.registry.actions[idx]] = data.instant_regret[idx] / pos_sum;
            data.instant_regret[idx] = 0.;
         }
      } else {
         const double uniform_prob = 1. / static_cast< double >(n_actions);
         for(const auto& action : data.registry.actions) {
            policy_out[action] = uniform_prob;
         }
         for(double& scratch : data.instant_regret) {
            scratch = 0.;
         }
      }
   }

   /// gamma-side average-policy multiplier; identical convention to the
   /// DiscountedCFR decorator: pow(t/(t+1), gamma) with t = raw index + 1
   [[nodiscard]] double policy_weight(size_t iteration) const
   {
      double t = double(iteration) + 1.;
      return std::pow(t / (t + 1.), m_params.gamma_at(iteration));
   }

   /// the stored parameters (exposes 'weight_by_cycle' to the solver)
   [[nodiscard]] const CFRDiscountedParameters& discounted_parameters() const { return m_params; }

  private:
   CFRDiscountedParameters m_params;
};

/// DCFR+ regret minimizer (arXiv:2404.13891, sec. 4): discount-before-add RM+
template < concepts::action Action >
using DiscountedRegretMatchingPlus = DiscountedPlusRegretMatching< Action, /*predictive=*/false >;

/// PDCFR+ regret minimizer (arXiv:2404.13891, sec. 4): DCFR+ + persistence prediction
template < concepts::action Action >
using DiscountedPredictiveRegretMatchingPlus = DiscountedPlusRegretMatching<
   Action,
   /*predictive=*/true >;

///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////// concept conformance checks //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

static_assert(
   regret_minimizer_for<
      DiscountedRegretMatchingPlus< int >,
      int,
      detail::minimal_policy_out_stub< int > >,
   "DiscountedRegretMatchingPlus does not satisfy the regret minimizer protocol."
);
static_assert(
   regret_minimizer_for<
      DiscountedPredictiveRegretMatchingPlus< int >,
      int,
      detail::minimal_policy_out_stub< int > >,
   "DiscountedPredictiveRegretMatchingPlus does not satisfy the regret minimizer protocol."
);

}  // namespace nor::rm

#endif  // NOR_RM_MINIMIZERS_DISCOUNTED_PREDICTIVE_HPP
