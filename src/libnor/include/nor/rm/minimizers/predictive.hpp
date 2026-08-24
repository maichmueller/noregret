
#ifndef NOR_RM_MINIMIZERS_PREDICTIVE_HPP
#define NOR_RM_MINIMIZERS_PREDICTIVE_HPP

// NOTE: this header relies on 'per_action_table' and the minimizer node data
// protocol which are defined in nor/rm/minimizers/minimizers.hpp BEFORE it
// includes this file. Include the former instead of this file directly.

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <vector>

namespace nor::rm {

///////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// prediction shift policies //////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief prediction-shift policies of the predictive minimizer.
 *
 * A Shift maps the buffered instantaneous regret rho(I,a) of an action to the
 * prediction term that the PRM+ recommendation adds on top of the clipped
 * cumulative regret:
 *    theta(a) = max(0, clip(z)(a) + Shift{}(rho(a)))
 *
 * PROTOCOL EXTENSION (contextual/stateful shifts). Some robustifications
 * (APCFR+) cannot express their prediction scale as a pure function of the
 * single rho(a) value: they adapt it per infostate from the iteration history.
 * Such a Shift opts out of the plain invocable interface and instead provides
 *
 *    context_type   the per-infostate auxiliary state it needs (stored inside
 *                   the minimizer's node_data_type as a
 *                   [[no_unique_address]] member 'shift_context')
 *    static void before_recommend(context_type&, node_data_type&, size_t)
 *                   invoked ONCE at the top of every recommend() call, before
 *                   any theta evaluation. It may read AND MUTATE the node data
 *                   (this is how the Stable-PRM+ restart resets z/rho) and is
 *                   expected to leave whatever scale it derived in its context.
 *    static double evaluate(const context_type&, double instant_regret)
 *                   the per-action theta contribution, replacing the pure
 *                   invocable call.
 *
 * The kernel detects the protocol through 'contextual_prediction_shift' below;
 * pure shifts (PCFR+/SAPCFR+/P2PCFR+) are completely unaffected. A contextual
 * shift that grows per-action tables must additionally provide
 *    static void register_context(context_type&, const Action&)
 * mirroring the node-data registration hook.
 */
template < typename Shift >
concept prediction_shift = std::regular_invocable< Shift, double >
                           and std::convertible_to< std::invoke_result_t< Shift, double >, double >;

template < typename Shift, typename NodeData >
concept contextual_prediction_shift = requires(
   typename Shift::context_type& context,
   const NodeData& const_data,
   NodeData& data,
   size_t iteration,
   double instant_regret
) {
   typename Shift::context_type;
   Shift::before_recommend(context, data, iteration);
   {
      Shift::evaluate(context, instant_regret)
   } -> std::convertible_to< double >;
};

/// optional per-action growth hook of a contextual shift's context
template < typename Shift, typename Action >
concept context_registering_shift = requires(
   typename Shift::context_type& context,
   const Action& action
) { Shift::register_context(context, action); };

/// any policy admissible as the kernel's Shift parameter: either a pure
/// per-action function ('prediction_shift') or a stateful contextual policy
/// signaling itself through its 'context_type' member
template < typename Shift >
concept shift_policy = prediction_shift< Shift > or requires { typename Shift::context_type; };

namespace detail {

struct empty_shift_context {};

/// resolves the per-infostate auxiliary state a shift requests through its
/// 'context_type' member (empty for pure shifts so that the [[no_unique_address]]
/// member below costs nothing)
template < typename Shift, typename = void >
struct shift_context_of {
   using type = empty_shift_context;
};

template < typename Shift >
struct shift_context_of< Shift, std::void_t< typename Shift::context_type > > {
   using type = typename Shift::context_type;
};

}  // namespace detail

template < typename Shift >
using shift_context_for_t = typename detail::shift_context_of< Shift >::type;

/// persistence prediction at full weight: shift = rho. This is PCFR+
/// (prediction scale s = 1).
struct IdentityShift {
   [[nodiscard]] constexpr double operator()(double instant_regret) const noexcept
   {
      return instant_regret;
   }
};

/// constant-damped persistence prediction: shift = ScaleFactor * rho.
/// The SAPCFR+ robustification is the instance with ScaleFactor = 1/(1 + alpha)
/// for alpha = 2 (arXiv:2503.12770), i.e. only the prediction term is damped
/// while the raw instantaneous regret contribution stays unscaled.
template < double ScaleFactor >
struct ConstantShift {
   static constexpr double scale = ScaleFactor;

   [[nodiscard]] constexpr double operator()(double instant_regret) const noexcept
   {
      return scale * instant_regret;
   }
};

/// robustification factor alpha of SAPCFR+ (arXiv:2503.12770)
inline constexpr double sap_alpha = 2.;

/// the SAPCFR+ prediction-shift policy: s = 1 / (1 + alpha), alpha = 2
using SapPredictionShift = ConstantShift< 1. / (1. + sap_alpha) >;

/**
 * @brief P2PCFR+ prediction-shift policy ("Pessimistic PCFR+", Meng, Liu et al.,
 * ICLR 2025 submission, OpenReview id njyZgDDeY4).
 *
 * PROVENANCE CAUTION: this is a non-archival OpenReview submission (not
 * accepted as far as we could verify; the forum page is bot-walled). The
 * update rule below was transcribed from the submitted PDF (section 4,
 * equations following Eq. (5)):
 *    R̂^t_I = [R^t_I + 1/(1+alpha) * r^{t-1}_I]^+ ,
 *    R^{t+1}_I = [R^t_I + r^t_I]^+ ,
 *    sigma^t_i(I) = R̂^t_I / ||R̂^t_I||_1 ,  alpha >= 0 constant.
 * Their regret analysis (Theorems 4.1/4.2) assumes alpha in [0, 1]; their
 * alternative bound (Theorem 4.4) holds for arbitrary alpha with the
 * interpolating property alpha -> 0 recovers PCFR+ and alpha -> infinity
 * recovers CFR+, and their experiments report alpha = 5 as the best empirical
 * choice ("consistently outperforms PCFR+ when alpha <= 10").
 *
 * Functionally this coincides with the SAPCFR+ family (fixed damping of the
 * prediction term only); the papers are independent derivations of the same
 * mechanism with different motivation (P2PCFR+: minimizing the discrepancy
 * between the strategies represented by implicit and explicit accumulated
 * regrets within one iteration) and different default alpha. It is kept as a
 * distinct named variant for reproducibility of the respective publications.
 */
inline constexpr double p2p_alpha = 5.;

/// the P2PCFR+ prediction-shift policy: s = 1 / (1 + alpha), alpha = 5
using P2PPredictionShift = ConstantShift< 1. / (1. + p2p_alpha) >;

/**
 * @brief APCFR+ adaptive-asymmetry prediction shift (arXiv:2503.12770v2,
 * "Faster Game Solving via Asymmetry of Step Sizes", Eq. (4) + Eq. (10)).
 *
 * Update rule (their Eq. (4)):
 *    R̂^t_I = [R^t_I + 1/(1 + alpha^t_I) * r^{t-1}_I]^+ ,
 *    R^{t+1}_I = [R^t_I + r^t_I]^+
 * where alpha^t_I >= 0 is learned PER INFOSET from the running sums (their
 * Eq. (10)):
 *    alpha^t_I = min( sqrt( sum_{tau=1}^{t-1} ||r^tau - r^{tau-1}||_2^2
 *                            / sum_{tau=1}^{t-1} ||R^{tau+1} - R^tau||_2^2 ),
 *                     alpha_max )
 * with alpha_max = 5 fixed by the authors. The numerator accumulates the
 * squared L2 norms of the prediction errors (observed minus predicted
 * instantaneous regret); the denominator those of the realized implicit-regret
 * increments. When the denominator vanishes the second bound term alpha*E is
 * zero for every alpha, so the bound-minimizing choice is alpha_max.
 *
 * Index correspondence to our solver loop: recommend() at the end of loop-n
 * derives the strategy of paper-iteration n+2 from the prediction r^{n+1},
 * i.e. exactly paper-step "R̂^{n+2}", whose alpha^{n+2} per Eq. (10) uses the
 * sums over tau <= n+1 — precisely what has been observed when before_recommend
 * runs. The delta bookkeeping therefore folds the JUST-completed iteration's
 * contributions first and computes alpha afterwards.
 *
 * This is the motivating instance of the contextual-shift protocol extension:
 * the scale depends on per-infostate iteration history, not on rho alone.
 */
template < double AlphaMax = 5. >
struct AdaptiveAsymmetryShift {
   static constexpr double alpha_max = AlphaMax;

   struct context_type {
      /// sum_tau ||r^tau_I - r^{tau-1}_I||_2^2 over all completed iterations
      double sum_sq_prediction_error = 0.;
      /// sum_tau ||R^{tau+1}_I - R^tau_I||_2^2 over all completed iterations
      double sum_sq_implicit_delta = 0.;
      /// prediction scale 1/(1 + alpha^t_I) valid for the current recommendation
      double current_scale = 1.;
      /// snapshot r^{t-1}_I of the previous recommend's instantaneous buffer
      std::vector< double > prev_instant_regret;
      /// snapshot R^t_I of the previous recommend's cumulative table
      std::vector< double > prev_cumul_regret;
   };

   static void register_context(context_type& ctx, const auto& /*action*/)
   {
      ctx.prev_instant_regret.emplace_back(0.);
      ctx.prev_cumul_regret.emplace_back(0.);
   }

   template < typename NodeData >
   static void before_recommend(context_type& ctx, const NodeData& data, size_t /*iteration*/)
   {
      const auto n_actions = data.regret.size();
      double sq_prediction_error = 0.;
      double sq_implicit_delta = 0.;
      for(auto idx : std::views::iota(size_t{0}, n_actions)) {
         const double delta_r = data.instant_regret[idx] - ctx.prev_instant_regret[idx];
         const double delta_z = data.regret[idx] - ctx.prev_cumul_regret[idx];
         sq_prediction_error += delta_r * delta_r;
         sq_implicit_delta += delta_z * delta_z;
      }
      ctx.sum_sq_prediction_error += sq_prediction_error;
      ctx.sum_sq_implicit_delta += sq_implicit_delta;
      // Eq. (10): adaptive per-infostate asymmetry coefficient, clamped by
      // alpha_max solely so that the Theorem-4.1 bound remains finite
      const double alpha = ctx.sum_sq_implicit_delta > 0. ? std::min(
                              std::sqrt(ctx.sum_sq_prediction_error / ctx.sum_sq_implicit_delta),
                              AlphaMax
                           )
                                                          : AlphaMax;
      ctx.current_scale = 1. / (1. + alpha);
      // snapshot the consumed tables for the next round's delta computation
      ctx.prev_instant_regret.assign(data.instant_regret.begin(), data.instant_regret.end());
      ctx.prev_cumul_regret.assign(data.regret.begin(), data.regret.end());
   }

   [[nodiscard]] static double evaluate(const context_type& ctx, double instant_regret) noexcept
   {
      return ctx.current_scale * instant_regret;
   }
};

/**
 * @brief Stable-PRM+ restart policy (Farina, Grand-Clément, Kroer, Lee, Luo,
 * NeurIPS 2023, arXiv:2305.14709, Algorithm 1 "Stable Predictive RM+").
 *
 * Keeps the cumulative regret vector at L1-distance at least Threshold from
 * the origin by RESTARTING: whenever every component of the stored cumulative
 * table satisfies z(I,a) <= Threshold, the table is reset componentwise to
 * Threshold and the prediction for the upcoming iteration is suppressed
 * (m^{t+1} = 0, cf. the prediction schedule of their Theorem 4.1). In our
 * kernel the check-and-reset happens at the top of recommend() -- i.e. after
 * the traversal-time observe() updates, matching the algorithm's order of
 * "update w^t, then restart if w^t <= R0*1".
 *
 * The paper proves O(T^{1/4}) individual-regret bounds for this schedule in
 * normal-form games; note that the APCFR+/P2PCFR+ papers report it never
 * beats plain PCFR+ empirically on IIG benchmarks -- it is provided here for
 * completeness of the robustification family.
 */
template < double Threshold = 1. >
struct StableRestartShift {
   static_assert(Threshold > 0., "the restart threshold must be positive");
   static constexpr double threshold = Threshold;

   struct context_type {
      /// whether the current recommendation follows a restart event
      bool restarted = false;
   };

   template < typename NodeData >
   static void before_recommend(context_type& ctx, NodeData& data, size_t /*iteration*/)
   {
      ctx.restarted = std::ranges::all_of(data.regret, [](double cumul_regret) {
         return cumul_regret <= Threshold;
      });
      if(ctx.restarted) {
         // reinitialize at the floor R0 * 1 (componentwise restart condition
         // w^t_i <= R0 * 1 => w^t_i <- R0, Algorithm 1 lines 8-9)
         for(double& cumul_regret : data.regret) {
            cumul_regret = Threshold;
         }
         // prediction m^{t+1} = 0 for the iteration following a restart
         for(double& instant_regret : data.instant_regret) {
            instant_regret = 0.;
         }
      }
   }

   [[nodiscard]] static constexpr double
   evaluate(const context_type& ctx, double instant_regret) noexcept
   {
      return ctx.restarted ? 0. : instant_regret;
   }
};

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
 * Sandholm's heuristic in section 7 of the paper). The prediction term's scale
 * is carried by the 'Shift' template parameter: IdentityShift (s = 1)
 * reproduces PCFR+ and SapPredictionShift (s = 1/(1 + alpha), alpha = 2)
 * reproduces SAPCFR+ (arXiv:2503.12770) bit-compatibly. Stateful shifts
 * (AdaptiveAsymmetryShift for APCFR+, StableRestartShift for Stable-PRM+)
 * route through the contextual protocol described at
 * 'contextual_prediction_shift'.
 *
 * The 'NormFloorEpsilon' parameter implements the SMOOTH-PRM+ robustification
 * (arXiv:2305.14709, Algorithm 2): when positive, the predicted-regret vector
 * theta is kept at 1-norm at least epsilon by spreading any deficit uniformly,
 *    theta(a) += max(0, epsilon - ||theta||_1) / |A(I)| ,
 * before normalization ("chopping off" the origin; for non-negative theta this
 * uniform spread IS the exact Euclidean projection onto {x >= 0, ||x||_1 >=
 * eps}). The default of 0 compiles the floor away entirely.
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
template <
   concepts::action Action,
   shift_policy Shift = IdentityShift,
   double NormFloorEpsilon = 0. >
struct PredictiveRegretMatchingPlus {
   using shift_type = Shift;
   static constexpr double norm_floor_epsilon = NormFloorEpsilon;

   struct node_data_type {
      detail::action_registry< Action > registry;
      /// cumulative counterfactual regret z(I,a); entry i belongs to registry.actions[i]
      per_action_table< Action > regret;
      /// instantaneous counterfactual regret buffer of the current iteration
      per_action_table< Action > instant_regret;
      /// strategy snapshot written by the previous recommend() call
      per_action_table< Action > strategy_snapshot;
      /// per-infostate auxiliary state of stateful (contextual) shifts; empty
      /// and zero-sized for pure shifts such as IdentityShift/ConstantShift
      [[no_unique_address]] shift_context_for_t< Shift > shift_context;

      void register_action(const Action& action)
      {
         registry.register_action(action);
         regret.emplace_back(0.);
         instant_regret.emplace_back(0.);
         strategy_snapshot.emplace_back(0.);
         if constexpr(context_registering_shift< Shift, Action >) {
            Shift::register_context(shift_context, action);
         }
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

   static void observe(node_data_type& data, const Action& action, double increment)
   {
      const auto idx = data.registry.index_of(action);
      // Algorithm 5 line 7: the cumulative table is clipped at fold-in time
      auto& cumul_regret = data.regret[idx];
      cumul_regret = std::max(0., cumul_regret + increment);
      data.instant_regret[idx] += increment;
   }

   template < typename PolicyOut >
   static void recommend(node_data_type& data, PolicyOut& policy_out, size_t iteration)
   {
      const auto n_actions = data.regret.size();

      // defensive re-clip of the stored table (no-op under regular observe()
      // flow; guards externally injected table states)
      for(double& cumul_regret : data.regret) {
         cumul_regret = std::max(0., cumul_regret);
      }

      // contextual-shift protocol hook: stateful shifts update their scale /
      // apply their restart logic once per recommendation, before any theta
      // evaluation (and may mutate the tables, as the Stable-PRM+ restart does)
      if constexpr(contextual_prediction_shift< Shift, node_data_type >) {
         Shift::before_recommend(data.shift_context, data, iteration);
      }

      [[maybe_unused]] const constexpr Shift shift{};

      // theta(a) = clip(z)(a) + s * rho(a), clamped from below at 0
      const auto predicted_regret = [&](auto idx, double cumul_regret) {
         if constexpr(contextual_prediction_shift< Shift, node_data_type >) {
            return cumul_regret + Shift::evaluate(data.shift_context, data.instant_regret[idx]);
         } else {
            return cumul_regret + shift(data.instant_regret[idx]);
         }
      };

      // first pass: normalizer over the positive parts of the predicted regret.
      // note that the cumulative table itself is NOT clamped in place.
      double pos_sum{0.};
      for(auto idx : std::views::iota(size_t{0}, n_actions)) {
         pos_sum += std::max(0., predicted_regret(idx, data.regret[idx]));
      }

      // Smooth-PRM+ floor (arXiv:2305.14709): lift a vanishing 1-norm of theta
      // to exactly epsilon by spreading the deficit uniformly over the actions
      // (the exact Euclidean projection onto {x >= 0, ||x||_1 >= eps} for
      // non-negative x). Compiled out entirely when NormFloorEpsilon == 0.
      double floor_deficit = 0.;
      if constexpr(NormFloorEpsilon > 0.) {
         if(pos_sum < NormFloorEpsilon and n_actions > 0) {
            floor_deficit = (NormFloorEpsilon - pos_sum) / static_cast< double >(n_actions);
            pos_sum = NormFloorEpsilon;
         }
      }

      // second pass: derive the policy and mirror it into the snapshot such
      // that the NEXT round's recommend pairs it against the then-observed
      // instantaneous regret (predictive x^{t-1} semantics)
      if(pos_sum > 0.) {
         for(auto idx : std::views::iota(size_t{0}, n_actions)) {
            const double prob = (std::max(0., predicted_regret(idx, data.regret[idx]))
                                 + floor_deficit)
                                / pos_sum;
            policy_out[data.registry.actions[idx]] = prob;
            data.strategy_snapshot[idx] = prob;
         }
      } else {
         const double uniform_prob = 1. / static_cast< double >(n_actions);
         for(auto idx : std::views::iota(size_t{0}, n_actions)) {
            policy_out[data.registry.actions[idx]] = uniform_prob;
            data.strategy_snapshot[idx] = uniform_prob;
         }
      }

      // consume the instantaneous buffer: the next accumulation phase starts
      // from scratch so that rho always reflects exactly one full iteration
      for(double& instant_regret : data.instant_regret) {
         instant_regret = 0.;
      }
   }

   /// averaging weights are supplied externally through the quadratic average
   /// (gamma = 2 side of the DiscountedCFR decorator); nothing to add here
   static constexpr double policy_weight(size_t /*iteration*/) { return 1.; }
};

/// alias for the SAPCFR+-robustified flavor of the predictive minimizer
template < concepts::action Action >
using SAPPredictiveRegretMatchingPlus = PredictiveRegretMatchingPlus< Action, SapPredictionShift >;

/// alias for the P2PCFR+-robustified flavor of the predictive minimizer
/// (OpenReview njyZgDDeY4; see 'P2PPredictionShift' for provenance caveats)
template < concepts::action Action >
using P2PPredictiveRegretMatchingPlus = PredictiveRegretMatchingPlus< Action, P2PPredictionShift >;

/// alias for the APCFR+-robustified flavor of the predictive minimizer
/// (adaptive asymmetry of step sizes, arXiv:2503.12770)
template < concepts::action Action >
using APPredictiveRegretMatchingPlus = PredictiveRegretMatchingPlus<
   Action,
   AdaptiveAsymmetryShift<> >;

/// the Smooth-PRM+ norm floor epsilon: the papers chop off the origin at
/// ||R||_1 >= 1 (arXiv:2305.14709 defines Delta_>= with unit threshold)
inline constexpr double smooth_prm_plus_epsilon = 1.;

/// alias for the Smooth-PRM+-robustified flavor of the predictive minimizer
template < concepts::action Action >
using SmoothPredictiveRegretMatchingPlus = PredictiveRegretMatchingPlus<
   Action,
   IdentityShift,
   smooth_prm_plus_epsilon >;

/// alias for the Stable-PRM+ (restart) robustification of the predictive
/// minimizer; restart threshold R0 = 1 per the paper's WLOG normalization
inline constexpr double stable_prm_plus_threshold = 1.;

template < concepts::action Action >
using StablePredictiveRegretMatchingPlus = PredictiveRegretMatchingPlus<
   Action,
   StableRestartShift< stable_prm_plus_threshold > >;

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
static_assert(
   regret_minimizer_for<
      P2PPredictiveRegretMatchingPlus< int >,
      int,
      detail::minimal_policy_out_stub< int > >,
   "P2PPredictiveRegretMatchingPlus does not satisfy the regret minimizer protocol."
);
static_assert(
   regret_minimizer_for<
      APPredictiveRegretMatchingPlus< int >,
      int,
      detail::minimal_policy_out_stub< int > >,
   "APPredictiveRegretMatchingPlus (contextual shift) does not satisfy the regret minimizer "
   "protocol."
);
static_assert(
   regret_minimizer_for<
      SmoothPredictiveRegretMatchingPlus< int >,
      int,
      detail::minimal_policy_out_stub< int > >,
   "SmoothPredictiveRegretMatchingPlus does not satisfy the regret minimizer protocol."
);
static_assert(
   regret_minimizer_for<
      StablePredictiveRegretMatchingPlus< int >,
      int,
      detail::minimal_policy_out_stub< int > >,
   "StablePredictiveRegretMatchingPlus (contextual shift) does not satisfy the regret minimizer "
   "protocol."
);
static_assert(
   contextual_prediction_shift<
      AdaptiveAsymmetryShift<>,
      PredictiveRegretMatchingPlus< int, AdaptiveAsymmetryShift<> >::node_data_type >,
   "AdaptiveAsymmetryShift must model the contextual prediction-shift protocol."
);
static_assert(
   not contextual_prediction_shift<
      IdentityShift,
      PredictiveRegretMatchingPlus< int >::node_data_type >,
   "IdentityShift must remain a pure (non-contextual) shift."
);

}  // namespace nor::rm

#endif  // NOR_RM_MINIMIZERS_PREDICTIVE_HPP
