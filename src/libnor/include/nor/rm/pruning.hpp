
#ifndef NOR_RM_PRUNING_HPP
#define NOR_RM_PRUNING_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "nor/concepts.hpp"
#include "nor/rm/cfr_tabular/cfr_config.hpp"

namespace nor::rm::pruning {

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// payoff bounds ///////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

/// conservative stand-in for the paper's U/L quantities: the lower/upper payoff reachable for
/// one player. Resolved once per solver (B4 env trait when supported, otherwise a single
/// full-tree probe -- see VanillaCFR::_payoff_bound in cfr.tcc).
struct PayoffBound {
   double lower = 0.;
   double upper = 0.;
   [[nodiscard]] double range() const { return upper - lower; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// per-(infostate,action) tables
///////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Worst-case number of iterations D(I,a) may be skipped once its regret is known.
///
/// Transcription of Theorem 1 of Brown & Sandholm, "Regret-Based Pruning in Extensive-Form
/// Games", NIPS 2015:
///    m = floor( |R(I,a)| / (U(I,a) - L(I)) )
/// where U(I,a)/L(I) bound the maximum/minimum payoff reachable for P(I) after action a and
/// from I respectively. The proof uses r^t(I,a) <= U(I,a) - L(I), so R^t(I,a) <= 0 for all
/// T0 <= t <= T0 + m provided sigma(I,a) = 0 whenever R(I,a) <= 0.
///
/// 'payoff_range' is the conservative stand-in for U(I,a) - L(I) (e.g. the probed per-player
/// payoff range hi - lo). A non-positive range (degenerate subtree) yields the maximum
/// representable window.
[[nodiscard]] inline size_t theorem1_window(double cumulative_regret, double payoff_range)
{
   if(cumulative_regret >= 0.) {
      return 0;
   }
   const double range = payoff_range > 0. ? payoff_range : std::numeric_limits< double >::min();
   constexpr double max_m = double(std::numeric_limits< size_t >::max() / 2);
   const double m = std::min(std::floor(std::abs(cumulative_regret) / range), max_m);
   return static_cast< size_t >(m);
}

/// Upper bound on one iteration's regret increment r^t(I,a) = pi_{-i}(I) * (v(I->a) - v(I))
/// used by the window-length computation. Without re-weighting, r^t <= U(I,a) - L(I). Under
/// exponential weighting (ExponentialCFR) increments are scaled by the L1 factor
/// exp(r(I,a) - mean_r(I)), itself bounded by exp(range) since all instantaneous entries of one
/// iteration lie within a range-sized band around their mean -- hence the inflated bound.
[[nodiscard]] inline double window_increment_bound(CFRWeightingMode weighting, double payoff_range)
{
   if(weighting == CFRWeightingMode::exponential) {
      return payoff_range * std::exp(payoff_range);
   }
   return payoff_range;
}

/// Dynamic continuation condition of a pruning window -- transcription of eq. (9) of NIPS'15:
/// D(I,a) stays pruned from T0 until T1 as long as
///    sum_{t<=T0} v(I,a) + sum_{T0<t<=T1} pi_{-i}(I) * U(I,a)  <=  sum_{t<=T1} v(I).
/// The running left-hand side minus right-hand side (the 'pessimistic_regret' table entry)
/// upper-bounds the true cumulative regret because v(I->a) <= U(I,a) pointwise; while it is
/// non-positive the window continues, once it turns positive the buffered best-response values
/// fold in and normal traversal resumes.
[[nodiscard]] inline bool pessimistic_unfold_required(double pessimistic_regret_estimate)
{
   return pessimistic_regret_estimate > 0.;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////// dynamic thresholding schedules (Brown, Kroer, Sandholm '17)
//////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Dynamic-thresholding schedule for REGRET MATCHING recommenders -- transcription of Theorem 2
/// of Brown, Kroer, Sandholm, "Dynamic Thresholding and Pruning for Regret Minimization",
/// AAAI 2017 (DOI 10.1609/aaai.v31i1.10603):
///    tau_t = (C^2 - 1) / (2 C |A(I)|^2 sqrt(t))
/// with C >= 1 and t the LOGICAL (1-based) iteration. Zeroing actions below tau_t and
/// renormalizing keeps R^T(I) <= C Delta(I) sqrt(|A(I)| sqrt(T)).
[[nodiscard]] inline double
rm_dynamic_threshold(size_t n_actions, size_t logical_iteration, double c)
{
   if(c <= 1.) {
      return 0.;
   }
   const double n = static_cast< double >(std::max(n_actions, size_t{1}));
   const double t = static_cast< double >(std::max(logical_iteration, size_t{1}));
   return ((c * c) - 1.) / (2. * c * n * n * std::sqrt(t));
}

/// Dynamic-thresholding schedule for HEDGE/exponential-weighted recommenders -- transcription of
/// their Theorem 1:
///    tau_t = (C - 1) sqrt(ln |A(I)|) / (sqrt(2) |A(I)|^2 sqrt(t))
/// keeping R^T(I) <= C sqrt(2) Delta(I) sqrt(ln(|A(I)|)) sqrt(T). Our ExponentialCFR recommends
/// through regret matching on its cumulative table (see minimizers.hpp), so the RM schedule is
/// the theoretically matching choice there; this Hedge variant transcribes their headline
/// theorem for completeness and for unit tests of both formulas.
[[nodiscard]] inline double
hedge_dynamic_threshold(size_t n_actions, size_t logical_iteration, double c)
{
   if(c <= 1.) {
      return 0.;
   }
   const double n = static_cast< double >(std::max(n_actions, size_t{1}));
   const double t = static_cast< double >(std::max(logical_iteration, size_t{1}));
   return ((c - 1.) * std::sqrt(std::log(n))) / (std::sqrt(2.) * n * n * std::sqrt(t));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// per-(infostate,action) tables
///////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Per-(infostate,action) bookkeeping of regret-based pruning windows. One instance lives inside
/// the node data of every minimizer that participates in RBP-style gating (RegretMatchingPlusRBP
/// directly, Thresholded<Inner> as its own member); entries stay index-aligned with the
/// infostate's action registry via register_action().
///
/// Field semantics (the named fields required by the solver spec):
///    last_pruned_iteration -- iteration at which the most recent window was ARMED (its T0);
///                             0 == never pruned.
///    pruned_until          -- exclusive deadline: edges are skipped while iteration() <
///                             pruned_until; 0 == no window active. Set to T0 + m with m from
///                             theorem1_window().
///    br_regret_buffer      -- sum over window visits of pi_{-i}(h) * (v_BR(h,a) - v(I)); folded
///                             into the regret table at unfold time ("update the regrets to
///                             match this", NIPS'15 sec. 4).
///    pessimistic_regret    -- running left-hand side of the eq-(9) continuation condition,
///                             initialized to R^{T0}(I,a), incremented per visit by
///                             pi_{-i}(h) * (U(I,a) - v(I)).
///    cached_br_value       -- memoized value of the last periodic BR traversal of D(h,a).
///    visits_since_refresh  -- visits since cached_br_value was recomputed (refresh cadence).
struct RBPTables {
   std::vector< size_t > last_pruned_iteration;
   std::vector< size_t > pruned_until;
   std::vector< double > br_regret_buffer;
   std::vector< double > pessimistic_regret;
   std::vector< double > cached_br_value;
   std::vector< size_t > visits_since_refresh;

   void register_action()
   {
      last_pruned_iteration.emplace_back(0);
      pruned_until.emplace_back(0);
      br_regret_buffer.emplace_back(0.);
      pessimistic_regret.emplace_back(0.);
      cached_br_value.emplace_back(0.);
      visits_since_refresh.emplace_back(0);
   }

   [[nodiscard]] bool is_pruned(size_t idx, size_t current_iteration) const
   {
      return pruned_until[idx] != 0 and current_iteration < pruned_until[idx];
   }
};

}  // namespace nor::rm::pruning

#endif  // NOR_RM_PRUNING_HPP
