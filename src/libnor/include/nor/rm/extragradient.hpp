
#ifndef NOR_RM_EXTRAGRADIENT_HPP
#define NOR_RM_EXTRAGRADIENT_HPP

#include <cstddef>

namespace nor::rm {

/**
 * @brief Extragradient RM+ ("ExRM+") / Clairvoyant CFR solver-level engine.
 *
 * Implements the extragradient member of the stabilized-predictive-RM+ family
 * (Farina, Grand-Clément, Kroer, Lee, Luo, "Regret Matching+: (In)Stability and
 * Fast Convergence in Games", NeurIPS 2023, arXiv:2305.14709) -- their
 * Algorithm 5, which is Algorithm 4 ("Conceptual RM+ with approximate
 * fixed-point") restricted to a SINGLE fixed-point iteration (k = 1), the same
 * one-extrapolation instantiation the authors evaluate as "Clairvoyant CFR" for
 * extensive-form games (their sec. 6).
 *
 * PAPER ALGORITHM (normal form; X_>= is the chopped-off orthant, g the L1
 * normalization, F the joint gradient operator):
 *    w^t = Pi_{z^{t-1}, X_>=}(eta F(z^{t-1}))     (extrapolation/probe step)
 *    z^t = Pi_{z^{t-1}, X_>=}(eta F(w^t))         (real update AT the probe point)
 *    x^t = g(w^t)                                 (the PLAYED strategy)
 * With the RM+ prox over the CHOPPED-OFF orthant (Euclidean regularizer,
 * eta = 1) both projections collapse to componentwise clipping plus a lift of
 * a vanishing 1-norm to the floor epsilon ("chopping off" the origin, the
 * paper's Smooth-PRM+ device -- an INTEGRAL part of Algorithm 5, whose
 * decision set is X_>=):
 *    w^t = floor([z^{t-1} + eta r(z^{t-1})]^+),
 *    z^t = floor([z^{t-1} + eta r(w^t)]^+)
 * where r(x) denotes the instantaneous-regret vector generated under x (r =
 * -F by the f(x, l) = l - <x, l> 1 convention of the paper's Lemma 2.1) and
 * floor(.) = . + max(0, epsilon - ||.||_1)/n 1. The floor is ESSENTIAL in
 * practice: without it z can stay pinned at the origin whenever the realized
 * regrets at the intermediate point are all non-positive (observed immediately
 * on kuhn poker -- the exact instability-at-the-origin phenomenon the paper
 * introduces stabilization for), deadlocking the scheme at uniform play.
 *
 * CFR INSTANTIATION (two full traversals per global iteration; alternating
 * updates with updating player p):
 *    1. ANCHOR PROBE traversal: values-only walk under the current profile
 *       (p plays g(z^{t-1})); buffers the counterfactual increments
 *       r_anchor(I, a) = pi_{-p}(h) (v(h, a) - v(h)) for every infoset I of p.
 *       No minimizer/average state is mutated -- this pass only MEASURES
 *       F at the anchor point g(z^{t-1}).
 *    2. INTERMEDIATE STEP (no traversal): per infoset,
 *          w^t(I) = [z^{t-1}_I + eta r_anchor(I)]^+,
 *       written into the current-policy tables (g(w^t) = sigma^{t+1/2});
 *       the stored cumulative table z^{t-1} remains untouched so that it can
 *       serve as the ANCHOR of the real update below (both proxes of
 *       Algorithm 5 share the base z^{t-1}).
 *    3. REAL traversal under sigma^{t+1/2}: defers its counterfactual
 *       increments into a buffer and accumulates the average strategy from
 *       sigma^{t+1/2} weighted by own reach -- Algorithm 5 line 6 makes the
 *       INTERMEDIATE strategy the algorithm's decision x^t, so it (not the
 *       post-update recommendation) is what enters the average.
 *    4. End-of-iteration sweep: fold z^t = [z^{t-1} + eta r_real]^+ (one batch
 *       observe per action; the RM+ kernel clamps at recommendation time,
 *       reproducing the clipped sum exactly) and refresh the recommendations
 *       to g(z^t), restoring the invariant "current policy = g(z)" that the
 *       next iteration's anchor probe relies on.
 *
 * THEORY VS. THIS CARRIER (documented deviations):
 *    - The paper's O(1)-social-regret guarantee (Theorem 5.6, with
 *      eta = (sqrt(2) L_F)^{-1}) covers SIMULTANEOUS updates of all players
 *      inside the joint operator F. Following the house convention for the
 *      whole predictive/stabilized family, this carrier statically pins
 *      ALTERNATING updates instead (each player's probes measure only his own
 *      counterfactual components). The scheme stays a sound regret-minimizing
 *      CFR variant, but the constant-regret constants are not claimed here --
 *      exactly like the authors' own EFG experiments, which use the
 *      single-fixed-point variant as a heuristic step-size hyperparameter
 *      method ("clairvoyant only in spirit", sec. 6).
 *    - The X_>= origin-chopping IS applied (with epsilon = 1, the paper's WLOG
 *      normalization; knob CFRConfig::extragradient_norm_floor_epsilon) --
 *      Algorithm 5's decision set requires it, and the plain orthant variant
 *      deadlocks at the origin on poker-like games. Composing the anchor-probe
 *      engine with the RESTARTING (Stable-PRM+) stabilization is left as
 *      future work.
 *    - The paper reports weak EFG performance when the contraction-mandated
 *      (prohibitively small) step sizes are used; larger eta restores
 *      practical convergence but voids the fixed-point existence argument.
 *      The step size is therefore exposed as a knob
 *      (CFRConfig::extragradient_stepsize, default 1).
 */
struct ExtragradientStats {
   /// number of anchor probe traversals launched (one per global iteration)
   size_t probe_traversals = 0;
   /// number of real update traversals launched (one per global iteration)
   size_t real_traversals = 0;
};

}  // namespace nor::rm

#endif  // NOR_RM_EXTRAGRADIENT_HPP
