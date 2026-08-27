
#ifndef NOR_RNR_HPP
#define NOR_RNR_HPP

#include <stdexcept>

#include "nor/rm/cfr_tabular/cfr.hpp"
#include "nor/rm/opponent_aware/opponent_model.hpp"
#include "nor/rm/opponent_aware/response.hpp"

namespace nor::opponent_aware {

/**
 * RESTRICTED NASH RESPONSE (RNR) -- Johanson, Zinkevich & Bowling, "Computing Robust
 * Counter-Strategies to Opponent Models", NeurIPS 2007.
 *
 * Computes a counter-strategy to the opponent model 'sigma_fix' by solving (with CFR) the
 * modified game in which the modeled player is FORCED to play the model with probability p and
 * may play freely (an adversarially optimized strategy) with probability 1 - p. The responding
 * player's component of that modified game's equilibrium interpolates between best response
 * against the model and safe equilibrium play:
 *
 * PARAMETER CONVENTION (paper-faithful; see their eq. (5) and sec. 4):
 *   forcing_probability = 1  ->  pure best response against the model (maximally exploitative,
 *                               maximally exploitable);
 *   forcing_probability = 0  ->  unrestricted self-play, i.e. a Nash equilibrium of the
 *                               unmodified game (minimally exploitable, non-exploitative).
 * In between, exploitation of the model rises while worst-case exploitability falls as p
 * shrinks (their Figure 1 trade-off envelope; every RNR lies on the epsilon-safe-best-response
 * frontier for its induced epsilon -- their Thm. 1).
 *
 * IMPLEMENTATION NOTES / DEVIATIONS:
 *   - Mechanics: at every traversal visit to an infostate of the modeled player the PLAYED
 *     edge probabilities are blended as p * model(I) + (1 - p) * sigma_current(I)
 *     (rm::OpponentBlendPolicy); regret/counterfactual-value updates run unmodified on that
 *     play. This realizes the restricted strategy set Sigma^{p, sigma_fix} = {mix_p(sigma_fix,
 *     sigma') : sigma'} through PER-INFOSTATE behavioral blending instead of the paper's single
 *     unobserved root chance node ("forced on the current hand", correlating all decisions of
 *     one hand). The two constructions coincide exactly when every history visits the modeled
 *     player at most once (e.g. two-player kuhn poker and rock-paper-scissors); for deeper
 *     trees per-infostate blending is the same device the DBR follow-up prescribes explicitly,
 *     and it preserves both edge cases and the qualitative safety/exploitation interpolation of
 *     their Thm. 1.
 *   - Bookkeeping: only the RESPONDING player's average strategy is consumed; the modeled
 *     player's stored tables keep tracking their free component (the blend is applied strictly
 *     per visit and restored afterwards), so their realized-play profile is reconstructible
 *     downstream but not stored.
 *
 * SAFETY CAVEATS (prominent, from the papers' setting):
 *   - ZERO-SUM ONLY. All guarantees (epsilon-safety of the response, equilibrium recovery at
 *     p = 0) are two-player zero-sum statements. In general-sum or multi-player settings the
 *     computed strategy carries NO safety property whatsoever.
 *   - MODEL-CLASS ASSUMPTIONS. Robustness is relative to opponents that either play like the
 *     model or adversarially optimize AROUND it in the modified-game sense. There is NO
 *     guarantee against arbitrary adversaries outside this class, and no protection when the
 *     model itself was fitted adversarially (poisoned observations).
 *   - CONVERGENCE IS APPROXIMATE. CFR on the modified game converges asymptotically; finite
 *     iteration counts trade exploitability against residual model-overfitting noise exactly
 *     like plain CFR trades it against exploitability.
 *
 * @tparam config the rm::CFRConfig of the underlying vanilla CFR solver; the opponent-blend
 *                 mode is enabled automatically. Invalid combinations (warm start, pruning,
 *                 lazy updates, extragradient iterations) are statically rejected by the
 *                 solver's configuration check.
 * @param env, root_state the game.
 * @param responder the player computing the response (the other participant is the modeled one).
 * @param opponent_model_policy the fixed model table; any type exposing .at(infostate) whose
 *                              mapped type exposes .at(action) -> double covering EVERY legal
 *                              action at every infostate of the modeled player (e.g.
 *                              nor::TabularPolicy over HashmapActionPolicy).
 * @param forcing_probability the global blend weight p in [0, 1] (paper-faithful direction:
 *                            1 = best response, 0 = Nash equilibrium).
 * @param n_iterations number of CFR iterations to run.
 */
template < rm::CFRConfig config = {}, typename Env, typename ModelPolicyTable >
[[nodiscard]] auto rnr_response(
   Env&& env,
   const auto_world_state_type< std::remove_cvref_t< Env > >& root_state,
   Player responder,
   const ModelPolicyTable& opponent_model_policy,
   double forcing_probability,
   size_t n_iterations
)
{
   using env_type = std::remove_cvref_t< Env >;
   using info_state_type = auto_info_state_type< env_type >;
   using action_type = auto_action_type< env_type >;
   if(forcing_probability < 0. or forcing_probability > 1.) {
      throw std::invalid_argument("rnr_response: the forcing probability must lie in [0, 1].");
   }
   auto model = make_fixed_opponent_model< info_state_type, action_type >(
      opponent_model_policy, forcing_probability
   );
   return detail::solve_modified_game< config >(
      std::forward< Env >(env), root_state, responder, model, n_iterations
   );
}

}  // namespace nor::opponent_aware

#endif  // NOR_RNR_HPP
