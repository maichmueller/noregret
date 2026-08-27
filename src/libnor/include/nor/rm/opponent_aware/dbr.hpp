
#ifndef NOR_DBR_HPP
#define NOR_DBR_HPP

#include <concepts>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "nor/rm/cfr_tabular/cfr.hpp"
#include "nor/rm/opponent_aware/opponent_model.hpp"
#include "nor/rm/opponent_aware/response.hpp"

namespace nor::opponent_aware {

/**
 * DATA BIASED ROBUST COUNTER-STRATEGY (DBR) -- Johanson & Bowling, "Data Biased Robust
 * Counter Strategies", AISTATS 2009.
 *
 * Generalizes RNR from one global forcing weight p to a PER-INFOSTATE confidence
 *    Pconf(I) = p_max(n_I),
 * a function of the number n_I of observations recorded at infostate I (the paper's sec. 5
 * families: n-Step, 0-10 Linear, s-Curve -- see step_confidence/linear_confidence/
 * scurve_confidence). The modified game forces the modeled player at infostate I onto the
 * empirical frequency distribution of the data with probability Pconf(I) and leaves them free
 * with probability 1 - Pconf(I); solving it with CFR yields the responding player's robust
 * counter-strategy. Where nothing was observed (Pconf = 0) the response degrades to equilibrium
 * play instead of trusting a default policy -- precisely the paper's remedy against RNR's three
 * failure modes (default-policy overfitting, data-quantity brittleness, source-of-data
 * brittleness).
 *
 * Edges: total absence of data everywhere behaves as plain CFR self-play; uniformly constant
 * confidence degenerates to RNR with that weight.
 *
 * IMPLEMENTATION NOTES / DEVIATIONS:
 *   - Same per-infostate, visit-scoped blend mechanics and bookkeeping deviations as
 *     rnr_response (see there for the full list); DBR's own formulation is already
 *     per-infostate, so no root-chance-node deviation arises at all here.
 *   - The model distribution is the maximum-likelihood (raw normalized frequency) estimate;
 *     no smoothing is applied. All-or-nothing support estimates are inherent to small samples
 *     at Pconf close to 1.
 *
 * SAFETY CAVEATS (prominent, from the papers' setting):
 *   - ZERO-SUM ONLY: guarantees hold for two-player zero-sum games relative to the model class.
 *   - DATA REPRESENTATIVENESS: exploitation follows the SAMPLED frequencies; biased or poisoned
 *     observation streams are faithfully exploited into mistakes. No safety statement holds
 *     against opponents whose strategy drifts away from the sampling distribution.
 *
 * @tparam config the rm::CFRConfig of the underlying vanilla CFR solver (blend mode enabled
 *                 automatically).
 * @param env, root_state the game.
 * @param responder the player computing the response (other participant is the modeled one).
 * @param counts the FrequencyTable<Infostate, Action> of observed opponent actions.
 * @param pconf the confidence function over observation counts into [0, 1]; build it with
 *              step_confidence / linear_confidence / scurve_confidence or supply any callable
 *              double(double) -> [0,1].
 * @param n_iterations number of CFR iterations to run.
 */
template < rm::CFRConfig config = {}, typename Env, typename ConfidenceFn >
   requires std::invocable< ConfidenceFn, double >
            and std::convertible_to< std::invoke_result_t< ConfidenceFn, double >, double >
[[nodiscard]] auto dbr_response(
   Env&& env,
   const auto_world_state_type< std::remove_cvref_t< Env > >& root_state,
   Player responder,
   const FrequencyTable<
      auto_info_state_type< std::remove_cvref_t< Env > >,
      auto_action_type< std::remove_cvref_t< Env > > >& counts,
   ConfidenceFn&& pconf,
   size_t n_iterations
)
{
   using info_state_type = auto_info_state_type< std::remove_cvref_t< Env > >;
   using action_type = auto_action_type< std::remove_cvref_t< Env > >;
   auto model = make_frequency_opponent_model< info_state_type, action_type >(
      counts,
      [pconf = std::forward< ConfidenceFn >(pconf), &counts](const info_state_type& infostate) {
         return static_cast< double >(pconf(observation_count(counts, infostate)));
      }
   );
   return detail::solve_modified_game< config >(
      std::forward< Env >(env), root_state, responder, model, n_iterations
   );
}

}  // namespace nor::opponent_aware

#endif  // NOR_DBR_HPP
