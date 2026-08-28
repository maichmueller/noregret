
#ifndef NOR_OPPONENT_AWARE_RESPONSE_HPP
#define NOR_OPPONENT_AWARE_RESPONSE_HPP

#include <ranges>
#include <stdexcept>
#include <utility>

#include "common/common.hpp"
#include "nor/game_defs.hpp"
#include "nor/policy/policy.hpp"
#include "nor/rm/cfr_tabular/cfr.hpp"
#include "nor/rm/opponent_aware/opponent_model.hpp"
#include "nor/utils/utils.hpp"

namespace nor::opponent_aware {

/**
 * Outcome of an opponent-aware solving pass of the modified game (RNR / DBR family).
 *
 * SAFETY REMINDER: the underlying guarantees (Johanson, Zinkevich & Bowling, NeurIPS 2007,
 * Thm. 1; Johanson & Bowling, AISTATS 2009) hold for TWO-PLAYER ZERO-SUM games and are
 * relative to the assumed model class -- they say nothing about adversaries that behave
 * outside the model-or-adapt-around-it envelope.
 */
template < typename Env >
struct ResponseResult {
   using info_state_type = auto_info_state_type< Env >;
   using action_type = auto_action_type< Env >;
   /// normalized behavioral strategy of the responding player (the robust counter-strategy)
   std::unordered_map< info_state_type, HashmapActionPolicy< action_type > > policy{};
   /// root value of every player under the FINAL played (blended) profile of the last
   /// iteration
   player_hashmap< double > final_root_values{};
};

/// enables the per-infostate opponent blend on a copy of 'solver_config' (compile-time)
[[nodiscard]] inline consteval rm::CFRConfig with_blend_enabled(rm::CFRConfig solver_config)
{
   solver_config.opponent_blend_mode = rm::CFROpponentBlendMode::per_infostate_blend;
   return solver_config;
}

namespace detail {

/**
 * Value-hashed behavioral policy table used canonically inside the opponent-aware family --
 * keyed by infostate VALUE, exactly how CFR engine tables and frequency tables identify
 * infostates. Environment-provided plain std::hash specializations are frequently coarser
 * than the solver's internal identity notion, so raw tables are FUNNELED through this
 * representation before any cross-machinery lookup.
 */
template < typename InfoStateType, typename ActionType >
using canonical_policy_table = std::unordered_map<
   InfoStateType,
   HashmapActionPolicy< ActionType >,
   common::value_hasher< InfoStateType >,
   common::value_comparator< InfoStateType > >;

/**
 * Copies any infostate->action-policy style container (raw std-hashed / value-hashed maps,
 * TabularPolicy, ...) into the canonical value-hashed table above.
 */
template < typename InfoStateType, typename ActionType, typename StrategyTable >
[[nodiscard]] auto canonicalize_strategy(const StrategyTable& strategy)
   -> canonical_policy_table< InfoStateType, ActionType >
{
   using action_policy_type = HashmapActionPolicy< ActionType >;
   canonical_policy_table< InfoStateType, ActionType > out{};
   for(const auto& [infostate, action_policy] : strategy) {
      action_policy_type copied{};
      for(const auto& [action, prob] : normalize_action_policy(action_policy)) {
         copied[action] = prob;
      }
      out.emplace(infostate, std::move(copied));
   }
   return out;
}

}  // namespace detail

/**
 * Expected value of 'responder' playing behavioral strategy 'responder_strategy' against
 * 'opponent_strategy' (both may be raw infostate->action-policy maps or nor::TabularPolicy).
 */
template < typename Env, typename ResponderStrategy, typename OpponentStrategy >
[[nodiscard]] double value_against(
   Env&& env,
   const auto_world_state_type< std::remove_cvref_t< Env > >& root_state,
   Player responder,
   const ResponderStrategy& responder_strategy,
   const OpponentStrategy& opponent_strategy
)
{
   using env_type = std::remove_cvref_t< Env >;
   using info_state_type = auto_info_state_type< env_type >;
   using action_type = auto_action_type< env_type >;
   auto players = env.players(root_state);
   std::erase(players, Player::chance);
   if(players.size() != 2) {
      throw std::invalid_argument(
         "opponent_aware: the RNR/DBR/DBBR family is implemented for two-player games only."
      );
   }
   // canonicalize both sides so heterogeneously hashed input tables compose in one profile
   using canonical_table = detail::canonical_policy_table< info_state_type, action_type >;
   using tabular_policy_type = TabularPolicy<
      info_state_type,
      HashmapActionPolicy< action_type >,
      canonical_table >;
   tabular_policy_type responder_policy{
      detail::canonicalize_strategy< info_state_type, action_type >(responder_strategy)};
   tabular_policy_type opponent_policy{
      detail::canonicalize_strategy< info_state_type, action_type >(opponent_strategy)};
   Player opponent = players.front() == responder ? players.back() : players.front();
   return rm::policy_value(
             std::forward< Env >(env),
             root_state,
             player_hashmap{
                std::pair{responder, StatePolicyView{responder_policy}},
                std::pair{opponent, StatePolicyView{opponent_policy}}}
   )
      .get()
      .at(responder);
}

namespace detail {

/// runs vanilla CFR on the blend-modified game prescribed by 'model' (see
/// rm::CFROpponentBlendMode::per_infostate_blend): at every traversal visit to an infostate
/// of the modeled player the PLAYED edge probabilities follow
///    Pconf(I) * model(I) + (1 - Pconf(I)) * sigma_current(I),
/// while regret and counterfactual-value updates run unmodified on that play. The
/// responding player's average strategy converges to the robust counter-strategy of the
/// modified game; the modeled player's stored tables keep tracking their free component.
template < rm::CFRConfig config, typename Env, typename OpponentModelType >
[[nodiscard]] auto solve_modified_game(
   Env&& env,
   const auto_world_state_type< std::remove_cvref_t< Env > >& root_state,
   Player responder,
   const OpponentModelType& model,
   size_t n_iterations
)
{
   using env_type = std::remove_cvref_t< Env >;
   using info_state_type = auto_info_state_type< env_type >;
   using action_type = auto_action_type< env_type >;
   using tabular_policy_type = TabularPolicy< info_state_type, HashmapActionPolicy< action_type > >;

   auto all_players = env.players(root_state);
   std::erase(all_players, Player::chance);
   if(all_players.size() != 2) {
      throw std::invalid_argument(
         "opponent_aware: the RNR/DBR family is implemented for two-player games only."
      );
   }
   if(not common::isin(responder, all_players)) {
      throw std::invalid_argument("opponent_aware: the responder is not a participant.");
   }
   Player modeled_player = all_players.front() == responder ? all_players.back()
                                                            : all_players.front();

   // the per-infostate blend selector handed to the solver (cf. the WarmStartPolicy mechanics)
   rm::opponent_blend_policy_selector_t< env_type > selector{};
   selector.blend = [&model, modeled_player](
                       const info_state_type& infostate, const std::vector< action_type >& actions
                    )
      -> std::optional<
         typename rm::OpponentBlendPolicy< info_state_type, action_type >::BlendSpec > {
      if(infostate.player() != modeled_player) {
         return std::nullopt;
      }
      const double confidence = model.confidence(infostate);
      if(not (confidence >= 0. and confidence <= 1.)) {
         throw std::invalid_argument(
            "opponent_aware: the model confidence must lie in [0, 1] at every infostate."
         );
      }
      if(confidence <= 0.) {
         // no trust in the model here: play the free component only
         return std::nullopt;
      }
      typename rm::OpponentBlendPolicy< info_state_type, action_type >::BlendSpec spec{};
      spec.forced_probability = confidence;
      spec.model_distribution = model.policy(infostate, actions);
      for(const auto& action : actions) {
         if(not spec.model_distribution.contains(action)) {
            throw std::invalid_argument(
               "opponent_aware: the opponent model does not cover every legal action at the "
               "requested infostate."
            );
         }
      }
      return spec;
   };

   rm::VanillaCFR< with_blend_enabled(config), env_type, tabular_policy_type, tabular_policy_type >
      solver{
         std::move(selector),
         std::forward< Env >(env),
         utils::static_unique_ptr_downcast< auto_world_state_type< env_type > >(
            utils::clone_any_way(root_state)
         ),
         tabular_policy_type{},
         tabular_policy_type{}};

   auto root_values_per_iteration = solver.iterate(n_iterations);

   ResponseResult< env_type > result{};
   result.final_root_values = root_values_per_iteration.back();
   result.policy = normalize_state_policy(solver.average_policy().at(responder).table());
   return result;
}

}  // namespace detail

}  // namespace nor::opponent_aware

#endif  // NOR_OPPONENT_AWARE_RESPONSE_HPP
