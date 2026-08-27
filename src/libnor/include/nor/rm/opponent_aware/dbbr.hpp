
#ifndef NOR_DBBR_HPP
#define NOR_DBBR_HPP

#include <algorithm>
#include <cmath>
#include <concepts>
#include <ranges>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "common/common.hpp"
#include "nor/factory.hpp"
#include "nor/game_defs.hpp"
#include "nor/policy/best_response.hpp"
#include "nor/policy/policy.hpp"
#include "nor/rm/forest.hpp"
#include "nor/rm/opponent_aware/opponent_model.hpp"
#include "nor/rm/opponent_aware/response.hpp"
#include "nor/rm/policy_value.hpp"
#include "nor/utils/utils.hpp"

namespace nor::opponent_aware {

/**
 * Behavioral policy table alias used throughout the DBBR machinery. Keyed by infostate VALUE
 * (value_hasher/value_comparator), mirroring how the CFR engine and the frequency tables
 * identify infostates; plain std::hash-keyed tables have repeatedly proven unsafe to mix with
 * those code paths because environment-provided std::hash specializations may be coarser than
 * the internal equality notion.
 */
template < typename InfoStateType, typename ActionType >
using policy_table_type = std::unordered_map<
   InfoStateType,
   HashmapActionPolicy< ActionType >,
   common::value_hasher< InfoStateType >,
   common::value_comparator< InfoStateType > >;

/**
 * Outcome of a DBBR pass: the projected model handed to the best response, the resulting exact
 * best response, and the root values under (response vs. model).
 */
template < typename Env >
struct DbbResult {
   using info_state_type = auto_info_state_type< Env >;
   using action_type = auto_action_type< Env >;
   using policy_table_type = nor::opponent_aware::policy_table_type< info_state_type, action_type >;

   /// fully specified PROJECTED opponent model feeding the best-response computation
   policy_table_type model{};
   /// the exact best response against 'model' (the counter-strategy to play)
   policy_table_type policy{};
   /// root value of every player under (policy, model)
   player_hashmap< double > root_values{};
};

/**
 * Enumerates the exact set {infostate -> legal actions} reachable by every actual player via
 * one full game-tree walk (shared edge mechanics with the best-response walk).
 */
template < typename Env >
[[nodiscard]] auto enumerate_infostate_actions(
   Env& env,
   const auto_world_state_type< std::remove_cvref_t< Env > >& root_state
)
{
   using env_type = std::remove_cvref_t< Env >;
   using info_state_type = auto_info_state_type< env_type >;
   using action_type = auto_action_type< env_type >;
   using observation_type = auto_observation_type< env_type >;

   struct VisitData {
      std::unordered_map< Player, info_state_type > infostates{};
      std::unordered_map< Player, std::vector< std::pair< observation_type, observation_type > > >
         observation_buffer{};
   };

   using table_type = std::unordered_map<
      info_state_type,
      std::vector< action_type >,
      common::value_hasher< info_state_type >,
      common::value_comparator< info_state_type > >;

   table_type table{};

   // the root's actor owns an infostate before any edge is ever taken
   if(auto root_player = env.active_player(root_state); root_player != Player::chance) {
      info_state_type root_infostate{root_player};
      table.emplace(std::move(root_infostate), env.actions(root_player, root_state));
   }

   forest::GameTreeTraverser< env_type > traverser(env);
   // seed the root visitation data for every ACTUAL player exactly like the CFR engine's
   // _iterate does: one fresh infostate plus an empty observation buffer per player (the
   // buffering helpers .at() into these containers along every edge)
   VisitData root_visit{};
   for(auto player : env.players(root_state)) {
      if(player == Player::chance) {
         continue;
      }
      root_visit.infostates.emplace(player, info_state_type{player});
      root_visit.observation_buffer.emplace(
         player, std::vector< std::pair< observation_type, observation_type > >{}
      );
   }
   traverser.walk(
      utils::static_unique_ptr_downcast< auto_world_state_type< env_type > >(
         utils::clone_any_way(root_state)
      ),
      std::move(root_visit),
      forest::TraversalHooks{
         .child_hook =
            [&env, &table](
               VisitData& visit_data, auto&& curr_action, auto&& curr_state, auto&& next_state
            ) -> VisitData {
            if(env.is_terminal(*next_state)) {
               return {};
            }
            auto [child_buffer, child_infostates] = std::visit(
               common::Overload{
                  [&](const auto& action_or_outcome) {
                     return next_infostate_and_obs_buffers(
                        env,
                        visit_data.observation_buffer,
                        visit_data.infostates,
                        *curr_state,
                        action_or_outcome,
                        *next_state
                     );
                  },
                  [&](const std::monostate&)
                     -> std::pair<
                        std::unordered_map<
                           Player,
                           std::vector< std::pair< observation_type, observation_type > > >,
                        std::unordered_map< Player, info_state_type > > {
                     throw std::logic_error("We entered a std::monostate visit branch.");
                  }},
               *curr_action
            );
            if(auto player = env.active_player(*next_state); player != Player::chance) {
               if(auto found = child_infostates.find(player); found != child_infostates.end()) {
                  table.emplace(found->second, env.actions(player, *next_state));
               }
            }
            return VisitData{
               .infostates = std::move(child_infostates),
               .observation_buffer = std::move(child_buffer)};
         }}
   );
   return table;
}

namespace detail {

/**
 * PROJECTION + COMPLETION step of DBBR: builds the fully specified behavioral strategy of
 * 'modeled_player' from partial frequency observations. Observed infostates take the
 * maximum-likelihood empirical distribution over their LEGAL actions; every other infostate is
 * completed through 'completer(infostate, legal_actions)' which must return a normalized
 * distribution as pairs.
 */
template < typename Env, typename CountsTable, typename CompleterFn >
[[nodiscard]] auto project_frequency_strategy(
   Env&& env,
   const auto_world_state_type< std::remove_cvref_t< Env > >& root_state,
   Player modeled_player,
   const CountsTable& counts,
   CompleterFn completer
)
{
   using env_type = std::remove_cvref_t< Env >;
   using info_state_type = auto_info_state_type< env_type >;
   using action_type = auto_action_type< env_type >;

   policy_table_type< info_state_type, action_type > model{};
   for(const auto& [infostate, actions] : enumerate_infostate_actions(env, root_state)) {
      if(infostate.player() != modeled_player) {
         continue;
      }
      if(observation_count(counts, infostate) > 0.) {
         model.emplace(
            infostate,
            HashmapActionPolicy< action_type >{normalized_frequencies(counts, infostate, actions)}
         );
      } else {
         auto& entry = model.emplace(infostate, HashmapActionPolicy< action_type >{}).first->second;
         for(const auto& [action, prob] : completer(infostate, actions)) {
            entry[action] = prob;
         }
      }
   }
   return model;
}

/**
 * RESPONSE step of DBBR: exact best response against the projected model plus bookkeeping
 * (DbbResult assembly). Split out so the public overloads only differ in their completion
 * rule.
 */
template < typename Env, typename CountsTable, typename CompleterFn >
[[nodiscard]] auto solve_dbb(
   Env&& env,
   const auto_world_state_type< std::remove_cvref_t< Env > >& root_state,
   Player responder,
   const CountsTable& counts,
   CompleterFn completer
)
{
   using env_type = std::remove_cvref_t< Env >;
   using info_state_type = auto_info_state_type< env_type >;
   using action_type = auto_action_type< env_type >;
   using canonical_table = detail::canonical_policy_table< info_state_type, action_type >;
   using tabular_policy_type = TabularPolicy<
      info_state_type,
      HashmapActionPolicy< action_type >,
      canonical_table >;

   auto all_players = env.players(root_state);
   std::erase(all_players, Player::chance);
   if(all_players.size() != 2) {
      throw std::invalid_argument(
         "opponent_aware: the RNR/DBR/DBBR family is implemented for two-player games only."
      );
   }
   if(not common::isin(responder, all_players)) {
      throw std::invalid_argument("opponent_aware: the responder is not a participant.");
   }
   Player modeled_player = all_players.front() == responder ? all_players.back()
                                                            : all_players.front();

   DbbResult< env_type > result{};
   auto model_map = project_frequency_strategy(
      std::forward< Env >(env), root_state, modeled_player, counts, completer
   );
   result.model = model_map;

   // the best-response machinery never consults the responding player's own probabilities, so
   // an empty placeholder table suffices on that side
   BestResponsePolicy< info_state_type, action_type > best_response(responder, {});
   best_response.allocate(
      std::forward< Env >(env),
      root_state,
      player_hashmap{
         std::pair{responder, tabular_policy_type{}},
         std::pair{modeled_player, tabular_policy_type{std::move(model_map)}}}
   );

   for(const auto& [infostate, best_action] : best_response.table(responder)) {
      auto& entry = result.policy.emplace(infostate, HashmapActionPolicy< action_type >{})
                       .first->second;
      entry[best_action] = 1.;
   }

   // evaluated against the MATERIALIZED response table: policy_value's profile must carry one
   // uniform StatePolicyView element type; direct table construction because the factory's
   // requires-clause rejects deduced reference types
   result.root_values = rm::policy_value(
                           std::forward< Env >(env),
                           root_state,
                           player_hashmap{
                              std::pair{
                                 responder, StatePolicyView{tabular_policy_type{result.policy}}},
                              std::pair{
                                 modeled_player,
                                 StatePolicyView{tabular_policy_type{result.model}}}}
   )
                           .get()
                           .to_hashmap();
   return result;
}

}  // namespace detail

/**
 * DEV-BASED BEST RESPONSE (DBBR) with an explicit equilibrium/base anchor
 * (Ganzfried & Sandholm, AAMAS 2011): unobserved infostates complete through 'base_policy'
 * -- any type exposing .at(infostate).at(action) -> double covering the modeled player's
 * infostates (their precomputed equilibrium strategy sigma*). See the module-level algorithm
 * description and SAFETY CAVEATS above.
 *
 * @param env, root_state the game.
 * @param responder the player computing the response (the other participant is the modeled one).
 * @param counts FrequencyTable<Infostate, Action> of observed opponent actions.
 */
template < typename Env, typename CountsTable, typename BasePolicyTable >
[[nodiscard]] auto dbbr_response(
   Env&& env,
   const auto_world_state_type< std::remove_cvref_t< Env > >& root_state,
   Player responder,
   const CountsTable& counts,
   const BasePolicyTable& base_policy
)
{
   using action_type = auto_action_type< std::remove_cvref_t< Env > >;
   return detail::solve_dbb(
      std::forward< Env >(env),
      root_state,
      responder,
      counts,
      [&base_policy](
         const auto& infostate, const auto& actions
      ) -> std::vector< std::pair< action_type, double > > {
         double sum = 0.;
         auto distribution = std::ranges::to< std::vector >(
            actions | std::views::transform([&](const action_type& action) {
               const double prob = base_policy.at(infostate).at(action);
               sum += prob;
               return std::pair{action, prob};
            })
         );
         if(std::abs(sum - 1.) > 1e-6) {
            throw std::invalid_argument(
               "dbbr_response: the base policy must be normalized at every completion point."
            );
         }
         return distribution;
      }
   );
}

/**
 * DBBR without a supplied anchor: unobserved infostates complete uniformly (see the deviation
 * note on the COMPLETION step above).
 */
template < typename Env, typename CountsTable >
[[nodiscard]] auto dbbr_response(
   Env&& env,
   const auto_world_state_type< std::remove_cvref_t< Env > >& root_state,
   Player responder,
   const CountsTable& counts
)
{
   return detail::solve_dbb(
      std::forward< Env >(env),
      root_state,
      responder,
      counts,
      [](const auto& /*unused*/, const auto& actions)
         -> std::vector<
            std::pair< typename std::decay_t< decltype(actions) >::value_type, double > > {
         const double uniform_prob = 1. / static_cast< double >(std::ranges::distance(actions));
         return std::ranges::to< std::vector >(
            actions
            | std::views::transform(
               [uniform_prob](const typename std::decay_t< decltype(actions) >::value_type& action
               ) {
                  return std::pair{action, uniform_prob};
               }
            )
         );
      }
   );
}

}  // namespace nor::opponent_aware

#endif  // NOR_DBBR_HPP
