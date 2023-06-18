
#ifndef NOR_POLICY_VALUE_HPP
#define NOR_POLICY_VALUE_HPP

#include "nor/concepts.hpp"
#include "rm_utils.hpp"

namespace nor::rm {

namespace detail {

template < typename Observation >
using ObservationbufferMap = fluent::NamedType<
   player_hash_map< std::vector< std::pair< Observation, Observation > > >,
   struct observation_buffer_tag >;

struct policy_value_impl {
   template < typename Env, typename Policy >
      requires concepts::fosg< std::remove_cvref_t< Env > >
   static StateValueMap traverse(
      Env&& env,
      const player_hash_map< Policy >& policy_profile,
      uptr< auto_world_state_type< std::remove_cvref_t< Env > > > state,
      ReachProbabilityMap reach_probability,
      ObservationbufferMap< auto_observation_type< std::remove_cvref_t< Env > > >
         observation_buffer,
      player_hash_map< auto_info_state_type< std::remove_cvref_t< Env > > > infostates
   )
   {
      using env_type = std::remove_cvref_t< Env >;
      if(env.is_terminal(*state)) {
         return StateValueMap{collect_rewards(env, *state)};
      }

      if(ranges::all_of(reach_probability.get(), [&](const auto& player_rp_pair) {
            // A mere check on ONE of the opponents having reach prob 0 and the active player
            // having reach prob 0 would not suffice in the multiplayer case as some average
            // strategy updates of other opponent with reach prob > 0 would be missed
            const auto& [player, rp] = player_rp_pair;
            return player != Player::chance and rp <= std::numeric_limits< double >::epsilon();
         })) {
         // if the entire subtree is pruned then the only values that could be found are all 0 for
         // each player
         return StateValueMap{std::invoke([&] {
            StateValueMap::UnderlyingType map;
            for(auto player : env.players(*state)) {
               if(utils::is_actual_player_pred(player)) {
                  map[player] = 0.;
               }
            }
            return map;
         })};
      }

      Player active_player = env.active_player(*state);
      // the state's value for each player. To be filled by the action traversal functions.
      StateValueMap state_value{{}};
      // each action's value for each player. To be filled by the action traversal functions.
      std::unordered_map< auto_action_variant_type< env_type >, StateValueMap > action_value;
      // traverse all child states from this state. The constexpr check for determinism in the env
      // allows deterministic envs to not provide certain functions that are only needed in the
      // stochastic case.
      // First define the default branch for an active non-chance player
      auto nonchance_player_traverse = [&] {
         traverse_player_actions(
            env,
            policy_profile,
            active_player,
            std::move(state),
            reach_probability,
            std::move(observation_buffer),
            std::move(infostates),
            state_value,
            action_value
         );
      };
      // now we check first if we even need to consider a chance player, as the env could be simply
      // deterministic. In that case we might need to traverse the chance player's actions or an
      // active player's actions
      if constexpr(concepts::stochastic_env< env_type >) {
         if(active_player == Player::chance) {
            traverse_chance_actions(
               env,
               policy_profile,
               active_player,
               std::move(state),
               reach_probability,
               std::move(observation_buffer),
               std::move(infostates),
               state_value,
               action_value
            );
            // if this is a chance node then we don't need to update any regret or average policy
            // after the traversal
            return state_value;
         } else {
            nonchance_player_traverse();
         }
      } else {
         nonchance_player_traverse();
      }

      return StateValueMap{std::move(state_value)};
   }

   template < typename Env, typename Policy >
   static void traverse_player_actions(
      Env&& env,
      const player_hash_map< Policy >& policy_profile,
      Player active_player,
      uptr< auto_world_state_type< std::remove_cvref_t< Env > > > state,
      const ReachProbabilityMap& reach_probability,
      const ObservationbufferMap< auto_observation_type< std::remove_cvref_t< Env > > >&
         observation_buffer,
      player_hash_map< auto_info_state_type< std::remove_cvref_t< Env > > > infostate_map,
      StateValueMap& state_value,
      std::unordered_map< auto_action_variant_type< std::remove_cvref_t< Env > >, StateValueMap >&
         action_value
   )
   {
      using env_type = std::remove_cvref_t< Env >;

      auto& this_infostate = infostate_map.at(active_player);
      auto&& action_policy = policy_profile.at(active_player).at(this_infostate);
      //      double normalizing_factor = ranges::accumulate(
      //         action_policy | ranges::views::values, double(0.), std::plus{}
      //      );
      //      if(std::abs(normalizing_factor) < 1e-20) {
      //         throw std::invalid_argument(
      //            "Average policy likelihoods accumulate to 0. Such values cannot be normalized."
      //         );
      //      }

      for(const auto_action_type< env_type >& action : env.actions(active_player, *state)) {
         auto action_prob = action_policy.at(action);

         auto child_reach_prob = reach_probability.get();
         child_reach_prob[active_player] *= action_prob;

         uptr< auto_world_state_type< env_type > > next_wstate_uptr = child_state(
            env, *state, action
         );
         auto [child_observation_buffer, child_infostate_map] = next_infostate_and_obs_buffers(
            env, observation_buffer.get(), infostate_map, *state, action, *next_wstate_uptr
         );

         StateValueMap child_rewards_map = traverse(
            env,
            policy_profile,
            std::move(next_wstate_uptr),
            ReachProbabilityMap{std::move(child_reach_prob)},
            ObservationbufferMap< auto_observation_type< env_type > >{
               std::move(child_observation_buffer)},
            player_hash_map< auto_info_state_type< env_type > >{std::move(child_infostate_map)}
         );
         // add the child state's value to the respective player's value table, multiplied by the
         // policies likelihood of playing this action
         for(auto [player, child_value] : child_rewards_map.get()) {
            state_value.get()[player] += action_prob * child_value;
         }
         action_value.emplace(action, std::move(child_rewards_map));
      }
   }

   template < typename Env, typename Policy >
   static void traverse_chance_actions(
      Env&& env,
      const player_hash_map< Policy >& policy_profile,
      Player active_player,
      uptr< auto_world_state_type< std::remove_cvref_t< Env > > > state,
      const ReachProbabilityMap& reach_probability,
      const ObservationbufferMap< auto_observation_type< std::remove_cvref_t< Env > > >&
         observation_buffer,
      player_hash_map< auto_info_state_type< std::remove_cvref_t< Env > > > infostate_map,
      StateValueMap& state_value,
      std::unordered_map< auto_action_variant_type< std::remove_cvref_t< Env > >, StateValueMap >&
         action_value
   )
   {
      using env_type = std::remove_cvref_t< Env >;
      for(auto&& outcome : env.chance_actions(*state)) {
         uptr< auto_world_state_type< env_type > > next_wstate_uptr = child_state(
            env, *state, outcome
         );

         auto child_reach_prob = reach_probability.get();
         auto outcome_prob = env.chance_probability(*state, outcome);
         child_reach_prob[active_player] *= outcome_prob;

         auto [child_observation_buffer, child_infostate_map] = next_infostate_and_obs_buffers(
            env, observation_buffer.get(), infostate_map, *state, outcome, *next_wstate_uptr
         );

         StateValueMap child_rewards_map = traverse(
            env,
            policy_profile,
            std::move(next_wstate_uptr),
            ReachProbabilityMap{std::move(child_reach_prob)},
            ObservationbufferMap< auto_observation_type< env_type > >{
               std::move(child_observation_buffer)},
            player_hash_map< auto_info_state_type< env_type > >{std::move(child_infostate_map)}
         );
         // add the child state's value to the respective player's value table, multiplied by the
         // policies likelihood of playing this action
         for(auto [player, child_value] : child_rewards_map.get()) {
            state_value.get()[player] += outcome_prob * child_value;
         }
         action_value.emplace(std::move(outcome), std::move(child_rewards_map));
      }
   }
};

}  // namespace detail

template <
   typename Env,
   typename Policy,
   typename ActionPolicy = auto_action_policy_type< Policy > >
   requires concepts::state_policy_no_default<
               Policy,
               auto_info_state_type< std::remove_cvref_t< Env > >,
               auto_action_type< std::remove_cvref_t< Env > >,
               ActionPolicy >
            or concepts::state_policy_view<
               Policy,
               auto_info_state_type< std::remove_cvref_t< Env > >,
               auto_action_type< std::remove_cvref_t< Env > > >
StateValueMap policy_value(
   Env&& env,
   const auto_world_state_type< std::remove_cvref_t< Env > >& root_state,
   const player_hash_map< Policy >& policy_profile
)
{
   using env_type = std::remove_cvref_t< Env >;
   auto root_players = env.players(root_state);
   return detail::policy_value_impl::traverse(
      env,
      policy_profile,
      utils::static_unique_ptr_downcast< auto_world_state_type< env_type > >(
         utils::clone_any_way(root_state)
      ),
      std::invoke([&] {
         ReachProbabilityMap rp_map{{}};
         for(auto player : root_players) {
            rp_map.get().emplace(player, 1.);
         }
         return rp_map;
      }),
      std::invoke([&] {
         detail::ObservationbufferMap< auto_observation_type< env_type > > obs_map{{}};
         for(auto player : root_players | utils::is_actual_player_filter) {
            obs_map.get().try_emplace(player);
         }
         return obs_map;
      }),
      std::invoke([&] {
         player_hash_map< auto_info_state_type< env_type > > infostates;
         for(auto player : root_players | utils::is_actual_player_filter) {
            infostates.emplace(player, auto_info_state_type< env_type >{player});
         }
         return infostates;
      })
   );
}  // namespace nor::rm

}  // namespace nor::rm

#endif  // NOR_POLICY_VALUE_HPP
