

#ifndef NOR_TABULAR_CFR_BASE_HPP
#define NOR_TABULAR_CFR_BASE_HPP

#include <execution>
#include <iostream>
#include <list>
#include <map>
#include <named_type.hpp>
#include <queue>
#include <range/v3/all.hpp>
#include <stack>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cfr_utils.hpp"
#include "common/common.hpp"
#include "forest.hpp"
#include "node.hpp"
#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "nor/policy.hpp"
#include "nor/type_defs.hpp"
#include "nor/utils/utils.hpp"

namespace nor::rm {

/**
 * A Counterfactual Regret Minimization algorithm base class following the
 * terminology of the Factored-Observation Stochastic Games (FOSG) formulation.
 *
 * This class defines the common template and constructor definitions, as well as the tabular
 * members to be used by specific implementations.
 *
 */
template < bool alternating_update, typename Env, typename Policy, typename AveragePolicy >
   requires concepts::tabular_cfr_requirements< Env, Policy, AveragePolicy >
class TabularCFRBase {
   ////////////////////////////
   /// API: public typedefs ///
   ////////////////////////////
  public:
   /// aliases for the template types
   using env_type = Env;
   using policy_type = Policy;
   /// store the flag for alternating updates
   static constexpr bool alternating_updates = alternating_update;
   /// import all fosg aliases to be used in this class from the env type.
   using action_type = typename fosg_auto_traits< Env >::action_type;
   using world_state_type = typename fosg_auto_traits< Env >::world_state_type;
   using info_state_type = typename fosg_auto_traits< Env >::info_state_type;
   using public_state_type = typename fosg_auto_traits< Env >::public_state_type;
   using observation_type = typename fosg_auto_traits< Env >::observation_type;
   using chance_outcome_type = typename fosg_auto_traits< Env >::chance_outcome_type;
   using chance_distribution_type = typename fosg_auto_traits< Env >::chance_distribution_type;

   /// the data to store per infostate entry
   using infostate_data_type = InfostateNodeData< action_type >;
   /// strong-types for argument passing
   using InfostateMap = fluent::
      NamedType< std::unordered_map< Player, sptr< info_state_type > >, struct reach_prob_tag >;

   using ObservationbufferMap = fluent::NamedType<
      std::unordered_map< Player, std::vector< observation_type > >,
      struct observation_buffer_tag >;

   ////////////////////
   /// Constructors ///
   ////////////////////

  protected:
   /// the constructors are protected since this class is not intended to be instantiated on its own
   /// but only used as base class for cfr algorithms

   TabularCFRBase(
      Env game,
      uptr< world_state_type > root_state,
      Policy policy = Policy(),
      AveragePolicy avg_policy = AveragePolicy())
      // clang-format off
      requires
         all_predicate_v<
            std::is_copy_constructible,
            Policy,
            AveragePolicy >
   // clang-format on
   : m_env(std::move(game)), m_root_state(std::move(root_state)), m_curr_policy(), m_avg_policy()
   {
      _assert_sequential_game();
      for(auto player : game.players() | utils::is_nonchance_player_filter) {
         m_curr_policy.emplace(player, policy);
         m_avg_policy.emplace(player, avg_policy);
      }
      _init_player_update_schedule();
   }

   TabularCFRBase(Env env, Policy policy = Policy(), AveragePolicy avg_policy = AveragePolicy())
      // clang-format off
      requires
         concepts::has::method::initial_world_state< Env >
   // clang-format on
   :
       TabularCFRBase(
          std::move(env),
          std::make_unique< world_state_type >(env.initial_world_state()),
          std::move(policy),
          std::move(avg_policy))
   {
      _init_player_update_schedule();
   }

   TabularCFRBase(
      Env game,
      uptr< world_state_type > root_state,
      std::unordered_map< Player, Policy > policy,
      std::unordered_map< Player, AveragePolicy > avg_policy)
       : m_env(std::move(game)),
         m_root_state(std::move(root_state)),
         m_curr_policy(std::move(policy)),
         m_avg_policy(std::move(avg_policy))
   {
      _assert_sequential_game();
      _init_player_update_schedule();
   }

   ////////////////////////////////////
   /// API: public member functions ///
   ////////////////////////////////////

  public:
   /**
    * @brief gets the current or average state policy of a node
    *
    * Depending on the template parameter either the current policy (true) or the average policy
    * (false) over the last iterations is queried. If the current node has not been emplaced in the
    * policy yet, then the default policy will be asked to provide an initial entry.
    * @param current_policy switch for querying either the current (true) or the average (false)
    * policy.
    * @return action_policy_type the player's state policy (distribution) over all actions at this
    * node
    */
   template < bool current_policy >
   auto& fetch_policy(
      const sptr< info_state_type >& infostate,
      const std::vector< action_type >& actions);
   /**
    * @brief Policy fetching overload for explicit naming of the policy.
    */
   template < PolicyLabel label >
   decltype(auto) fetch_policy(
      const sptr< info_state_type >& infostate,
      const std::vector< action_type >& actions)
   {
      static_assert(
         label == PolicyLabel::current or label == PolicyLabel::average,
         "Policy label has to be either 'current' or 'average'.");
      return fetch_policy< label == PolicyLabel::current >(infostate, actions);
   }

   /**
    *
    * @param action the action to select at this node
    * @return
    */
   template < bool current_policy >
   inline auto& fetch_policy(
      const sptr< info_state_type >& infostate,
      const std::vector< action_type >& actions,
      const action_type& action)
   {
      return fetch_policy< current_policy >(infostate, actions)[action];
   }

   /// getter methods for stored data

   [[nodiscard]] inline const auto& root_state() const { return *m_root_state; }
   [[nodiscard]] inline auto iteration() const { return m_iteration; }
   [[nodiscard]] inline const auto& policy() const { return m_curr_policy; }
   [[nodiscard]] inline const auto& average_policy() const { return m_avg_policy; }
   [[nodiscard]] inline const auto& env() const { return m_env; }

   //////////////////////////////////
   /// protected member functions ///
   //////////////////////////////////

  protected:
   /// protected member access for derived classes
   [[nodiscard]] inline auto& _env() { return m_env; }
   [[nodiscard]] inline const auto& _root_state_uptr() const { return m_root_state; }
   [[nodiscard]] inline auto& _iteration() { return m_iteration; }
   [[nodiscard]] inline auto& _policy() { return m_curr_policy; }
   [[nodiscard]] inline auto& _average_policy() { return m_avg_policy; }
   [[nodiscard]] inline auto& _player_update_schedule() { return m_player_update_schedule; }

   /**
    * @brief Cycles the update schedule by popping the next player to update and requeueing them as
    * last.
    *
    * The update schedule for alternative updates is a cycle P1-P2-...-PN. After each update the
    * schedule moves by returning the first player and reattaching it to the back of the queue. This
    * way every other player is pushed up by one position in the update queue.
    * @param player_to_update the optional player to update next. If not given, the next in line
    * will be chosen.
    * @return the player to update next.
    */
   Player _cycle_player_to_update(std::optional< Player > player_to_update = std::nullopt);

   /**
    * @brief simple check to see if the environment fulfills the necessary game dynamics
    */
   inline void _assert_sequential_game()
   {
      if(m_env.turn_dynamic() != TurnDynamic::sequential) {
         throw std::invalid_argument(
            "VanillaCFR can only be performed on a sequential turn-based game.");
      }
   }

   /**
    * @brief initializes the player cycle buffer with all available players at the current state
    */
   inline void _init_player_update_schedule()
   {
      if constexpr(alternating_updates) {
         for(auto player : m_env.players()) {
            if(player != Player::chance) {
               m_player_update_schedule.push_back(player);
            }
         }
      }
   }

   /**
    * @brief emplaces the environment rewards for a terminal state and stores them in the node.
    *
    * No terminality checking is done within this method! Hence only call this method if you are
    * already certain that the node is a terminal one. Whether the environment rewards for
    * non-terminal states would be problematic is dependant on the environment.
    * @param[in] terminal_wstate the terminal state to collect rewards for.
    */
   auto _collect_rewards(
      const_ref_if_t<  // the fosg concept asserts a reward function taking world_state_type.
                       // But if it can be passed a const world state then do so instead
         concepts::has::method::reward< env_type, const world_state_type& >,
         world_state_type > terminal_wstate) const;

   uptr< world_state_type > _child_state(const uptr< world_state_type >& state, const auto& action)
   {
      // clone the current world state first before transitioniong it with this action
      uptr< world_state_type >
         next_wstate_uptr = utils::static_unique_ptr_downcast< world_state_type >(
            utils::clone_any_way(state));
      // move the new world state forward by the current action
      _env().transition(*next_wstate_uptr, action);
      return next_wstate_uptr;
   }

   auto _fill_infostate_and_obs_buffers(
      ObservationbufferMap observation_buffer,
      InfostateMap infostate_map,
      const auto& action_or_outcome,
      const world_state_type& state);

   void _fill_infostate_and_obs_buffers_inplace(
      ObservationbufferMap& observation_buffer,
      InfostateMap& infostate_map,
      const auto& action_or_outcome,
      const world_state_type& state);

   ///////////////////////////////////////////
   /// private member variable definitions ///
   ///////////////////////////////////////////

  private:
   /// the environment object to maneuver the states with.
   env_type m_env;
   /// the game tree mapping information states to the associated game nodes.
   uptr< world_state_type > m_root_state;
   /// the current policy $\pi^t$ that each player is following in this iteration (t).
   std::unordered_map< Player, Policy > m_curr_policy;
   /// the average policy table. The values stored in this table are the UNNORMALIZED average state
   /// policies. This means that the state policy p(s, . ) for a given info state s needs to
   /// normalize its probabilities p(s, . ) by \sum_a p(s,a) when used for evaluation.
   std::unordered_map< Player, AveragePolicy > m_avg_policy;
   /// the next player to update when doing alternative updates. Otherwise this member will be
   /// unused.
   std::deque< Player > m_player_update_schedule{};
   /// the number of iterations we have run so far.
   size_t m_iteration = 0;
};

template < bool alternating_updates, typename Env, typename Policy, typename AveragePolicy >
   requires concepts::tabular_cfr_requirements< Env, Policy, AveragePolicy >
Player TabularCFRBase< alternating_updates, Env, Policy, AveragePolicy >::_cycle_player_to_update(
   std::optional< Player > player_to_update)
{
   auto player_q_iter = std::find(
      m_player_update_schedule.begin(),
      m_player_update_schedule.end(),
      player_to_update.value_or(m_player_update_schedule.front()));
   Player next_to_update = *player_q_iter;
   m_player_update_schedule.erase(player_q_iter);
   m_player_update_schedule.push_back(next_to_update);
   return next_to_update;
}

template < bool alternating_updates, typename Env, typename Policy, typename AveragePolicy >
   requires concepts::tabular_cfr_requirements< Env, Policy, AveragePolicy >
template < bool current_policy >
auto& TabularCFRBase< alternating_updates, Env, Policy, AveragePolicy >::fetch_policy(
   const sptr< info_state_type >& infostate,
   const std::vector< action_type >& actions)
{
   if constexpr(current_policy) {
      auto& player_policy = m_curr_policy[infostate->player()];
      return player_policy[std::pair{*infostate, actions}];
   } else {
      auto& player_policy = m_avg_policy[infostate->player()];
      return player_policy[std::pair{*infostate, actions}];
   }
}

template < bool altenating_updates, typename Env, typename Policy, typename AveragePolicy >
   requires concepts::tabular_cfr_requirements< Env, Policy, AveragePolicy >
auto TabularCFRBase< altenating_updates, Env, Policy, AveragePolicy >::_collect_rewards(
   const_ref_if_t<
      concepts::has::method::reward< env_type, const world_state_type& >,
      world_state_type > terminal_wstate) const
{
   std::unordered_map< Player, double > rewards;
   auto players = env().players();
   if constexpr(nor::concepts::has::method::reward_multi< Env >) {
      // if the environment has a method for returning all rewards for given players at
      // once, then we will assume this is a more performant alternative and use it
      // instead (e.g. when it is costly to compute the reward of each player
      // individually).
      std::erase(std::remove_if(players.begin(), players.end(), utils::is_chance_player_pred));
      auto all_rewards = env().reward(players, terminal_wstate);
      ranges::views::for_each(
         players, [&](Player player) { rewards.emplace(player, all_rewards[player]); });
   } else {
      // otherwise we just loop over the per player reward method
      for(auto player : players | utils::is_nonchance_player_filter) {
         rewards.emplace(player, env().reward(player, terminal_wstate));
      }
      return rewards;
   }
}

template < bool alternating_updates, typename Env, typename Policy, typename AveragePolicy >
   requires concepts::tabular_cfr_requirements< Env, Policy, AveragePolicy >
auto TabularCFRBase< alternating_updates, Env, Policy, AveragePolicy >::
   _fill_infostate_and_obs_buffers(
      ObservationbufferMap observation_buffer,
      InfostateMap infostate_map,
      const auto& action_or_outcome,
      const world_state_type& state)
{
   _fill_infostate_and_obs_buffers_inplace(
      observation_buffer, infostate_map, action_or_outcome, state);
   return std::tuple{observation_buffer, infostate_map};
}

template < bool alternating_updates, typename Env, typename Policy, typename AveragePolicy >
   requires concepts::tabular_cfr_requirements< Env, Policy, AveragePolicy >
void TabularCFRBase< alternating_updates, Env, Policy, AveragePolicy >::
   _fill_infostate_and_obs_buffers_inplace(
      ObservationbufferMap& observation_buffer,
      InfostateMap& infostate_map,
      const auto& action_or_outcome,
      const world_state_type& state)
{
   auto active_player = _env().active_player(state);
   for(auto player : _env().players()) {
      if(player == Player::chance) {
         continue;
      }
      if(player != active_player) {
         // for all but the active player we simply append action and state observation to the
         // buffer. They will be written to an actual infostate once that player becomes the
         // active player again
         auto& player_infostate = observation_buffer.get().at(player);
         player_infostate.emplace_back(_env().private_observation(player, action_or_outcome));
         player_infostate.emplace_back(_env().private_observation(player, state));
      } else {
         // for the active player we first append all recent action and state observations to a
         // info state copy, and then follow it up by adding the current action and state
         // observations
         auto& infostate_ptr_slot = infostate_map.get().at(active_player);
         auto cloned_infostate_ptr = utils::clone_any_way(infostate_ptr_slot);
         auto& obs_history = observation_buffer.get()[active_player];
         for(auto& obs : obs_history) {
            cloned_infostate_ptr->append(std::move(obs));
         }
         obs_history.clear();
         cloned_infostate_ptr->append(_env().private_observation(player, action_or_outcome));
         cloned_infostate_ptr->append(_env().private_observation(player, state));
         infostate_ptr_slot = std::move(cloned_infostate_ptr);
      }
   }
}

}  // namespace nor::rm

#endif  // NOR_TABULAR_CFR_BASE_HPP
