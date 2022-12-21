

#ifndef NOR_CFR_BASE_TABULAR_HPP
#define NOR_CFR_BASE_TABULAR_HPP

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

#include "common/common.hpp"
#include "forest.hpp"
#include "node.hpp"
#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "nor/type_defs.hpp"
#include "nor/utils/utils.hpp"
#include "rm_utils.hpp"

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
   requires concepts::tabular_cfr_requirements<
      Env,
      Policy,
      AveragePolicy,
      UniformPolicy<
         typename fosg_auto_traits< Env >::info_state_type,
         typename fosg_auto_traits< Policy >::action_policy_type >,
      ZeroDefaultPolicy<
         typename fosg_auto_traits< Env >::info_state_type,
         typename fosg_auto_traits< AveragePolicy >::action_policy_type > >
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

   using uniform_policy_type = UniformPolicy<
      typename fosg_auto_traits< Env >::info_state_type,
      typename fosg_auto_traits< Policy >::action_policy_type >;
   using zero_policy_type = UniformPolicy<
      typename fosg_auto_traits< Env >::info_state_type,
      typename fosg_auto_traits< Policy >::action_policy_type >;

   /// the data to store per infostate entry
   using infostate_data_type = InfostateNodeData< action_type >;
   /// strong-types for argument passing
   using InfostateSptrMap = fluent::
      NamedType< player_hash_map< sptr< info_state_type > >, struct reach_prob_tag >;

   using ObservationbufferMap = fluent::NamedType<
      player_hash_map< std::vector< std::pair< observation_type, observation_type > > >,
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
      const Policy& policy = Policy(),
      const AveragePolicy& avg_policy = AveragePolicy()
   )
      // clang-format off
      requires
         common::all_predicate_v<
            std::is_copy_constructible,
            Policy,
            AveragePolicy >
       // clang-format on
       : m_env(std::move(game)),
         m_root_state(std::move(root_state)),
         m_curr_policy(),
         m_avg_policy()
   {
      for(auto player : game.players(*m_root_state) | utils::is_actual_player_filter) {
         m_curr_policy.emplace(player, policy);
         m_avg_policy.emplace(player, avg_policy);
      }
      _init_player_update_schedule();
   }

   TabularCFRBase(
      Env env,
      const Policy& policy = Policy(),
      const AveragePolicy& avg_policy = AveragePolicy()
   )
      // clang-format off
      requires
         concepts::has::method::initial_world_state< Env >
       // clang-format on
       : TabularCFRBase(
          std::move(env),
          std::make_unique< world_state_type >(env.initial_world_state()),
          policy,
          avg_policy
       )
   {
      _init_player_update_schedule();
   }

   TabularCFRBase(
      Env game,
      uptr< world_state_type > root_state,
      std::unordered_map< Player, Policy > policy,
      std::unordered_map< Player, AveragePolicy > avg_policy
   )
       : m_env(std::move(game)),
         m_root_state(std::move(root_state)),
         m_curr_policy(std::move(policy)),
         m_avg_policy(std::move(avg_policy))
   {
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
   auto& fetch_policy(const info_state_type& infostate, const std::vector< action_type >& actions);
   /**
    * @brief Policy fetching overload for explicit naming of the policy.
    */
   template < PolicyLabel label >
   decltype(auto)
   fetch_policy(const info_state_type& infostate, const std::vector< action_type >& actions)
   {
      static_assert(
         label == PolicyLabel::current or label == PolicyLabel::average,
         "Policy label has to be either 'current' or 'average'."
      );
      return fetch_policy< label == PolicyLabel::current >(infostate, actions);
   }

   /**
    *
    * @param action the action to select at this node
    * @return
    */
   template < bool current_policy >
   inline auto& fetch_policy(
      const info_state_type& infostate,
      const std::vector< action_type >& actions,
      const action_type& action
   )
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

   Player _preview_next_player_to_update() const { return m_player_update_schedule.front(); }

   /**
    * @brief initializes the player cycle buffer with all available players at the current state
    */
   inline void _init_player_update_schedule()
   {
      if constexpr(alternating_updates) {
         for(auto player : m_env.players(*m_root_state)) {
            if(player != Player::chance) {
               m_player_update_schedule.push_back(player);
            }
         }
      }
   }

   ///////////////////////////////////////////
   /// private member variable definitions ///
   ///////////////////////////////////////////

  private:
   /// the environment object to maneuver the states with.
   env_type m_env;
   /// the root game state.
   uptr< world_state_type > m_root_state;
   /// a map of the current policy $\pi^t$ that each player is following in this iteration (t).
   player_hash_map< Policy > m_curr_policy;
   /// the average policy table. The values stored in this table are the UNNORMALIZED average state
   /// policy cumulative values. This means that the state policy p(s, . ) for a given info state s
   /// needs to normalize its probabilities p(s, . ) by \sum_a p(s,a) when used for evaluation.
   player_hash_map< AveragePolicy > m_avg_policy;
   /// the next player to update when doing alternative updates. Otherwise this member will be
   /// unused.
   std::deque< Player > m_player_update_schedule{};
   /// the number of iterations we have run so far.
   size_t m_iteration = 0;
};

template < bool alternating_updates, typename Env, typename Policy, typename AveragePolicy >
   requires concepts::tabular_cfr_requirements<
      Env,
      Policy,
      AveragePolicy,
      UniformPolicy<
         typename fosg_auto_traits< Env >::info_state_type,
         typename fosg_auto_traits< Policy >::action_policy_type >,
      ZeroDefaultPolicy<
         typename fosg_auto_traits< Env >::info_state_type,
         typename fosg_auto_traits< AveragePolicy >::action_policy_type > >
Player TabularCFRBase< alternating_updates, Env, Policy, AveragePolicy >::_cycle_player_to_update(
   std::optional< Player > player_to_update
)
{
   // we assert here that the chosen player to update is not the chance player.
   if(player_to_update.value_or(Player::alex) == Player::chance) {
      std::stringstream ssout;
      ssout << "Given combination of '";
      ssout << Player::chance;
      ssout << "' and '";
      ssout << "alternating updates'";
      ssout << "is incompatible. Did you forget to pass the correct player parameter?";
      throw std::invalid_argument(ssout.str());
   }

   auto player_q_iter = std::find(
      m_player_update_schedule.begin(),
      m_player_update_schedule.end(),
      player_to_update.value_or(m_player_update_schedule.front())
   );

   if(player_q_iter == m_player_update_schedule.end()) {
      std::stringstream ssout;
      ssout << "Given player to update ";
      ssout << player_to_update.value();
      ssout << " is not a member of the update schedule.";
      ssout << ranges::views::all(m_player_update_schedule) << ".";
      throw std::invalid_argument(ssout.str());
   }
   Player next_to_update = *player_q_iter;
   m_player_update_schedule.erase(player_q_iter);
   m_player_update_schedule.push_back(next_to_update);
   return next_to_update;
}

template < bool alternating_updates, typename Env, typename Policy, typename AveragePolicy >
   requires concepts::tabular_cfr_requirements<
      Env,
      Policy,
      AveragePolicy,
      UniformPolicy<
         typename fosg_auto_traits< Env >::info_state_type,
         typename fosg_auto_traits< Policy >::action_policy_type >,
      ZeroDefaultPolicy<
         typename fosg_auto_traits< Env >::info_state_type,
         typename fosg_auto_traits< AveragePolicy >::action_policy_type > >
template < bool current_policy >
auto& TabularCFRBase< alternating_updates, Env, Policy, AveragePolicy >::fetch_policy(
   const info_state_type& infostate,
   const std::vector< action_type >& actions
)
{
   if constexpr(current_policy) {
      auto& player_policy = m_curr_policy[infostate.player()];
      return player_policy(infostate, actions, uniform_policy_type{});
   } else {
      auto& player_policy = m_avg_policy[infostate.player()];
      return player_policy(infostate, actions, zero_policy_type{});
   }
}

}  // namespace nor::rm

#endif  // NOR_CFR_BASE_TABULAR_HPP
