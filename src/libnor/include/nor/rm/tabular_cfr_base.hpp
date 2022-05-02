

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

#include "algorithms.hpp"
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

   [[nodiscard]] auto& infonode(const sptr< info_state_type >& infostate) const
   {
      return m_infonode.at(infostate);
   }
   [[nodiscard]] const auto& root_state() const { return *m_root_state; }
   [[nodiscard]] auto iteration() const { return m_iteration; }
   [[nodiscard]] const auto& policy() const { return m_curr_policy; }
   [[nodiscard]] const auto& average_policy() const { return m_avg_policy; }
   [[nodiscard]] const auto& env() const { return m_env; }

   //////////////////////////////////
   /// protected member functions ///
   //////////////////////////////////

  protected:
   /// protected member access for derived classes
   [[nodiscard]] auto& _env() { return m_env; }
   [[nodiscard]] const auto& _root_state_uptr() const { return m_root_state; }
   [[nodiscard]] auto& _iteration() { return m_iteration; }
   [[nodiscard]] auto& _policy() { return m_curr_policy; }
   [[nodiscard]] auto& _average_policy() { return m_avg_policy; }
   [[nodiscard]] auto& _player_update_schedule() { return m_player_update_schedule; }
   [[nodiscard]] auto& _infonodes() { return m_infonode; }
   [[nodiscard]] auto& _infonode(const sptr< info_state_type >& infostate)
   {
      return m_infonode.at(infostate);
   }

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
   /// the relevant data stored at each infostate
   std::unordered_map<
      sptr< info_state_type >,
      infostate_data_type,
      decltype(
         [](const sptr< info_state_type >& ptr) { return std::hash< info_state_type >{}(*ptr); }),
      decltype([](const sptr< info_state_type >& ptr1, const sptr< info_state_type >& ptr2) {
         return *ptr1 == *ptr2;
      }) >
      m_infonode{};
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

}  // namespace nor::rm

#endif  // NOR_TABULAR_CFR_BASE_HPP
