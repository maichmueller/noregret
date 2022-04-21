
#ifndef NOR_VANILLA_HPP
#define NOR_VANILLA_HPP

//#include <cppitertools/enumerate.hpp>
//#include <cppitertools/reversed.hpp>
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
#include "node.hpp"
#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "nor/policy.hpp"
#include "nor/type_defs.hpp"
#include "nor/utils/utils.hpp"
#include "rm.hpp"

namespace nor::rm {

struct CFRConfig {
   bool alternating_updates = true;
   bool store_public_states = false;
   bool store_world_states = false;
};

/**
 * A (Vanilla) Counterfactual Regret Minimization algorithm class following the
 * terminology of the Factored-Observation Stochastic Games (FOSG) formulation.
 *
 * The implementation follows the algorithm detail of @cite Neller2013
 *
 *
 * @tparam Env, the environment/game type to run VanillaCFR on.
 *
 */
template <
   CFRConfig cfr_config,
   typename Env,
   typename Policy,
   typename DefaultPolicy,
   typename AveragePolicy = Policy >
requires concepts::vanilla_cfr_requirements< Env, Policy, DefaultPolicy, AveragePolicy >
class VanillaCFR {
   ////////////////////////////
   /// API: public typedefs ///
   ////////////////////////////
  public:
   /// aliases for the template types
   using env_type = Env;
   using policy_type = Policy;
   using default_policy_type = DefaultPolicy;
   /// import all fosg aliases to be used in this class from the env type.
   using action_type = typename fosg_auto_traits< Env >::action_type;
   using world_state_type = typename fosg_auto_traits< Env >::world_state_type;
   using info_state_type = typename fosg_auto_traits< Env >::info_state_type;
   using public_state_type = typename fosg_auto_traits< Env >::public_state_type;
   using observation_type = typename fosg_auto_traits< Env >::observation_type;
   using chance_outcome_type = typename fosg_auto_traits< Env >::chance_outcome_type;
   using chance_policy_type = typename fosg_auto_traits< Env >::chance_distribution_type;
   using action_variant_type = std::variant<
      action_type,
      std::conditional_t<
         std::is_same_v< chance_outcome_type, void >,
         std::monostate,
         chance_outcome_type > >;
   /// the data to store per infostate entry
   using infostate_data_type = InfostateNodeData< action_type >;
   /// strong-types for player based maps
   using ValueMap = fluent::NamedType< std::unordered_map< Player, double >, struct value_map_tag >;
   using ReachProbabilityMap = fluent::
      NamedType< std::unordered_map< Player, double >, struct reach_prob_tag >;
   using InfostateMap = fluent::
      NamedType< std::unordered_map< Player, sptr< info_state_type > >, struct reach_prob_tag >;
   using ObservationbufferMap = fluent::NamedType<
      std::unordered_map< Player, std::vector< observation_type > >,
      struct observation_buffer_tag >;

   ////////////////////
   /// Constructors ///
   ////////////////////

   VanillaCFR(
      Env game,
      uptr< world_state_type > root_state,
      Policy policy = Policy(),
      DefaultPolicy default_policy = DefaultPolicy(),
      AveragePolicy avg_policy = AveragePolicy())
      // clang-format off
      requires
         all_predicate_v<
            std::is_copy_constructible,
            Policy,
            AveragePolicy >
       // clang-format on
       :
       m_env(std::move(game)),
       m_root_state(std::move(root_state)),
       m_curr_policy(),
       m_avg_policy(),
       m_default_policy(std::move(default_policy))
   {
      _assert_sequential_game();
      for(auto player : game.players()) {
         m_curr_policy.emplace(player, policy);
         m_avg_policy.emplace(player, avg_policy);
      }
      _init_player_update_schedule();
   }

   VanillaCFR(
      Env env,
      Policy policy = Policy(),
      DefaultPolicy default_policy = DefaultPolicy(),
      AveragePolicy avg_policy = AveragePolicy())
      // clang-format off
      requires
         concepts::has::method::initial_world_state< Env >
       // clang-format on
       :
       VanillaCFR(
          std::move(env),
          std::make_unique< world_state_type >(env.initial_world_state()),
          std::move(policy),
          std::move(default_policy),
          std::move(avg_policy))
   {
      _init_player_update_schedule();
   }

   VanillaCFR(
      Env game,
      uptr< world_state_type > root_state,
      std::unordered_map< Player, Policy > policy,
      std::unordered_map< Player, AveragePolicy > avg_policy,
      DefaultPolicy default_policy = DefaultPolicy())
       : m_env(std::move(game)),
         m_root_state(std::move(root_state)),
         m_curr_policy(std::move(policy)),
         m_avg_policy(std::move(avg_policy)),
         m_default_policy(std::move(default_policy))
   {
      _assert_sequential_game();
      _init_player_update_schedule();
   }

   ////////////////////////////////////
   /// API: public member functions ///
   ////////////////////////////////////

  public:
   /**
    * @brief executes n iterations of the VanillaCFR algorithm in unrolled form (no recursion).
    *
    * The decision for doing alternating updates or simultaneous updates happens at compile time via
    * the cfr config. This optimizes some unncessary repeated if-branching away at the cost of
    * higher maintenance. The user can also decide whether to store the public state at each node
    * within the CFR config! This can save some memory, since the public states are not needed,
    * unless one wants to e.g. perform analysis.
    *
    * By returning a pointer to the constant current policy after the n iterations, the user can
    * select to store a copy of the policy at each step themselves.
    *
    * @param n_iters the number of iterations to perform.
    * @return a pointer to the constant current policy after the update
    */
   auto iterate(size_t n_iters);
   /**
    * @brief executes one iteration of alternating updates vanilla cfr.
    *
    * This overload only participates if the config defined alternating updates to be made.
    *
    * By returning a pointer to the constant current policy after the n iterations, the user can
    * select to store a copy of the policy at each step themselves.
    *
    * @param player_to_update the optional player to update this iteration. If not provided, the
    * function will continue with the regular update cycle. By providing this parameter the user can
    * expressly modify the update cycle to even update individual players multiple times in a row.
    * @return a pointer to the constant current policy after the update
    */
   auto iterate(std::optional< Player > player_to_update = std::nullopt) requires(
      cfr_config.alternating_updates);

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
   auto& fetch_policy(bool current_policy, const sptr< info_state_type >& infostate);

   /**
    * The const overload of this function (@see fetch_policy) will not insert or return a default
    * policy if one wasn't found, but throw an error instead.
    */
   auto& fetch_policy(bool current_policy, const sptr< info_state_type >& infostate) const;

   /**
    *
    * @param action the action to select at this node
    * @return
    */
   inline auto& fetch_policy(
      bool current_policy,
      const sptr< info_state_type >& infostate,
      const action_type& action)
   {
      return fetch_policy(current_policy, infostate)[action];
   }
   /**
    * Const overload for action selection at the node.
    */
   inline auto& fetch_policy(
      bool current_policy,
      const sptr< info_state_type >& infostate,
      const action_type& action) const
   {
      return fetch_policy(current_policy, infostate)[action];
   }
   /**
    * @brief updates the regret and policy tables of the node with the state-values. Then performs
    * regret-matching.
    *
    * The method implements lines 21-25 of @cite Neller2013.
    *
    * @param player the player whose values are to be updated
    */
   void update_regret_and_policy(
      const sptr< info_state_type >& infostate,
      ReachProbabilityMap reach_probability,
      const ValueMap& state_value,
      const std::unordered_map< action_variant_type, ValueMap >& action_value);

   /// getter methods for stored data

   auto& infodata(const info_state_type& infostate) { return m_infonode_data.at(infostate); }

   [[nodiscard]] auto& infodata(const info_state_type& infostate) const
   {
      return m_infonode_data.at(infostate);
   }
   [[nodiscard]] const auto& root_state() const { return *m_root_state; }
   [[nodiscard]] auto iteration() const { return m_iteration; }
   [[nodiscard]] const auto& policy() const { return m_curr_policy; }
   [[nodiscard]] const auto& average_policy() const { return m_avg_policy; }

   ////////////////////////////////
   /// private member functions ///
   ////////////////////////////////

  private:
   /**
    * @brief The internal vanilla cfr iteration routine.
    *
    * This function sets the algorithm scheme for vanilla cfr by delegating to the right functions.
    *
    * @param player_to_update the player to update (optionally). If not given either the next player
    * to update from the schedule is taken (alternating updates) or every player is updated at the
    * same time (simultaneous updates).
    */
   template < bool initializing_run >
   auto _iterate(std::optional< Player > player_to_update);

   inline void _assert_sequential_game()
   {
      if(m_env.turn_dynamic() != TurnDynamic::sequential) {
         throw std::invalid_argument(
            "VanillaCFR can only be performed on a sequential turn-based game.");
      }
   }
   inline void _init_player_update_schedule()
   {
      if constexpr(cfr_config.alternating_updates) {
         for(auto player : m_env.players()) {
            if(player != Player::chance) {
               m_player_update_schedule.push_back(player);
            }
         }
      }
   }

   /**
    * @brief traverses the game tree and fills the nodes with current policy weighted value updates.
    *
    * This function is the regular traversal function to call on iteration i > 0, after the nodes
    * have been emplaced by @see _first_traversal.
    */
   template < bool initialize_infonodes >
   ValueMap _traversal(
      std::optional< Player > player_to_update,
      uptr< world_state_type > curr_worldstate,
      ReachProbabilityMap reach_probability,
      ObservationbufferMap observation_buffer,
      InfostateMap infostate_map,
      bool use_current_policy = true);

   template < bool initialize_infonodes >
   void _traverse_player_actions(
      std::optional< Player > player_to_update,
      Player active_player,
      uptr< world_state_type > state,
      const ReachProbabilityMap& reach_probability,
      const ObservationbufferMap& observation_buffer,
      InfostateMap infostate_map,
      ValueMap& state_value,
      std::unordered_map< action_variant_type, ValueMap >& action_value,
      bool use_current_policy = true);

   template < bool initialize_infonodes >
   void _traverse_chance_actions(
      std::optional< Player > player_to_update,
      Player active_player,
      uptr< world_state_type > state,
      const ReachProbabilityMap& reach_probability,
      const ObservationbufferMap& observation_buffer,
      InfostateMap infostate_map,
      ValueMap& state_value,
      std::unordered_map< action_variant_type, ValueMap >& action_value,
      bool use_current_policy = true);

   auto _fill_infostate_and_obs_buffers(
      const ObservationbufferMap& observation_buffer,
      const InfostateMap& infostate_map,
      const auto& action_or_outcome,
      const world_state_type& state);

   /**
    * @brief emplaces the environment rewards for a terminal state and stores them in the node.
    *
    * No terminality checking is done within this method! Hence only call this method if you are
    * already certain that the node is a terminal one. Whether the environment rewards for
    * non-terminal states would be problematic is dependant on the environment.
    * @param[in] terminal_wstate the terminal state to collect rewards for.
    * @param[out] node the terminal node in which to store the reward values.
    */
   auto _collect_rewards(
      std::conditional_t<  // the fosg concept asserts a reward function taking world_state_type.
                           // But if it can be passed a const world state then do so instead
         concepts::has::method::reward< env_type, const world_state_type& >,
         const world_state_type&,
         world_state_type& > terminal_wstate) const;

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
   /// the fallback policy to use when the encountered infostate has not been obseved before
   default_policy_type m_default_policy;
   /// the relevant data stored at each infostate
   std::unordered_map<
      sptr< info_state_type >,
      infostate_data_type,
      decltype(
         [](const sptr< info_state_type >& ptr) { return std::hash< info_state_type >{}(*ptr); }),
      decltype([](const sptr< info_state_type >& ptr1, const sptr< info_state_type >& ptr2) {
         return *ptr1 == *ptr2;
      }) >
      m_infonode_data{};
   /// the next player to update when doing alternative updates. Otherwise this member will be
   /// unused.
   std::deque< Player > m_player_update_schedule{};
   /// the number of iterations we have run so far.
   size_t m_iteration = 0;
};
///
//////
/////////
////////////
///////////////
//////////////////
/////////////////////
////////////////////////
///////////////////////////
//////////////////////////////
/////////////////////////////////
///////////////////////////////////
///                              ///
/// member function definitions  ///
///                              ///
///////////////////////////////////
/////////////////////////////////
//////////////////////////////
///////////////////////////
////////////////////////
/////////////////////
//////////////////
///////////////
////////////
/////////
//////
///

template <
   CFRConfig cfr_config,
   typename Env,
   typename Policy,
   typename DefaultPolicy,
   typename AveragePolicy >
requires concepts::vanilla_cfr_requirements< Env, Policy, DefaultPolicy, AveragePolicy >
auto VanillaCFR< cfr_config, Env, Policy, DefaultPolicy, AveragePolicy >::iterate(size_t n_iters)
{
   std::vector< std::unordered_map< Player, double > > root_values_per_iteration;
   root_values_per_iteration.reserve(n_iters);
   for([[maybe_unused]] auto _ : ranges::views::iota(size_t(0), n_iters)) {
      LOGD2("Iteration number: ", m_iteration);
      ValueMap value = [&] {
         if constexpr(cfr_config.alternating_updates) {
            auto player_to_update = _cycle_player_to_update();
            if(m_iteration == 0) {
               return _iterate< true >(player_to_update);
            } else {
               return _iterate< false >(player_to_update);
            }
         } else {
            if(m_iteration == 0) {
               return _iterate< true >(std::nullopt);
            } else {
               return _iterate< false >(std::nullopt);
            }
         }
      }();
      root_values_per_iteration.emplace_back(std::move(value.get()));
      m_iteration++;
   }
   return root_values_per_iteration;
}

template <
   CFRConfig cfr_config,
   typename Env,
   typename Policy,
   typename DefaultPolicy,
   typename AveragePolicy >
requires concepts::vanilla_cfr_requirements< Env, Policy, DefaultPolicy, AveragePolicy >
auto VanillaCFR< cfr_config, Env, Policy, DefaultPolicy, AveragePolicy >::iterate(
   std::optional< Player > player_to_update) requires(cfr_config.alternating_updates)
{
   // we assert here that the chosen player to update is not the chance player as is defined
   // by default. Seeing the chance player here inidcates that the player forgot to set the
   // player parameter with this rm config.
   if(player_to_update.value_or(Player::alex) == Player::chance) {
      std::stringstream ssout;
      ssout << "Given combination of '";
      ssout << Player::chance;
      ssout << "' and '";
      ssout << "alternating updates'";
      ssout << "is incompatible. Did you forget to pass the correct player parameter?";
      throw std::invalid_argument(ssout.str());
   } else if(auto env_players = m_env.players();
             ranges::find(env_players, player_to_update) == env_players.end()) {
      std::stringstream ssout;
      ssout << "Given player to update ";
      ssout << player_to_update.value();
      ssout << " is not a member of the game's player list ";
      ssout << ranges::views::all(env_players) << ".";
      throw std::invalid_argument(ssout.str());
   }
   LOGD2("Iteration number: ", m_iteration);
   // run the iteration
   ValueMap values = [&] {
      if(m_iteration == 0)
         return _iterate< true >(_cycle_player_to_update(player_to_update));
      else
         return _iterate< false >(_cycle_player_to_update(player_to_update));
   }();
   // and increment our iteration counter
   m_iteration++;
   // we always return the current policy. This way the user can decide whether to store the n-th
   // iteraton's policies user-side or not.
   return std::vector{std::move(values.get())};
}

template <
   CFRConfig cfr_config,
   typename Env,
   typename Policy,
   typename DefaultPolicy,
   typename AveragePolicy >
requires concepts::vanilla_cfr_requirements< Env, Policy, DefaultPolicy, AveragePolicy >
template < bool initializing_run >
auto VanillaCFR< cfr_config, Env, Policy, DefaultPolicy, AveragePolicy >::_iterate(
   std::optional< Player > player_to_update)
{
   return _traversal< initializing_run >(
      player_to_update,
      utils::static_unique_ptr_downcast< world_state_type >(utils::clone_any_way(m_root_state)),
      [&] {
         std::unordered_map< Player, double > rp_map;
         for(auto player : m_env.players()) {
            rp_map.emplace(player, 1.);
         }
         return ReachProbabilityMap{rp_map};
      }(),
      [&] {
         std::unordered_map< Player, std::vector< observation_type > > obs_map;
         for(auto player : m_env.players()) {
            obs_map.emplace(player, std::vector< observation_type >{});
         }
         return ObservationbufferMap{obs_map};
      }(),
      [&] {
         std::unordered_map< Player, sptr< info_state_type > > infostates;
         for(auto player : m_env.players()) {
            if(player != Player::chance) {
               auto& infostate = infostates
                                    .emplace(player, std::make_shared< info_state_type >(player))
                                    .first->second;
               infostate->append(m_env.private_observation(player, *m_root_state));
            }
         }
         return InfostateMap{infostates};
      }());
}

template <
   CFRConfig cfr_config,
   typename Env,
   typename Policy,
   typename DefaultPolicy,
   typename AveragePolicy >
requires concepts::vanilla_cfr_requirements< Env, Policy, DefaultPolicy, AveragePolicy >
void VanillaCFR< cfr_config, Env, Policy, DefaultPolicy, AveragePolicy >::update_regret_and_policy(
   const sptr< info_state_type >& infostate,
   ReachProbabilityMap reach_probability,
   const ValueMap& state_value,
   const std::unordered_map< action_variant_type, ValueMap >& action_value)
{
   auto& curr_state_policy = fetch_policy(true, infostate);
   auto& avg_state_policy = fetch_policy(false, infostate);
   auto player = infostate->player();
   auto& infodata = m_infonode_data.at(infostate);
   for(const auto& [action_variant, q_value] : action_value) {
      // we only call this function with action values from a non-chance player, so we can safely
      // assume that the action is of action_type
      const auto& action = std::get< action_type >(action_variant);
      // update the regret according to the formula (let I be the infostate, p be the player):
      //    r = r + counterfactual_reach_prob_{p}(I) * (value_{p}(I, a) - value_{p}(I))
      infodata.regret(action) += rm::cf_reach_probability(reach_probability.get(), player)
                                 * (q_value.get().at(player) - state_value.get().at(player));
      avg_state_policy[action] += reach_probability.get()[player] * curr_state_policy[action];
   }
   regret_matching(curr_state_policy, infodata.regret());
}

template <
   CFRConfig cfr_config,
   typename Env,
   typename Policy,
   typename DefaultPolicy,
   typename AveragePolicy >
requires concepts::vanilla_cfr_requirements< Env, Policy, DefaultPolicy, AveragePolicy >
template < bool initialize_infonodes >
typename VanillaCFR< cfr_config, Env, Policy, DefaultPolicy, AveragePolicy >::ValueMap
VanillaCFR< cfr_config, Env, Policy, DefaultPolicy, AveragePolicy >::_traversal(
   std::optional< Player > player_to_update,
   uptr< world_state_type > state,
   ReachProbabilityMap reach_probability,
   ObservationbufferMap observation_buffer,
   InfostateMap infostates,
   bool use_current_policy)
{
   if(m_env.is_terminal(*state)) {
      return ValueMap{_collect_rewards(*state)};
   }

   Player active_player = m_env.active_player(*state);
   // the state's value for each player. To be filled by the action traversal functions.
   ValueMap state_value{std::unordered_map< Player, double >{}};
   // each actions's value for each player. To be filled by the action traversal functions.
   std::unordered_map< action_variant_type, ValueMap > action_value;
   // copy the current infostate ptr before the infostates are moved into the child traversal funcs
   sptr< info_state_type > this_infostate = nullptr;

   // traverse all child states from this state. The constexpr check for determinism in the env
   // allows deterministic envs to not provide certain functions that are only needed in the
   // stochastic case.
   // First define the default branch for an active non-chance player
   auto nonchance_player_traversal = [&] {
      this_infostate = infostates.get().at(active_player);
      if constexpr(initialize_infonodes) {
         m_infonode_data.emplace(
            this_infostate, infostate_data_type{m_env.actions(active_player, *state)});
      }
      _traverse_player_actions< initialize_infonodes >(
         player_to_update,
         active_player,
         std::move(state),
         reach_probability,
         std::move(observation_buffer),
         std::move(infostates),
         state_value,
         action_value);
   };
   // now we check first if we even need to consider a chance player, as the env could be simply
   // deterministic. In that case we might need to traverse the chance player's actions or an active
   // player's actions
   if constexpr(not concepts::deterministic_fosg< env_type >) {
      if(active_player == Player::chance) {
         _traverse_chance_actions< initialize_infonodes >(
            player_to_update,
            active_player,
            std::move(state),
            reach_probability,
            std::move(observation_buffer),
            std::move(infostates),
            state_value,
            action_value);
      } else {
         nonchance_player_traversal();
      }
   } else {
      nonchance_player_traversal();
   }

   //   for(auto [player, value] : state_value.get()) {
   //      LOGD2("Player", player);
   //      LOGD2("Value", value);
   //   }

   if(active_player != Player::chance) {
      if constexpr(cfr_config.alternating_updates) {
         // in alternating updates, we only update the regret and strategy if the current
         // player is the chosen player to update.
         if(active_player == player_to_update.value_or(Player::chance)) {
            update_regret_and_policy(
               this_infostate, reach_probability, ValueMap{state_value}, action_value);
         }
      } else {
         // if we do simultaenous updates, then we always update the regret and strategy
         // values of the node's active player.
         update_regret_and_policy(
            this_infostate, reach_probability, ValueMap{state_value}, action_value);
      }
   }
   return ValueMap{state_value};
}

template <
   CFRConfig cfr_config,
   typename Env,
   typename Policy,
   typename DefaultPolicy,
   typename AveragePolicy >
requires concepts::vanilla_cfr_requirements< Env, Policy, DefaultPolicy, AveragePolicy >
template < bool initialize_infonodes >
void VanillaCFR< cfr_config, Env, Policy, DefaultPolicy, AveragePolicy >::_traverse_player_actions(
   std::optional< Player > player_to_update,
   Player active_player,
   uptr< world_state_type > state,
   const ReachProbabilityMap& reach_probability,
   const ObservationbufferMap& observation_buffer,
   InfostateMap infostate_map,
   ValueMap& state_value,
   std::unordered_map< action_variant_type, ValueMap >& action_value,
   bool use_current_policy)
{
   auto& this_infostate = infostate_map.get().at(active_player);
   if constexpr(initialize_infonodes) {
      m_infonode_data.emplace(
         this_infostate, infostate_data_type{m_env.actions(active_player, *state)});
   }
   auto& curr_infostate_policy = fetch_policy(use_current_policy, this_infostate);

   for(const action_type& action :
       m_infonode_data.at(this_infostate).regret() | ranges::views::keys) {
      // clone the current world state to transition it uniquely in this action transition
      uptr< world_state_type >
         next_wstate_uptr = utils::static_unique_ptr_downcast< world_state_type >(
            utils::clone_any_way(state));
      // move the new world state forward by the current action
      m_env.transition(*next_wstate_uptr, action);

      auto child_reach_prob = reach_probability.get();
      auto action_prob = curr_infostate_policy[action];
      child_reach_prob[active_player] *= action_prob;

      auto [observation_buffer_copy, child_infostate_map] = _fill_infostate_and_obs_buffers(
         observation_buffer, infostate_map, action, *next_wstate_uptr);

      ValueMap child_rewards_map = _traversal< initialize_infonodes >(
         player_to_update,
         std::move(next_wstate_uptr),
         ReachProbabilityMap{std::move(child_reach_prob)},
         ObservationbufferMap{std::move(observation_buffer_copy)},
         InfostateMap{std::move(child_infostate_map)},
         use_current_policy);
      for(auto [player, child_value] : child_rewards_map.get()) {
         state_value.get()[player] += action_prob * child_value;
      }
      action_value.emplace(action, std::move(child_rewards_map));
   }
}

template <
   CFRConfig cfr_config,
   typename Env,
   typename Policy,
   typename DefaultPolicy,
   typename AveragePolicy >
requires concepts::vanilla_cfr_requirements< Env, Policy, DefaultPolicy, AveragePolicy >
template < bool initialize_infonodes >
void VanillaCFR< cfr_config, Env, Policy, DefaultPolicy, AveragePolicy >::_traverse_chance_actions(
   std::optional< Player > player_to_update,
   Player active_player,
   uptr< world_state_type > state,
   const ReachProbabilityMap& reach_probability,
   const ObservationbufferMap& observation_buffer,
   InfostateMap infostate_map,
   ValueMap& state_value,
   std::unordered_map< action_variant_type, ValueMap >& action_value,
   bool use_current_policy)
{
   for(auto&& outcome : m_env.chance_actions(*state)) {
      uptr< world_state_type >
         next_wstate_uptr = utils::static_unique_ptr_downcast< world_state_type >(
            utils::clone_any_way(state));
      // move the new world state forward by the current action
      m_env.transition(*next_wstate_uptr, outcome);

      auto child_reach_prob = reach_probability.get();
      auto outcome_prob = m_env.chance_probability(*state, outcome);
      child_reach_prob[active_player] *= outcome_prob;

      auto [observation_buffer_copy, child_infostate_map] = _fill_infostate_and_obs_buffers(
         observation_buffer, infostate_map, outcome, *next_wstate_uptr);

      ValueMap child_rewards_map = _traversal< initialize_infonodes >(
         player_to_update,
         std::move(next_wstate_uptr),
         ReachProbabilityMap{std::move(child_reach_prob)},
         ObservationbufferMap{std::move(observation_buffer_copy)},
         InfostateMap{std::move(child_infostate_map)},
         use_current_policy);
      for(auto [player, child_value] : child_rewards_map.get()) {
         state_value.get()[player] += outcome_prob * child_value;
      }
      action_value.emplace(std::move(outcome), std::move(child_rewards_map));
   }
}

template <
   CFRConfig cfr_config,
   typename Env,
   typename Policy,
   typename DefaultPolicy,
   typename AveragePolicy >
requires concepts::vanilla_cfr_requirements< Env, Policy, DefaultPolicy, AveragePolicy >
auto VanillaCFR< cfr_config, Env, Policy, DefaultPolicy, AveragePolicy >::
   _fill_infostate_and_obs_buffers(
      const ObservationbufferMap& observation_buffer,
      const InfostateMap& infostate_map,
      const auto& action_or_outcome,
      const world_state_type& state)
{
   auto active_player = m_env.active_player(state);
   std::unordered_map< Player, sptr< info_state_type > > child_infostate_map;
   auto observation_buffer_copy = observation_buffer.get();
   for(auto player : m_env.players()) {
      if(player == Player::chance) {
         continue;
      }
      if(player != active_player) {
         // for all but the active player we simply append action and state observation to the
         // buffer. They will be written to an actual infostate once that player becomes the
         // active player again
         auto& player_infostate = observation_buffer_copy.at(player);
         player_infostate.emplace_back(m_env.private_observation(player, action_or_outcome));
         player_infostate.emplace_back(m_env.private_observation(player, state));
         child_infostate_map.emplace(player, infostate_map.get().at(player));
      } else {
         // for the active player we first append all recent action and state observations to a
         // info state copy, and then follow it up by adding the current action and state
         // observations
         auto cloned_infostate_ptr = sptr< info_state_type >(
            utils::clone_any_way(infostate_map.get().at(active_player)));
         auto& obs_history = observation_buffer_copy[active_player];
         for(auto& obs : obs_history) {
            cloned_infostate_ptr->append(std::move(obs));
         }
         obs_history.clear();
         cloned_infostate_ptr->append(m_env.private_observation(player, action_or_outcome));
         cloned_infostate_ptr->append(m_env.private_observation(player, state));
         child_infostate_map.emplace(player, std::move(cloned_infostate_ptr));
      }
   }
   return std::tuple{observation_buffer_copy, child_infostate_map};
}

template <
   CFRConfig cfr_config,
   typename Env,
   typename Policy,
   typename DefaultPolicy,
   typename AveragePolicy >
requires concepts::vanilla_cfr_requirements< Env, Policy, DefaultPolicy, AveragePolicy >
auto VanillaCFR< cfr_config, Env, Policy, DefaultPolicy, AveragePolicy >::_collect_rewards(
   std::conditional_t<
      concepts::has::method::reward< env_type, const world_state_type& >,
      const world_state_type&,
      world_state_type& > terminal_wstate) const
{
   std::unordered_map< Player, double > rewards;
   if constexpr(nor::concepts::has::method::reward_multi< Env >) {
      // if the environment has a method for returning all rewards for given players at
      // once, then we will assume this is a more performant alternative and use it
      // instead (e.g. when it is costly to compute the reward of each player
      // individually).
      auto non_chance_filter = [](Player player) { return player != Player::chance; };
      auto all_rewards = m_env.reward(m_env.players(), terminal_wstate);
      ranges::views::for_each(
         m_env.players() | ranges::views::filter(non_chance_filter)
            | ranges::views::transform([&](Player player) {
                 return std::pair{player, all_rewards[player]};
              }),
         [&](const auto& player_reward) {
            rewards.emplace(std::get< 0 >(player_reward), std::get< 1 >(player_reward));
         });
   } else {
      // otherwise we just loop over the per player reward method
      for(auto player : m_env.players()) {
         if(player != Player::chance)
            rewards.emplace(player, m_env.reward(player, terminal_wstate));
      }
   }
   return rewards;
}

template <
   CFRConfig cfr_config,
   typename Env,
   typename Policy,
   typename DefaultPolicy,
   typename AveragePolicy >
requires concepts::vanilla_cfr_requirements< Env, Policy, DefaultPolicy, AveragePolicy > Player
VanillaCFR< cfr_config, Env, Policy, DefaultPolicy, AveragePolicy >::_cycle_player_to_update(
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

template <
   CFRConfig cfr_config,
   typename Env,
   typename Policy,
   typename DefaultPolicy,
   typename AveragePolicy >
requires concepts::vanilla_cfr_requirements< Env, Policy, DefaultPolicy, AveragePolicy >
auto& VanillaCFR< cfr_config, Env, Policy, DefaultPolicy, AveragePolicy >::fetch_policy(
   bool current_policy,
   const sptr< info_state_type >& infostate)
{
   if(current_policy) {
      auto& player_policy = m_curr_policy[infostate->player()];
      auto found_action_policy = player_policy.find(*infostate);
      if(found_action_policy == player_policy.end()) {
         auto actions = ranges::to< std::vector< action_type > >(
            m_infonode_data.at(infostate).regret() | ranges::views::keys);
         return player_policy.emplace(*infostate, m_default_policy[{*infostate, actions}])
            .first->second;
      }
      return found_action_policy->second;
   } else {
      auto& player_policy = m_avg_policy[infostate->player()];
      auto found_action_policy = player_policy.find(*infostate);
      if(found_action_policy == player_policy.end()) {
         // the average policy is necessary to be 0 initialized on all unseen entries,
         // since these entries are to be updated cumulatively.
         typename AveragePolicy::action_policy_type default_avg_policy;
         for(const auto& action : m_infonode_data.at(infostate).regret() | ranges::views::keys) {
            // we know that no chance outcome could be demanded here
            // so we ask for the action type to be returned
            default_avg_policy[action] = 0.;
         }
         return player_policy.emplace(*infostate, default_avg_policy).first->second;
      }
      return found_action_policy->second;
   }
}

}  // namespace nor::rm

#endif  // NOR_VANILLA_HPP
