
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
template < CFRConfig cfr_config, typename Env, typename Policy, typename AveragePolicy >
   requires concepts::vanilla_cfr_requirements< Env, Policy, AveragePolicy >
class VanillaCFR {
   ////////////////////////////
   /// API: public typedefs ///
   ////////////////////////////
  public:
   /// aliases for the template types
   using env_type = Env;
   using policy_type = Policy;
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

   VanillaCFR(Env env, Policy policy = Policy(), AveragePolicy avg_policy = AveragePolicy())
      // clang-format off
      requires
         concepts::has::method::initial_world_state< Env >
   // clang-format on
   :
       VanillaCFR(
          std::move(env),
          std::make_unique< world_state_type >(env.initial_world_state()),
          std::move(policy),
          std::move(avg_policy))
   {
      _init_player_update_schedule();
   }

   VanillaCFR(
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
   auto iterate(std::optional< Player > player_to_update = std::nullopt)
      requires(cfr_config.alternating_updates)
   ;

   ValueMap game_value() { return _iterate< false, false >(std::nullopt); }

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
      const ReachProbabilityMap& reach_probability,
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
   template < bool initializing_run, bool use_current_policy = true >
   auto _iterate(std::optional< Player > player_to_update);

   /**
    * @brief traverses the game tree and fills the nodes with current policy weighted value updates.
    *
    * This function is the regular traversal function to call on iteration i > 0, after the nodes
    * have been emplaced by @see _first_traversal.
    */
   template < bool initialize_infonodes, bool use_current_policy = true >
   ValueMap _traversal(
      std::optional< Player > player_to_update,
      uptr< world_state_type > curr_worldstate,
      ReachProbabilityMap reach_probability,
      ObservationbufferMap observation_buffer,
      InfostateMap infostate_map);

   template < bool initialize_infonodes, bool use_current_policy = true >
   void _traverse_player_actions(
      std::optional< Player > player_to_update,
      Player active_player,
      uptr< world_state_type > state,
      const ReachProbabilityMap& reach_probability,
      const ObservationbufferMap& observation_buffer,
      InfostateMap infostate_map,
      ValueMap& state_value,
      std::unordered_map< action_variant_type, ValueMap >& action_value);

   template < bool initialize_infonodes, bool use_current_policy = true >
   void _traverse_chance_actions(
      std::optional< Player > player_to_update,
      Player active_player,
      uptr< world_state_type > state,
      const ReachProbabilityMap& reach_probability,
      const ObservationbufferMap& observation_buffer,
      InfostateMap infostate_map,
      ValueMap& state_value,
      std::unordered_map< action_variant_type, ValueMap >& action_value);

   auto _fill_infostate_and_obs_buffers(
      const ObservationbufferMap& observation_buffer,
      const InfostateMap& infostate_map,
      const auto& action_or_outcome,
      const world_state_type& state);

   void _apply_regret_matching(const std::optional< Player >& player_to_update);

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
      const_ref_if_t<  // the fosg concept asserts a reward function taking world_state_type.
                       // But if it can be passed a const world state then do so instead
         concepts::has::method::reward< env_type, const world_state_type& >,
         world_state_type > terminal_wstate) const;

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
      if constexpr(cfr_config.alternating_updates) {
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

template < CFRConfig cfr_config, typename Env, typename Policy, typename AveragePolicy >
   requires concepts::vanilla_cfr_requirements< Env, Policy, AveragePolicy >
auto VanillaCFR< cfr_config, Env, Policy, AveragePolicy >::iterate(size_t n_iters)
{
   std::vector< std::unordered_map< Player, double > > root_values_per_iteration;
   root_values_per_iteration.reserve(n_iters);
   for([[maybe_unused]] auto _ : ranges::views::iota(size_t(0), n_iters)) {
      LOGD2("Iteration number: ", m_iteration);
      ValueMap value = [&] {
         if constexpr(cfr_config.alternating_updates) {
            auto player_to_update = _cycle_player_to_update();
            if(m_iteration < m_env.players().size() - 1) {
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

template < CFRConfig cfr_config, typename Env, typename Policy, typename AveragePolicy >
   requires concepts::vanilla_cfr_requirements< Env, Policy, AveragePolicy >
auto VanillaCFR< cfr_config, Env, Policy, AveragePolicy >::iterate(
   std::optional< Player > player_to_update)
   requires(cfr_config.alternating_updates)
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
      if(m_iteration < m_env.players().size() - 1)
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

template < CFRConfig cfr_config, typename Env, typename Policy, typename AveragePolicy >
   requires concepts::vanilla_cfr_requirements< Env, Policy, AveragePolicy >
template < bool initializing_run, bool use_current_policy >
auto VanillaCFR< cfr_config, Env, Policy, AveragePolicy >::_iterate(
   std::optional< Player > player_to_update)
{
   auto root_game_value = _traversal< initializing_run, use_current_policy >(
      player_to_update,
      utils::static_unique_ptr_downcast< world_state_type >(utils::clone_any_way(m_root_state)),
      [&] {
         std::unordered_map< Player, double > rp_map;
         for(auto player : m_env.players()) {
            rp_map.emplace(player, 1.);
         }
         return ReachProbabilityMap{std::move(rp_map)};
      }(),
      [&] {
         std::unordered_map< Player, std::vector< observation_type > > obs_map;
         auto players = m_env.players();
         for(auto player : players | utils::is_nonchance_player_filter) {
            obs_map.emplace(player, std::vector< observation_type >{});
         }
         return ObservationbufferMap{std::move(obs_map)};
      }(),
      [&] {
         std::unordered_map< Player, sptr< info_state_type > > infostates;
         auto players = m_env.players();
         for(auto player : players | utils::is_nonchance_player_filter) {
            auto& infostate = infostates
                                 .emplace(player, std::make_shared< info_state_type >(player))
                                 .first->second;
            infostate->append(m_env.private_observation(player, *m_root_state));
         }
         return InfostateMap{std::move(infostates)};
      }());

   if constexpr(use_current_policy) {
      _apply_regret_matching(player_to_update);
   }
   return root_game_value;
}

template < CFRConfig cfr_config, typename Env, typename Policy, typename AveragePolicy >
   requires concepts::vanilla_cfr_requirements< Env, Policy, AveragePolicy >
void VanillaCFR< cfr_config, Env, Policy, AveragePolicy >::_apply_regret_matching(
   const std::optional< Player >& player_to_update)
{
   auto call_regret_matching =
      [&](const sptr< info_state_type >& infostate_ptr, const infostate_data_type& infodata) {
         rm::regret_matching(
            fetch_policy< true >(infostate_ptr, infodata.actions()),
            infodata.regret(),
            // we provide the accessor to get the underlying referenced action, as the infodata
            // stores only reference wrappers to the actions
            [](const action_type& action) {
               return std::ref(action);
            });
      };

   if constexpr(cfr_config.alternating_updates) {
      Player update_player = player_to_update.value_or(Player::chance);
      for(auto& [infostate_ptr, infodata] : m_infonode_data) {
         if(infostate_ptr->player() == update_player) {
            call_regret_matching(infostate_ptr, infodata);
         }
      }
   } else {
      for(auto& [infostate_ptr, infodata] : m_infonode_data) {
         call_regret_matching(infostate_ptr, infodata);
      }
   }
}

template < CFRConfig cfr_config, typename Env, typename Policy, typename AveragePolicy >
   requires concepts::vanilla_cfr_requirements< Env, Policy, AveragePolicy >
template < bool initialize_infonodes, bool use_current_policy >
typename VanillaCFR< cfr_config, Env, Policy, AveragePolicy >::ValueMap
VanillaCFR< cfr_config, Env, Policy, AveragePolicy >::_traversal(
   std::optional< Player > player_to_update,
   uptr< world_state_type > state,
   ReachProbabilityMap reach_probability,
   ObservationbufferMap observation_buffer,
   InfostateMap infostates)
{
   if(m_env.is_terminal(*state)) {
      return ValueMap{_collect_rewards(*state)};
   }

   Player active_player = m_env.active_player(*state);
   sptr< info_state_type > this_infostate = nullptr;
   // the state's value for each player. To be filled by the action traversal functions.
   ValueMap state_value{{}};
   // each actions's value for each player. To be filled by the action traversal functions.
   std::unordered_map< action_variant_type, ValueMap > action_value;
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
      _traverse_player_actions< initialize_infonodes, use_current_policy >(
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
         _traverse_chance_actions< initialize_infonodes, use_current_policy >(
            player_to_update,
            active_player,
            std::move(state),
            reach_probability,
            std::move(observation_buffer),
            std::move(infostates),
            state_value,
            action_value);
         // if this is a chance node then we dont need to update any regret or average policy after
         // the traversal
         return state_value;
      } else {
         nonchance_player_traversal();
      }
   } else {
      nonchance_player_traversal();
   }

   if constexpr(use_current_policy) {
      // we can only update our regrets and policies if we are traversing with the current policy,
      // since the average policy is not to be changed directly (but through averaging up all
      // current policies)
      if constexpr(cfr_config.alternating_updates) {
         // in alternating updates, we only update the regret and strategy if the current
         // player is the chosen player to update.
         if(active_player == player_to_update.value_or(Player::chance)) {
            update_regret_and_policy(this_infostate, reach_probability, state_value, action_value);
         }
      } else {
         // if we do simultaenous updates, then we always update the regret and strategy
         // values of the node's active player.
         update_regret_and_policy(this_infostate, reach_probability, state_value, action_value);
      }
   }
   return ValueMap{state_value};
}

template < CFRConfig cfr_config, typename Env, typename Policy, typename AveragePolicy >
   requires concepts::vanilla_cfr_requirements< Env, Policy, AveragePolicy >
template < bool initialize_infonodes, bool use_current_policy >
void VanillaCFR< cfr_config, Env, Policy, AveragePolicy >::_traverse_player_actions(
   std::optional< Player > player_to_update,
   Player active_player,
   uptr< world_state_type > state,
   const ReachProbabilityMap& reach_probability,
   const ObservationbufferMap& observation_buffer,
   InfostateMap infostate_map,
   ValueMap& state_value,
   std::unordered_map< action_variant_type, ValueMap >& action_value)
{
   auto& this_infostate = infostate_map.get().at(active_player);
   if constexpr(initialize_infonodes) {
      m_infonode_data.emplace(
         this_infostate, infostate_data_type{m_env.actions(active_player, *state)});
   }
   auto& action_policy = fetch_policy< use_current_policy >(
      this_infostate, m_infonode_data.at(this_infostate).actions());
   double normalizing_factor = 1.;
   if constexpr(not use_current_policy) {
      // we try to normalize only for the average policy, since iterations with the current policy
      // are for the express purpose of updating the average strategy. As such, we should not
      // intervene to change these values, as that may alter the values incorrectly
      auto action_policy_value_view = action_policy | ranges::views::values;
      normalizing_factor = std::reduce(
         action_policy_value_view.begin(), action_policy_value_view.end(), 0., std::plus{});

      if(std::abs(normalizing_factor) < 1e-20) {
         throw std::invalid_argument(
            "Average policy likelihoods accumulate to 0. Such values cannot be normalized.");
      }
   }

   for(const action_type& action :
       m_infonode_data.at(this_infostate).regret() | ranges::views::keys) {
      // clone the current world state first before transitioniong it with this action
      uptr< world_state_type >
         next_wstate_uptr = utils::static_unique_ptr_downcast< world_state_type >(
            utils::clone_any_way(state));
      // move the new world state forward by the current action
      m_env.transition(*next_wstate_uptr, action);

      auto child_reach_prob = reach_probability.get();
      auto action_prob = action_policy[action] / normalizing_factor;
      child_reach_prob[active_player] *= action_prob;

      auto [observation_buffer_copy, child_infostate_map] = _fill_infostate_and_obs_buffers(
         observation_buffer, infostate_map, action, *next_wstate_uptr);

      ValueMap child_rewards_map = _traversal< initialize_infonodes, use_current_policy >(
         player_to_update,
         std::move(next_wstate_uptr),
         ReachProbabilityMap{std::move(child_reach_prob)},
         ObservationbufferMap{std::move(observation_buffer_copy)},
         InfostateMap{std::move(child_infostate_map)});
      // add the child state's value to the respective player's value table, multiplied by the
      // policies likelihood of playing this action
      for(auto [player, child_value] : child_rewards_map.get()) {
         state_value.get()[player] += action_prob * child_value;
      }
      action_value.emplace(action, std::move(child_rewards_map));
   }
}

template < CFRConfig cfr_config, typename Env, typename Policy, typename AveragePolicy >
   requires concepts::vanilla_cfr_requirements< Env, Policy, AveragePolicy >
template < bool initialize_infonodes, bool use_current_policy >
void VanillaCFR< cfr_config, Env, Policy, AveragePolicy >::_traverse_chance_actions(
   std::optional< Player > player_to_update,
   Player active_player,
   uptr< world_state_type > state,
   const ReachProbabilityMap& reach_probability,
   const ObservationbufferMap& observation_buffer,
   InfostateMap infostate_map,
   ValueMap& state_value,
   std::unordered_map< action_variant_type, ValueMap >& action_value)
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

      ValueMap child_rewards_map = _traversal< initialize_infonodes, use_current_policy >(
         player_to_update,
         std::move(next_wstate_uptr),
         ReachProbabilityMap{std::move(child_reach_prob)},
         ObservationbufferMap{std::move(observation_buffer_copy)},
         InfostateMap{std::move(child_infostate_map)});
      // add the child state's value to the respective player's value table, multiplied by the
      // policies likelihood of playing this action
      for(auto [player, child_value] : child_rewards_map.get()) {
         state_value.get()[player] += outcome_prob * child_value;
      }
      action_value.emplace(std::move(outcome), std::move(child_rewards_map));
   }
}

template < CFRConfig cfr_config, typename Env, typename Policy, typename AveragePolicy >
   requires concepts::vanilla_cfr_requirements< Env, Policy, AveragePolicy >
auto VanillaCFR< cfr_config, Env, Policy, AveragePolicy >::_fill_infostate_and_obs_buffers(
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
         child_infostate_map.emplace(player, infostate_map.get().at(player));
         auto& player_infostate = observation_buffer_copy.at(player);
         player_infostate.emplace_back(m_env.private_observation(player, action_or_outcome));
         player_infostate.emplace_back(m_env.private_observation(player, state));
      } else {
         // for the active player we first append all recent action and state observations to a
         // info state copy, and then follow it up by adding the current action and state
         // observations
         auto cloned_infostate_ptr = utils::clone_any_way(infostate_map.get().at(active_player));
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

template < CFRConfig cfr_config, typename Env, typename Policy, typename AveragePolicy >
   requires concepts::vanilla_cfr_requirements< Env, Policy, AveragePolicy >
void VanillaCFR< cfr_config, Env, Policy, AveragePolicy >::update_regret_and_policy(
   const sptr< info_state_type >& infostate,
   const ReachProbabilityMap& reach_probability,
   const ValueMap& state_value,
   const std::unordered_map< action_variant_type, ValueMap >& action_value)
{
   auto& infodata = m_infonode_data.at(infostate);
   auto& curr_action_policy = fetch_policy< true >(infostate, infodata.actions());
   auto& avg_action_policy = fetch_policy< false >(infostate, infodata.actions());
   auto player = infostate->player();
   double cf_reach_prob = rm::cf_reach_probability(reach_probability.get(), player);
   double player_reach_prob = reach_probability.get().at(player);
   double player_state_value = state_value.get().at(player);

   for(const auto& [action_variant, q_value] : action_value) {
      // we only call this function with action values from a non-chance player, so we can safely
      // assume that the action is of action_type
      const auto& action = std::get< action_type >(action_variant);
      // update the cumulative regret according to the formula:
      // let I be the infostate, p be the player, r the cumulative regret
      //    r = \sum_a counterfactual_reach_prob_{p}(I) * (value_{p}(I-->a) - value_{p}(I))
      infodata.regret(action) += cf_reach_prob * (q_value.get().at(player) - player_state_value);
      avg_action_policy[action] += player_reach_prob * curr_action_policy[action];
   }
}

template < CFRConfig cfr_config, typename Env, typename Policy, typename AveragePolicy >
   requires concepts::vanilla_cfr_requirements< Env, Policy, AveragePolicy >
auto VanillaCFR< cfr_config, Env, Policy, AveragePolicy >::_collect_rewards(
   const_ref_if_t<
      concepts::has::method::reward< env_type, const world_state_type& >,
      world_state_type > terminal_wstate) const
{
   std::unordered_map< Player, double > rewards;
   auto players = m_env.players();
   if constexpr(nor::concepts::has::method::reward_multi< Env >) {
      // if the environment has a method for returning all rewards for given players at
      // once, then we will assume this is a more performant alternative and use it
      // instead (e.g. when it is costly to compute the reward of each player
      // individually).
      std::erase(std::remove_if(players.begin(), players.end(), utils::is_chance_player_pred));
      auto all_rewards = m_env.reward(players, terminal_wstate);
      ranges::views::for_each(
         players, [&](Player player) { rewards.emplace(player, all_rewards[player]); });
   } else {
      // otherwise we just loop over the per player reward method
      for(auto player : players | utils::is_nonchance_player_filter) {
         rewards.emplace(player, m_env.reward(player, terminal_wstate));
      }
      return rewards;
   }
}

template < CFRConfig cfr_config, typename Env, typename Policy, typename AveragePolicy >
   requires concepts::vanilla_cfr_requirements< Env, Policy, AveragePolicy >
Player VanillaCFR< cfr_config, Env, Policy, AveragePolicy >::_cycle_player_to_update(
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

template < CFRConfig cfr_config, typename Env, typename Policy, typename AveragePolicy >
   requires concepts::vanilla_cfr_requirements< Env, Policy, AveragePolicy >
template < bool current_policy >
auto& VanillaCFR< cfr_config, Env, Policy, AveragePolicy >::fetch_policy(
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

#endif  // NOR_VANILLA_HPP
