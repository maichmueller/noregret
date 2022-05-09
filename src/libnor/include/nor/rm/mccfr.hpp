
#ifndef NOR_MCCFR_HPP
#define NOR_MCCFR_HPP

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

enum class MCCFRAlgorithm { outcome_sampling = 0, external_sampling = 1 };

enum class MCCFRWeightingMode { lazy = 0, optimistic = 1 };

enum class MCCFRExplorationMode {
   epsilon_on_policy = 0,
};

struct MCCFRConfig {
   bool alternating_updates = true;
   MCCFRAlgorithm algorithm = MCCFRAlgorithm::outcome_sampling;
   MCCFRExplorationMode exploration = MCCFRExplorationMode::epsilon_on_policy;
   MCCFRWeightingMode weighting = MCCFRWeightingMode::lazy;
};

/**
 * The Monte-Carlo Counterfactual Regret Minimization algorithm class following the
 * terminology of the Factored-Observation Stochastic Games (FOSG) formulation.
 *
 * @tparam Env, the environment/game type to run VanillaCFR on.
 *
 */
template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
class MCCFR: public TabularCFRBase< config.alternating_updates, Env, Policy, AveragePolicy > {
   ////////////////////////////
   /// API: public typedefs ///
   ////////////////////////////
  public:
   /// aliases for the template types
   using base = TabularCFRBase< config.alternating_updates, Env, Policy, AveragePolicy >;
   using env_type = Env;
   using policy_type = Policy;
   /// import all fosg aliases to be used in this class from the env type.
   using typename base::action_type;
   using typename base::world_state_type;
   using typename base::info_state_type;
   using typename base::public_state_type;
   using typename base::observation_type;
   using typename base::chance_outcome_type;
   using typename base::chance_distribution_type;
   using action_variant_type = std::variant<
      action_type,
      std::conditional_t<
         std::is_same_v< chance_outcome_type, void >,
         std::monostate,
         chance_outcome_type > >;
   /// the data to store per infostate entry
   using infostate_data_type = InfostateNodeData< action_type >;

   ////////////////////
   /// Constructors ///
   ////////////////////

   /// inherit all constructors from base
   MCCFR(auto... args, double epsilon, size_t seed = std::random_device{}())
       : base(std::forward< decltype(args) >(args)...), m_epsilon(epsilon), m_rng(seed)
   {
      _sanity_check_config();
   }

   ////////////////////////////////////
   /// API: public member functions ///
   ////////////////////////////////////

   /// import public getters

   using base::env;
   using base::policy;
   using base::average_policy;
   using base::iteration;
   using base::root_state;

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
      requires(config.alternating_updates)
   ;

   //   ValueMap game_value() { return _iterate< false, false >(std::nullopt); }

   /**
    * @brief updates the regret and policy tables of the node with the state-values. Then performs
    * regret-matching.
    *
    * @param player the player whose values are to be updated
    */
   void update_regret_and_policy(
      const sptr< info_state_type >& infostate,
      const ReachProbabilityMap& reach_probability,
      const ValueMap& state_value,
      const std::unordered_map< action_variant_type, ValueMap >& action_value);

   ////////////////////////////////
   /// private member functions ///
   ////////////////////////////////

  private:
   /// import the parent's member variable accessors
   using base::_env;
   using base::_iteration;
   using base::_root_state_uptr;
   using base::_policy;
   using base::_average_policy;
   using base::_player_update_schedule;
   using base::_infonodes;
   using base::_infonode;
   using base::_cycle_player_to_update;
   using base::_child_state;

   /// the parameter to control the epsilon-on-policy epxloration
   double m_epsilon;
   /// the rng state to produce random numbers with
   common::random::RNG m_rng;
   /// the standard 0 to 1. floating point uniform distribution
   std::uniform_real_distribution< double > m_uniform_01_dist{0., 1.};

   /// define the implementation details of the

   /**
    * @brief The internal vanilla cfr iteration routine.
    *
    * This function sets the algorithm scheme for vanilla cfr by delegating to the right functions.
    *
    * @param player_to_update the player to update (optionally). If not given either the next player
    * to update from the schedule is taken (alternating updates) or every player is updated at the
    * same time (simultaneous updates).
    */
   template < bool use_current_policy = true >
   auto _iterate(std::optional< Player > player_to_update);

   /**
    * @brief traverses the game tree and fills the nodes with current policy weighted value updates.
    *
    * This function is the regular traversal function to call on iteration i > 0, after the nodes
    * have been emplaced by @see _first_traversal.
    */
   template < bool use_current_policy = true >
      requires(
         config.algorithm == MCCFRAlgorithm::outcome_sampling
         and config.weighting == MCCFRWeightingMode::optimistic)
   std::pair< StateValue, Probability > _traverse(
      Player player_to_update,
      uptr< world_state_type > curr_worldstate,
      ReachProbabilityMap reach_probability,
      ObservationbufferMap observation_buffer,
      InfostateMap infostates,
      Probability sample_probability);

   template < bool use_current_policy = true >
      requires(
         config.algorithm == MCCFRAlgorithm::outcome_sampling
         and config.weighting == MCCFRWeightingMode::lazy)
   std::pair< StateValue, Probability > _traverse(
      Player player_to_update,
      uptr< world_state_type > state,
      Probability own_reach_probability,
      Probability opp_reach_probability,
      ObservationbufferMap observation_buffer,
      InfostateMap infostates,
      Probability sample_probability,
      Weight own_weight,
      Weight opp_weight);

   template < bool use_current_policy = true >
      requires(config.algorithm == MCCFRAlgorithm::external_sampling)
   ValueMap _traverse(
      std::optional< Player > player_to_update,
      uptr< world_state_type > curr_worldstate,
      ReachProbabilityMap reach_probability,
      ObservationbufferMap observation_buffer,
      InfostateMap infostates,
      Probability sample_probability,
      ReachProbabilityMap sampling_weights);

   auto _fill_infostate_and_obs_buffers(
      const ObservationbufferMap& observation_buffer,
      const InfostateMap& infostate_map,
      const auto& action_or_outcome,
      const world_state_type& state);

   void _apply_regret_matching(const std::optional< Player >& player_to_update);

   constexpr bool _sanity_check_config();
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

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
auto MCCFR< config, Env, Policy, AveragePolicy >::iterate(size_t n_iters)
{
   std::vector< std::unordered_map< Player, double > > root_values_per_iteration;
   root_values_per_iteration.reserve(n_iters);
   for([[maybe_unused]] auto _ : ranges::views::iota(size_t(0), n_iters)) {
      LOGD2("Iteration number: ", _iteration());
      ValueMap value = [&] {
         if constexpr(config.alternating_updates) {
            auto player_to_update = _cycle_player_to_update();
            if(_iteration() < _env().players().size() - 1) {
               return _iterate< true >(player_to_update);
            } else {
               return _iterate< false >(player_to_update);
            }
         } else {
            if(_iteration() == 0) {
               return _iterate< true >(std::nullopt);
            } else {
               return _iterate< false >(std::nullopt);
            }
         }
      }();
      root_values_per_iteration.emplace_back(std::move(value.get()));
      _iteration()++;
   }
   return root_values_per_iteration;
}

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
auto MCCFR< config, Env, Policy, AveragePolicy >::iterate(std::optional< Player > player_to_update)
   requires(config.alternating_updates)
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
   } else if(auto env_players = _env().players();
             ranges::find(env_players, player_to_update) == env_players.end()) {
      std::stringstream ssout;
      ssout << "Given player to update ";
      ssout << player_to_update.value();
      ssout << " is not a member of the game's player list ";
      ssout << ranges::views::all(env_players) << ".";
      throw std::invalid_argument(ssout.str());
   }
   LOGD2("Iteration number: ", _iteration());
   // run the iteration
   ValueMap values = [&] {
      if(_iteration() < _env().players().size() - 1)
         return _iterate< true >(_cycle_player_to_update(player_to_update));
      else
         return _iterate< false >(_cycle_player_to_update(player_to_update));
   }();
   // and increment our iteration counter
   _iteration()++;
   return std::vector{std::move(values.get())};
}

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
template < bool use_current_policy >
auto MCCFR< config, Env, Policy, AveragePolicy >::_iterate(std::optional< Player > player_to_update)
{
   auto root_game_value = _traverse< use_current_policy >(
      player_to_update,
      utils::static_unique_ptr_downcast< world_state_type >(
         utils::clone_any_way(_root_state_uptr())),
      [&] {
         std::unordered_map< Player, double > rp_map;
         for(auto player : _env().players()) {
            rp_map.emplace(player, 1.);
         }
         return ReachProbabilityMap{std::move(rp_map)};
      }(),
      [&] {
         std::unordered_map< Player, std::vector< observation_type > > obs_map;
         auto players = _env().players();
         for(auto player : players | utils::is_nonchance_player_filter) {
            obs_map.emplace(player, std::vector< observation_type >{});
         }
         return ObservationbufferMap{std::move(obs_map)};
      }(),
      [&] {
         std::unordered_map< Player, sptr< info_state_type > > infostates;
         auto players = _env().players();
         for(auto player : players | utils::is_nonchance_player_filter) {
            auto& infostate = infostates
                                 .emplace(player, std::make_shared< info_state_type >(player))
                                 .first->second;
            infostate->append(_env().private_observation(player, root_state()));
         }
         return InfostateMap{std::move(infostates)};
      }());

   if constexpr(use_current_policy) {
      _apply_regret_matching(player_to_update);
   }
   return root_game_value;
}

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
void MCCFR< config, Env, Policy, AveragePolicy >::_apply_regret_matching(
   const std::optional< Player >& player_to_update)
{
   auto call_regret_matching =
      [&](const sptr< info_state_type >& infostate_ptr, const infostate_data_type& istate_data) {
         rm::regret_matching(
            fetch_policy< true >(infostate_ptr, istate_data.actions()),
            istate_data.regret(),
            // we provide the accessor to get the underlying referenced action, as the infodata
            // stores only reference wrappers to the actions
            [](const action_type& action) { return std::ref(action); });
      };

   if constexpr(config.alternating_updates) {
      Player update_player = player_to_update.value_or(Player::chance);
      for(auto& [infostate_ptr, data] : _infonodes()) {
         if(infostate_ptr->player() == update_player) {
            call_regret_matching(infostate_ptr, data);
         }
      }
   } else {
      for(auto& [infostate_ptr, data] : _infonodes()) {
         call_regret_matching(infostate_ptr, data);
      }
   }
}

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
template < bool use_current_policy >
   requires(
      config.algorithm == MCCFRAlgorithm::outcome_sampling
      and config.weighting == MCCFRWeightingMode::lazy)
std::pair<
   typename MCCFR< config, Env, Policy, AveragePolicy >::StateValue,
   typename MCCFR< config, Env, Policy, AveragePolicy >::
      Probability > MCCFR< config, Env, Policy, AveragePolicy >::
   _traverse(
      Player player_to_update,
      uptr< world_state_type > state,
      Probability own_reach_probability,
      Probability opp_reach_probability,
      ObservationbufferMap observation_buffer,
      InfostateMap infostates,
      Probability sample_probability,
      Weight own_weight,
      Weight opp_weight)
{
   Player active_player = _env().active_player(*state);

   if(_env().is_terminal(*state)) {
      return std::pair{_env().reward(active_player, *state) / sample_probability, 1.};
   }

   sptr< info_state_type > this_infostate = nullptr;
   // the state's value for each player. To be filled by the action traversal functions.
   StateValue state_value{{}};

   auto nonchance_player_traversal = [&] {
      this_infostate = infostates.get().at(active_player);
      auto [data_iter, success] = _infonodes().try_emplace(this_infostate, infostate_data_type{});
      if(success) {
         data_iter->emplace(_env().actions(active_player, *state));
      }
      auto next_own_weight = own_weight;
      auto next_opp_weight = opp_weight;

      auto choose_action_via_policy = [&] {
         auto& on_policy = fetch_policy(this_infostate, data_iter->actions());
         auto& chosen_action = common::random::choose(data_iter->actions(), on_policy, m_rng);
         return std::pair{chosen_action, on_policy[chosen_action]};
      };

      if(active_player == player_to_update) {
         auto [next_action, action_prob] = [&, this]() {
            if(m_uniform_01_dist(m_rng) < m_epsilon) {
               return std::pair{
                  common::random::choose(data_iter->actions(), m_rng),
                  1. / static_cast< double >(data_iter->actions().size())};
            } else {
               return choose_action_via_policy();
            }
         }();
         auto next_state = _child_state(state, next_action);

         auto [child_value, tail_prob] = _traverse(
            player,
            ,
            Probability{own_reach_probability.get() * action_prob},
            opp_reach_probability,
            ObservationbufferMap observation_buffer,
            InfostateMap infostates,
            Probability sample_probability,
            Weight own_weight,
            Weight opp_weight);
      } else {
         auto [next_action, action_prob] = choose_action_via_policy();
      }
   };
   // now we check first if we even need to consider a chance player, as the env could be simply
   // deterministic. In that case we might need to traverse the chance player's actions or an active
   // player's actions
   if constexpr(not concepts::deterministic_fosg< env_type >) {
      if(active_player == Player::chance) {
         auto chance_actions = _env().chance_actions(*state);
         auto& chosen_action = common::random::choose(chance_actions, m_rng);
         return _traverse(
            player_to_update,
            _child_state(state, chosen_action),
            std::move(own_reach_probability),
            std::move(opp_reach_probability),
            std::move(observation_buffer),
            std::move(infostates),
            std::move(sample_probability),
            std::move(own_weight),
            std::move(opp_weight));

         _traverse_chance_actions< use_current_policy >(
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
      if constexpr(config.alternating_updates) {
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

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
template < bool use_current_policy >
void MCCFR< config, Env, Policy, AveragePolicy >::_traverse_player_actions(
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
      _infonodes().emplace(
         this_infostate, infostate_data_type{_env().actions(active_player, *state)});
   }
   auto& action_policy = fetch_policy< use_current_policy >(
      this_infostate, _infonode(this_infostate).actions());
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

   for(const action_type& action : _infonode(this_infostate).regret() | ranges::views::keys) {
      // clone the current world state first before transitioniong it with this action
      uptr< world_state_type > next_wstate_uptr = _child_state(*state, action);

      auto child_reach_prob = reach_probability.get();
      auto action_prob = action_policy[action] / normalizing_factor;
      child_reach_prob[active_player] *= action_prob;

      auto [observation_buffer_copy, child_infostate_map] = _fill_infostate_and_obs_buffers(
         observation_buffer, infostate_map, action, *next_wstate_uptr);

      ValueMap child_rewards_map = _traverse< initialize_infonodes, use_current_policy >(
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

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
template < bool use_current_policy >
void MCCFR< config, Env, Policy, AveragePolicy >::_traverse_chance_actions(
   std::optional< Player > player_to_update,
   Player active_player,
   uptr< world_state_type > state,
   const ReachProbabilityMap& reach_probability,
   const ObservationbufferMap& observation_buffer,
   InfostateMap infostate_map,
   ValueMap& state_value,
   std::unordered_map< action_variant_type, ValueMap >& action_value)
{
   for(auto&& outcome : _env().chance_actions(*state)) {
      uptr< world_state_type >
         next_wstate_uptr = utils::static_unique_ptr_downcast< world_state_type >(
            utils::clone_any_way(state));
      // move the new world state forward by the current action
      _env().transition(*next_wstate_uptr, outcome);

      auto child_reach_prob = reach_probability.get();
      auto outcome_prob = _env().chance_probability(*state, outcome);
      child_reach_prob[active_player] *= outcome_prob;

      auto [observation_buffer_copy, child_infostate_map] = _fill_infostate_and_obs_buffers(
         observation_buffer, infostate_map, outcome, *next_wstate_uptr);

      ValueMap child_rewards_map = _traverse< initialize_infonodes, use_current_policy >(
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

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
auto MCCFR< config, Env, Policy, AveragePolicy >::_fill_infostate_and_obs_buffers(
   const ObservationbufferMap& observation_buffer,
   const InfostateMap& infostate_map,
   const auto& action_or_outcome,
   const world_state_type& state)
{
   auto active_player = _env().active_player(state);
   std::unordered_map< Player, sptr< info_state_type > > child_infostate_map;
   auto observation_buffer_copy = observation_buffer.get();
   for(auto player : _env().players()) {
      if(player == Player::chance) {
         continue;
      }
      if(player != active_player) {
         // for all but the active player we simply append action and state observation to the
         // buffer. They will be written to an actual infostate once that player becomes the
         // active player again
         child_infostate_map.emplace(player, infostate_map.get().at(player));
         auto& player_infostate = observation_buffer_copy.at(player);
         player_infostate.emplace_back(_env().private_observation(player, action_or_outcome));
         player_infostate.emplace_back(_env().private_observation(player, state));
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
         cloned_infostate_ptr->append(_env().private_observation(player, action_or_outcome));
         cloned_infostate_ptr->append(_env().private_observation(player, state));
         child_infostate_map.emplace(player, std::move(cloned_infostate_ptr));
      }
   }
   return std::tuple{observation_buffer_copy, child_infostate_map};
}

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
void MCCFR< config, Env, Policy, AveragePolicy >::update_regret_and_policy(
   const sptr< info_state_type >& infostate,
   const ReachProbabilityMap& reach_probability,
   const ValueMap& state_value,
   const std::unordered_map< action_variant_type, ValueMap >& action_value)
{
   auto& istatedata = _infonode(infostate);
   auto& curr_action_policy = fetch_policy< true >(infostate, istatedata.actions());
   auto& avg_action_policy = fetch_policy< false >(infostate, istatedata.actions());
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
      istatedata.regret(action) += cf_reach_prob * (q_value.get().at(player) - player_state_value);
      avg_action_policy[action] += player_reach_prob * curr_action_policy[action];
   }
}

}  // namespace nor::rm

#endif  // NOR_MCCFR_HPP
