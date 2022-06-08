
#ifndef NOR_VCFR_HPP
#define NOR_VCFR_HPP

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

#include "cfr_config.hpp"
#include "cfr_utils.hpp"
#include "common/common.hpp"
#include "forest.hpp"
#include "node.hpp"
#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "nor/policy.hpp"
#include "nor/type_defs.hpp"
#include "nor/utils/utils.hpp"
#include "tabular_cfr_base.hpp"

namespace nor::rm {

namespace detail {

template < CFRConfig config, typename Env >
struct VCFRNodeDataSelector {
  private:
   using action_type = typename fosg_auto_traits< Env >::action_type;
   /// for vanilla cfr we need no extra weight data stored
   using default_data_type = InfostateNodeData< action_type >;

  public:
   using type = default_data_type;
};
}  // namespace detail

/**
 * A (Vanilla) Counterfactual Regret Minimization algorithm class following the
 * terminology of the Factored-Observation Stochastic Games (FOSG) formulation.
 *
 * @tparam Env, the environment type to run VanillaCFR on.
 * @tparam Policy, the policy type to store a player's current policy in.
 * @tparam AveragePolicy, the policy type to store a player's average policy in.
 *
 */
template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
class VanillaCFR:
    public TabularCFRBase<
       config.update_mode == UpdateMode::alternating,
       Env,
       Policy,
       AveragePolicy > {
   ////////////////////////////
   /// API: public typedefs ///
   ////////////////////////////
  public:
   /// aliases for the template types
   using base =
      TabularCFRBase< config.update_mode == UpdateMode::alternating, Env, Policy, AveragePolicy >;
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
   using infostate_data_type = typename detail::VCFRNodeDataSelector< config, env_type >::type;
   /// strong-types for player based maps
   using InfostateMap = typename base::InfostateMap;
   using ObservationbufferMap = typename base::ObservationbufferMap;

   ////////////////////
   /// Constructors ///
   ////////////////////

   /// inherit all constructors from base
   VanillaCFR(const VanillaCFR&) = delete;
   VanillaCFR(VanillaCFR&&) = default;
   ~VanillaCFR() = default;
   VanillaCFR& operator=(const VanillaCFR&) = delete;
   VanillaCFR& operator=(VanillaCFR&&) = default;

   template < typename... Args >
   VanillaCFR(Args&&... args) : base(std::forward< Args >(args)...)
   {
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
      requires(config.update_mode == UpdateMode::alternating)
   ;

   StateValueMap game_value() { return _iterate< false, false >(std::nullopt); }

   /**
    * @brief updates the regret and policy tables of the infostate with the state-values.
    */
   void update_regret_and_policy(
      const sptr< info_state_type >& infostate,
      const ReachProbabilityMap& reach_probability,
      const StateValueMap& state_value,
      const std::unordered_map< action_variant_type, StateValueMap >& action_value);

   ////////////////////////////////
   /// private member functions ///
   ////////////////////////////////

   [[nodiscard]] inline auto& _infonodes() { return m_infonode; }
   [[nodiscard]] inline auto& infonode(const sptr< info_state_type >& infostate) const
   {
      return m_infonode.at(infostate);
   }
   [[nodiscard]] inline auto& _infonode(const sptr< info_state_type >& infostate)
   {
      return m_infonode.at(infostate);
   }

  private:
   /// import the parent's member variable accessors
   using base::_env;
   using base::_iteration;
   using base::_root_state_uptr;
   using base::_policy;
   using base::_average_policy;
   using base::_player_update_schedule;
   using base::_cycle_player_to_update;
   using base::_child_state;
   using base::_fill_infostate_and_obs_buffers;

   /// the relevant data stored at each infostate
   std::unordered_map<
      sptr< info_state_type >,
      infostate_data_type,
      common::sptr_value_hasher< info_state_type >,
      common::sptr_value_comparator< info_state_type > >
      m_infonode{};

   /// define the implementation details of the API

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
    * @brief traverses the game tree and fills the nodes with policy weighted regret updates.
    */
   template < bool initialize_infonodes, bool use_current_policy = true >
   StateValueMap _traverse(
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
      StateValueMap& state_value,
      std::unordered_map< action_variant_type, StateValueMap >& action_value);

   template < bool initialize_infonodes, bool use_current_policy = true >
   void _traverse_chance_actions(
      std::optional< Player > player_to_update,
      Player active_player,
      uptr< world_state_type > state,
      const ReachProbabilityMap& reach_probability,
      const ObservationbufferMap& observation_buffer,
      InfostateMap infostate_map,
      StateValueMap& state_value,
      std::unordered_map< action_variant_type, StateValueMap >& action_value);

   void _apply_regret_matching(const std::optional< Player >& player_to_update);
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

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
auto VanillaCFR< config, Env, Policy, AveragePolicy >::iterate(size_t n_iters)
{
   std::vector< std::unordered_map< Player, double > > root_values_per_iteration;
   root_values_per_iteration.reserve(n_iters);
   for([[maybe_unused]] auto _ : ranges::views::iota(size_t(0), n_iters)) {
      LOGD2("Iteration number: ", _iteration());
      StateValueMap value = [&] {
         if constexpr(config.update_mode == UpdateMode::alternating) {
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

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
auto VanillaCFR< config, Env, Policy, AveragePolicy >::iterate(
   std::optional< Player > player_to_update)
   requires(config.update_mode == UpdateMode::alternating)
{
   LOGD2("Iteration number: ", _iteration());
   // run the iteration
   StateValueMap values = [&] {
      if(_iteration() < _env().players().size() - 1)
         return _iterate< true >(_cycle_player_to_update(player_to_update));
      else
         return _iterate< false >(_cycle_player_to_update(player_to_update));
   }();
   // and increment our iteration counter
   _iteration()++;
   return std::vector{std::move(values.get())};
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
template < bool initializing_run, bool use_current_policy >
auto VanillaCFR< config, Env, Policy, AveragePolicy >::_iterate(
   std::optional< Player > player_to_update)
{
   auto root_game_value = _traverse< initializing_run, use_current_policy >(
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

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
void VanillaCFR< config, Env, Policy, AveragePolicy >::_apply_regret_matching(
   const std::optional< Player >& player_to_update)
{
   auto call_regret_matching =
      [&](
         const sptr< info_state_type >& infostate_ptr,
         const_if_t<
            config.regret_minimizing_mode == RegretMinimizingMode::regret_matching,
            infostate_data_type >& istate_data) {
         if constexpr(config.regret_minimizing_mode == RegretMinimizingMode::regret_matching) {
            rm::regret_matching(
               fetch_policy< PolicyLabel::current >(infostate_ptr, istate_data.actions()),
               istate_data.regret(),
               // we provide the accessor to get the underlying referenced action, as the infodata
               // stores only reference wrappers to the actions
               [](const action_type& action) { return std::ref(action); });
         } else if constexpr(
            config.regret_minimizing_mode == RegretMinimizingMode::regret_matching_plus) {
            rm::regret_matching_plus(
               fetch_policy< true >(infostate_ptr, istate_data.actions()),
               istate_data.regret(),
               // we provide the accessor to get the underlying referenced action, as the infodata
               // stores only reference wrappers to the actions
               [](const action_type& action) { return std::ref(action); });
         }
      };

   if constexpr(config.update_mode == UpdateMode::alternating) {
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

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
template < bool initialize_infonodes, bool use_current_policy >
StateValueMap VanillaCFR< config, Env, Policy, AveragePolicy >::_traverse(
   std::optional< Player > player_to_update,
   uptr< world_state_type > state,
   ReachProbabilityMap reach_probability,
   ObservationbufferMap observation_buffer,
   InfostateMap infostates)
{
   if(_env().is_terminal(*state)) {
      return StateValueMap{collect_rewards(_env(), *state)};
   }

   Player active_player = _env().active_player(*state);
   sptr< info_state_type > this_infostate = nullptr;
   // the state's value for each player. To be filled by the action traversal functions.
   StateValueMap state_value{{}};
   // each actions's value for each player. To be filled by the action traversal functions.
   std::unordered_map< action_variant_type, StateValueMap > action_value;
   // traverse all child states from this state. The constexpr check for determinism in the env
   // allows deterministic envs to not provide certain functions that are only needed in the
   // stochastic case.
   // First define the default branch for an active non-chance player
   auto nonchance_player_traverse = [&] {
      this_infostate = infostates.get().at(active_player);
      if constexpr(initialize_infonodes) {
         _infonodes().emplace(
            this_infostate, infostate_data_type{_env().actions(active_player, *state)});
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
         nonchance_player_traverse();
      }
   } else {
      nonchance_player_traverse();
   }

   if constexpr(use_current_policy) {
      // we can only update our regrets and policies if we are traversing with the current policy,
      // since the average policy is not to be changed directly (but through averaging up all
      // current policies)
      if constexpr(config.update_mode == UpdateMode::alternating) {
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
   return StateValueMap{state_value};
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
template < bool initialize_infonodes, bool use_current_policy >
void VanillaCFR< config, Env, Policy, AveragePolicy >::_traverse_player_actions(
   std::optional< Player > player_to_update,
   Player active_player,
   uptr< world_state_type > state,
   const ReachProbabilityMap& reach_probability,
   const ObservationbufferMap& observation_buffer,
   InfostateMap infostate_map,
   StateValueMap& state_value,
   std::unordered_map< action_variant_type, StateValueMap >& action_value)
{
   auto& this_infostate = infostate_map.get().at(active_player);
   if constexpr(initialize_infonodes) {
      _infonodes().emplace(
         this_infostate, infostate_data_type{_env().actions(active_player, *state)});
   }
   const auto& actions = _infonode(this_infostate).actions();
   auto& action_policy = fetch_policy< use_current_policy >(this_infostate, actions);
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

   for(const action_type& action : actions) {
      uptr< world_state_type > next_wstate_uptr = _child_state(state, action);

      auto child_reach_prob = reach_probability.get();
      auto action_prob = action_policy[action] / normalizing_factor;
      child_reach_prob[active_player] *= action_prob;

      auto [child_observation_buffer, child_infostate_map] = _fill_infostate_and_obs_buffers(
         observation_buffer, infostate_map, action, *next_wstate_uptr);

      StateValueMap child_rewards_map = _traverse< initialize_infonodes, use_current_policy >(
         player_to_update,
         std::move(next_wstate_uptr),
         ReachProbabilityMap{std::move(child_reach_prob)},
         ObservationbufferMap{std::move(child_observation_buffer)},
         InfostateMap{std::move(child_infostate_map)});
      // add the child state's value to the respective player's value table, multiplied by the
      // policies likelihood of playing this action
      for(auto [player, child_value] : child_rewards_map.get()) {
         state_value.get()[player] += action_prob * child_value;
      }
      action_value.emplace(action, std::move(child_rewards_map));
   }
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
template < bool initialize_infonodes, bool use_current_policy >
void VanillaCFR< config, Env, Policy, AveragePolicy >::_traverse_chance_actions(
   std::optional< Player > player_to_update,
   Player active_player,
   uptr< world_state_type > state,
   const ReachProbabilityMap& reach_probability,
   const ObservationbufferMap& observation_buffer,
   InfostateMap infostate_map,
   StateValueMap& state_value,
   std::unordered_map< action_variant_type, StateValueMap >& action_value)
{
   for(auto&& outcome : _env().chance_actions(*state)) {
      uptr< world_state_type > next_wstate_uptr = _child_state(state, outcome);

      auto child_reach_prob = reach_probability.get();
      auto outcome_prob = _env().chance_probability(*state, outcome);
      child_reach_prob[active_player] *= outcome_prob;

      auto [child_observation_buffer, child_infostate_map] = _fill_infostate_and_obs_buffers(
         observation_buffer, infostate_map, outcome, *next_wstate_uptr);

      StateValueMap child_rewards_map = _traverse< initialize_infonodes, use_current_policy >(
         player_to_update,
         std::move(next_wstate_uptr),
         ReachProbabilityMap{std::move(child_reach_prob)},
         ObservationbufferMap{std::move(child_observation_buffer)},
         InfostateMap{std::move(child_infostate_map)});
      // add the child state's value to the respective player's value table, multiplied by the
      // policies likelihood of playing this action
      for(auto [player, child_value] : child_rewards_map.get()) {
         state_value.get()[player] += outcome_prob * child_value;
      }
      action_value.emplace(std::move(outcome), std::move(child_rewards_map));
   }
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
void VanillaCFR< config, Env, Policy, AveragePolicy >::update_regret_and_policy(
   const sptr< info_state_type >& infostate,
   const ReachProbabilityMap& reach_probability,
   const StateValueMap& state_value,
   const std::unordered_map< action_variant_type, StateValueMap >& action_value)
{
   auto& istatedata = _infonode(infostate);
   const auto& actions = istatedata.actions();
   auto& curr_action_policy = fetch_policy< PolicyLabel::current >(infostate, actions);
   auto& avg_action_policy = fetch_policy< PolicyLabel::average >(infostate, actions);
   auto player = infostate->player();
   double cf_reach_prob = rm::cf_reach_probability(player, reach_probability.get());
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
      avg_action_policy[action] += [&] {
         if constexpr(config.weighting_mode == CFRWeightingMode::uniform) {
            // uniform weighting simply means that every update (old and new) is treated with the
            // same priority.
            return player_reach_prob * curr_action_policy[action];
         } else if constexpr(config.weighting_mode == CFRWeightingMode::linear) {
            // linear weighting simply multiplies the current iteration onto the increment. The
            // normalization factor from the papers is irrelevant, as it is absorbed by the
            // normalization constant of each action policy individually afterwards.
            return (player_reach_prob * curr_action_policy[action]) * double(_iteration());
         } else {
            static_assert(
               utils::always_false_v< action_type >,
               "Weighting Mode has to be either uniform or linear.");
         }
      }();
   }
}

}  // namespace nor::rm

#endif  // NOR_VCFR_HPP