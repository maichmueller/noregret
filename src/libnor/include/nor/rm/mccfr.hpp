
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

template < MCCFRConfig config, typename Env >
struct MCCFRNodeDataSelector {
  private:
   using action_type = typename fosg_auto_traits< Env >::action_type;
   /// for lazy weighting we need a weight for each individual action at each infostate
   using lazy_node_type = InfostateNodeData<
      action_type,
      std::unordered_map<
         std::reference_wrapper< const action_type >,
         double,
         common::default_ref_hasher< const action_type >,
         common::default_ref_comparator< const action_type > > >;
   /// for optimistic weghting we merely need a counter for each infostate
   using optimistic_node_type = InfostateNodeData< action_type, size_t >;
   /// for eg external sampling or stochastic weighetd outcome sampling we merely need the regret
   /// storage
   using basic_node_type = InfostateNodeData< action_type >;

  public:
   using type = std::conditional_t<
      config.algorithm == MCCFRAlgorithmMode::external_sampling,
      basic_node_type,
      std::conditional_t<
         config.algorithm == MCCFRAlgorithmMode::outcome_sampling,
         std::conditional_t<
            config.weighting == MCCFRWeightingMode::lazy,
            lazy_node_type,
            std::conditional_t<
               config.weighting == MCCFRWeightingMode::optimistic,
               optimistic_node_type,
               std::conditional_t<
                  config.weighting == MCCFRWeightingMode::stochastic,
                  basic_node_type,
                  void > > >,
         void > >;
};

}  // namespace detail

/**
 * The Monte-Carlo Counterfactual Regret Minimization algorithm class following the
 * terminology of the Factored-Observation Stochastic Games (FOSG) formulation.
 *
 * @tparam Env, the environment/game type to run VanillaCFR on.
 *
 */
template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
class MCCFR:
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
   using infostate_data_type = typename detail::MCCFRNodeDataSelector< config, env_type >::type;

   /// assert that the node data type did not turn out to be void
   static_assert(
      not std::is_same_v< infostate_data_type, void >,
      "Infostate node data is void type.");
   /// strong-types for player based maps
   using WeightMap = fluent::
      NamedType< std::unordered_map< Player, double >, struct weight_map_tag >;
   using InfostateMap = typename base::InfostateMap;
   using ObservationbufferMap = typename base::ObservationbufferMap;

   ////////////////////
   /// Constructors ///
   ////////////////////

   MCCFR(
      Env env_,
      uptr< world_state_type > root_state_,
      Policy policy_ = Policy(),
      AveragePolicy avg_policy_ = AveragePolicy(),
      double epsilon = 0.6,
      size_t seed = std::random_device{}())
       : base(std::move(env_), std::move(root_state_), std::move(policy_), std::move(avg_policy_)),
         m_epsilon(epsilon),
         m_rng(seed)
   {
      _sanity_check_config();
   }

   MCCFR(
      Env env_,
      Policy policy_ = Policy(),
      AveragePolicy avg_policy_ = AveragePolicy(),
      double epsilon = 0.6,
      size_t seed = std::random_device{}())
       : MCCFR(
          std::move(env_),
          std::make_unique< world_state_type >(env.initial_world_state()),
          std::move(policy_),
          std::move(avg_policy_),
          epsilon,
          seed)
   {
   }

   MCCFR(
      Env env_,
      uptr< world_state_type > root_state_,
      std::unordered_map< Player, Policy > policy_,
      std::unordered_map< Player, AveragePolicy > avg_policy_,
      double epsilon = 0.6,
      size_t seed = std::random_device{}())
       : base(std::move(env_), std::move(root_state_), std::move(policy_), std::move(avg_policy_)),
         m_epsilon(epsilon),
         m_rng(seed)
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
      requires(config.update_mode == UpdateMode::alternating)
   ;

   StateValue game_value(Player player) { return StateValue{0.}; }
   StateValueMap game_value()
   {
      StateValueMap values{{}};
      auto players = _env().players();
      for(auto player : players | utils::is_nonchance_player_filter) {
         values.get()[player] = 0.;
      }
      return values;
   }

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
   using base::_preview_next_player_to_update;
   using base::_child_state;
   using base::_fill_infostate_and_obs_buffers;
   using base::_fill_infostate_and_obs_buffers_inplace;

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
   auto _iterate(std::optional< Player > player_to_update);

   /**
    * @brief traverses the game tree and fills the nodes with current policy weighted value updates.
    *
    * This function is the regular traversal function to call on iteration i > 0, after the nodes
    * have been emplaced by @see _first_traversal.
    */

   std::pair< StateValue, Probability > _traverse(
      std::optional< Player > player_to_update,
      world_state_type& state,
      ReachProbabilityMap reach_probability,
      ObservationbufferMap observation_buffer,
      InfostateMap infostates,
      Probability sample_probability)
      requires(
         config.algorithm == MCCFRAlgorithmMode::outcome_sampling
         and (config.weighting == MCCFRWeightingMode::optimistic or config.weighting == MCCFRWeightingMode::stochastic))
   ;

   std::pair< StateValue, Probability > _traverse(
      std::optional< Player > player_to_update,
      world_state_type& state,
      ReachProbabilityMap reach_probability,
      ObservationbufferMap observation_buffer,
      InfostateMap infostates,
      Probability sample_probability,
      WeightMap weights)
      requires(
         config.algorithm == MCCFRAlgorithmMode::outcome_sampling
         and config.weighting == MCCFRWeightingMode::lazy)
   ;

   StateValue _traverse(
      Player player_to_update,
      uptr< world_state_type > state,
      ObservationbufferMap observation_buffer,
      InfostateMap infostates)
      requires(config.algorithm == MCCFRAlgorithmMode::external_sampling)
   ;

   void _update_regrets(
      const ReachProbabilityMap& reach_probability,
      Player active_player,
      infostate_data_type& infostate_data,
      const action_type& sampled_action,
      Probability sample_prob,
      StateValue action_value,
      Probability tail_prob) const
      requires(config.algorithm == MCCFRAlgorithmMode::outcome_sampling)
   ;

   void _update_average_policy(
      const sptr< info_state_type >& infostate,
      infostate_data_type& infonode_data,
      const auto& current_policy,
      Probability reach_prob,
      [[maybe_unused]] Probability sample_prob)
      requires(
         config.algorithm == MCCFRAlgorithmMode::outcome_sampling
         and (config.weighting == MCCFRWeightingMode::optimistic or config.weighting == MCCFRWeightingMode::stochastic))
   ;

   void _update_average_policy(
      const sptr< info_state_type >& infostate,
      infostate_data_type& infonode_data,
      const auto& current_policy,
      const action_type& sampled_action,
      Weight weight,
      Probability reach_prob)
      requires(
         config.algorithm == MCCFRAlgorithmMode::outcome_sampling
         and config.weighting == MCCFRWeightingMode::lazy)
   ;

   auto _sample_action(
      Player active_player,
      std::optional< Player > player_to_update,
      const infostate_data_type& infonode_data,
      auto& player_policy);

   template < bool return_likelihood = true >
   auto _sample_outcome(const world_state_type& state);

   constexpr void _sanity_check_config();
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
constexpr void MCCFR< config, Env, Policy, AveragePolicy >::_sanity_check_config()
{
   constexpr bool eval = [&] {
      if constexpr(config.algorithm == MCCFRAlgorithmMode::outcome_sampling) {
         if constexpr(config.update_mode != UpdateMode::alternating) {
            return false;
         }
      } else if constexpr(config.algorithm == MCCFRAlgorithmMode::external_sampling) {
         if constexpr(config.update_mode != UpdateMode::alternating) {
            return false;
         }
         if constexpr(config.weighting != MCCFRWeightingMode::stochastic) {
            return false;
         }
      }
      return true;
   }();
   static_assert(eval, "Config did not pass the check for correctness.");
};

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
auto MCCFR< config, Env, Policy, AveragePolicy >::iterate(size_t n_iters)
{
   std::vector< std::pair< Player, double > > root_values_per_iteration;
   root_values_per_iteration.reserve(n_iters);
   for([[maybe_unused]] auto _ : ranges::views::iota(size_t(0), n_iters)) {
      LOGD2("Iteration number", _iteration());
      auto player_to_update = _cycle_player_to_update();
      auto value_estim = [&] {
         if constexpr(config.algorithm == MCCFRAlgorithmMode::outcome_sampling) {
            return std::get< 0 >(_iterate(player_to_update)).get();
         } else {
            return _iterate(player_to_update).get();
         }
      }();
      root_values_per_iteration.emplace_back(std::pair{player_to_update, std::move(value_estim)});
      _iteration()++;
   }
   return root_values_per_iteration;
}

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
auto MCCFR< config, Env, Policy, AveragePolicy >::iterate(std::optional< Player > player_to_update)
   requires(config.update_mode == UpdateMode::alternating)
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
   auto updated_player = _cycle_player_to_update(player_to_update);
   auto value = std::vector{std::pair{updated_player, std::get< 0 >(_iterate()).get()}};
   // and increment our iteration counter
   _iteration()++;
   return value;
}

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
auto MCCFR< config, Env, Policy, AveragePolicy >::_iterate(std::optional< Player > player_to_update)
{
   auto players = _env().players();
   auto init_infostates = [&] {
      std::unordered_map< Player, sptr< info_state_type > > infostates;
      for(auto player : players | utils::is_nonchance_player_filter) {
         auto& infostate = infostates.emplace(player, std::make_shared< info_state_type >(player))
                              .first->second;
         infostate->append(_env().private_observation(player, root_state()));
      }
      return InfostateMap{std::move(infostates)};
   };
   auto init_reach_probs = [&] {
      std::unordered_map< Player, double > rp_map;
      for(auto player : players) {
         rp_map.emplace(player, 1.);
      }
      return ReachProbabilityMap{std::move(rp_map)};
   };
   auto init_obs_buffer = [&] {
      std::unordered_map< Player, std::vector< observation_type > > obs_map;
      for(auto player : players | utils::is_nonchance_player_filter) {
         obs_map.emplace(player, std::vector< observation_type >{});
      }
      return ObservationbufferMap{std::move(obs_map)};
   };
   if constexpr(config.algorithm == MCCFRAlgorithmMode::outcome_sampling) {
      // in outcome-sampling we only have a single trajectory to traverse in the tree. Hence we
      // can maintain the lifetime of that world state in this upstream function call and merely
      // pass in the state as reference
      auto init_world_state = utils::static_unique_ptr_downcast< world_state_type >(
         utils::clone_any_way(_root_state_uptr()));
      if constexpr(config.weighting == MCCFRWeightingMode::lazy) {
         return _traverse(
            player_to_update,
            *init_world_state,
            init_reach_probs(),
            init_obs_buffer(),
            init_infostates(),
            Probability{1.},
            [&] {
               std::unordered_map< Player, double > weights;
               for(auto player : players | utils::is_nonchance_player_filter) {
                  weights.emplace(player, 0.);
               }
               return WeightMap{std::move(weights)};
            }());
      }
      if constexpr(
         config.weighting == MCCFRWeightingMode::optimistic
         or config.weighting == MCCFRWeightingMode::stochastic) {
         return _traverse(
            player_to_update,
            *init_world_state,
            init_reach_probs(),
            init_obs_buffer(),
            init_infostates(),
            Probability{1.});
      }
   }
   if constexpr(config.algorithm == MCCFRAlgorithmMode::external_sampling) {
      if constexpr(config.weighting == MCCFRWeightingMode::stochastic) {
         return _traverse(
            player_to_update.value(),
            utils::static_unique_ptr_downcast< world_state_type >(
               utils::clone_any_way(_root_state_uptr())),
            init_obs_buffer(),
            init_infostates());
      }
   }
}

////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// Outcome-Sampling MCCFR /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >

std::pair< StateValue, Probability > MCCFR< config, Env, Policy, AveragePolicy >::_traverse(
   std::optional< Player > player_to_update,
   world_state_type& state,
   ReachProbabilityMap reach_probability,
   ObservationbufferMap observation_buffer,
   InfostateMap infostates,
   Probability sample_probability,
   WeightMap weights)
   requires(
      config.algorithm == MCCFRAlgorithmMode::outcome_sampling
      and config.weighting == MCCFRWeightingMode::lazy)
{
   if(_env().is_terminal(state)) {
      if constexpr(config.update_mode == UpdateMode::alternating) {
         return std::pair{
            StateValue{_env().reward(player_to_update.value(), state) / sample_probability.get()},
            Probability{1.}};
      }
      //      if constexpr(config.update_mode == UpdateMode::simultaneous) {
      //         auto rewards_map = collect_rewards(_env(), *state);
      //         for(auto& [player, reward] : rewards_map) {
      //            reward /= sample_probability.get();
      //         }
      //         return std::pair{StateValueMap{std::move(rewards_map)}, Probability{1.}};
      //      } else {
      //         static_assert(utils::always_false_v< Env >, "Update Mode not one of alternating
      //         or simultaneous");
      //      }
   }

   Player active_player = _env().active_player(state);

   // now we check first if we even need to consider a chance player, as the env could be
   // simply deterministic. In that case we might need to traverse the chance player's actions
   // or an active player's actions
   if constexpr(not concepts::deterministic_fosg< env_type >) {
      if(active_player == Player::chance) {
         auto [chosen_outcome, chance_prob] = _sample_outcome(state);

         reach_probability.get()[Player::chance] *= chance_prob;

         _env().transition(state, chosen_outcome);

         _fill_infostate_and_obs_buffers_inplace(
            observation_buffer, infostates, chosen_outcome, state);

         return _traverse(
            player_to_update,
            state,
            std::move(reach_probability),
            std::move(observation_buffer),
            std::move(infostates),
            Probability{sample_probability.get() * chance_prob},
            std::move(weights));
      }
   }

   auto infostate = infostates.get().at(active_player);
   auto [infostate_and_data_iter, success] = _infonodes().try_emplace(
      infostate, infostate_data_type{});
   auto& infonode_data = infostate_and_data_iter->second;
   if(success) {
      // success means we have indeed emplaced a new data node, instead of simply fetching an
      // existing one. We thus need to fill it with the legal actions at this node.
      infonode_data.emplace(_env().actions(active_player, state));
   }

   auto& player_policy = fetch_policy< PolicyLabel::current >(infostate, infonode_data.actions());

   // apply one round of regret matching on the current policy before using it. MCCFR only
   // updates the policy once you revisit it, as it is a lazy update schedule. As such, one would
   // need to update all infostates after the last iteration to ensure that the policy is fully
   // up-to-date

   rm::regret_matching(
      player_policy,
      infonode_data.regret(),
      // we provide the accessor to get the underlying referenced action, as the infodata
      // stores only reference wrappers to the actions
      [](const action_type& action) { return std::cref(action); });

   auto [sampled_action, action_sampling_prob, action_policy_prob] = _sample_action(
      active_player, player_to_update, infonode_data, player_policy);

   auto next_reach_prob = reach_probability.get();
   auto next_weights = weights.get();
   next_reach_prob[active_player] *= action_policy_prob;
   next_weights[active_player] = next_weights[active_player] * action_policy_prob
                                 + infonode_data.weight()[std::cref(sampled_action)];

   _env().transition(state, sampled_action);

   _fill_infostate_and_obs_buffers_inplace(observation_buffer, infostates, sampled_action, state);

   auto [action_value_or_map, tail_prob] = _traverse(
      player_to_update,
      state,
      ReachProbabilityMap{std::move(next_reach_prob)},
      std::move(observation_buffer),
      std::move(infostates),
      Probability{sample_probability.get() * action_sampling_prob},
      WeightMap{std::move(next_weights)});

   if constexpr(config.update_mode == UpdateMode::simultaneous) {
      _update_regrets(
         reach_probability,
         active_player,
         infonode_data,
         sampled_action,
         Probability{action_policy_prob},
         action_value_or_map,
         tail_prob);

      _update_average_policy(
         infostate,
         infonode_data,
         player_policy,
         sampled_action,
         Weight{weights.get()[active_player]},
         Probability{reach_probability.get()[active_player]});

   } else {
      static_assert(
         config.update_mode == UpdateMode::alternating,
         "The update mode has to be either alternating or simultaneous.");
      // in alternating updates we update the regret only for the player_to_update and the
      // strategy only if the current player is the next one in line to traverse the tree and
      // update
      if(active_player == player_to_update.value()) {
         _update_regrets(
            reach_probability,
            active_player,
            infonode_data,
            sampled_action,
            Probability{action_policy_prob},
            action_value_or_map,
            tail_prob);
      } else if(active_player == _preview_next_player_to_update()) {
         // the check in this if statement collapses to a simple true in the 2-player case
         _update_average_policy(
            infostate,
            infonode_data,
            player_policy,
            sampled_action,
            Weight{weights.get()[active_player]},
            Probability{reach_probability.get()[active_player]});
      }
   }

   return std::pair{
      std::move(action_value_or_map), Probability{tail_prob.get() * action_policy_prob}};
}

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
std::pair< StateValue, Probability > MCCFR< config, Env, Policy, AveragePolicy >::_traverse(
   std::optional< Player > player_to_update,
   world_state_type& state,
   ReachProbabilityMap reach_probability,
   ObservationbufferMap observation_buffer,
   InfostateMap infostates,
   Probability sample_probability)
   requires(
      config.algorithm == MCCFRAlgorithmMode::outcome_sampling
      and (config.weighting == MCCFRWeightingMode::optimistic or config.weighting == MCCFRWeightingMode::stochastic))
{
   if(_env().is_terminal(state)) {
      if constexpr(config.update_mode == UpdateMode::alternating) {
         return std::pair{
            StateValue{_env().reward(player_to_update.value(), state) / sample_probability.get()},
            Probability{1.}};
      }
      //      if constexpr(config.update_mode == UpdateMode::simultaneous) {
      //         auto rewards_map = collect_rewards(_env(), *state);
      //         for(auto& [player, reward] : rewards_map) {
      //            reward /= sample_probability.get();
      //         }
      //         return std::pair{StateValueMap{std::move(rewards_map)}, Probability{1.}};
      //      } else {
      //         static_assert(utils::always_false_v< Env >, "Update Mode not one of alternating
      //         or simultaneous");
      //      }
   }

   Player active_player = _env().active_player(state);

   // now we check first if we even need to consider a chance player, as the env could be
   // simply deterministic. In that case we might need to traverse the chance player's actions
   // or an active player's actions
   if constexpr(not concepts::deterministic_fosg< env_type >) {
      if(active_player == Player::chance) {
         auto [chosen_outcome, chance_prob] = _sample_outcome(state);

         reach_probability.get()[Player::chance] *= chance_prob;

         _env().transition(state, chosen_outcome);

         _fill_infostate_and_obs_buffers_inplace(
            observation_buffer, infostates, chosen_outcome, state);

         return _traverse(
            player_to_update,
            state,
            std::move(reach_probability),
            std::move(observation_buffer),
            std::move(infostates),
            Probability{sample_probability.get() * chance_prob});
      }
   }

   auto infostate = infostates.get().at(active_player);
   auto [infostate_and_data_iter, success] = _infonodes().try_emplace(
      infostate, infostate_data_type{});
   auto& infonode_data = infostate_and_data_iter->second;
   if(success) {
      // success means we have indeed emplaced a new data node, instead of simply fetching an
      // existing one. We thus need to fill it with the legal actions at this node.
      infonode_data.emplace(_env().actions(active_player, state));
   }

   auto& player_policy = fetch_policy< PolicyLabel::current >(infostate, infonode_data.actions());

   // apply one round of regret matching on the current policy before using it. MCCFR only
   // updates the policy once you revisit it, as it is a lazy update schedule. As such, one would
   // need to update all infostates after the last iteration to ensure that the policy is fully
   // up-to-date
   rm::regret_matching(
      player_policy,
      infonode_data.regret(),
      // we provide the accessor to get the underlying referenced action, as the infodata
      // stores only reference wrappers to the actions
      [](const action_type& action) { return std::cref(action); });

   LOGD2("Active Player", active_player);
   LOGD2("Player to update", player_to_update.value());
   LOGD2("Infostate", infostate->history().back());
   auto [sampled_action, action_sampling_prob, action_policy_prob] = _sample_action(
      active_player, player_to_update, infonode_data, player_policy);

   LOGD2("Action", sampled_action);
   LOGD2("Action sampling prob", action_sampling_prob);
   LOGD2("Action policy prob", action_policy_prob);
   auto next_reach_prob = reach_probability.get();
   next_reach_prob[active_player] *= action_policy_prob;
   LOGD2(
      "New Reach Probs",
      std::to_string(next_reach_prob[active_player]) + " "
         + std::to_string(next_reach_prob[Player(1 - static_cast< int >(active_player))]) + " "
         + std::to_string(next_reach_prob[Player::chance]) + " "
         + std::to_string(sample_probability.get() * action_sampling_prob));
   _env().transition(state, sampled_action);

   _fill_infostate_and_obs_buffers_inplace(observation_buffer, infostates, sampled_action, state);

   auto [action_value, tail_prob] = _traverse(
      player_to_update,
      state,
      ReachProbabilityMap{std::move(next_reach_prob)},
      std::move(observation_buffer),
      std::move(infostates),
      Probability{sample_probability.get() * action_sampling_prob});

   if constexpr(config.update_mode == UpdateMode::simultaneous) {
      _update_regrets(
         reach_probability,
         active_player,
         infonode_data,
         sampled_action,
         Probability{action_policy_prob},
         action_value,
         tail_prob);

      _update_average_policy(
         infostate,
         infonode_data,
         player_policy,
         Probability{reach_probability.get()[active_player]},
         sample_probability);

   } else {
      static_assert(
         config.update_mode == UpdateMode::alternating,
         "The update mode has to be either alternating or simultaneous.");
      // in alternating updates we update the regret only for the player_to_update and the
      // strategy only if the current player is the next one in line to traverse the tree and
      // update
      if(active_player == player_to_update.value()) {
         _update_regrets(
            reach_probability,
            active_player,
            infonode_data,
            sampled_action,
            Probability{action_policy_prob},
            action_value,
            tail_prob);
      } else if(active_player == _preview_next_player_to_update()) {
         // the check in this if statement collapses to a simple true in the 2-player case
         _update_average_policy(
            infostate,
            infonode_data,
            player_policy,
            Probability{reach_probability.get()[active_player]},
            sample_probability);
      }
   }

   return std::pair{action_value, Probability{tail_prob.get() * action_policy_prob}};
}

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
void MCCFR< config, Env, Policy, AveragePolicy >::_update_regrets(
   const ReachProbabilityMap& reach_probability,  // = pi(z[I])
   Player active_player,
   infostate_data_type& infostate_data,  // = -->r(I) and A(I)
   const action_type& sampled_action,  // = a', the sampled action
   Probability sampled_action_policy_prob,  // = sigma(I, a) for the sampled action
   StateValue action_value,  // = u(z[I]a)
   Probability tail_prob  // = pi^sigma(z[I]a, z)
) const
   requires(config.algorithm == MCCFRAlgorithmMode::outcome_sampling)
{
   LOGD2("Action Value", action_value.get());
   LOGD2("CF Probability", cf_reach_probability(active_player, reach_probability.get()));
   auto cf_value_weight = action_value.get()
                          * cf_reach_probability(active_player, reach_probability.get());
   LOGD2("CF Value Weight", cf_value_weight);
   for(const auto& action : infostate_data.actions()) {
      // compute the estimated counterfactual regret and add it to the cumulative regret table
      infostate_data.regret(action) += [&] {
         if(action == sampled_action) {
            // note that tail_prob = pi^sigma(z[I]a, z)
            // the probability pi^sigma(z[I]a, z) - pi^sigma(z[I], z) can also be expressed as
            // pi^sigma(z[I]a, z) * (1 - sigma(I, a)), since
            //    pi(h, z) = pi(z) / pi(h)   and    pi(ha) = pi(h) * sigma(I[h], a)
            // --> pi(ha, z) - pi(h, z) = pi(z) / (pi(h) * sigma(I, a)) - pi(z) / pi(h)
            //                          = pi(z) / (pi(h) * sigma(I, a)) * ( 1 - sigma(I, a))
            //                          = pi(ha, z) * ( 1 - sigma(I, a))
            LOGD2(
               "Regret update sampled action",
               cf_value_weight * tail_prob.get() * (1. - sampled_action_policy_prob.get()));
            return cf_value_weight * tail_prob.get() * (1. - sampled_action_policy_prob.get());
         } else {
            // we are returning here the formula: -W * pi(z[I], z)

            LOGD2(
               "Regret update other action",
               -cf_value_weight * tail_prob.get() * sampled_action_policy_prob.get());
            return -cf_value_weight * tail_prob.get() * sampled_action_policy_prob.get();
         }
      }();
   }
}

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
void MCCFR< config, Env, Policy, AveragePolicy >::_update_average_policy(
   const sptr< info_state_type >& infostate,
   infostate_data_type& infonode_data,
   const auto& current_policy,
   const action_type& sampled_action,
   Weight weight,
   Probability reach_prob)
   requires(
      config.algorithm == MCCFRAlgorithmMode::outcome_sampling
      and config.weighting == MCCFRWeightingMode::lazy)
{
   auto& avg_policy = fetch_policy< false >(infostate, infonode_data.actions());
   for(const action_type& action : infonode_data.actions()) {
      auto policy_incr = (weight.get() + reach_prob.get()) * current_policy[action];
      avg_policy[action] += policy_incr;
      if(action == sampled_action) [[unlikely]] {
         infonode_data.weight()[std::cref(action)] = 0.;
      } else [[likely]] {
         infonode_data.weight()[std::cref(action)] += policy_incr;
      }
   }
}

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
void MCCFR< config, Env, Policy, AveragePolicy >::_update_average_policy(
   const sptr< info_state_type >& infostate,
   infostate_data_type& infonode_data,
   const auto& current_policy,
   Probability reach_prob,
   Probability sample_prob)
   requires(
      config.algorithm == MCCFRAlgorithmMode::outcome_sampling
      and (config.weighting == MCCFRWeightingMode::optimistic or config.weighting == MCCFRWeightingMode::stochastic))
{
   auto& avg_policy = fetch_policy< false >(infostate, infonode_data.actions());

   if constexpr(config.weighting == MCCFRWeightingMode::optimistic) {
      auto& infostate_last_visit = infonode_data.weight();
      auto current_iter = _iteration();
      // we add + 1 to the current iter counter, since the iterations start counting at 0
      auto last_visit_difference = static_cast< double >(1 + current_iter - infostate_last_visit);
      for(const action_type& action : infonode_data.actions()) {
         LOGD2("Policy update", reach_prob.get() * current_policy[action] * last_visit_difference);
         avg_policy[action] += reach_prob.get() * current_policy[action] * last_visit_difference;
      }
      // mark this infostate as visited during this iteration. This will offset the delay
      // weight for future updates to reference the current one instead.
      infostate_last_visit = current_iter;
   }

   if constexpr(config.weighting == MCCFRWeightingMode::stochastic) {
      // the correct avg strategy increment is
      // avg_strategy(I, a) += pi^sigma_{currentPlayer}(h) * sigma(I, a)
      // In stochastic weighting the update is boosted by the sample probability, i.e. by
      // multiplying 1 / pi^{sigma'}(h) with the increment
      for(const action_type& action : infonode_data.actions()) {
         avg_policy[action] += reach_prob.get() * current_policy[action] / sample_prob.get();
      }
   }
}

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
auto MCCFR< config, Env, Policy, AveragePolicy >::_sample_action(
   Player active_player,
   std::optional< Player > player_to_update,
   const infostate_data_type& infonode_data,
   auto& player_policy)
{
   // we first define the sampling schemes:
   // 1. Sampling directly from policy calls the policy map as many times as there are options to
   // choose from and returns the sampled action, its policy probability, and its policy
   // probability again (for API consistency)
   // 2. Epsilon-On-Policy sampling with respect to the policy map executes two steps: first, it
   // decides whether we sample uniformly from the actions or not. If so, it executes a separate
   // branch for uniform sampling. Alternatively it reverts back to sampling procedure 1) and
   // adapts the sampling likelihood for the chosen sample.

   auto policy_sampling = [&] {
      // in the non-epsilon case we simply use the player's policy to sample the next move
      // from. Thus in this case, the action's sample probability and action's policy
      // probability are the same, i.e. action_sample_prob = action_policy_prob

      LOGD2("Sample Policy", ranges::views::values(player_policy));
      LOGD2("Sample Policy Keys", ranges::views::keys(player_policy));
      LOGD2("Normal Policy", ranges::views::values(player_policy));
      LOGD2("Normal Policy Keys", ranges::views::keys(player_policy));
      auto& chosen_action = common::random::choose(
         infonode_data.actions(), [&](const auto& act) { return player_policy[act]; }, m_rng);
      auto action_prob = player_policy[chosen_action];
      return std::tuple{chosen_action, action_prob, action_prob};
   };
   auto epsilon_on_policy_sampling = [&] {
      double uniform_prob = 1. / static_cast< double >(infonode_data.actions().size());
      if(m_uniform_01_dist(m_rng) < m_epsilon) {
         // with probability epsilon we do exploration, i.e. uniform sampling, over all actions
         // available. This is a tiny speedup over querying the actual policy map for the
         // epsilon-on-policy enhanced likelihoods
         auto& chosen_action = common::random::choose(infonode_data.actions(), m_rng);
         LOGD2(
            "Sample Policy",
            (ranges::views::values(player_policy) | ranges::views::transform([&](auto val) {
                return m_epsilon * uniform_prob + (1 - m_epsilon) * val;
             })));
         LOGD2("Sample Policy Keys", ranges::views::keys(player_policy));
         LOGD2("Normal Policy", ranges::views::values(player_policy));
         LOGD2("Normal Policy Keys", ranges::views::keys(player_policy));
         return std::tuple{
            chosen_action,
            m_epsilon * uniform_prob + (1 - m_epsilon) * player_policy[chosen_action],
            player_policy[chosen_action]};
      } else {
         // if we don't explore, then we simply sample according to the policy.
         // BUT: Since in theory we have done epsilon-on-policy exploration, yet merely in two
         // separate steps, we need to adapt the returned sampling probability to the
         // epsilon-on-policy probability of the sampled action
         auto [chosen_action, _, action_prob] = policy_sampling();
         return std::tuple{
            std::move(chosen_action),
            m_epsilon * uniform_prob + (1 - m_epsilon) * action_prob,
            action_prob};
      }
   };

   //   auto epsilon_on_policy_sampling = [&] {
   //      double uniform_prob = 1. / static_cast< double >(infonode_data.actions().size());
   //      auto pol_functor = [&](const auto& act) {
   //         return m_epsilon * uniform_prob + (1 - m_epsilon) * player_policy[act];
   //      };
   //      // with probability epsilon we do exploration, i.e. uniform sampling, over all actions
   //      // available. This is a tiny speedup over querying the actual policy map for the
   //      // epsilon-on-policy enhanced likelihoods
   //
   //      auto& chosen_action = common::random::choose(infonode_data.actions(), pol_functor,
   //      m_rng); LOGD2(
   //         "Sample Policy",
   //         (ranges::views::values(player_policy) | ranges::views::transform([&](auto val) {
   //             return m_epsilon * uniform_prob + (1 - m_epsilon) * val;
   //          })));
   //      LOGD2("Sample Policy Keys", ranges::views::keys(player_policy));
   //      LOGD2("Normal Policy", ranges::views::values(player_policy));
   //      LOGD2("Normal Policy Keys", ranges::views::keys(player_policy));
   //      return std::tuple{chosen_action, pol_functor(chosen_action),
   //      player_policy[chosen_action]};
   //   };

   // here we now decide what sampling procedure is exactly executed. It depends on the MCCFR
   // config given and then on the specific algorithm's sampling scheme
   if constexpr(config.algorithm == MCCFRAlgorithmMode::outcome_sampling) {
      if((config.update_mode == UpdateMode::simultaneous
          or active_player == player_to_update.value_or(Player::chance))) {
         return epsilon_on_policy_sampling();
      } else {
         return policy_sampling();
      }
   } else if constexpr(config.algorithm == MCCFRAlgorithmMode::external_sampling) {
      // for external sampling we always sample according to the policy
      return policy_sampling();
   }
}

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
template < bool return_likelihood >
auto MCCFR< config, Env, Policy, AveragePolicy >::_sample_outcome(const world_state_type& state)
{
   auto chance_actions = _env().chance_actions(state);
   auto chance_probabilities = ranges::to< std::unordered_map< chance_outcome_type, double > >(
      chance_actions | ranges::views::transform([this, &state](const auto& outcome) {
         return std::pair{outcome, _env().chance_probability(state, outcome)};
      }));
   auto& chosen_outcome = common::random::choose(
      chance_actions, [&](const auto& outcome) { return chance_probabilities[outcome]; }, m_rng);

   if constexpr(return_likelihood) {
      double chance_prob = chance_probabilities[chosen_outcome];
      return std::tuple{std::move(chosen_outcome), chance_prob};
   } else {
      return chosen_outcome;
   }
}

////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// External-Sampling MCCFR ////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >

StateValue MCCFR< config, Env, Policy, AveragePolicy >::_traverse(
   Player player_to_update,
   uptr< world_state_type > state,
   ObservationbufferMap observation_buffer,
   InfostateMap infostates)
   requires(config.algorithm == MCCFRAlgorithmMode::external_sampling)
{
   Player active_player = _env().active_player(*state);

   if(_env().is_terminal(*state)) {
      return StateValue{_env().reward(player_to_update, *state)};
   }

   // now we check first if we even need to consider a chance player, as the env could be
   // simply deterministic. In that case we might need to traverse the chance player's actions
   // or an active player's actions
   if constexpr(not concepts::deterministic_fosg< env_type >) {
      if(active_player == Player::chance) {
         auto chosen_outcome = _sample_outcome< false >(*state);

         _env().transition(*state, chosen_outcome);

         _fill_infostate_and_obs_buffers_inplace(
            observation_buffer, infostates, chosen_outcome, *state);

         return _traverse(
            player_to_update,
            std::move(state),
            std::move(observation_buffer),
            std::move(infostates));
      }
   }

   auto infostate = infostates.get().at(active_player);
   auto [infostate_and_data_iter, success] = _infonodes().try_emplace(
      infostate, infostate_data_type{});
   auto& infonode_data = infostate_and_data_iter->second;
   if(success) {
      // success means we have indeed emplaced a new data node, instead of simply fetching an
      // existing one. We thus need to fill it with the legal actions at this node.
      infonode_data.emplace(_env().actions(active_player, *state));
   }

   auto& player_policy = fetch_policy< PolicyLabel::current >(infostate, infonode_data.actions());

   rm::regret_matching(
      player_policy,
      infonode_data.regret(),
      // we provide the accessor to get the underlying referenced action, as the infodata
      // stores only reference wrappers to the actions
      [](const action_type& action) { return std::cref(action); });

   if(active_player == player_to_update) {
      // for the traversing player we explore all actions possible

      // the first round of action iteration we will traverse the tree further to find all action
      // values from this node and compute the state value of the current node
      double state_value_estimate = 0.;
      std::unordered_map< action_type, double > value_estimates;
      value_estimates.reserve(infonode_data.actions().size());

      for(const auto& action : infonode_data.actions()) {
         auto next_state = utils::static_unique_ptr_downcast< world_state_type >(
            utils::clone_any_way(state));
         _env().transition(*next_state, action);

         auto [next_observation_buffer, next_infostates] = _fill_infostate_and_obs_buffers(
            observation_buffer, infostates, action, *next_state);

         double action_value_estimate = _traverse(
                                           player_to_update,
                                           std::move(next_state),
                                           std::move(next_observation_buffer),
                                           std::move(next_infostates))
                                           .get();
         value_estimates.emplace(action, action_value_estimate);
         state_value_estimate += action_value_estimate * player_policy[action];
      }
      // in the second round of action iteration we update the regret of each action through the
      // previously found action and state values
      for(const auto& action : infonode_data.actions()) {
         infonode_data.regret(action) += value_estimates[action] - state_value_estimate;
      }

      return StateValue{state_value_estimate};

   } else {
      // for the non-traversing player we sample a single action and continue
      auto& sampled_action = common::random::choose(
         infonode_data.actions(), [&](const auto& act) { return player_policy[act]; }, m_rng);
      //      auto& sampled_action = infonode_data.actions()[0];

      _env().transition(*state, sampled_action);

      _fill_infostate_and_obs_buffers_inplace(
         observation_buffer, infostates, sampled_action, *state);

      double action_value_estimate = _traverse(
                                        player_to_update,
                                        std::move(state),
                                        std::move(observation_buffer),
                                        std::move(infostates))
                                        .get();

      if(active_player == _preview_next_player_to_update()) {
         // this update scheme represents the 'simple' update plan mentioned in open_spiel. We
         // are updating the policy if the active player is the next player to be updated in the
         // update cycle. Updates the average policy with the current policy
         auto& average_player_policy = fetch_policy< PolicyLabel::average >(
            infostate, infonode_data.actions());
         for(const auto& action : infonode_data.actions()) {
            average_player_policy[action] += player_policy[action];
         }
      }
      return StateValue{action_value_estimate};
   }
}

}  // namespace nor::rm

#endif  // NOR_MCCFR_HPP
