
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

#include "cfr_base.hpp"
#include "cfr_config.hpp"
#include "common/common.hpp"
#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "nor/rm/forest.hpp"
#include "nor/rm/minimizers/minimizers.hpp"
#include "nor/rm/node.hpp"
#include "nor/rm/rm_utils.hpp"
#include "nor/type_defs.hpp"
#include "nor/utils/utils.hpp"

namespace nor::rm {

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

   /// import all fosg aliases to be used in this class from the env type.
   using typename base::env_type;
   using typename base::policy_type;
   using typename base::action_type;
   using typename base::world_state_type;
   using typename base::info_state_type;
   using typename base::public_state_type;
   using typename base::observation_type;
   using typename base::chance_outcome_type;
   using typename base::chance_distribution_type;
   using typename base::InfostateSptrMap;
   using typename base::ObservationbufferMap;
   using action_variant_type = auto_action_variant_type< env_type >;
   /// the regret minimizer selected by the configuration
   using minimizer_type = mccfr_minimizer_for_t< config, action_type >;
   /// the data to store per infostate entry
   using infostate_data_type = InfostateNodeData<
      action_type,
      typename minimizer_type::node_data_type >;

   /// assert that the node data type did not turn out to be void
   static_assert(
      not std::is_same_v< infostate_data_type, void >,
      "Infostate node data is void type."
   );
   /// strong-types for player based maps
   using WeightMap = fluent::
      NamedType< std::unordered_map< Player, double >, struct weight_map_tag >;

   using ConditionalWeightMap = std::
      conditional_t< config.weighting == MCCFRWeightingMode::lazy, WeightMap, utils::empty >;
   using ConditionalWeight = std::
      conditional_t< config.weighting == MCCFRWeightingMode::lazy, Weight, utils::empty >;

   /// a hash set storing which infostates and their associated data need to be
   /// updated in terms of regret minimization POST cfr iteration. Identity is
   /// defined by the infostate pointer alone.
   using istate_and_data_pair = std::pair< const info_state_type*, infostate_data_type* >;
   struct istate_and_data_hash {
      size_t operator()(const istate_and_data_pair& pair) const
      {
         return common::value_hasher< info_state_type >{}(*pair.first);
      }
   };
   struct istate_and_data_equal {
      bool operator()(const istate_and_data_pair& left, const istate_and_data_pair& right) const
      {
         return common::value_comparator< info_state_type >{}(*left.first, *right.first);
      }
   };
   using delayed_update_set = std::
      unordered_set< istate_and_data_pair, istate_and_data_hash, istate_and_data_equal >;

   ////////////////////
   /// Constructors ///
   ////////////////////

  public:
   // forwarding wrapper constructor around all constructors
   template < typename T1, typename... Args >
   // exclude potential recursion traps
      requires common::is_none_v<
         std::remove_cvref_t< T1 >,  // remove cvref to avoid checking each ref-case individually
         tag::internal_construct,  // don't recurse back from internal constructors or self
         MCCFR  // don't steal the copy/move constructor calls (std::remove_cvref ensures both)
         >
   MCCFR(T1&& t, Args&&... args)
       : MCCFR(tag::internal_construct{}, std::forward< T1 >(t), std::forward< Args >(args)...)
   {
      this->_sanity_check_config();
      assert_serialized_and_unrolled(_env());
   }

  private:
   MCCFR(
      tag::internal_construct,
      Env env_,
      uptr< world_state_type > root_state_,
      Policy policy_ = Policy(),
      AveragePolicy avg_policy_ = AveragePolicy(),
      double epsilon = 0.6,
      size_t seed = common::default_seed  // consistent with factory.hpp defaults
   )
       : base(std::move(env_), std::move(root_state_), std::move(policy_), std::move(avg_policy_)),
         m_epsilon(epsilon),
         m_rng(seed)
   {
   }

   MCCFR(
      tag::internal_construct,
      Env env_,
      Policy policy_ = Policy(),
      AveragePolicy avg_policy_ = AveragePolicy(),
      double epsilon = 0.6,
      size_t seed = common::default_seed  // consistent with factory.hpp defaults
   )
       : MCCFR(
          tag::internal_construct{},
          std::move(env_),
          std::make_unique< world_state_type >(env.initial_world_state()),
          std::move(policy_),
          std::move(avg_policy_),
          epsilon,
          seed
       )
   {
   }

   MCCFR(
      tag::internal_construct,
      Env env_,
      uptr< world_state_type > root_state_,
      std::unordered_map< Player, Policy > policy_,
      std::unordered_map< Player, AveragePolicy > avg_policy_,
      double epsilon = 0.6,
      size_t seed = common::default_seed  // consistent with factory.hpp defaults
   )
       : base(std::move(env_), std::move(root_state_), std::move(policy_), std::move(avg_policy_)),
         m_epsilon(epsilon),
         m_rng(seed)
   {
   }

   ////////////////////////////////////
   /// API: public member functions ///
   ////////////////////////////////////

  public:
   /// import public getters

   using base::env;
   using base::policy;
   using base::average_policy;
   using base::iteration;
   using base::root_state;
   using base::fetch_policy;

  public:
   /**
    * @brief executes n iterations of the VanillaCFR algorithm.
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
    * @param player_to_update the optional player to update this iteration. If not provided, the
    * function will continue with the regular update cycle. By providing this parameter the user can
    * expressly modify the update cycle to even update individual players multiple times in a row.
    * @return the game value of the iteration with the current policy
    */
   auto iterate(std::optional< Player > player_to_update = std::nullopt)
      requires(config.update_mode == UpdateMode::alternating);

   ////////////////////////////////
   /// private member functions ///
   ////////////////////////////////

   inline auto& _infonodes() { return m_infonode; }
   inline auto& infonode(const sptr< info_state_type >& infostate) const
   {
      return m_infonode.at(infostate);
   }
   inline auto& _infonode(const sptr< info_state_type >& infostate)
   {
      return m_infonode.at(infostate);
   }
   inline auto& infonode(const info_state_type& infostate) const
   {
      auto found = m_infonode.find(infostate);
      if(found == m_infonode.end()) {
         throw std::out_of_range{"Infostate not found."};
      }
      return found->second;
   }
   inline auto& _infonode(const info_state_type& infostate)
   {
      auto found = m_infonode.find(infostate);
      if(found == m_infonode.end()) {
         throw std::out_of_range{"Infostate not found."};
      }
      return found->second;
   }

  private:
   /// import the parent's member variable accessors and protected utilities
   using base::_env;
   using base::_iteration;
   using base::_root_state_uptr;
   using base::_policy;
   using base::_average_policy;
   using base::_player_update_schedule;
   using base::_cycle_player_to_update;
   using base::_preview_next_player_to_update;
   using base::_partial_pruning_condition;

   /// the relevant data stored at each infostate
   std::unordered_map<
      sptr< info_state_type >,
      infostate_data_type,
      common::value_hasher< info_state_type >,
      common::value_comparator< info_state_type > >
      m_infonode{};
   /// the parameter to control the epsilon-on-policy exploration
   double m_epsilon;
   /// the rng state to produce random numbers with
   common::RNG m_rng;
   /// the standard 0 to 1. floating point uniform distribution
   std::uniform_real_distribution< double > m_uniform_01_dist{0., 1.};
   /// the actual regret minimizing method we will apply on the infostates
   [[no_unique_address]] minimizer_type m_regret_minimizer{};

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

   std::pair< StateValueMap, Probability > _traverse(
      std::optional< Player > player_to_update,
      world_state_type& state,
      ReachProbabilityMap reach_probability,
      ObservationbufferMap observation_buffer,
      InfostateSptrMap infostates,
      Probability sample_probability,
      ConditionalWeightMap weights
   )
      requires(config.algorithm == MCCFRAlgorithmMode::outcome_sampling);

   StateValue _traverse(
      Player player_to_update,
      uptr< world_state_type > state,
      ObservationbufferMap observation_buffer,
      InfostateSptrMap infostates,
      delayed_update_set& infostates_to_update
   )  // clang-format off
      requires(
         config.algorithm == MCCFRAlgorithmMode::external_sampling
         or (
            config.algorithm == MCCFRAlgorithmMode::pure_cfr
            and config.update_mode == UpdateMode::alternating
         )
      );
   // clang-format on

   StateValueMap _traverse(
      std::optional< Player > player_to_update,
      uptr< world_state_type > curr_worldstate,
      ReachProbabilityMap reach_probability,
      ObservationbufferMap observation_buffer,
      InfostateSptrMap infostates,
      [[maybe_unused]] delayed_update_set& infostates_to_update
   )  // clang-format off
      requires(
         config.algorithm == MCCFRAlgorithmMode::chance_sampling
         or (
            config.algorithm == MCCFRAlgorithmMode::pure_cfr
            and config.update_mode == UpdateMode::simultaneous
         )
      );
   // clang-format on

   void _update_regrets(
      const ReachProbabilityMap& reach_probability,
      Player active_player,
      infostate_data_type& infostate_data,
      const action_type& sampled_action,
      Probability sampled_action_policy_prob,
      StateValue action_value,
      Probability tail_prob
   ) const
      requires(config.algorithm == MCCFRAlgorithmMode::outcome_sampling);

   void _update_average_policy(
      const info_state_type& infostate,
      infostate_data_type& infonode_data,
      const auto& current_policy,
      Probability reach_prob,
      [[maybe_unused]] Probability sample_prob,
      [[maybe_unused]] const action_type& sampled_action,
      [[maybe_unused]] ConditionalWeight weight
   )
      requires(config.algorithm == MCCFRAlgorithmMode::outcome_sampling);

   /**
    * @brief add the regret and policy increments to the respective tables.
    */
   void update_regret_and_policy(
      const info_state_type& infostate,
      const ReachProbabilityMap& reach_probability,
      const StateValueMap& state_value,
      const std::unordered_map< action_variant_type, StateValueMap >& action_value_map,
      auto& avg_action_policy,
      [[maybe_unused]] auto& curr_action_policy
   )
      requires(
         config.algorithm == MCCFRAlgorithmMode::chance_sampling
         or (config.algorithm == MCCFRAlgorithmMode::pure_cfr and config.update_mode == UpdateMode::simultaneous)
      );

   auto _terminal_value(
      world_state_type& state,
      std::optional< Player > player_to_update,
      Probability sample_probability
   )
      requires(config.algorithm == MCCFRAlgorithmMode::outcome_sampling);

   auto _sample_action(
      Player active_player,
      std::optional< Player > player_to_update,
      const std::vector< action_type >& actions,
      auto& action_policy
   );

   auto _sample_action_on_policy(const std::vector< action_type >& actions, auto& action_policy);

   template < bool return_likelihood = true >
   auto _sample_outcome(const world_state_type& state);

   void _initiate_regret_minimization(const delayed_update_set& update_set);

   void _invoke_regret_minimizer(const info_state_type& infostate, infostate_data_type& data);

   constexpr void _sanity_check_config();
};

}  // namespace nor::rm

// include the actual template definitions of the member functions
#include "mccfr.tcc"

#endif  // NOR_MCCFR_HPP
