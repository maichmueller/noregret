
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
 * A (Vanilla) Counterfactual Regret Minimization algorithm class following the
 * terminology of the Factored-Observation Stochastic Games (FOSG) formulation.
 *
 * The implementation follows the algorithm detail of @cite Neller2013
 *
 *
 * @tparam Env, the environment/game type to run VanillaCFR on.
 *
 */
template < bool alternating_updates, typename Env, typename Policy, typename AveragePolicy >
class MCCFR:
    public TabularCFRBase< alternating_updates, Env, Policy, AveragePolicy > {
   ////////////////////////////
   /// API: public typedefs ///
   ////////////////////////////
  public:
   /// aliases for the template types
   using base = TabularCFRBase< alternating_updates, Env, Policy, AveragePolicy >;
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

   /// inherit all constructors from base
   MCCFR(auto... args) : base(std::forward< decltype(args) >(args)...) {}

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
      requires(alternating_updates)
   ;

//   ValueMap game_value() { return _iterate< false, false >(std::nullopt); }

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

   /// the rng state to produce random numbers with
   common::random::RNG m_rng;


   /// define the implementation details of the API


};

}  // namespace nor::rm

#endif  // NOR_MCCFR_HPP
