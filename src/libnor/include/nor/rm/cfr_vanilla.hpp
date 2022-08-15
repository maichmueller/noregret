
#ifndef NOR_CFR_VANILLA_HPP
#define NOR_CFR_VANILLA_HPP

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

#include "cfr_base_tabular.hpp"
#include "cfr_config.hpp"
#include "common/common.hpp"
#include "forest.hpp"
#include "node.hpp"
#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "nor/policy.hpp"
#include "nor/type_defs.hpp"
#include "nor/utils/utils.hpp"
#include "rm_utils.hpp"

namespace nor::rm {

namespace detail {

template < CFRConfig config, typename Env >
struct VCFRNodeDataSelector {
  private:
   using action_type = typename fosg_auto_traits< Env >::action_type;
   /// for vanilla cfr we need no extra weight data stored
   using default_data_type = InfostateNodeData< action_type >;
   /// for exponential CFR we need to store:
   /// 1) the L1 weight for each action at each infostate intermittently per iteration.
   /// 2) the reach probability for each infostate intermittently per iteration because of 3)
   /// 3) Since the L1 weight is actually L1 = L1(t, I, a) and thus different
   /// for each action, we cannot merely keep track of the average policy numerator and then
   /// normalize at each infostate to get the actual average policy. As such we need to also store
   /// the average policy denominator. Since L1 is action dependent, this will need to be stored
   /// independently for each action at each infostate
   using exp_node_type = InfostateNodeData<
      action_type,
      // storage_element< 1 >: the instantaneous regret r(I, a) = sum_h r(h, a) will be stored in
      // the action map
      std::unordered_map<
         std::reference_wrapper< const action_type >,
         double,
         common::ref_wrapper_hasher< const action_type >,
         common::ref_wrapper_comparator< const action_type > >,
      // storage_element< 2 >: the reach probability pi^t(I)
      double,
      // storage_element< 3 >: the average policy cumulative denominator
      // sum_t pi^t(I) * exp(L1^t(I, a))
      std::unordered_map<
         std::reference_wrapper< const action_type >,
         double,
         common::ref_wrapper_hasher< const action_type >,
         common::ref_wrapper_comparator< const action_type > > >;
   /// For regret-based-pruning with CFR+ we need to store the instantenous regret and determine
   /// after the tree traversal whether to replace the current cumulative regret with the
   /// instantaneous one (i.e. if r(I, a) > 0 and R^T(I,a) < 0) or do a normal addition to the
   /// cumulative regret
   using rbp_cfr_plus_node_type = InfostateNodeData<
      action_type,
      // storage_element< 1 >: the instantaneous regret r(I, a) = sum_h r(h, a) will be stored in
      // the action map
      std::unordered_map<
         std::reference_wrapper< const action_type >,
         double,
         common::ref_wrapper_hasher< const action_type >,
         common::ref_wrapper_comparator< const action_type > > >;

  public:
   using type = std::conditional_t<
      config.weighting_mode == CFRWeightingMode::exponential,
      exp_node_type,  // if true, then we need extra tables per node
      std::conditional_t<
         config.pruning_mode == CFRPruningMode::regret_based
            and config.regret_minimizing_mode == RegretMinimizingMode::regret_matching_plus,
         rbp_cfr_plus_node_type,  // if true, then we need an extra table for the instant. regret
         default_data_type > >;  // otherwise the standard tables suffice
};

inline double _zero(double, size_t)
{
   return 0.;
}

}  // namespace detail

struct CFRDiscountedParameters {
   /// the parameter to exponentiate the weight of positive cumulative regrets with
   double alpha = 1.5;
   /// the parameter to exponentiate the weight of negative cumulative regrets with
   double beta = 0.;
   /// the parameter to exponentiate the weight of the cumulative policy with
   double gamma = 2.;
};

struct CFRExponentialParameters {
   /// the parameter function beta (can depend on the instantaneous regret of the action) to limit
   /// negative regrets to
   double (*beta)(double, size_t) = &detail::_zero;
};

namespace detail {
// a verification of the current config correctness
template < CFRConfig config >
consteval bool sanity_check_cfr_config();

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
  public:
   static_assert(
      detail::sanity_check_cfr_config< config >(),
      "The configuration check did not return TRUE.");

   ////////////////////////////
   /// API: public typedefs ///
   ////////////////////////////

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
      requires(not common::contains(
         std::array{CFRWeightingMode::discounted, CFRWeightingMode::exponential},
         config.weighting_mode))
   VanillaCFR(Args&&... args) : base(std::forward< Args >(args)...)
   {
   }

   template < typename... Args >
      requires(config.weighting_mode == CFRWeightingMode::discounted)
   VanillaCFR(CFRDiscountedParameters params, Args&&... args)
       : base(std::forward< Args >(args)...), m_dcfr_params(std::move(params))
   {
   }

   template < typename... Args >
      requires(config.weighting_mode == CFRWeightingMode::exponential)
   VanillaCFR(CFRExponentialParameters params, Args&&... args)
       : base(std::forward< Args >(args)...), m_expcfr_params(std::move(params))
   {
   }

   ////////////////////////////////////
   /// API: public member functions ///
   ////////////////////////////////////

   /// import public getters

   using base::env;
   using base::policy;
   using base::iteration;
   using base::root_state;

   auto& average_policy() const
      requires(config.weighting_mode != CFRWeightingMode::exponential)
   {
      return base::average_policy();
   }

   auto average_policy() const
      requires(config.weighting_mode == CFRWeightingMode::exponential)
   {
      // we need to build the average policy now on demand as the denominator is no longer
      // attainable via mere normalization, but is stored separately.
      auto avg_policy_out = base::average_policy();
      for(auto& [_, avg_player_policy_out] : avg_policy_out) {
         for(auto& [infostate_ptr, action_policy] : avg_player_policy_out) {
            const auto& action_policy_denominator = _infonode(infostate_ptr)
                                                       .template storage_element< 3 >();
            for(auto& [action_ref, policy_prob] : action_policy) {
               policy_prob /= action_policy_denominator.at(action_ref);
            }
         }
      }

      return avg_policy_out;
   }

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

  private:
   ////////////////////////////////
   /// private member functions ///
   ////////////////////////////////

   [[nodiscard]] inline auto& _infonodes() { return m_infonode; }
   [[nodiscard]] inline auto& _infonode(const info_state_type& infostate) const
   {
      return m_infonode.find(infostate)->second;
   }
   [[nodiscard]] inline auto& _infonode(const info_state_type& infostate)
   {
      return m_infonode.find(infostate)->second;
   }
   [[nodiscard]] inline auto& _infonode(const sptr< info_state_type >& infostate) const
   {
      return m_infonode.at(infostate);
   }
   [[nodiscard]] inline auto& _infonode(const sptr< info_state_type >& infostate)
   {
      return m_infonode.at(infostate);
   }

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

   /// Discounted CFR specific parameters
   CFRDiscountedParameters m_dcfr_params;
   /// Exponential CFR specific parameters
   CFRExponentialParameters m_expcfr_params;
   /// the actual regret minimizing method we will apply on the infostates
   static constexpr auto m_regret_minimizer = []< typename... Args >(Args&&... args) {
      if constexpr(config.regret_minimizing_mode == RegretMinimizingMode::regret_matching) {
         return rm::regret_matching(std::forward< Args >(args)...);
      }
      if constexpr(config.regret_minimizing_mode == RegretMinimizingMode::regret_matching_plus) {
         if constexpr(config.pruning_mode == CFRPruningMode::regret_based) {
            return rm::regret_matching_plus_rbp(std::forward< Args >(args)...);
         } else {
            return rm::regret_matching_plus(std::forward< Args >(args)...);
         }
      }
   };

   /////////////////////////////////////////////////
   /// private implementation details of the API ///
   /////////////////////////////////////////////////

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

   void _invoke_regret_minimizer(
      const sptr< info_state_type >& infostate_ptr,
      infostate_data_type& istate_data,
      [[maybe_unused]] double policy_weight,
      [[maybe_unused]] const auto& regret_weights);

   void _invoke_regret_minimizer(
      const sptr< info_state_type >& infostate_ptr,
      infostate_data_type& istate_data,
      auto&&...)
      requires(config.weighting_mode == CFRWeightingMode::exponential)
   ;
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
            if(_iteration() < _env().players(*_root_state_uptr()).size() - 1) {
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
      if(_iteration() < _env().players(*_root_state_uptr()).size() - 1)
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
         for(auto player : _env().players(*_root_state_uptr())) {
            rp_map.emplace(player, 1.);
         }
         return ReachProbabilityMap{std::move(rp_map)};
      }(),
      [&] {
         std::unordered_map< Player, std::vector< observation_type > > obs_map;
         auto players = _env().players(*_root_state_uptr());
         for(auto player : players | utils::is_actual_player_filter) {
            obs_map.emplace(player, std::vector< observation_type >{});
         }
         return ObservationbufferMap{std::move(obs_map)};
      }(),
      [&] {
         std::unordered_map< Player, sptr< info_state_type > > infostates;
         auto players = _env().players(*_root_state_uptr());
         for(auto player : players | utils::is_actual_player_filter) {
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
   auto policy_weight = [&]([[maybe_unused]] infostate_data_type& data_node) {
      if constexpr(
         config.weighting_mode == CFRWeightingMode::linear
         or config.weighting_mode == CFRWeightingMode::discounted) {
         // weighting by an iteration dependant factor multiplies the current iteration t as
         // t^gamma onto the update INCREMENT. The numerically more stable approach, however, is
         // to multiply the ACCUMULATED strategy with (t/(t+1))^gamma, as the risk of reaching
         // numerical ceilings is reduced. This is mathematically equivalent (e.g. see Noam
         // Brown's PhD thesis "Equilibrium Finding for Large Adversarial Imperfect-Information
         // Games").

         // normalization factor from the papers is irrelevant, as it is absorbed by the
         // normalization constant of each action policy afterwards.
         auto t = double(_iteration());
         double weighting_factor = t / (t + 1);
         if constexpr(config.weighting_mode == CFRWeightingMode::discounted) {
            weighting_factor = std::pow(weighting_factor, m_dcfr_params.gamma);
         }
         return weighting_factor;
      } else {
         // when no weighting is needed simply return 0. This will be ignored anyway
         return 0;
      }
   };

   auto regret_weights = [&]([[maybe_unused]] infostate_data_type& data_node) {
      if constexpr(config.weighting_mode == CFRWeightingMode::discounted) {
         // normalization factor from the papers is irrelevant, as it is absorbed by the
         // normalization constant of each action policy afterwards.
         auto t = double(_iteration());
         double t_alpha = std::pow(t, m_dcfr_params.alpha);
         double t_beta = std::pow(t, m_dcfr_params.beta);
         return std::array< double, 2 >{t_beta / (t_beta + 1), t_alpha / (t_alpha + 1)};
      } else {
         // when no weighting is needed simply return 0. This will be ignored anyway
         return 0;
      }
   };

   // here we now invoke the actual regret minimization procedure for each infostate individually
   if constexpr(config.update_mode == UpdateMode::alternating) {
      Player update_player = player_to_update.value();
      for(auto& [infostate_ptr, data] : _infonodes()) {
         if(infostate_ptr->player() == update_player) {
            _invoke_regret_minimizer(
               infostate_ptr, data, policy_weight(data), regret_weights(data));
         }
      }
   } else {
      // for simultaneous updates we simply update all infostates
      for(auto& [infostate_ptr, data] : _infonodes()) {
         _invoke_regret_minimizer(infostate_ptr, data, policy_weight(data), regret_weights(data));
      }
   }
};

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
void VanillaCFR< config, Env, Policy, AveragePolicy >::_invoke_regret_minimizer(
   const sptr< info_state_type >& infostate_ptr,
   infostate_data_type& istate_data,
   [[maybe_unused]] double policy_weight,
   [[maybe_unused]] const auto& regret_weights)
{
   // since we are reusing this variable a few times we alias it here
   auto& current_policy = fetch_policy< PolicyLabel::current >(
      infostate_ptr, istate_data.actions());

   // Discounted CFR only:
   // we first multiply the accumulated regret by the correct weight as per discount setting
   auto& regret_table = istate_data.regret();
   if constexpr(config.weighting_mode == CFRWeightingMode::discounted) {
      static_assert(
         std::is_same_v< std::array< double, 2 >, std::remove_cvref_t< decltype(regret_weights) > >,
         "Expected a regret weight array of length 2.");
      for(auto& cumul_regret : regret_table | ranges::views::values) {
         // index 0 is beta based weight, index 1 is alpha based weight
         cumul_regret *= regret_weights[cumul_regret > 0.];
      }
   }

   // here we now perform the actual regret minimizing update step as we update the current
   // policy through a regret matching algorithm. The specific algorihtm is determined by the
   // config we input

   // we provide the accessor lambda to get the underlying referenced action, as the infodata
   // stores only reference wrappers to the actions
   if constexpr(
      config.pruning_mode == CFRPruningMode::regret_based
      and config.regret_minimizing_mode == RegretMinimizingMode::regret_matching_plus) {
      m_regret_minimizer(
         current_policy,
         regret_table,
         [](const action_type& action) { return std::ref(action); },
         istate_data.template storage_element< 1 >());
   } else {
      m_regret_minimizer(
         current_policy, regret_table, [](const action_type& action) { return std::ref(action); });
   }

   // now we update the current accumulated policy by the iteration factor, again as per
   // discount setting.
   if constexpr(
      config.weighting_mode == CFRWeightingMode::linear
      or config.weighting_mode == CFRWeightingMode::discounted) {
      // we are expecting to be given the right weight for the configuration here
      for(auto& policy_prob :
          fetch_policy< PolicyLabel::average >(infostate_ptr, istate_data.actions())
             | ranges::views::values) {
         policy_prob *= policy_weight;
      }
   }
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
void VanillaCFR< config, Env, Policy, AveragePolicy >::_invoke_regret_minimizer(
   const sptr< info_state_type >& infostate_ptr,
   infostate_data_type& istate_data,
   auto&&...)
   requires(config.weighting_mode == CFRWeightingMode::exponential)
{
   auto exp_l1_weights = [&] {
      // we need to reset this infostate data node's accumulated weights to prepare empty
      // buffers for the next iteration
      std::unordered_map<
         std::reference_wrapper< const action_type >,
         double,
         common::ref_wrapper_hasher< const action_type >,
         common::ref_wrapper_comparator< const action_type > >
         l1;
      LOGD2(
         "Instant regret values",
         ranges::views::values(istate_data.template storage_element< 1 >()));
      double average_instant_regret = 0.;
      ranges::for_each(
         istate_data.template storage_element< 1 >(), [&](auto& actionref_to_instant_regret) {
            auto& [action_ref, instant_regret] = actionref_to_instant_regret;
            // instant_regret is r(I, a), not r(h, a)
            average_instant_regret += instant_regret;
         });
      average_instant_regret /= double(istate_data.template storage_element< 1 >().size());

      ranges::for_each(
         istate_data.template storage_element< 1 >(), [&](auto& actionref_to_instant_regret) {
            auto& [action_ref, instant_regret] = actionref_to_instant_regret;
            // instant_regret is r(I, a), not R(I, a), i.e. the cumulative regret
            l1[action_ref] = std::exp(instant_regret - average_instant_regret);
         });
      return l1;
   }();
   // exponential cfr requires weighting the cumulative regret by the L1 factor to EACH (I, a) pair.
   // Yet L1, which is actually L1(I, a), is only known after the entire tree has been traversed and
   // thus can't be done during the traversal. Hence, we need to update our cumulative regret by the
   // correct weight now here upon iteration over all infostates
   auto& curr_policy = fetch_policy< PolicyLabel::current >(infostate_ptr, istate_data.actions());
   auto& regret_table = istate_data.regret();
   for(auto& [action, cumul_regret] : regret_table) {
      auto action_ref = std::cref(action);
      auto& instant_regret = istate_data.template storage_element< 1 >()[action_ref];
      LOGD2("Action", action);
      LOGD2("L1 weight", exp_l1_weights[action_ref]);
      LOGD2("Instant regret", instant_regret);
      LOGD2(
         "All instant regret", ranges::views::values(istate_data.template storage_element< 1 >()));
      LOGD2("Cumul regret before", ranges::views::values(istate_data.regret()));

      if(instant_regret >= 0) {
         LOGD2("Cumul regret update (>=0)", exp_l1_weights[action_ref] * instant_regret);
         cumul_regret += exp_l1_weights[action_ref] * instant_regret;
      } else {
         LOGD2(
            "Cumul regret update (<0)",
            exp_l1_weights[action_ref] * m_expcfr_params.beta(instant_regret, _iteration()));
         cumul_regret += exp_l1_weights[action_ref]
                         * m_expcfr_params.beta(instant_regret, _iteration());
      }
      // reset the instant regret, so that the next round's storage is starting fresh
      instant_regret = 0.;
      LOGD2("Cumul regret after", ranges::views::values(istate_data.regret()));
   }

   LOGD2("Cumul Policy before", ranges::views::values(istate_data.template storage_element< 3 >()));
   // now we update the current accumulated policy numerator and denominator
   for(auto& [action, avg_policy_prob] :
       fetch_policy< PolicyLabel::average >(infostate_ptr, istate_data.actions())) {
      double reach_prob = istate_data.template storage_element< 2 >();
      double l1_weight = exp_l1_weights[std::cref(action)];
      // this is the cumulative enumerator update
      LOGD2("Cumul Policy Reach prob", reach_prob);
      LOGD2("Cumul Policy L1 Weight prob", l1_weight);
      LOGD2("Cumul Policy Current Policy", curr_policy[action]);
      LOGD2("Cumul Policy numerator update", l1_weight * reach_prob * curr_policy[action]);
      avg_policy_prob += l1_weight * reach_prob * curr_policy[action];
      // this is the cumulative denominator update
      LOGD2("Cumul Policy denominator update", l1_weight * reach_prob);
      istate_data.template storage_element< 3 >(std::cref(action)) += l1_weight * reach_prob;
   }

   LOGD2("Cumul Policy after", ranges::views::values(istate_data.template storage_element< 3 >()));
   // here we now perform the actual regret minimizing update step as we update the current
   // policy through a regret matching algorithm. The specific algorihtm is determined by the
   // config we entered to the class

   // we provide the accessor lambda to get the underlying referenced action, as the infodata
   // stores only reference wrappers to the actions
   m_regret_minimizer(
      curr_policy, regret_table, [](const action_type& action) { return std::ref(action); });
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

   if constexpr(config.pruning_mode == CFRPruningMode::partial) {
      // if all players have 0 reach probability for reaching this infostate then the
      // entire subtree visited from this infostate can be pruned, since both the regret
      // updates (depending on the counterfactual values, i.e. pi_{-i}) will be 0, as well
      // as the average strategy updates (depending on pi_i) will be 0. If only the
      // opponent reach prob is 0, then we can only skip regret updates which does not
      // improve the speed much in this implementation.
      if([&] {
            if constexpr(config.update_mode == UpdateMode::alternating) {
               // if one of the opponents' (non traversers') reach prob is 0. then the regret
               // updates will be skipped. If also the traversing player's reach probability is 0
               // then the entire subtree is prunable, since the average strategy updates would also
               // be 0.
               Player traverser = player_to_update.value();
               auto traversing_player_rp_is_zero = reach_probability.get()[traverser]
                                                   <= std::numeric_limits< double >::epsilon();
               return traversing_player_rp_is_zero
                      and ranges::any_of(reach_probability.get(), [&](const auto& player_rp_pair) {
                             const auto& [player, rp] = player_rp_pair;
                             return player != traverser
                                    and rp <= std::numeric_limits< double >::epsilon();
                          });
            } else {
               // A mere check on ONE of the opponents having reach prob 0 and the active player
               // having reach prob 0 would not suffice in the multiplayer case as some average
               // strategy updates of other opponent with reach prob > 0 would be missed in the case
               // of simultaneous updates.
               return ranges::all_of(reach_probability.get(), [&](const auto& player_rp_pair) {
                  const auto& [player, rp] = player_rp_pair;
                  return player != Player::chance
                         and rp <= std::numeric_limits< double >::epsilon();
               });
            }
         }()) {
         // if the entire subtree is pruned then the values that could be found are all 0. for each
         // player
         return StateValueMap{[&] {
            StateValueMap::UnderlyingType map;
            for(auto player : _env().players(*state) | utils::is_actual_player_pred) {
               map[player] = 0.;
            }
            return map;
         }()};
      }
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
   // deterministic. In that case we might need to traverse the chance player's actions or an
   // active player's actions
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
         // if this is a chance node then we dont need to update any regret or average policy
         // after the traversal
         return state_value;
      } else {
         nonchance_player_traverse();
      }
   } else {
      nonchance_player_traverse();
   }

   if constexpr(use_current_policy) {
      // we can only update our regrets and policies if we are traversing with the current
      // policy, since the average policy is not to be changed directly (but through averaging up
      // all current policies)
      if constexpr(config.update_mode == UpdateMode::alternating) {
         // in alternating updates, we only update the regret and strategy if the current
         // player is the chosen player to update.
         if(active_player == player_to_update.value()) {
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
      // we try to normalize only for the average policy, since iterations with the current
      // policy are for the express purpose of updating the average strategy. As such, we should
      // not intervene to change these values, as that may alter the values incorrectly
      auto action_policy_value_view = action_policy | ranges::views::values;
      normalizing_factor = std::reduce(
         action_policy_value_view.begin(), action_policy_value_view.end(), 0., std::plus{});

      if(std::abs(normalizing_factor) < 1e-20) {
         throw std::invalid_argument(
            "Average policy likelihoods accumulate to 0. Such values cannot be normalized.");
      }
   }

   for(const action_type& action : actions) {
      auto action_prob = action_policy[action] / normalizing_factor;

      auto child_reach_prob = reach_probability.get();
      child_reach_prob[active_player] *= action_prob;

      uptr< world_state_type > next_wstate_uptr = _child_state(state, action);
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
   auto& avg_action_policy = [&]() -> auto&
   {
      if constexpr(config.weighting_mode != CFRWeightingMode::exponential) {
         return fetch_policy< PolicyLabel::average >(infostate, actions);
      } else {
         // this value will be ignored so we can simply return anything that is cheap to fetch (it
         // has to be an l-value so can't return an r-value like simply 0
         return _env();
      }
   }
   ();
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
      if constexpr(config.weighting_mode != CFRWeightingMode::exponential) {
         if(cf_reach_prob > 0) {
            // this if statement effectively introduces partial pruning. But this is such a slight
            // modification (and gain, if any) that it is to be included in all variants of CFR
            //
            // all other cfr variants currently implemented need the average regret update at
            // history update time
            istatedata.regret(action) += cf_reach_prob
                                         * (q_value.get().at(player) - player_state_value);
         }
      } else {
         // for the exponential cfr method we need to remember these regret increments of
         // iteration t, until the end of iteration t, and apply them once we have computed the L1
         // weights (i.e. at infostate update time, not history update time). After iteration t ends
         // we have to delete them again, so that this is only a memory of the current iteration!
         // Each history h that passed through infostate I will increment here the
         // instantaneous regret values r(h,a), in order to accumulate r(I, a) = sum_h r(h, a)

         // We emplace the action first into the cumul regret map (if not already there) to receive
         // the action key's reference back. This reference is then going to be ref-wrapped and
         // added to the instant temp regret map. We avoid copying the action this way. For games
         // with elaborate action types, this may be worth the extra runtime, for small action
         // values probably not. We do however save some memory since ref wrappers are merely as big
         // as one pointer.
         auto [iter, _] = istatedata.regret().try_emplace(action, 0.);
         // if we emplaced merely std::cref(action) here then we would have silent segfaults that
         // lead to erroenuous memory storing
         istatedata.template storage_element< 1 >(std::cref(
            iter->first)) += cf_reach_prob * (q_value.get().at(player) - player_state_value);
      }
      if constexpr(config.weighting_mode != CFRWeightingMode::exponential) {
         // update the cumulative policy according to the formula:
         // let I be the infostate, p be the player, a the chosen action, sigma^t the current
         // policy
         //    avg_sigma^{t+1} = \sum_a reach_prob_{p}(I) * sigma^t(I, a)
         avg_action_policy[action] += player_reach_prob * curr_action_policy[action];
         // For exponential CFR we update the average policy after the tree traversal
      }
   }
   if constexpr(config.weighting_mode == CFRWeightingMode::exponential) {
      // For exponential CFR we need to store the reach probability of the active player until
      // the end of the iteration
      istatedata.template storage_element< 2 >() = player_reach_prob;
   }
}

namespace detail {
// a verification of the current config correctness
template < CFRConfig config >
consteval bool sanity_check_cfr_config()
{
   if constexpr(
      config.weighting_mode == CFRWeightingMode::exponential
      and config.pruning_mode == CFRPruningMode::regret_based
      and config.regret_minimizing_mode == RegretMinimizingMode::regret_matching_plus) {
      // there is currently no theoretic work on combining these methods and the update rule for the
      // cumulative regret in both clash with different approaches (exp weighting wants e^L1
      // weighted updates) while regret-based-pruning with CFR+ wants to replace the
      // cumulative regret with r(I,a) only if r(I,a) > 0 and R^T(I,a) < 0, otherwise do a normal
      // cumulative regret update (i.e. R^t+1(I,a) = R^t(I,a) + r(I,a))
      return false;
   }
   return true;
}

}  // namespace detail

}  // namespace nor::rm

#endif  // NOR_CFR_VANILLA_HPP
