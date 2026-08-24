
#ifndef NOR_CFR_HPP
#define NOR_CFR_HPP

#include <iostream>
#include <list>
#include <map>
#include <named_type.hpp>
#include <queue>
#include <ranges>
#include <stack>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cfr_base.hpp"
#include "cfr_config.hpp"
#include "common/common.hpp"
#include "nor/at_runtime.hpp"
#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "nor/rm/forest.hpp"
#include "nor/rm/minimizers/minimizers.hpp"
#include "nor/rm/node.hpp"
#include "nor/rm/rm_utils.hpp"
#include "nor/tag.hpp"
#include "nor/type_defs.hpp"
#include "nor/utils/utils.hpp"

namespace nor::rm {

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
      "The configuration check did not return TRUE."
   );

   ////////////////////////////
   /// API: public typedefs ///
   ////////////////////////////

   /// aliases for the template types
   using base =
      TabularCFRBase< config.update_mode == UpdateMode::alternating, Env, Policy, AveragePolicy >;
   using env_type = Env;
   using policy_type = Policy;
   using average_policy_type = AveragePolicy;
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
   /// the regret minimizer selected by the configuration
   using minimizer_type = minimizer_for_t< config, action_type >;
   /// the data to store per infostate entry
   using infostate_data_type = InfostateNodeData<
      action_type,
      typename minimizer_type::node_data_type >;
   /// strong-types for player based maps
   using InfostateSptrMap = typename base::InfostateSptrMap;
   using ObservationbufferMap = typename base::ObservationbufferMap;

   ////////////////////
   /// Constructors ///
   ////////////////////

   /// inherit all constructors from base
  public:
   VanillaCFR(const VanillaCFR&) = delete;
   VanillaCFR(VanillaCFR&&) = default;
   ~VanillaCFR() = default;
   VanillaCFR& operator=(const VanillaCFR&) = delete;
   VanillaCFR& operator=(VanillaCFR&&) noexcept = default;

   // forwarding wrapper constructor around all constructors
   template < typename T1, typename... Args >
   // exclude potential recursion traps
      requires common::is_none_v<
         std::remove_cvref_t< T1 >,  // remove cvref to avoid checking each ref-case individually
         tag::internal_construct,  // don't recurse back from internal constructors or self
         VanillaCFR  // don't steal the copy/move constructor calls (std::remove_cvref ensures both)
         >
   VanillaCFR(T1&& t, Args&&... args)
       : VanillaCFR(tag::internal_construct{}, std::forward< T1 >(t), std::forward< Args >(args)...)
   {
      assert_serialized_and_unrolled(_env());
   }

  private:
   template < typename... Args >
      requires(not common::contains(
         std::array{CFRWeightingMode::discounted, CFRWeightingMode::exponential},
         config.weighting_mode
      ))
   VanillaCFR(tag::internal_construct, Args&&... args) : base(std::forward< Args >(args)...)
   {
   }

   template < typename... Args >
      requires(config.weighting_mode == CFRWeightingMode::discounted)
   VanillaCFR(tag::internal_construct, CFRDiscountedParameters params, Args&&... args)
       : base(std::forward< Args >(args)...), m_regret_minimizer(params)
   {
   }

   template < typename... Args >
      requires(config.weighting_mode == CFRWeightingMode::exponential)
   VanillaCFR(tag::internal_construct, CFRExponentialParameters params, Args&&... args)
       : base(std::forward< Args >(args)...), m_expcfr_params(params)
   {
   }

   ////////////////////////////////////
   /// API: public member functions ///
   ////////////////////////////////////

  public:
   /// import public getters

   using base::env;
   using base::policy;
   using base::iteration;
   using base::cycle;
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
            const auto& node_data = _infonode(infostate_ptr).data();
            const auto& action_policy_denominator = node_data.avg_policy_denominator;
            for(auto& [action, policy_prob] : action_policy) {
               policy_prob /= action_policy_denominator[node_data.index_of(action)];
            }
         }
      }

      return avg_policy_out;
   }

   /**
    * @brief executes n iterations of the VanillaCFR algorithm in unrolled form (no recursion).

    * @param n_iters the number of iterations to perform.
    * @return game value per iteration
    */
   auto iterate(size_t n_iters);
   /**
    * @brief executes one iteration of alternating updates vanilla cfr.
    *
    * This overload only participates if the config defined alternating updates to be made.

    * @param player_to_update the optional player to update this iteration. If not provided, the
    * function will continue with the regular update cycle. By providing this parameter the user can
    * expressly modify the update cycle to even update individual players multiple times in a row.
    * @return game value of the iteration of the player
    */
   auto iterate(std::optional< Player > player_to_update = std::nullopt)
      requires(config.update_mode == UpdateMode::alternating);

   StateValueMap game_value() { return _iterate< false, false >(std::nullopt); }

   /**
    * @brief updates the regret and policy tables of the infostate with the state-values.
    */
   void update_regret_and_policy(
      const info_state_type& infostate,
      const ReachProbabilityMap& reach_probability,
      const StateValueMap& state_value,
      const std::unordered_map< action_variant_type, StateValueMap >& action_value_map
   );

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

   /// import the parent's member variable accessors and protected utilities
   using base::_env;
   using base::_iteration;
   using base::_root_state_uptr;
   using base::_policy;
   using base::_average_policy;
   using base::_player_update_schedule;
   using base::_cycle_player_to_update;
   using base::_partial_pruning_condition;

   /// the relevant data stored at each infostate
   std::unordered_map<
      sptr< info_state_type >,
      infostate_data_type,
      common::value_hasher< info_state_type >,
      common::value_comparator< info_state_type > >
      m_infonode{};

   /// Exponential CFR specific parameters
   [[no_unique_address]] std::conditional_t<
      config.weighting_mode == CFRWeightingMode::exponential,
      CFRExponentialParameters,
      utils::empty >
      m_expcfr_params;
   /// the actual regret minimizing method we will apply on the infostates
   [[no_unique_address]] minimizer_type m_regret_minimizer{};

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
    *
    * The world state lives in a depth-indexed arena slot owned by the solver and is
    * reused across the whole recursion of an iteration; reach probabilities,
    * observation buffers and infostate maps are passed by reference and restored
    * at every recursion boundary (save/restore instead of per-edge copies).
    */
   template < bool initialize_infonodes, bool use_current_policy = true >
   StateValueMap _traverse(
      std::optional< Player > player_to_update,
      world_state_type& state,
      size_t depth,
      ReachProbabilityMap& reach_probability,
      ObservationbufferMap& observation_buffer,
      InfostateSptrMap& infostates
   );

   template < bool initialize_infonodes, bool use_current_policy = true >
   void _traverse_player_actions(
      std::optional< Player > player_to_update,
      Player active_player,
      world_state_type& state,
      size_t depth,
      ReachProbabilityMap& reach_probability,
      ObservationbufferMap& observation_buffer,
      InfostateSptrMap& infostates,
      StateValueMap& state_value,
      std::unordered_map< action_variant_type, StateValueMap >& action_value
   );

   template < bool initialize_infonodes, bool use_current_policy = true >
   void _traverse_chance_actions(
      std::optional< Player > player_to_update,
      Player active_player,
      world_state_type& state,
      size_t depth,
      ReachProbabilityMap& reach_probability,
      ObservationbufferMap& observation_buffer,
      InfostateSptrMap& infostates,
      StateValueMap& state_value,
      std::unordered_map< action_variant_type, StateValueMap >& action_value
   );

   /// returns the arena slot for recursion 'depth', copy-assigned from 'source'.
   /// Slots are allocated on first use and then reused for the remainder of
   /// the solver's lifetime (one live clone per active path position).
   world_state_type& _arena_state(size_t depth, const world_state_type& source);

   /// depth-indexed world-state clones reused across the recursion within one
   /// iteration (and across iterations); slot d holds the state reached after
   /// d transitions along the currently active traversal path. Slots are
   /// reconstructed in place per edge (no heap traffic, no copy-assignment
   /// requirement on the state type).
   /// NOTE: a deque on purpose -- growing it never moves existing slots, so
   /// references into deeper slots held by active recursion frames stay valid
   std::deque< utils::ReusableSlot< world_state_type > > m_traversal_state_arena;

   void _initiate_regret_minimization(const std::optional< Player >& player_to_update);

   /// performs the end-of-traversal regret minimization step for one infostate
   void _invoke_regret_minimizer(const info_state_type& infostate);
};

template < typename Env, typename Policy, typename AveragePolicy >
using CFRPlus = VanillaCFR<
   CFRConfig{
      .update_mode = UpdateMode::alternating,
      .regret_minimizing_mode = RegretMinimizingMode::regret_matching_plus,
      .weighting_mode = CFRWeightingMode::uniform},
   Env,
   Policy,
   AveragePolicy >;

/**
 * PCFR+ (Farina, Kroer, Sandholm — "Faster Game Solving via Predictive
 * Blackwell Approachability", AAAI 2021): CFR+ whose regret-matching step is
 * conditioned on a persistence prediction of the next instantaneous regret.
 *
 * The config rides the discounted weighting machinery purely for its gamma-side
 * quadratic average-policy accumulation (the default CFRDiscountedParameters
 * already provide gamma = 2); the alpha/beta regret discounts are compiled out
 * by the minimizer selection for this mode. Constructed like CFRDiscounted,
 * i.e. with a leading (possibly default) rm::CFRDiscountedParameters argument.
 */
template < CFRDiscountedConfig config, typename Env, typename Policy, typename AveragePolicy >
using PCFRPlus = VanillaCFR<
   CFRConfig{
      .update_mode = config.update_mode,
      .regret_minimizing_mode = RegretMinimizingMode::predictive_regret_matching_plus,
      .weighting_mode = CFRWeightingMode::discounted},
   Env,
   Policy,
   AveragePolicy >;

/**
 * SAPCFR+ (arXiv:2503.12770): robustified PCFR+ in which only the prediction
 * shift term is damped by 1/(1 + alpha) with alpha = 2. Same construction
 * constraints as PCFRPlus.
 */
template < CFRDiscountedConfig config, typename Env, typename Policy, typename AveragePolicy >
using SAPCFRPlus = VanillaCFR<
   CFRConfig{
      .update_mode = config.update_mode,
      .regret_minimizing_mode = RegretMinimizingMode::sap_predictive_regret_matching_plus,
      .weighting_mode = CFRWeightingMode::discounted},
   Env,
   Policy,
   AveragePolicy >;

/**
 * APCFR+ (Meng et al., arXiv:2503.12770v2, "Faster Game Solving via Asymmetry
 * of Step Sizes"): PCFR+ with an adaptive per-infostate asymmetry between the
 * prediction-carrying implicit update and the explicit accumulated regret
 * update; the asymmetry coefficient is learned from running squared-L2-norm
 * sums (their Eq. (10), alpha_max = 5). Same construction constraints as
 * PCFRPlus.
 */
template < CFRDiscountedConfig config, typename Env, typename Policy, typename AveragePolicy >
using APCFRPlus = VanillaCFR < CFRConfig
{
   .update_mode = config.update_mode,
   .regret_minimizing_mode = RegretMinimizingMode::ap_predictive_regret_matching_plus,
   *DCFR
         + (Xu et al., IJCAI 2024, arXiv : 2404.13891, sec .4)
       : RM + folding with the* positive
         - part alpha discount applied BEFORE adding the instantaneous regret and* clipping the
            sum: * R
      ^ t = [R ^ {t - 1} * (t - 1) ^ alpha / ((t - 1) ^ alpha + 1) + r ^ t]
            ^ +*Constructed like CFRDiscounted,
   i.e.with a leading(possibly default) * rm::CFRDiscountedParameters argument;
   the paper defaults are available via
 * rm::dcfrplus_default_parameters() (alpha = 1.5, gamma = 4).
 */
template < CFRDiscountedConfig config, typename Env, typename Policy, typename AveragePolicy >
using DCFRPlus = VanillaCFR<
   CFRConfig{
      .update_mode = config.update_mode,
      .regret_minimizing_mode = RegretMinimizingMode::discounted_regret_matching_plus,
      .weighting_mode = CFRWeightingMode::discounted},
   Env,
   Policy,
   AveragePolicy >;

   /**
    * P2PCFR+ ("Pessimistic PCFR+", ICLR'25 submission, OpenReview njyZgDDeY4):
    * PCFR+ with a fixed pessimistic prediction damping 1/(1 + alpha), alpha = 5.
    * Same construction constraints as PCFRPlus.
    */
   template < CFRDiscountedConfig config, typename Env, typename Policy, typename AveragePolicy >
   using P2PCFRPlus = VanillaCFR < CFRConfig
   {
      .update_mode = config.update_mode,
      .regret_minimizing_mode = RegretMinimizingMode::p2p_predictive_regret_matching_plus,
      *PDCFR
         + (arXiv : 2404.13891, sec .4)
          : DCFR + whose recommendations are computed * from the persistence
            - predicted next cumulative regret
                 * R ~^{ t + 1 } = [R ^ t * t ^ alpha / (t ^ alpha + 1) + v ^ {t + 1}] ^ +,
      v ^ { t + 1 } = r
                      ^ t.
                           * Paper defaults via rm::pcfrplus_default_parameters()(
                                alpha = 2.3, gamma = 5
                           )
                                .
                           *
                           / template <
                              CFRDiscountedConfig config,
                              typename Env,
                              typename Policy,
                              typename AveragePolicy >
                           using PDCFRPlus = VanillaCFR<
         CFRConfig{
            .update_mode = config.update_mode,
            .regret_minimizing_mode = RegretMinimizingMode::
               discounted_predictive_regret_matching_plus,
            .weighting_mode = CFRWeightingMode::discounted},
         Env,
         Policy,
         AveragePolicy >;

      /**
       * Smooth PCFR+ / Smooth PRM+ (Farina, Grand-Clément, Kroer, Lee, Luo, NeurIPS
       * 2023, arXiv:2305.14709, Algorithm 2): PCFR+ whose predicted-regret vector is
       * kept at 1-norm >= 1 before normalization. Same construction constraints as
       * PCFRPlus.
       */
      template < CFRDiscountedConfig config, typename Env, typename Policy, typename AveragePolicy >
      using SmoothPCFRPlus = VanillaCFR<
         CFRConfig{
            .update_mode = config.update_mode,
            .regret_minimizing_mode = RegretMinimizingMode::smooth_predictive_regret_matching_plus,
            .weighting_mode = CFRWeightingMode::discounted},
         Env,
         Policy,
         AveragePolicy >;

      /**
       * Stable PCFR+ / Stable PRM+ (arXiv:2305.14709, Algorithm 1): PCFR+ with
       * componentwise restart of the cumulative regret table at the R0 = 1 floor.
       * Same construction constraints as PCFRPlus.
       */
      template < CFRDiscountedConfig config, typename Env, typename Policy, typename AveragePolicy >
      using StablePCFRPlus = VanillaCFR<
         CFRConfig{
            .update_mode = config.update_mode,
            .regret_minimizing_mode = RegretMinimizingMode::stable_predictive_regret_matching_plus,
            .weighting_mode = CFRWeightingMode::discounted},
         Env,
         Policy,
         AveragePolicy >;
      *HS - DCFR / HS - DCFRPlus / HS - PDCFRPlus / HS
            - PCFRPlus(Zhang, McAleer, Sandholm, *AAAI 2026, arXiv : 2404.09097)
          : these are NOT distinct solver types-- a
            * hyperparameter schedule is a runtime property of rm::CFRDiscountedParameters.
            * Build them by passing one of the assembled bundles to make_cfr
            / make_cfr_discounted: *HS
            - DCFR(n)->rm::hs_dcfr_parameters(n, variant) with
                 * RegretMinimizingMode::regret_matching * HS
            - PCFR
            + (n)->rm::hs_pcfrplus_parameters(n, variant) with
                 * RegretMinimizingMode::predictive_regret_matching_plus * HS
            - DCFR
            + (n)->rm::hs_dcfrplus_parameters(n, variant) with
                 * RegretMinimizingMode::discounted_regret_matching_plus * HS
            - PDCFR
            + (n)->rm::hs_pdcfrplus_parameters(n, variant) with
                 * RegretMinimizingMode::discounted_predictive_regret_matching_plus
                 * where 'variant' selects HSVariant::gamma30(paper - recommended default)
         or *HSVariant::gamma15
               and 'n' is the total planned iteration count(bound into the
                                                               * schedule callables at construction;
                                                            see discounted_predictive.hpp)
                         .
                      * /

                      template <
                         CFRExponentialConfig config,
                         typename Env,
                         typename Policy,
                         typename AveragePolicy >
                      using CFRExponential = VanillaCFR<
         CFRConfig{
            .update_mode = config.update_mode,
            .regret_minimizing_mode = config.regret_minimizing_mode,
            .weighting_mode = CFRWeightingMode::exponential},
         Env,
         Policy,
         AveragePolicy >;

      template < CFRDiscountedConfig config, typename Env, typename Policy, typename AveragePolicy >
      using CFRDiscounted = VanillaCFR<
         CFRConfig{
            .update_mode = config.update_mode,
            .regret_minimizing_mode = config.regret_minimizing_mode,
            .weighting_mode = CFRWeightingMode::discounted},
         Env,
         Policy,
         AveragePolicy >;

      template < CFRLinearConfig config, typename Env, typename Policy, typename AveragePolicy >
      using CFRLinear = VanillaCFR<
         CFRConfig{
            .update_mode = config.update_mode,
            .regret_minimizing_mode = config.regret_minimizing_mode,
            .weighting_mode = CFRWeightingMode::discounted},
         Env,
         Policy,
         AveragePolicy >;

   }  // namespace nor::rm

// include the actual template implementations of this class
#include "cfr.tcc"

#endif  // NOR_CFR_HPP
