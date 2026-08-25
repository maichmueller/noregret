
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
#include "nor/rm/lazy.hpp"
#include "nor/rm/minimizers/minimizers.hpp"
#include "nor/rm/node.hpp"
#include "nor/rm/pruning.hpp"
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

   /// exponential weighting WITHOUT explicit parameters (purely additive): lets generic
   /// CFRConfig-driven factories construct exponential-weighted solvers -- including the
   /// dynamic-thresholding combinations -- with default beta
   template < typename T1, typename... Args >
      requires(config.weighting_mode == CFRWeightingMode::exponential)
                 and common::is_none_v<
                    std::remove_cvref_t< T1 >,
                    tag::internal_construct,
                    VanillaCFR,
                    CFRExponentialParameters >
   VanillaCFR(tag::internal_construct, T1&& t, Args&&... args)
       : base(std::forward< T1 >(t), std::forward< Args >(args)...), m_expcfr_params{}
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

   /// activity counters of the pruning engine (regret-based / dynamic-thresholding modes):
   /// windows armed at recommendation time, subtree descents avoided by the traversal gate,
   /// window folds (buffered best-response regret folded back in) and periodic best-response
   /// refresh traversals. Always available; only ever non-zero under the pruning modes.
   struct PruningStats {
      size_t windows_armed = 0;
      size_t skipped_edge_visits = 0;
      size_t window_folds = 0;
      size_t br_refreshes = 0;
   };

   [[nodiscard]] PruningStats pruning_stats() const { return m_pruning_stats; }

   /// activity counters of the lazy-update engine (lazy_update_mode != off): closed segments
   /// (buffered folds + re-recommendation executed) and end-of-iteration recommendations
   /// avoided while an infostate's strategy stayed FROZEN. Always available; only ever
   /// non-zero under CFRLazyUpdateMode::reach_threshold.
   struct LazyStats {
      size_t segment_refreshes = 0;
      size_t skipped_refreshes = 0;
   };

   [[nodiscard]] LazyStats lazy_stats() const { return m_lazy_stats; }

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

   /// ---- pruning engine state (empty cost when pruning_mode == none) ----------------------
   PruningStats m_pruning_stats{};
   /// ---- lazy-update engine state (empty cost when lazy_update_mode == off) ---------------
   /// per-player maps of per-infostate open-segment bookkeeping, keyed by infostate VALUE
   /// (mirrors m_edge_bounds)
   [[no_unique_address]] std::conditional_t<
      config.lazy_update_mode != CFRLazyUpdateMode::off,
      player_hashmap< std::unordered_map<
         info_state_type,
         lazy::SegmentState,
         common::value_hasher< info_state_type >,
         common::value_comparator< info_state_type > > >,
      utils::empty >
      m_lazy_segments{};
   LazyStats m_lazy_stats{};
   /// memoized GLOBAL per-player bounds (B4 trait path, or degenerate fallback)
   player_hashmap< pruning::PayoffBound > m_payoff_bounds{};
   /// per-(infostate,action) probed ranges: lower = L(I) (min payoff below I), upper = U(I,a)
   /// (max payoff below h*a). Filled by the one-shot probe when the env lacks B4 bounds.
   player_hashmap< std::unordered_map<
      info_state_type,
      std::unordered_map< action_type, pruning::PayoffBound, common::value_hasher< action_type > >,
      common::value_hasher< info_state_type >,
      common::value_comparator< info_state_type > > >
      m_edge_bounds{};
   /// root participant order backing _probe_dfs's interval vectors (chance excluded)
   std::vector< Player > m_root_player_order{};

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

   ///////////////////////////////////////////////////////////////////////////////////////////
   ////////////////////// lazy-update segmentation engine (Lazy-CFR) /////////////////////////
   ///////////////////////////////////////////////////////////////////////////////////////////

   /// resolves (creating/resizing on demand) the OPEN-segment state of 'infostate'; the
   /// buffered-regret table is kept index-aligned with 'actions' (the infostate's registry)
   lazy::SegmentState&
   _lazy_segment(const info_state_type& infostate, const std::vector< action_type >& actions);

   /// closes the OPEN segment of 'infostate': folds its buffered counterfactual regret
   /// increments into the minimizer tables via observe(), applies the segment's accumulated
   /// own-reach-weighted average-strategy mass in one deferred step (exact under the frozen
   /// strategy) and flags the end-of-iteration sweep to recompute the recommendation
   void _lazy_fold_segment(const info_state_type& infostate, lazy::SegmentState& seg);

   /// consumes (reads AND clears) the pending-refresh flag of 'infostate's segment; false
   /// when the infostate has no segment state or its open segment was never closed, i.e.
   /// when its strategy must stay frozen for this iteration
   bool _lazy_consume_refresh(const info_state_type& infostate);

   ///////////////////////////////////////////////////////////////////////////////////////////
   ////////////////////// regret-based pruning / dynamic thresholding ////////////////////////
   ///////////////////////////////////////////////////////////////////////////////////////////

   /// lazily resolves (and memoizes) the payoff-bound stand-ins for the paper's U/L quantities:
   /// environments supporting the B4 trait report per-player global bounds; everything else is
   /// probed once from the tree as PER-(infostate,action) terminal-reward ranges -- the faithful
   /// (much tighter) reading of U(I,a) and L(I)
   [[nodiscard]] pruning::PayoffBound
   _edge_bound(const info_state_type& infostate, const action_type& action);

   /// one-shot acquisition of the per-(infostate,action) payoff ranges via a full-tree
   /// enumeration (shared edge-advance mechanics with the best-response walk)
   void _probe_edge_bounds();

   /// recursive worker of _probe_edge_bounds; returns the {lo,hi} interval of every ROOT
   /// player's terminal reward below 'state' (order aligned with m_root_player_order)
   [[nodiscard]] std::vector< pruning::PayoffBound > _probe_dfs(
      world_state_type& state,
      size_t depth,
      player_hashmap< sptr< info_state_type > >& infostates,
      player_hashmap< std::vector< std::pair< observation_type, observation_type > > >&
         observation_buffers
   );

   /// arms pruning windows on an infostate whose post-recommend regret entries cleared the
   /// minimum-skip filter; deadline = T0 + Theorem-1 window, pessimistic eq-(9) tracker seeded
   /// with R^{T0}(I,a)
   template < typename NodeData >
   void _arm_pruning_windows(const info_state_type& infostate, NodeData& node_data);

   /// traversal gate for edge (infostate of 'active_player', action at 'action_idx').
   /// Returns true when the subtree below the edge must be SKIPPED this traversal (the caller
   /// records a deferred window visit); when the edge's window deadline has expired the buffered
   /// best-response regret folds into the minimizer tables here and false is returned so the
   /// caller resumes normal recursion.
   template < typename NodeData >
   bool _rbp_gate(
      std::optional< Player > player_to_update,
      Player active_player,
      NodeData& node_data,
      size_t action_idx,
      const action_type& action,
      InfostateSptrMap& infostates,
      ObservationbufferMap& observation_buffer,
      const world_state_type& state,
      size_t depth
   );

   /// deferred per-visit bookkeeping pushed by the caller once its state_value aggregation is
   /// complete: buffer pi_{-i} * (v_BR - v(I)) into the best-response regret and advance the
   /// eq-(9) pessimistic tracker, folding early when the window provably cannot continue
   template < typename NodeData >
   void _push_window_visit(
      NodeData& node_data,
      Player active_player,
      const info_state_type& infostate,
      size_t action_idx,
      double cf_reach_prob,
      double state_value_for_player
   );

   /// folds a window's buffered best-response regret back into the minimizer tables ("update
   /// the regrets to match this", NIPS'15 sec. 4) and clears the window
   template < typename NodeData >
   void _rbp_fold(NodeData& node_data, size_t action_idx, const action_type& action);

   /// one-edge advancement shared by the best-response walk and the bounds probe: transitions
   /// into the arena slot of depth+1 and applies the observation flush/buffering IN PLACE on
   /// the passed containers (callers needing restoration snapshot beforehand)
   template < typename ActionOrOutcome >
   world_state_type& _br_advance(
      const ActionOrOutcome& action_or_outcome,
      const world_state_type& state,
      size_t depth,
      player_hashmap< sptr< info_state_type > >& infostates,
      player_hashmap< std::vector< std::pair< observation_type, observation_type > > >&
         observation_buffers
   );

   /// value of D(state, action_or_outcome) for 'br_player': best response against the
   /// OPPONENTS' AVERAGE strategies (greedy maxima at own nodes, average-policy expectation at
   /// opponent nodes, chance-probability expectation at chance nodes). Mutates the containers
   /// along each descent but restores them per edge (cheap targeted snapshots, no map copies).
   template < typename ActionOrOutcome >
   double _br_expectimax_from_edge(
      Player br_player,
      const ActionOrOutcome& action_or_outcome,
      const world_state_type& state,
      size_t depth,
      player_hashmap< sptr< info_state_type > >& infostates,
      player_hashmap< std::vector< std::pair< observation_type, observation_type > > >&
         observation_buffers
   );

   /// normalized (current) AVERAGE strategy of an infostate; handles the exponential-weighting
   /// numerator/denominator representation
   [[nodiscard]] std::vector< double > _normalized_average_policy(
      const info_state_type& infostate,
      const std::vector< action_type >& actions
   );
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
 * Lazy-CFR (Zhou et al., "Lazy-CFR: fast and near-optimal regret minimization for extensive
 * games", ICLR 2020, arXiv:1810.04433): vanilla CFR whose per-infoset recommendation step is
 * amortized through reach-budget segmentation -- an infoset's strategy stays FROZEN while its
 * accumulated opponent reach since the last refresh is below 'config.threshold_b'; the
 * buffered counterfactual contributions fold and the regret-matching recommendation is
 * recomputed only at segment close (see CFRLazyUpdateMode::reach_threshold).
 */
template < CFRLazyConfig config, typename Env, typename Policy, typename AveragePolicy >
using LazyCFR = VanillaCFR<
   CFRConfig{
      .update_mode = config.update_mode,
      .regret_minimizing_mode = config.regret_minimizing_mode,
      .weighting_mode = CFRWeightingMode::uniform,
      .pruning_mode = CFRPruningMode::none,
      .lazy_update_mode = CFRLazyUpdateMode::reach_threshold,
      .lazy_update_threshold_b = config.threshold_b},
   Env,
   Policy,
   AveragePolicy >;

/**
 * Lazy-CFR+ (Zhou et al., ICLR 2020, arXiv:1810.04433): the same lazy-update segmentation as
 * rm::LazyCFR riding the RegretMatchingPlus (CFR+) kernel.
 */
template < CFRLazyConfig config, typename Env, typename Policy, typename AveragePolicy >
using LazyCFRPlus = VanillaCFR<
   CFRConfig{
      .update_mode = config.update_mode,
      .regret_minimizing_mode = RegretMinimizingMode::regret_matching_plus,
      .weighting_mode = CFRWeightingMode::uniform,
      .pruning_mode = CFRPruningMode::none,
      .lazy_update_mode = CFRLazyUpdateMode::reach_threshold,
      .lazy_update_threshold_b = config.threshold_b},
   Env,
   Policy,
   AveragePolicy >;

/**
 * @brief InternalRegretCFR: vanilla CFR driven by the swap-basis phi-regret
 * kernel (rm::InternalRegretMatching; Hart-Mas-Colell style internal-regret
 * matching, "RM-X" family). Alternating updates over the uniform weighting
 * mode; see rm::RegretMinimizingMode::internal_regret_matching for the
 * statically enforced configuration constraints. Constructed like
 * factory::make_cfr_vanilla (the generic factory::make_cfr dispatches any
 * rm::CFRConfig carrying RegretMinimizingMode::internal_regret_matching here).
 */
template < typename Env, typename Policy, typename AveragePolicy >
using InternalRegretCFR = VanillaCFR<
   CFRConfig{
      .update_mode = UpdateMode::alternating,
      .regret_minimizing_mode = RegretMinimizingMode::internal_regret_matching,
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
using APCFRPlus = VanillaCFR<
   CFRConfig{
      .update_mode = config.update_mode,
      .regret_minimizing_mode = RegretMinimizingMode::ap_predictive_regret_matching_plus,
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
using P2PCFRPlus = VanillaCFR<
   CFRConfig{
      .update_mode = config.update_mode,
      .regret_minimizing_mode = RegretMinimizingMode::p2p_predictive_regret_matching_plus,
      .weighting_mode = CFRWeightingMode::discounted},
   Env,
   Policy,
   AveragePolicy >;

/**
 * Smooth PCFR+ / Smooth PRM+ (Farina, Grand-Clément, Kroer, Lee, Luo, NeurIPS
 * 2023, arXiv:2305.14709, Algorithm 2): PCFR+ whose predicted-regret vector is
 * kept at 1-norm >= 1 before normalization ("chopping off" the origin). Same
 * construction constraints as PCFRPlus.
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

/**
 * DCFR+ (Xu et al., "Minimizing Weighted Counterfactual Regret with Optimistic
 * Online Mirror Descent", IJCAI 2024, arXiv:2404.13891, sec. 4):
 *    R^t = [ R^{t-1} * (t-1)^alpha / ((t-1)^alpha + 1) + r^t ]^+
 * i.e. RM+-style folding WITH the positive-part alpha discount applied BEFORE
 * adding the instantaneous regret and clipping the sum. Constructed like
 * CFRDiscounted (leading, possibly default rm::CFRDiscountedParameters); paper
 * defaults available via rm::dcfrplus_default_parameters() (alpha = 1.5,
 * gamma = 4).
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
 * PDCFR+ (arXiv:2404.13891, sec. 4): DCFR+ whose recommendations are computed
 * from the persistence-predicted next cumulative regret
 *    R~^{t+1} = [ R^t * t^alpha / (t^alpha + 1) + v^{t+1} ]^+,  v^{t+1} = r^t.
 * Paper defaults via rm::pcfrplus_default_parameters() (alpha = 2.3, gamma = 5).
 */
template < CFRDiscountedConfig config, typename Env, typename Policy, typename AveragePolicy >
using PDCFRPlus = VanillaCFR<
   CFRConfig{
      .update_mode = config.update_mode,
      .regret_minimizing_mode = RegretMinimizingMode::discounted_predictive_regret_matching_plus,
      .weighting_mode = CFRWeightingMode::discounted},
   Env,
   Policy,
   AveragePolicy >;

/**
 * HS-DCFR / HS-PCFR+ / HS-DCFR+ / HS-PDCFR+ (Zhang, McAleer, Sandholm, AAAI
 * 2026, arXiv:2404.09097): NOT distinct solver types -- a hyperparameter
 * schedule is a runtime property of rm::CFRDiscountedParameters. Build them by
 * passing one of the assembled bundles to make_cfr_discounted:
 *   HS-DCFR(n)     -> rm::hs_dcfr_parameters(n, variant)     with
 *                     RegretMinimizingMode::regret_matching
 *   HS-PCFR+(n)    -> rm::hs_pcfrplus_parameters(n, variant) with
 *                     RegretMinimizingMode::predictive_regret_matching_plus
 *   HS-DCFR+(n)    -> rm::hs_dcfrplus_parameters(n, variant) with
 *                     RegretMinimizingMode::discounted_regret_matching_plus
 *   HS-PDCFR+(n)   -> rm::hs_pdcfrplus_parameters(n, variant) with
 *                     RegretMinimizingMode::discounted_predictive_regret_matching_plus
 * where 'variant' selects HSVariant::gamma30 (paper-recommended default) or
 * HSVariant::gamma15, and 'n' is the total planned iteration count (bound into
 * the schedule callables at construction; see discounted_predictive.hpp).
 */

template < CFRExponentialConfig config, typename Env, typename Policy, typename AveragePolicy >
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
