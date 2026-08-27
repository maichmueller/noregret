
#ifndef NOR_CFR_HPP
#define NOR_CFR_HPP

#include <functional>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <named_type.hpp>
#include <optional>
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
#include "nor/rm/action_value_table.hpp"
#include "nor/rm/cfr_tabular/mccfr.hpp"
#include "nor/rm/extragradient.hpp"
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

/// activity statistics of the greedy weighting engine. 'weight_draws' counts the
/// computed iteration weights (one per player update under alternating updates,
/// one per joint iteration under simultaneous updates); 'total_weight'/
/// 'min_weight'/'max_weight'/'last_weight' aggregate the effective (post-floor,
/// post-discard-guard) weights. Under uniform averaging all of these would be
/// identically 1. 'history_discards' counts how often the line search chose a
/// near-infinite weight -- i.e. an iteration that invalidates the accumulated
/// history (realized as a 1e-6 dilation + weight-1 update, see
/// rm::GreedyWeights).
struct GreedyWeightStats {
   size_t weight_draws = 0;
   size_t history_discards = 0;
   double total_weight = 0.;
   double min_weight = std::numeric_limits< double >::infinity();
   double max_weight = 0.;
   double last_weight = 0.;
};

/// solver-side bookkeeping of the greedy weighting engine (Zhang, Lerer & Brown,
/// AAAI 2022). The solver accumulates the total weight w_sum and the number of
/// completed weighted updates t (the paper's 'wsum' and loop counter). Greedy
/// weights is restricted to SIMULTANEOUS updates: a single joint instance draws
/// one weight per iteration from the potential summed over all players -- exactly
/// the published scheme.
struct GreedySolverState {
   GreedyWeightStats stats{};
   double wsum = 0.;
   size_t updates = 0;
};

/// reusable line-search scratch space so that repeated iterations do not reallocate:
/// 'regret_pairs' holds the (cumulative regret R_j, buffered instantaneous regret r_j)
/// aggregates of the current sweep, 'breakpoints' the sign-change candidates of the
/// line search
struct GreedyScratchBuffers {
   std::vector< std::pair< double, double > > regret_pairs{};
   std::vector< double > breakpoints{};
};

/// output buffer standing in for the former TabularPolicy write-target of the
/// minimizers' recommend()/finalize_iteration hooks (the node records are the
/// single source of truth; their 'current_strategy' cache is published from
/// here after each recommendation). Supports both PolicyOut idioms the
/// minimizer library uses:
///   - 'out[action] = p' assignments (all recommend kernels)
///   - pair-style iteration mutating 'entry.second'
///     (Thresholded::_apply_threshold)
template < concepts::action Action >
struct recommendation_scratch {
   std::vector< std::pair< Action, double > > entries{};

   /// aligns the buffer with the infostate's registry order and zeroes values
   void prepare(const std::vector< Action >& actions)
   {
      entries.clear();
      entries.reserve(actions.size());
      for(const auto& action : actions) {
         entries.emplace_back(action, 0.);
      }
   }

   double& operator[](const Action& action)
   {
      const auto found = std::ranges::find(entries, action, &std::pair< Action, double >::first);
      if(found == entries.end()) {
         throw std::out_of_range("recommendation target is not a registered action");
      }
      return found->second;
   }

   auto begin() { return entries.begin(); }
   auto end() { return entries.end(); }
};

/// AveragePolicy write-view for ExponentialCFR::finalize_iteration: routes the
/// numerator accumulations into the owning node record's strategy-sum table,
/// creating the uniform baseline on first touch exactly like the former
/// fetch_policy<average> entry creation did.
template < typename NodeRecord >
struct average_strategy_accumulator {
   NodeRecord* node;

   double& operator[](const auto& action)
   {
      node->activate_average();
      auto& sums = node->strategy_sum();
      return sums[node->index_of(action)];
   }
};

}  // namespace detail

/**
 * Fixed-policy carrier of the warm-start pre-play phase (see
 * rm::CFRConfig::warm_start_iterations). During the phase the solver forces every player's
 * PLAYED strategy to the distribution this selector reports; counterfactual regret updates
 * run unmodified, seeding the cumulative regret tables away from zero (the pre-play rounds
 * contribute nothing to the average strategy).
 *
 * CAVEAT on that last clause: it describes the EAGER average-strategy increment exactly,
 * but two deferred accumulation paths carry pre-play own-reach mass into their folds --
 * LazyCFR's segment accumulator and GreedyWeights' reach_prob_snapshot feed
 * unconditionally by design. This reproduces the historical (develop) arithmetic
 * bit-for-bit; golden-trajectory reproducibility was chosen over contract purity.
 *
 * The 'distribution' callable receives an infostate (which carries its owner) and that
 * infostate's registered actions and must return the FULL normalized action distribution for
 * it. An EMPTY callable selects the default uniform distribution over the legal actions.
 */
template < typename InfoState, typename Action >
struct WarmStartPolicy {
   using distribution_type = std::unordered_map< Action, double >;

   /// custom fixed-profile selector; empty => uniform over the infostate's legal actions
   std::function< distribution_type(const InfoState&, const std::vector< Action >&) >
      distribution{};
};

/// resolves the warm-start policy selector type of an environment
template < typename Env >
using warm_start_policy_selector_t = WarmStartPolicy<
   auto_info_state_type< Env >,
   auto_action_type< Env > >;

/**
 * Fixed-opposition carrier of opponent-aware solving (RNR / DBR; see
 * rm::CFROpponentBlendMode::per_infostate_blend). During EVERY traversal visit to an
 * infostate of a modeled player, the PLAYED edge probabilities are taken from the convex
 * blend of the fixed model distribution reported by this selector and the player's own
 * current-policy recommendation:
 *
 *    played(I, a) <- P(I) * model(I, a) + (1 - P(I)) * current(I, a)
 *
 * The mixing is applied PER VISIT and restored afterwards: the stored current-policy tables
 * keep holding the UNBLENDED free component (the regret minimizer's raw recommendation), so
 * revisits of one infostate within a traversal -- every chance branch above the infostate
 * produces one -- blend the SAME clean component instead of compounding.
 *
 * Counterfactual regret updates are NOT modified: everybody's counterfactual values measure
 * the actual (blended) play, i.e. exactly the modified game of Johanson, Zinkevich &
 * Bowling, NIPS 2007 / Johanson & Bowling, AISTATS 2009. Two documented bookkeeping
 * deviations follow from the visit-scoped realization:
 *   - the MODELED player's average-policy accumulation happens after the restoration and
 *     therefore tracks the FREE component only (their realized-play profile is trivially
 *     reconstructible downstream as P(I)*model + (1-P(I))*average); only the responding
 *     player's average is consumed by the entry points,
 *   - the modeled player's instantaneous regret deltas mix a baseline over the blended
 *     distribution; up to the additive Pconf term these coincide with the external-regret
 *     deltas of the free component (positive rescalings do not move regret matching's
 *     recommendation).
 *
 * The 'blend' callable receives an infostate (carrying its owner) and that infostate's
 * registered actions. Returning nullopt marks the infostate as UNCONSTRAINED (pure table
 * play); returning a spec forces the blend with 'forced_probability' in [0, 1] and requires
 * a FULL, normalized 'model_distribution' over the registered actions (violations throw). An
 * EMPTY callable disables blending entirely.
 */
template < typename InfoState, typename Action >
struct OpponentBlendPolicy {
   struct BlendSpec {
      /// per-infostate forcing weight P(I): probability mass played from the model (RNR: the
      /// global p; DBR: the confidence Pconf(I))
      double forced_probability = 0.;
      /// full normalized action distribution of the opponent model at this infostate
      std::unordered_map< Action, double, common::value_hasher< Action > > model_distribution{};
   };

   /// custom per-infostate blend selector; empty => no blending anywhere
   std::function< std::optional< BlendSpec >(const InfoState&, const std::vector< Action >&) >
      blend{};
};

/// resolves the opponent-blend policy selector type of an environment
template < typename Env >
using opponent_blend_policy_selector_t = OpponentBlendPolicy<
   auto_info_state_type< Env >,
   auto_action_type< Env > >;

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
      "The configuration check did not return TRUE. This solver was instantiated with a "
      "config combination that rm::CFRConfig documents as unanalyzed or broken -- see "
      "the clause comments above each 'return false' in detail::sanity_check_cfr_config "
      "(rm/cfr_tabular/cfr.tcc). Usual culprits: predictive/discounted kernels outside "
      "alternating full-tree discounted weighting; extragradient outside its analyzed "
      "core config; lazy segmentation with non-uniform weighting / pruning / those "
      "kernels; constrained RM+ with non-uniform weighting, pruning, lazy or warm start; "
      "the internal-regret kernel with non-uniform weighting, pruning or warm start; "
      "greedy weights with alternating updates or dynamic thresholding; regret-based "
      "pruning outside alternating + uniform + RM+; exponential/greedy weighting under "
      "partial pruning; exponential weighting under warm start."
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

   // The B8 floor-overriding trait requires the buffered (paper-exact) kernel path of
   // rm::ConstrainedRMPlus, which is compiled only for positive uniform floors: with
   // perturbation_floor == 0 the kernel degenerates bit-for-bit to plain RM+, whose
   // arithmetic cannot honor environment-provided floors at all.
   static_assert(
      config.regret_minimizing_mode != RegretMinimizingMode::constrained_regret_matching_plus
         or config.perturbation_floor > 0.
         or not concepts::has::method::
               action_probability_floors< Env, info_state_type, action_type >,
      "environments providing 'action_probability_floors' (B8 trait) combined with "
      "RegretMinimizingMode::constrained_regret_matching_plus require a positive "
      "CFRConfig::perturbation_floor"
   );
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

   /// attaches the opponent-model blend selector of the RNR/DBR family (see
   /// OpponentBlendPolicy). Only participates when the configuration enables
   /// CFROpponentBlendMode::per_infostate_blend; with the mode off a passed selector is a hard
   /// compile error instead of being silently ignored. The selector is forwarded as the FIRST
   /// constructor argument, ahead of all base-class arguments.
   template < typename... Args >
      requires(config.opponent_blend_mode != CFROpponentBlendMode::off)
   VanillaCFR(
      tag::internal_construct,
      opponent_blend_policy_selector_t< Env > selector,
      Args&&... args
   )
       : base(std::forward< Args >(args)...), m_opponent_blend(std::move(selector))
   {
   }

   /// attaches a custom fixed warm-start policy (see WarmStartPolicy). Only participates when
   /// the configuration actually enables the warm-start pre-play phase; with
   /// warm_start_iterations == 0 a passed selector is a hard compile error instead of being
   /// silently ignored. The selector is forwarded as the FIRST constructor argument, ahead of
   /// all base-class arguments.
   template < typename... Args >
      requires(config.warm_start_iterations > 0)
   VanillaCFR(tag::internal_construct, warm_start_policy_selector_t< Env > selector, Args&&... args)
       : base(std::forward< Args >(args)...), m_warm_start_policy(std::move(selector))
   {
   }

   ////////////////////////////////////
   /// API: public member functions ///
   ////////////////////////////////////

  public:
   /// import public getters

   using base::env;
   using base::iteration;
   using base::cycle;
   using base::root_state;

   /// the CURRENT policy profile, materialized from the infostate node
   /// records (D1: the records are the single source of truth; this view is
   /// rebuilt lazily whenever the records changed since the last call).
   /// CONTRACT: node records only exist for infosets visited by a traversal,
   /// so -- before the FIRST initializing traversal (or when zero iterations
   /// have run) -- this view is EMPTY; it never contains pre-play/uniform
   /// defaults (warm-start forced play happens at the fetch point and does
   /// not create records either). Callers wanting a complete profile must run
   /// at least one iteration first.
   /// SHARP EDGE: references/iterators taken from a returned view stay valid
   /// only until the next SOLVER MOVE that dirties the records (iterate(),
   /// finalize_iteration, ...); the very next accessor call after such a move
   /// CLEARS AND REBUILDS the whole view in place, dangling everything held
   /// across the rebuild. Copy out anything you need to keep.
   const auto& policy() const
   {
      if(m_curr_view_dirty) {
         _materialize_current_policy();
         m_curr_view_dirty = false;
      }
      return m_curr_policy_view;
   }

   /// the AVERAGE policy profile, materialized from the node records'
   /// cumulative strategy sums (raw unnormalized numerators, identical to the
   /// former table contents). Same CONTRACT as policy(): only infosets touched
   /// by at least one completed, average-initializing traversal appear here --
   /// before the first such iteration the view is empty. Same SHARP EDGE as
   /// policy(): references into the view invalidate on the rebuild triggered
   /// by the first accessor call after any solver move.
   const auto& average_policy() const
      requires(config.weighting_mode != CFRWeightingMode::exponential)
   {
      if(m_avg_view_dirty) {
         _materialize_average_policy();
         m_avg_view_dirty = false;
      }
      return m_avg_policy_view;
   }

   auto average_policy() const
      requires(config.weighting_mode == CFRWeightingMode::exponential)
   {
      // we need to build the average policy on demand as the denominator is no
      // longer attainable via mere normalization, but is stored separately.
      // The numerator lives in the node records' strategy sums; dividing by
      // the per-action denominators reproduces the former table division
      // entry-by-entry.
      player_hashmap< AveragePolicy > avg_policy_out;
      for(auto player : env().players(root_state()) | utils::is_actual_player_filter) {
         avg_policy_out.emplace(player, AveragePolicy());
      }
      for(auto& [infostate_ptr, node] : m_infonode) {
         if(not node.average_active()) {
            continue;
         }
         const auto& actions = node.actions();
         const auto& action_policy_denominator = node.data().avg_policy_denominator;
         auto& action_policy = avg_policy_out[infostate_ptr->player()](
            *infostate_ptr, actions, typename base::zero_policy_type{}
         );
         const auto& sums = node.strategy_sum();
         for(auto [idx, action] : std::views::enumerate(actions)) {
            action_policy[action] = sums[idx] / action_policy_denominator[idx];
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

   /// true iff global iteration 'iteration' lies inside the warm-start pre-play phase
   /// (see rm::CFRConfig::warm_start_iterations). Always false when the phase is disabled.
   [[nodiscard]] static constexpr bool warm_start_active(size_t iteration)
   {
      return config.warm_start_iterations > 0 and iteration < config.warm_start_iterations;
   }
   /// whether THIS solver currently runs inside the warm-start pre-play phase
   [[nodiscard]] bool in_warm_start() const { return warm_start_active(iteration()); }

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

   /// activity counters of the extragradient engine (extragradient_mode !=
   /// off): anchor probe and real update traversals, exactly one of each per
   /// global iteration. Always available; only ever non-zero under
   /// CFRExtragradientMode::anchor_probe.
   using ExtragradientStats = rm::ExtragradientStats;

   [[nodiscard]] ExtragradientStats extragradient_stats() const { return m_extragradient_stats; }

   /**
    * @brief read-only visitor over every REGISTERED infostate's cumulative
    * counterfactual regret table (the 'regret' member of the configured
    * minimizer's node data, index-aligned to each infostate's action registry).
    *
    * Enables runtime feature extractors -- e.g. the rm::ddcfr dynamic-
    * discounting layer's regret statistics -- without exposing mutable solver
    * internals. Traversal follows the infostate hashmap's order; callers must
    * not rely on any specific order (aggregations are order-insensitive up to
    * fp associativity). Empty for solvers that have not run an initializing
    * traversal yet.
    */
   template < typename Fn >
   void visit_regret_tables(Fn&& fn) const
   {
      for(const auto& [infostate_ptr, node] : m_infonode) {
         fn(node.data().regret);
      }
   }

   /// activity statistics of the greedy weighting engine (only available when
   /// weighting_mode == greedy) -- see rm::detail::GreedyWeightStats
   using GreedyWeightStats = detail::GreedyWeightStats;

   [[nodiscard]] const GreedyWeightStats& greedy_weight_stats() const
      requires(config.weighting_mode == CFRWeightingMode::greedy)
   {
      return m_greedy_state.stats;
   }

   /**
    * @brief updates the regret and policy tables of the infostate with the state-values.
    */
   void update_regret_and_policy(
      const info_state_type& infostate,
      const ReachProbabilityMap& reach_probability,
      const StateValueMap& state_value,
      const ActionValueTable< action_variant_type >& action_value_map
   );

  private:
   ////////////////////////////////
   /// private member functions ///
   ////////////////////////////////

   [[nodiscard]] inline auto& _infonodes() { return m_infonode; }
   /// Throwing lookups (mirroring MCCFR's): node records populate only during the first
   /// initializing traversal, so a lookup on an untrained solver (e.g. game_value() before
   /// any iterate()) must fail loudly with std::out_of_range instead of dereferencing an
   /// end() iterator (UB).
   [[nodiscard]] inline auto& _infonode(const info_state_type& infostate) const
   {
      auto found = m_infonode.find(infostate);
      if(found == m_infonode.end()) {
         throw std::out_of_range{
            "VanillaCFR: no infostate record found -- records are populated by the "
            "first initializing traversal; run at least one iteration before "
            "querying infoset data."};
      }
      return found->second;
   }
   [[nodiscard]] inline auto& _infonode(const info_state_type& infostate)
   {
      auto found = m_infonode.find(infostate);
      if(found == m_infonode.end()) {
         throw std::out_of_range{
            "VanillaCFR: no infostate record found -- records are populated by the "
            "first initializing traversal; run at least one iteration before "
            "querying infoset data."};
      }
      return found->second;
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
   /// Greedy-weights solver state: accumulated weight sum, update count and the
   /// weight statistics (single joint instance; greedy weights is
   /// simultaneous-updates only)
   [[no_unique_address]] std::conditional_t<
      config.weighting_mode == CFRWeightingMode::greedy,
      detail::GreedySolverState,
      utils::empty >
      m_greedy_state;
   /// scratch buffers reused by the greedy line search across iterations (avoids
   /// per-iteration reallocation)
   [[no_unique_address]] std::conditional_t<
      config.weighting_mode == CFRWeightingMode::greedy,
      detail::GreedyScratchBuffers,
      utils::empty >
      m_greedy_scratch;
   /// the actual regret minimizing method we will apply on the infostates
   [[no_unique_address]] minimizer_type m_regret_minimizer{};

   /// fixed-policy carrier of the warm-start pre-play phase (empty cost when
   /// config.warm_start_iterations == 0)
   [[no_unique_address]] std::conditional_t<
      (config.warm_start_iterations > 0),
      WarmStartPolicy< info_state_type, action_type >,
      utils::empty >
      m_warm_start_policy{};

   /// fixed-opposition carrier of opponent-aware solving (RNR/DBR; empty cost when
   /// config.opponent_blend_mode == off)
   [[no_unique_address]] std::conditional_t<
      config.opponent_blend_mode != CFROpponentBlendMode::off,
      OpponentBlendPolicy< info_state_type, action_type >,
      utils::empty >
      m_opponent_blend{};

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
   /// ---- extragradient engine state (empty cost when extragradient_mode == off) ----------
   /// per-player maps of deferred counterfactual-regret increment buffers,
   /// keyed by infostate VALUE (mirrors m_lazy_segments); index-aligned with
   /// the owning infostate's action registry. Temporally shared by the anchor
   /// probe (pass 1) and the real traversal's deferred updates (pass 2)
   [[no_unique_address]] std::conditional_t<
      config.extragradient_mode != CFRExtragradientMode::off,
      player_hashmap< std::unordered_map<
         info_state_type,
         std::vector< double >,
         common::value_hasher< info_state_type >,
         common::value_comparator< info_state_type > > >,
      utils::empty >
      m_ex_regret_buffers{};
   ExtragradientStats m_extragradient_stats{};
   /// memoized GLOBAL per-player bounds (B4 trait path, or degenerate fallback)
   player_hashmap< pruning::PayoffBound > m_payoff_bounds{};
   /// per-(infostate,action) probed ranges (lower = L(I), upper = U(I,a)) live INSIDE the
   /// infostate node records (InfostateNodeData::edge_bounds), index-aligned with their action
   /// registries; filled by the one-shot probe when the env lacks B4 bounds.
   /// root participant order backing _probe_dfs's interval vectors (chance excluded)
   std::vector< Player > m_root_player_order{};

   /// ---- touched-infoset sweep state (D3) -------------------------------------------------
   /// nodes whose update_regret_and_policy ran during the current iteration,
   /// in pre-order traversal order; consumed by _initiate_regret_minimization
   /// and cleared every iteration (memory bounded per cycle).
   /// NOTE: the infostate is stored BY VALUE on purpose. During traversals the
   /// live infostate objects are per-edge clones swapped into the seat-indexed
   /// slot table and destroyed again at the recursion-boundary restore (only
   /// the initializing run's node registration keeps a shared owner alive);
   /// a raw pointer would dangle by sweep time (first surfaced as lazy-CFR
   /// segment lookups missing their own freshly-folded segments).
   std::vector< std::pair< info_state_type, infostate_data_type* > > m_touched_infonodes{};
   /// monotone stamp compared against InfostateNodeData::sweep_stamp to keep
   /// the touched list free of duplicates within one iteration
   size_t m_sweep_clock = 0;
   /// D4: set once the infoset population has been pre-sized after the first
   /// full traversal
   bool m_infonode_presized = false;

   /// ---- lazily materialized policy views (D1) --------------------------------------------
   mutable player_hashmap< Policy > m_curr_policy_view{};
   mutable player_hashmap< AveragePolicy > m_avg_policy_view{};
   mutable bool m_curr_view_dirty = true;
   mutable bool m_avg_view_dirty = true;
   /// scratch output buffer of the minimizers' recommend/finalize calls
   detail::recommendation_scratch< action_type > m_recommend_scratch{};

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
   /// EXTRAGRADIENT anchor-probe mode: 'probe_pass = true' turns the traversal
   /// into a values-only measurement pass -- counterfactual increments of the
   /// updating player are buffered for the intermediate step instead of being
   /// applied, and no minimizer/average state is mutated. All existing
   /// configurations instantiate with the default false (compiled out
   /// entirely).
   template < bool initialize_infonodes, bool use_current_policy = true, bool probe_pass = false >
   StateValueMap _traverse(
      std::optional< Player > player_to_update,
      world_state_type& state,
      size_t depth,
      ReachProbabilityMap& reach_probability,
      ObservationbufferMap& observation_buffer,
      InfostateSptrMap& infostates,
      ActionValueArena< action_variant_type >& action_value_arena
   );

   template < bool initialize_infonodes, bool use_current_policy = true, bool probe_pass = false >
   void _traverse_player_actions(
      std::optional< Player > player_to_update,
      Player active_player,
      world_state_type& state,
      size_t depth,
      ReachProbabilityMap& reach_probability,
      ObservationbufferMap& observation_buffer,
      InfostateSptrMap& infostates,
      StateValueMap& state_value,
      ActionValueTable< action_variant_type >& action_value,
      ActionValueArena< action_variant_type >& action_value_arena
   );

   template < bool initialize_infonodes, bool use_current_policy = true, bool probe_pass = false >
   void _traverse_chance_actions(
      std::optional< Player > player_to_update,
      Player active_player,
      world_state_type& state,
      size_t depth,
      ReachProbabilityMap& reach_probability,
      ObservationbufferMap& observation_buffer,
      InfostateSptrMap& infostates,
      StateValueMap& state_value,
      ActionValueTable< action_variant_type >& action_value,
      ActionValueArena< action_variant_type >& action_value_arena
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
   void _invoke_regret_minimizer(
      [[maybe_unused]] const info_state_type& infostate,
      infostate_data_type& istate_data
   );

   /// publishes the scratch recommendation buffer into the node record's
   /// current-strategy cache (the traversal-visible current policy)
   void _publish_recommendation(infostate_data_type& istate_data);

   /// registers 'node' for the end-of-iteration regret-minimization sweep
   /// (once per iteration; stamp-deduplicated)
   void _touch(const info_state_type& infostate, infostate_data_type& node);

   /// overlays user-seeded starting strategies from the constructor-time
   /// policy tables onto a freshly created node record
   void _seed_node_from_user_tables(
      Player active_player,
      const info_state_type& infostate,
      infostate_data_type& node
   );

   /// rebuilds the materialized policy views from the node records
   void _materialize_current_policy() const;
   void _materialize_average_policy() const;

   /// WARM START pre-play phase worker: overwrites the CURRENT-policy cache 'node' (the
   /// infostate's node record) with the fixed warm-start distribution (uniform by default,
   /// else as reported by the attached WarmStartPolicy selector). Invoked at every traversal
   /// visit while warm_start_active(_iteration()) holds, so all players play the fixed profile
   /// during the phase while their regret/average updates stay unmodified.
   void _force_warm_start_policy(
      const info_state_type& infostate,
      const std::vector< action_type >& actions,
      infostate_data_type& node
   );

   /// OPPONENT-AWARE blend worker (RNR/DBR): mixes the fixed opponent-model distribution into
   /// the CURRENT policy table 'action_policy' of a MODELED player's infostate at the fetch
   /// point,    action_policy <- P(I) * model(I) + (1 - P(I)) * action_policy,
   /// exactly like _force_warm_start_policy but with a per-infostate weight/model pair reported
   /// by the attached OpponentBlendPolicy selector. Returns the PRE-BLEND entries it overwrote;
   /// the CALLER must restore them into the table once this visit's edge probabilities have
   /// been consumed, keeping the stored tables free-component-only so that revisit visits --
   /// one per chance branch above the infostate -- never compound a previously blended value.
   /// Returns an empty vector when nothing was blended.
   template < typename ActionPolicyTable >
   [[nodiscard]] std::vector< std::pair< action_type, double > > _apply_opponent_blend(
      const info_state_type& infostate,
      const std::vector< action_type >& actions,
      ActionPolicyTable& action_policy
   );

   /// behaviorally-constrained kernels: refreshes the per-action probability floors of the
   /// infostate's node data ahead of every recommendation. With an environment exposing the
   /// B8 trait 'action_probability_floors' the reported vector REPLACES the seeded uniform
   /// floor; without it the floors registered at table creation (the uniform config value)
   /// already are authoritative and this is a no-op.
   template < typename NodeData >
   void _refresh_probability_floors(const info_state_type& infostate, NodeData& node_data);

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

   /// end-of-iteration greedy-weights sweep (weighting_mode == greedy): pass 1 aggregates
   /// the (cumulative regret, buffered instantaneous regret) pairs of every swept infostate
   /// and computes the paper's greedy iteration weight by an exact piecewise line search
   /// over the aggregated potential; pass 2 folds the weighted regret/average-policy updates
   /// into all swept infostates and refreshes their recommendations. See rm::GreedyWeights
   /// for references.
   template < typename NodeView >
   void _finalize_greedy_iteration(NodeView&& node_view);

   ///////////////////////////////////////////////////////////////////////////////////////////
   ////////////////////// extragradient engine (ExRM+ / Clairvoyant CFR) /////////////////////
   ///////////////////////////////////////////////////////////////////////////////////////////

   /// resolves (creating/resizing/zeroing on demand) the deferred-increment
   /// buffer of 'infostate'; kept index-aligned with 'actions' (the infostate's
   /// registry). Serves BOTH engine phases temporally disjointly: the anchor
   /// probe fills it first (consumed + cleared by the intermediate step), the
   /// real traversal fills it second (consumed + erased by the end-of-iteration
   /// fold)
   std::vector< double >& _extragradient_buffer(const info_state_type& infostate);

   /// ANCHOR PROBE collection: buffers one visited history's counterfactual
   /// regret increments for the updating player WITHOUT touching any minimizer
   /// or average-policy state (values-only measurement of F at g(z^{t-1}))
   void _probe_collect(
      const info_state_type& infostate,
      const ReachProbabilityMap& reach_probability,
      const StateValueMap& state_value,
      const ActionValueTable< action_variant_type >& action_value_map
   );

   /// INTERMEDIATE STEP between the two traversals: derives
   ///    w^t(I) = [z^{t-1}_I + eta * r_anchor(I)]^+
   /// per infostate of 'update_player' from the STORED cumulative table plus
   /// the eta-scaled anchor-probe buffer and writes the normalized w^t into the
   /// current-policy tables (the played intermediate strategy g(w^t)). The
   /// stored table z^{t-1} is left untouched so both proxes of Algorithm 5 stay
   /// anchored at it; consumes and clears the anchor buffers.
   void _extragradient_intermediate_recommendation(Player update_player);

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

template < CFRPlusConfig config, typename Env, typename Policy, typename AveragePolicy >
using CFRPlus = VanillaCFR<
   CFRConfig{
      .update_mode = config.update_mode,
      .regret_minimizing_mode = RegretMinimizingMode::regret_matching_plus,
      .weighting_mode = CFRWeightingMode::uniform,
      .warm_start_iterations = config.warm_start_iterations},
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
 * Extragradient RM+ (Farina, Grand-Clément, Kroer, Lee, Luo, NeurIPS 2023,
 * arXiv:2305.14709, Algorithm 5) -- a.k.a. Clairvoyant CFR, the
 * single-fixed-point instantiation of Conceptual RM+ the authors evaluate on
 * extensive-form games: TWO traversals per global iteration. An anchor probe
 * under g(z^{t-1}) measures F there, the intermediate strategy
 * g(w^t) = g([z^{t-1} + eta r_anchor]^+) is derived, and the real traversal
 * under g(w^t) supplies the anchored fold z^t = [z^{t-1} + eta r_real]^+.
 * The intermediate strategy is what enters the average policy (Algorithm 5
 * line 6). See rm/extragradient.hpp for the exact mapping and the documented
 * deviations from the paper's simultaneous-update theory.
 */
template < CFRExtragradientConfig config, typename Env, typename Policy, typename AveragePolicy >
using ExtragradientCFR = VanillaCFR<
   CFRConfig{
      .update_mode = config.update_mode,
      .regret_minimizing_mode = config.regret_minimizing_mode,
      .weighting_mode = CFRWeightingMode::uniform,
      .pruning_mode = CFRPruningMode::none,
      .extragradient_stepsize = config.stepsize,
      .extragradient_mode = CFRExtragradientMode::anchor_probe},
   Env,
   Policy,
   AveragePolicy >;

/// the paper's "Clairvoyant CFR" naming for the same algorithm: Conceptual-RM+
/// with its fixed-point equation approximated by ONE extrapolation step (= the
/// extragradient scheme above; arXiv:2305.14709, secs. 5-6)
template < CFRExtragradientConfig config, typename Env, typename Policy, typename AveragePolicy >
using ClairvoyantCFR = ExtragradientCFR< config, Env, Policy, AveragePolicy >;

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

/**
 * GreedyWeightedCFR (Zhang, Lerer & Brown, "Equilibrium Finding in Normal-Form
 * Games Via Greedy Regret Minimization", AAAI 2022, arXiv:2204.04826):
 * vanilla CFR whose completed iterations are weighed dynamically by greedily
 * minimizing the aggregated counterfactual-regret potential -- see
 * rm::CFRWeightingMode::greedy and rm::GreedyWeights for the full formulation.
 * The floor knob lives on rm::CFRConfig::greedy_weight_floor_fraction (paper
 * default 1.0 = "100% of the average weight accrued thus far", Appendix F).
 */
template < CFRGreedyConfig config, typename Env, typename Policy, typename AveragePolicy >
using GreedyWeightedCFR = VanillaCFR<
   CFRConfig{
      .update_mode = config.update_mode,
      .regret_minimizing_mode = config.regret_minimizing_mode,
      .weighting_mode = CFRWeightingMode::greedy},
   Env,
   Policy,
   AveragePolicy >;

/**
 * Linear CFR (Brown & Sandholm, "Solving Imperfect-Information Games via
 * Discounted Regret Minimization", AAAI 2014): the DCFR family member with all
 * exponents fixed to one (alpha = beta = gamma = 1). Backed by the STATELESS
 * rm::LinearCFR weighting carrier instead of the parameterized discounted
 * machinery, which additionally makes this alias composable with
 * CFRPruningMode::dynamic_thresholding (see sanity_check_cfr_config). The
 * schedules are bit-for-bit identical to the historical spelling that injected
 * CFRDiscountedParameters{alpha=1, beta=1, gamma=1} into the discounted carrier.
 */
template < CFRLinearConfig config, typename Env, typename Policy, typename AveragePolicy >
using CFRLinear = VanillaCFR<
   CFRConfig{
      .update_mode = config.update_mode,
      .regret_minimizing_mode = config.regret_minimizing_mode,
      .weighting_mode = CFRWeightingMode::linear},
   Env,
   Policy,
   AveragePolicy >;

///////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// MCCFR+ (predictive OS-MCCFR) ///////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief configuration carrier of the MCCFR+ family: OUTCOME-SAMPLING MCCFR
 * whose per-infoset regret update is driven by a predictive RM+ kernel of the
 * PCFR+ family instead of plain regret matching.
 *
 * The recommendation step derives the strategy from theta(a) =
 * max(0, clip(z)(a) + s*rho(a)) where rho is the last realized
 * importance-weighted sampled instantaneous regret vector (persistence
 * prediction on samples) and s the kernel's prediction scale; the increments
 * themselves are the UNCHANGED conditionally-unbiased OS-MCCFR estimator.
 * See rm::MCCFRMinimizer for the precise statement of which guarantees survive
 * this composition (unbiasedness + pathwise PRM+ external-regret machinery)
 * and what degrades under sampling (prediction quality, quadratic averaging).
 *
 * 'regret_minimizing_mode' selects the kernel:
 *   predictive_regret_matching_plus            -> PCFR+      (s = 1)
 *   sap_predictive_regret_matching_plus        -> SAPCFR+    (s = 1/3)
 *   ap_predictive_regret_matching_plus         -> APCFR+     (adaptive per-infostate s)
 *   p2p_predictive_regret_matching_plus        -> P2PCFR+    (s = 1/6)
 *   smooth_predictive_regret_matching_plus     -> Smooth-PRM+ (origin floor)
 *   stable_predictive_regret_matching_plus     -> Stable-PRM+ (componentwise restart)
 * All modes statically require MCCFRAlgorithmMode::outcome_sampling.
 */
struct MCCFRPlusConfig {
   UpdateMode update_mode = UpdateMode::alternating;
   /// average-policy accumulation scheme of the sampling engine (lazy weighting
   /// reproduces Lanctot's unbiased deferred scheme; kept orthogonal to the
   /// regret kernel on purpose -- see the composition-theory notes in
   /// rm::MCCFRMinimizer for why PCFR+'s quadratic averaging is NOT imposed here)
   MCCFRWeightingMode weighting = MCCFRWeightingMode::lazy;
   /// the predictive RM+ kernel driving the regret updates (must be one of the
   /// predictive_* modes above; anything else either falls back to plain RM or
   /// trips the static admissibility checks)
   RegretMinimizingMode
      regret_minimizing_mode = RegretMinimizingMode::predictive_regret_matching_plus;
};

template < MCCFRPlusConfig config, typename Env, typename Policy, typename AveragePolicy >
using MCCFRPlus = MCCFR<
   MCCFRConfig{
      .update_mode = config.update_mode,
      .algorithm = MCCFRAlgorithmMode::outcome_sampling,
      .weighting = config.weighting,
      .regret_minimizing_mode = config.regret_minimizing_mode},
   Env,
   Policy,
   AveragePolicy >;

}  // namespace nor::rm

// include the actual template implementations of this class
#include "cfr.tcc"

#endif  // NOR_CFR_HPP
