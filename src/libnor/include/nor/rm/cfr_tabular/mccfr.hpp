
#ifndef NOR_MCCFR_HPP
#define NOR_MCCFR_HPP

#include <deque>
#include <iostream>
#include <list>
#include <map>
#include <named_type.hpp>
#include <queue>
#include <ranges>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cfr_base.hpp"
#include "cfr_config.hpp"
#include "common/common.hpp"
#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "nor/rm/action_value_table.hpp"
#include "nor/rm/cfr_tabular/solver_operations.hpp"
#include "nor/rm/forest.hpp"
#include "nor/rm/minimizers/minimizers.hpp"
#include "nor/rm/node.hpp"
#include "nor/rm/rm_utils.hpp"
#include "nor/rm/sampling_rules.hpp"
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
template <
   MCCFRConfig config,
   typename Env,
   typename Policy,
   typename AveragePolicy,
   typename SamplingRule = EpsilonOnPolicySamplingRule >
class MCCFR:
    public detail::TabularSolverOperations<
       MCCFR< config, Env, Policy, AveragePolicy, SamplingRule >,
       auto_info_state_type< Env >,
       auto_action_type< Env > >,
    public TabularCFRBase<
       config.update_mode == UpdateMode::alternating,
       Env,
       Policy,
       AveragePolicy > {
   ////////////////////////////
   /// API: public typedefs ///
   ////////////////////////////
  public:
   friend class detail::TabularSolverOperations<
      MCCFR< config, Env, Policy, AveragePolicy, SamplingRule >,
      auto_info_state_type< Env >,
      auto_action_type< Env > >;

   /// aliases for the template types
   using base =
      TabularCFRBase< config.update_mode == UpdateMode::alternating, Env, Policy, AveragePolicy >;
   using operation_layer = detail::TabularSolverOperations<
      MCCFR< config, Env, Policy, AveragePolicy, SamplingRule >,
      auto_info_state_type< Env >,
      auto_action_type< Env > >;

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

   /// resolved variance-reduction mode of this instantiation (legacy boolean
   /// shim applied; see 'effective_variance_reduction')
   static constexpr VarianceReductionMode vr_mode = effective_variance_reduction(config);
   /// B8: whether the probing value estimator (Gibson et al., AAAI 2012) is
   /// attached to this instantiation via a tagged rm::ProbingSamplingRule
   static constexpr bool probing_active = probing_sampling_rule< SamplingRule >;
   /// ESCHER-style history values V(h) keyed by world-state-edge hashes
   static constexpr bool vr_history_active = vr_mode == VarianceReductionMode::history_value;
   /// predictive baseline maintenance (Davis et al., ICML 2020, eq (8)) needs a
   /// side channel carrying the next-strategy value stream up the recursion
   static constexpr bool vr_predictive_active = config.baseline_update_rule
                                                   == BaselineUpdateRule::predictive
                                                and vr_mode != VarianceReductionMode::none;

   /// storage of the ESCHER history-value function: one scalar per visited
   /// world-state EDGE (h --a--> h a) at which the updating player acts. Keys
   /// are a rolling hash of the sampled trajectory prefix combined with the
   /// action's registry index; entries materialize lazily on first regression.
   struct HistoryValueStore {
      std::unordered_map< size_t, double > edge_values{};
   };
   /// strong-types for player based maps
   using WeightMap = fluent::NamedType< PlayerValueTable, struct weight_map_tag >;

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
      size_t seed = common::default_seed,  // consistent with factory.hpp defaults
      SamplingRule sampling_rule = SamplingRule{}
   )
       : base(std::move(env_), std::move(root_state_), std::move(policy_), std::move(avg_policy_)),
         m_epsilon(epsilon),
         m_rng(seed),
         m_sampling_rule(std::move(sampling_rule))
   {
      static_assert(
         config.exploration != MCCFRExplorationMode::custom_sampling_policy
            or rm::sampling_rule_for< SamplingRule, action_type, decltype([](const action_type&) {
                                         return 1.;
                                      }) >,
         "MCCFRExplorationMode::custom_sampling_policy requires an injectable "
         "SamplingRule satisfying rm::sampling_rule_for."
      );
   }

   MCCFR(
      tag::internal_construct,
      Env env_,
      Policy policy_ = Policy(),
      AveragePolicy avg_policy_ = AveragePolicy(),
      double epsilon = 0.6,
      size_t seed = common::default_seed,  // consistent with factory.hpp defaults
      SamplingRule sampling_rule = SamplingRule{}
   )
       : MCCFR(
          tag::internal_construct{},
          std::move(env_),
          std::make_unique< world_state_type >(env.initial_world_state()),
          std::move(policy_),
          std::move(avg_policy_),
          epsilon,
          seed,
          std::move(sampling_rule)
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
      size_t seed = common::default_seed,  // consistent with factory.hpp defaults
      SamplingRule sampling_rule = SamplingRule{}
   )
       : base(std::move(env_), std::move(root_state_), std::move(policy_), std::move(avg_policy_)),
         m_epsilon(epsilon),
         m_rng(seed),
         m_sampling_rule(std::move(sampling_rule))
   {
      static_assert(
         config.exploration != MCCFRExplorationMode::custom_sampling_policy
            or rm::sampling_rule_for< SamplingRule, action_type, decltype([](const action_type&) {
                                         return 1.;
                                      }) >,
         "MCCFRExplorationMode::custom_sampling_policy requires an injectable "
         "SamplingRule satisfying rm::sampling_rule_for."
      );
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
   using base::cycle;
   using base::root_state;
   using operation_layer::iterate;

   /**
    * @brief Legacy mutable policy access with direct-view invalidation.
    *
    * fetch_policy() may insert or mutate a table entry. Keep its historical
    * reference-returning API, but invalidate non-owning policy views before
    * exposing the mutable reference.
    */
   template < bool current_policy >
   auto& fetch_policy(const info_state_type& infostate, const std::vector< action_type >& actions)
   {
      this->_invalidate_policy_views();
      return base::template fetch_policy< current_policy >(infostate, actions);
   }

   template < PolicyLabel label >
   decltype(auto)
   fetch_policy(const info_state_type& infostate, const std::vector< action_type >& actions)
   {
      static_assert(
         label == PolicyLabel::current or label == PolicyLabel::average,
         "Policy label has to be either 'current' or 'average'."
      );
      return fetch_policy< label == PolicyLabel::current >(infostate, actions);
   }

   template < bool current_policy >
   auto& fetch_policy(
      const info_state_type& infostate,
      const std::vector< action_type >& actions,
      const action_type& action
   )
   {
      this->_invalidate_policy_views();
      return base::template fetch_policy< current_policy >(infostate, actions, action);
   }

   [[nodiscard]] size_t history_value_entry_count() const
   {
      if constexpr(vr_history_active) {
         return m_vr_history_store.edge_values.size();
      } else {
         return 0;
      }
   }

   /**
    * @brief B8 (probing): per-iteration root-side game-value diagnostic -- the
    * sigma-mixture of the probed counterfactual value vector at the SHALLOWEST
    * visited infoset of the updating player (closest to the root), on the
    * usual outcome-sampling importance scale. Reset by every outcome-sampling
    * iteration and republished during the post-order unwind; read it after
    * @see iterate for variance studies of the probed value estimates (Gibson
    * et al., AAAI 2012). Empty when probing is inactive or no eligible
    * infoset was visited during the last iteration. The iterate() return
    * values remain identical to the vanilla outcome-sampling flow.
    */
   [[nodiscard]] std::optional< double > probe_root_estimate() const
   {
      if constexpr(probing_active) {
         return m_probe_root_value;
      } else {
         return std::nullopt;
      }
   }

  public:
   /**
    * @brief executes n iterations of the VanillaCFR algorithm.
    *
    * @param n_iters the number of iterations to perform.
    * @return the root value map from each iteration. This is the historical
    *         collection API; use advance(), advance_last(), or trace() when a
    *         different collection policy is wanted.
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
    * @return a one-element vector containing the historical
    *         (updating-player, value) result. Use iterate() for the efficient
    *         root StateValueMap operation.
    */
   auto iterate(std::optional< Player > player_to_update)
      requires(config.update_mode == UpdateMode::alternating);

   ////////////////////////////////
   /// private member functions ///
   ////////////////////////////////

   inline auto& _infonodes() { return m_infonode; }
   [[nodiscard]] auto _policy_source() const
   {
      using node_map_type = std::remove_cvref_t< decltype(m_infonode) >;
      using current_policy_map_type = std::remove_cvref_t< decltype(this->policy()) >;
      using average_policy_map_type = std::remove_cvref_t< decltype(this->average_policy()) >;
      return detail::TablePolicySource<
         node_map_type,
         info_state_type,
         action_type,
         current_policy_map_type,
         average_policy_map_type >{m_infonode, this->policy(), this->average_policy()};
   }
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
   /// B6: injectable sampling rule (active only under
   /// MCCFRExplorationMode::custom_sampling_policy; default rule reproduces
   /// the epsilon-on-policy behavior draw-for-draw)
   [[no_unique_address]] SamplingRule m_sampling_rule{};
   /// depth-indexed world-state slots reused across the recursion within one
   /// iteration (deque: growing never moves slots, keeping references of
   /// active frames valid)
   std::deque< utils::ReusableSlot< world_state_type > > m_traversal_state_arena;
   /// B8: per-iteration root-side probed value diagnostic (see
   /// probe_root_estimate); empty unless probing_active
   [[no_unique_address]] std::conditional_t< probing_active, std::optional< double >, utils::empty >
      m_probe_root_value{};
   /// ESCHER history-value table V(h->a); empty (and overlapped via
   /// no_unique_address) unless vr_history_active
   [[no_unique_address]] std::conditional_t< vr_history_active, HistoryValueStore, utils::empty >
      m_vr_history_store{};
   /// per-depth side channel of the predictive baseline: slot d holds the
   /// outgoing next-strategy (sigma^{t+1}-weighted) value stream of the frame
   /// at depth d, consumed by its parent after the recursive call returns.
   /// Empty unless vr_predictive_active.
   [[no_unique_address]] std::
      conditional_t< vr_predictive_active, std::vector< double >, utils::empty >
         m_vr_secondary_streams{};

   /// grows (if needed) and returns the secondary-stream slot of 'depth'
   double& _vr_secondary_slot(size_t depth)
   {
      if(m_vr_secondary_streams.size() <= depth) {
         m_vr_secondary_streams.resize(depth + 1, 0.);
      }
      return m_vr_secondary_streams[depth];
   }

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

   /// one regular iteration hook consumed by the shared operation layer; the
   /// caller owns view invalidation and this hook advances the solver counter.
   StateValueMap _iterate_one();
   StateValueMap _iterate_one(std::optional< Player > player_to_update);

   /**
    * @brief traverses the game tree and fills the nodes with current policy weighted value updates.
    *
    * This function is the regular traversal function to call on iteration i > 0, after the nodes
    * have been emplaced by @see _first_traversal.
    */

   std::pair< StateValueMap, Probability > _traverse(
      std::optional< Player > player_to_update,
      world_state_type& state,
      size_t depth,
      ReachProbabilityMap& reach_probability,
      ObservationbufferMap& observation_buffer,
      InfostateSptrMap& infostates,
      Probability sample_probability,
      ConditionalWeightMap& weights,
      size_t path_hash
   )
      requires(config.algorithm == MCCFRAlgorithmMode::outcome_sampling);

   StateValue _traverse(
      Player player_to_update,
      world_state_type& state,
      size_t depth,
      ObservationbufferMap& observation_buffer,
      InfostateSptrMap& infostates,
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
      world_state_type& curr_worldstate,
      size_t depth,
      ReachProbabilityMap& reach_probability,
      ObservationbufferMap& observation_buffer,
      InfostateSptrMap& infostates,
      [[maybe_unused]] delayed_update_set& infostates_to_update,
      ActionValueArena< action_variant_type >& action_value_arena
   )  // clang-format off
      requires(
         config.algorithm == MCCFRAlgorithmMode::chance_sampling
         or (
            config.algorithm == MCCFRAlgorithmMode::pure_cfr
            and config.update_mode == UpdateMode::simultaneous
         )
      );
   // clang-format on

   /// returns the arena slot for recursion 'depth', reconstructed from
   /// 'source' in place (no allocation after the first visit to this depth).
   world_state_type& _arena_state(size_t depth, const world_state_type& source);

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

   /**
    * @brief VR-MCCFR regret accumulation at one infoset (paper step (d)).
    *
    * Accumulates R(I,a) += v̂ᵇ(I,a) − v̂ᵇ(I) for every legal action, with
    * v̂ᵇ(I,a) = π₋ᵢ(h)/q(h) · ûᵇ(h,a) (eq 11): the sampled action carries the
    * baseline-corrected continuation value, every off-trajectory action is
    * valued by its baseline. Baselines are read through the 'baseline_at'
    * accessor so both the per-infostate action-baseline table (VR-MCCFR) and
    * the engine-side history-value store (ESCHER mode) share this code path.
    */
   template < typename BaselineAt >
   void _vr_accumulate_regrets(
      infostate_data_type& infostate_data,
      const std::vector< action_type >& actions,
      const auto& action_policy,
      size_t sampled_idx,
      double cf_weight,
      double sampled_value,
      BaselineAt&& baseline_at
   ) const
      requires(config.algorithm == MCCFRAlgorithmMode::outcome_sampling);

   /**
    * @brief VR-MCCFR baseline regression of the sampled edge (paper step (e)):
    * running-mean blend toward the corrected estimate. Kept for the
    * 'running_mean' rule; the predictive rule is handled inline in _traverse.
    */
   void _vr_regress_running_mean(
      size_t sampled_idx,
      double snapshot,
      double target,
      auto&& regress_sampled
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
      const ActionValueTable< action_variant_type >& action_value_map,
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
      Probability sample_probability,
      size_t depth
   )
      requires(config.algorithm == MCCFRAlgorithmMode::outcome_sampling);

   auto _sample_action(
      Player active_player,
      std::optional< Player > player_to_update,
      const std::vector< action_type >& actions,
      auto& action_policy,
      const info_state_type& infostate
   );

   /**
    * @brief whether the epsilon-on-policy action mixture is in effect at an
    * infoset of 'active_player' during the current traversal.
    *
    * This mirrors exactly the selection logic inside @see _sample_action: the
    * mixture epsilon*uniform + (1-epsilon)*policy is sampled at the updating
    * player's infosets (at every actual player's infosets under simultaneous
    * updates), while pure on-policy sampling applies elsewhere.
    */
   static constexpr bool
   _epsilon_mixed_sampling_active(std::optional< Player > player_to_update, Player active_player)
   {
      return config.update_mode == UpdateMode::simultaneous
             or active_player == player_to_update.value_or(Player::chance);
   }

   auto _sample_action_on_policy(const std::vector< action_type >& actions, auto& action_policy);

   template < bool return_likelihood = true >
   auto _sample_outcome(const world_state_type& state);

   /**
    * @brief B7 (PCS): chance-outcome resolution under an injected
    * rm::PublicChanceSamplingRule.
    *
    * Public chance events are drawn exactly like @see _sample_outcome;
    * private events are resolved deterministically to their FIRST legal
    * outcome without consuming RNG, reporting that outcome's true chance
    * probability as the sampling likelihood (IS correction threaded into the
    * sample-probability accumulator by the caller). See
    * rm::PublicChanceSamplingRule for the single-trajectory deviation note.
    */
   template < bool return_likelihood = true >
   auto _sample_outcome_pcs(const world_state_type& state);

   /**
    * @brief B8 (probing): one on-policy Monte-Carlo probe rollout (Gibson et
    * al., AAAI 2012, Algorithm 1 'Probe').
    *
    * Descends from 'state' along a single trajectory, sampling chance from the
    * true distribution and every player's actions from the CURRENT strategy,
    * until a terminal history is reached whose updater reward is returned. The
    * rollout is value-only: no infonodes are created, no regrets or average
    * policies are touched, and no importance weighting is applied (on-policy
    * rollouts are unbiased estimates of v_sigma by themselves). Observation
    * folding happens against the CALLER-PROVIDED local containers so the
    * engine's live traversal state is untouched; world states reuse the
    * depth-indexed arena slots at depths strictly below the invoking frame.
    */
   auto _probe_rollout(
      world_state_type& state,
      size_t depth,
      player_hashmap< std::vector< std::pair< observation_type, observation_type > > >&
         observation_buffer,
      player_hashmap< sptr< info_state_type > >& infostates,
      Player updating_player
   ) -> double
      requires(config.algorithm == MCCFRAlgorithmMode::outcome_sampling);

   /**
    * @brief B8 (probing): regret accumulation from the probed counterfactual
    * value vector at one visited infoset of the updating player (paper
    * Algorithm 1 lines 40-44).
    *
    * r(I,a) += pi_-i(h_I) * (v_hat(a) - sum_a' sigma(I,a') v_hat(a')) with
    * v_hat(a*) = u(z)/q_prefix (the sampled branch's trajectory estimate) and
    * v_hat(a') = probe_rollout(h_I a')/(q_prefix * xi(I,a*)) for unsampled a'
    * (probes carry no prefix weight of their own, so they are lifted onto the
    * trajectory's per-unit scale with the same divisor as the sampled branch).
    */
   void _update_regrets_probing(
      const ReachProbabilityMap& reach_probability,
      Player active_player,
      infostate_data_type& infostate_data,
      const std::vector< action_type >& actions,
      const auto& action_policy,
      const std::vector< double >& probed_action_values
   ) const
      requires(config.algorithm == MCCFRAlgorithmMode::outcome_sampling);

   void _initiate_regret_minimization(const delayed_update_set& update_set);

   void _invoke_regret_minimizer(const info_state_type& infostate, infostate_data_type& data);

   constexpr void _sanity_check_config();
};

}  // namespace nor::rm

// include the actual template definitions of the member functions
#include "mccfr.tcc"

#endif  // NOR_MCCFR_HPP
