#pragma once

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>

#include "mccfr.hpp"

namespace nor::rm {

template <
   MCCFRConfig config,
   typename Env,
   typename Policy,
   typename AveragePolicy,
   typename SamplingRule >
constexpr void MCCFR< config, Env, Policy, AveragePolicy, SamplingRule >::_sanity_check_config()
{
   static_assert(
      not std::invoke([&] {
         constexpr bool pruning_in_non_full_traversal_modes =
            // clang-format off
         config.pruning_mode != CFRPruningMode::none
         and (
            config.algorithm != MCCFRAlgorithmMode::chance_sampling
            or (
               config.algorithm == MCCFRAlgorithmMode::pure_cfr
               and config.update_mode != UpdateMode::simultaneous
            )
         );
         // clang-format on
         constexpr bool ext_sampling_bad_combo =
            // clang-format off
         config.algorithm == MCCFRAlgorithmMode::external_sampling
         and (
            config.update_mode != UpdateMode::alternating
            or config.weighting != MCCFRWeightingMode::stochastic
         );
         // clang-format on
         return pruning_in_non_full_traversal_modes or ext_sampling_bad_combo;
      }),
      "Config did not pass the check for correctness."
   );
   static_assert(
      not (
         effective_variance_reduction(config) != VarianceReductionMode::none
         and config.update_mode != UpdateMode::alternating
      ),
      "VR-MCCFR/ESCHER baselines are currently restricted to alternating updates. Under "
      "simultaneous updates the baseline-corrected low-variance increments make both "
      "players' current policies chase each other within a single trajectory, and "
      "average-strategy convergence stalls (empirically established on Kuhn poker for "
      "epsilon-on-policy exploration >= 0.3 across baseline rates beta in {0, 0.01, "
      "0.03, 0.1, 1}: exploitability plateaus near 0.11-0.23 instead of descending). "
      "Alternating updates damp the chase and converge fast."
   );
   static_assert(
      not (
         config.updater_sampling == UpdaterSamplingMode::fixed_uniform
         and vr_mode != VarianceReductionMode::history_value
      ),
      "UpdaterSamplingMode::fixed_uniform (ESCHER's fixed sampling policy) drops ALL "
      "importance-sampling corrections -- both the 1/xi deviation factor of the sampled "
      "branch and the pi_-i/q(h) reach weighting. Dropping them is only sound when the "
      "baselines are history values V(h) tracked at world-state granularity (ESCHER sec. "
      "3/Theorem 1); with per-infoset action baselines the resulting estimator loses its "
      "unbiasedness guarantee. Select variance_reduction = history_value."
   );
   // B7 sanity guards: both new rule families are defined on (and hooked only
   // into) the outcome-sampling traversal.
   static_assert(
      not public_chance_sampling_rule< SamplingRule >
         or config.algorithm == MCCFRAlgorithmMode::outcome_sampling,
      "PublicChanceSamplingRule reroutes chance resolution of the outcome-sampling "
      "traversal and therefore requires MCCFRAlgorithmMode::outcome_sampling."
   );
   static_assert(
      not average_strategy_sampling_rule< SamplingRule >
         or config.algorithm == MCCFRAlgorithmMode::outcome_sampling,
      "AverageStrategySamplingRule requires MCCFRAlgorithmMode::outcome_sampling: ASS "
      "(Gibson et al., NIPS 2012) is formulated on outcome sampling, and only that "
      "traversal consumes an injectable action-sampling rule."
   );
   static_assert(
      not average_strategy_sampling_rule< SamplingRule >
         or config.exploration == MCCFRExplorationMode::custom_sampling_policy,
      "AverageStrategySamplingRule is injected through the sampling-rule slot and "
      "therefore requires MCCFRExplorationMode::custom_sampling_policy."
   );
   // B8 sanity guard: probing is a traversal-side value-estimation scheme of
   // the single-updater outcome-sampling walk; see rm::probing_supported for
   // the full rationale of each clause.
   static_assert(
      not probing_sampling_rule< SamplingRule > or probing_supported(config),
      "ProbingSamplingRule (Gibson et al., AAAI 2012) requires an outcome-sampling, "
      "alternating-update MCCFR configuration without VR/ESCHER baselines and with "
      "current-policy updater sampling (see rm::probing_supported). External/pure/chance "
      "sampling traversals do not consume probed value estimates."
   );
};

template <
   MCCFRConfig config,
   typename Env,
   typename Policy,
   typename AveragePolicy,
   typename SamplingRule >
auto MCCFR< config, Env, Policy, AveragePolicy, SamplingRule >::iterate(size_t n_iters)
{
   std::vector< StateValueMap > root_values_per_iteration;
   root_values_per_iteration.reserve(n_iters);
   for([[maybe_unused]] auto _ : std::views::iota(size_t(0), n_iters)) {
      SPDLOG_DEBUG("Iteration number: {}", _iteration());
      std::optional< Player > player_to_update = std::nullopt;
      if constexpr(config.update_mode == UpdateMode::alternating) {
         player_to_update = _cycle_player_to_update();
      }
      root_values_per_iteration.emplace_back(std::invoke([&] {
         if constexpr(config.algorithm == MCCFRAlgorithmMode::outcome_sampling) {
            return std::move(_iterate(player_to_update).first);
         }
         if constexpr(  // clang-format off
             (config.algorithm == MCCFRAlgorithmMode::chance_sampling)
             or (
                config.algorithm == MCCFRAlgorithmMode::pure_cfr
                and config.update_mode == UpdateMode::simultaneous
             )  // clang-format on
         ) {
            return std::move(_iterate(player_to_update).get());
         }
         if constexpr(  // clang-format off
             (config.algorithm == MCCFRAlgorithmMode::external_sampling)
             or (
                config.algorithm == MCCFRAlgorithmMode::pure_cfr
                and config.update_mode == UpdateMode::alternating
             )  // clang-format on
         ) {
            return StateValueMap{std::unordered_map< Player, double >{
               {*player_to_update, _iterate(*player_to_update).get()}}};
         }
      }));
      _iteration()++;
   }
   return root_values_per_iteration;
}

template <
   MCCFRConfig config,
   typename Env,
   typename Policy,
   typename AveragePolicy,
   typename SamplingRule >
auto MCCFR< config, Env, Policy, AveragePolicy, SamplingRule >::iterate(
   std::optional< Player > player_to_update
)
   requires(config.update_mode == UpdateMode::alternating)
{
   SPDLOG_DEBUG("Iteration number: {}", _iteration());
   // run the iteration
   auto updated_player = _cycle_player_to_update(player_to_update);
   auto value = std::vector{std::pair{updated_player, std::get< 0 >(_iterate()).get()}};
   // and increment our iteration counter
   _iteration()++;
   return value;
}

template <
   MCCFRConfig config,
   typename Env,
   typename Policy,
   typename AveragePolicy,
   typename SamplingRule >
auto MCCFR< config, Env, Policy, AveragePolicy, SamplingRule >::_arena_state(
   size_t depth,
   const world_state_type& source
) -> world_state_type&
{
   if(m_traversal_state_arena.size() <= depth) {
      m_traversal_state_arena.resize(depth + 1);
   }
   // reconstruct the slot from 'source' in place: no allocation after the
   // first visit to this depth and no requirement on copy assignment
   return m_traversal_state_arena[depth].construct_from(source);
}

template <
   MCCFRConfig config,
   typename Env,
   typename Policy,
   typename AveragePolicy,
   typename SamplingRule >
auto MCCFR< config, Env, Policy, AveragePolicy, SamplingRule >::_iterate(
   std::optional< Player > player_to_update
)
{
   auto players = _env().players(*_root_state_uptr());
   auto init_infostates = [&] {
      InfostateSptrMap infostates{};
      for(auto player : players | utils::is_actual_player_filter) {
         infostates.emplace(player, std::make_shared< info_state_type >(player));
      }
      return infostates;
   };
   auto init_reach_probs = [&] {
      ReachProbabilityMap rp_map{};
      for(auto player : players) {
         rp_map.get().emplace(player, 1.);
      }
      return rp_map;
   };
   auto init_obs_buffer = [&] {
      ObservationbufferMap obs_map{};
      for(auto player : players | utils::is_actual_player_filter) {
         obs_map.emplace(player);
      }
      return obs_map;
   };

   // the traversal containers are created once per iteration here and then
   // shared down the recursion by reference; world states live in the
   // depth-indexed arena (slot 0 holds the root copy).
   world_state_type& root_arena_state = _arena_state(0, *_root_state_uptr());

   if constexpr(config.algorithm == MCCFRAlgorithmMode::outcome_sampling) {
      auto weights = std::invoke([&] {
         if constexpr(config.weighting == MCCFRWeightingMode::lazy) {
            PlayerValueTable w;
            for(auto player : players | utils::is_actual_player_filter) {
               w.emplace(player, 0.);
            }
            return WeightMap{std::move(w)};
         } else {
            return utils::empty{};
         }
      });
      auto reach_probs = init_reach_probs();
      auto obs_buffer = init_obs_buffer();
      auto infostates = init_infostates();
      if constexpr(probing_active) {
         m_probe_root_value.reset();
      }
      return _traverse(
         player_to_update,
         root_arena_state,
         /*depth=*/0,
         reach_probs,
         obs_buffer,
         infostates,
         Probability{1.},
         weights,
         /*path_hash=*/0ul
      );
   }

   if constexpr( // clang-format off
      config.algorithm == MCCFRAlgorithmMode::external_sampling
      or (
         config.algorithm == MCCFRAlgorithmMode::pure_cfr
         and config.update_mode == UpdateMode::alternating
      )
   ) { // clang-format on
      delayed_update_set update_set{};
      auto obs_buffer = init_obs_buffer();
      auto infostates = init_infostates();
      auto value = _traverse(
         player_to_update.value(), root_arena_state, /*depth=*/0, obs_buffer, infostates, update_set
      );
      if constexpr(config.algorithm != MCCFRAlgorithmMode::external_sampling) {
         // external sampling is able to minimize the regret on the fly during the traversal, since
         // each infostate of the traverser is seen only once
         _initiate_regret_minimization(update_set);
      }
      update_set.clear();
      return value;
   }

   if constexpr( // clang-format off
      config.algorithm == MCCFRAlgorithmMode::chance_sampling
      or (
         config.algorithm == MCCFRAlgorithmMode::pure_cfr
         and config.update_mode == UpdateMode::simultaneous
      )
   ) { // clang-format on
      delayed_update_set update_set{};
      auto reach_probs = init_reach_probs();
      auto obs_buffer = init_obs_buffer();
      auto infostates = init_infostates();
      // per-depth reusable slots of the insertion-ordered action -> child-values
      // accumulator consumed by update_regret_and_policy (same reuse discipline
      // as in VanillaCFR::iterate)
      ActionValueArena< action_variant_type > action_value_arena{};
      auto values = _traverse(
         player_to_update,
         root_arena_state,
         /*depth=*/0,
         reach_probs,
         obs_buffer,
         infostates,
         update_set,
         action_value_arena
      );
      _initiate_regret_minimization(update_set);
      update_set.clear();
      return values;
   }
}

template <
   MCCFRConfig config,
   typename Env,
   typename Policy,
   typename AveragePolicy,
   typename SamplingRule >
void MCCFR< config, Env, Policy, AveragePolicy, SamplingRule >::_initiate_regret_minimization(
   const delayed_update_set& update_set
)
{
   // here we now invoke the actual regret minimization procedure for each infostate individually.
   // The sweep is intentionally serial (see the determinism note in rm_utils.hpp):
   // per-infostate workloads are tiny and parallel scheduling would make float
   // summation orders non-deterministic.
   auto node_view = std::invoke([&] {
      if constexpr(config.algorithm == MCCFRAlgorithmMode::pure_cfr) {
         return std::views::all(_infonodes());
      } else {
         return std::views::all(update_set);
      };
   });
   std::for_each(node_view.begin(), node_view.end(), [&](auto& entry) {
      // the pure-cfr branch iterates the whole infonode table while every
      // other algorithm iterates the delayed update set of pointer pairs
      if constexpr(config.algorithm == MCCFRAlgorithmMode::pure_cfr) {
         auto& [infostate_sptr, data] = entry;
         // reset the sampled plan per information state
         data.data().extras.sampled_action.reset();
         if(not (  // for alternating pure-cfr we have to check if this
                   // infostate was meant to be updated as well
               config.update_mode == UpdateMode::alternating
            )
            or update_set.find({infostate_sptr.get(), &data}) != update_set.end()) {
            _invoke_regret_minimizer(common::deref(infostate_sptr), data);
         }
      } else {
         const auto& [infostate_ptr, data_ptr] = entry;
         _invoke_regret_minimizer(common::deref(infostate_ptr), *data_ptr);
      }
   });
}
template <
   MCCFRConfig config,
   typename Env,
   typename Policy,
   typename AveragePolicy,
   typename SamplingRule >
void MCCFR< config, Env, Policy, AveragePolicy, SamplingRule >::_invoke_regret_minimizer(
   const info_state_type& infostate,
   infostate_data_type& data
)
{
   auto& current_policy = this->template fetch_policy< PolicyLabel::current >(
      infostate, data.actions()
   );
   m_regret_minimizer.recommend(data.data(), current_policy, _iteration());
}

////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// Outcome-Sampling MCCFR /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

template <
   MCCFRConfig config,
   typename Env,
   typename Policy,
   typename AveragePolicy,
   typename SamplingRule >
std::pair< StateValueMap, Probability >
MCCFR< config, Env, Policy, AveragePolicy, SamplingRule >::_traverse(
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
   requires(config.algorithm == MCCFRAlgorithmMode::outcome_sampling)
{
   if(_env().is_terminal(state)) {
      return _terminal_value(state, player_to_update, sample_probability, depth);
   }

   Player active_player = _env().active_player(state);

   // now we check first if we even need to consider a chance player, as the env could be
   // simply deterministic. In that case we might need to traverse the chance player's actions
   // or an active player's actions
   if constexpr(not concepts::deterministic_fosg< env_type >) {
      if(active_player == Player::chance) {
         // B7 (PCS): an injected PublicChanceSamplingRule reroutes chance-outcome
         // resolution; the default remains the vanilla distribution draw.
         auto [chosen_outcome, chance_prob] = [&] {
            if constexpr(public_chance_sampling_rule< SamplingRule >) {
               return _sample_outcome_pcs(state);
            } else {
               return _sample_outcome(state);
            }
         }();

         // single-trajectory descent: containers mutate in place; the chance
         // reach entry is restored after the recursive call returns because the
         // post-traversal updates read node-entry values
         // NOTE: no reference held across the recursion -- deeper frames may
         // insert keys and shift the compacted table's storage; restored via
         // operator[] re-lookup below.
         double& chance_reach_entry = reach_probability.get()[Player::chance];
         const double saved_chance_reach = chance_reach_entry;
         chance_reach_entry *= chance_prob;

         world_state_type& next_state = _arena_state(depth + 1, state);
         _env().transition(next_state, chosen_outcome);

         next_infostate_and_obs_buffers_seated(
            _env(), observation_buffer, infostates, state, chosen_outcome, next_state
         );

         // the edge identity of a chance move enters the rolling path hash via
         // the outcome's position in the state's chance-action list (stable,
         // and free of any hashability requirement on the outcome type itself)
         size_t child_path_hash = path_hash;
         if constexpr(vr_history_active) {
            auto chance_actions = _env().chance_actions(state);
            size_t outcome_index = 0;
            for(const auto& outcome : chance_actions) {
               if(outcome == chosen_outcome) {
                  break;
               }
               ++outcome_index;
            }
            common::hash_combine(child_path_hash, outcome_index);
         }

         auto [action_value_map, tail_prob] = _traverse(
            player_to_update,
            next_state,
            depth + 1,
            reach_probability,
            observation_buffer,
            infostates,
            Probability{sample_probability.get() * chance_prob},
            weights,
            child_path_hash
         );
         reach_probability.get()[Player::chance] = saved_chance_reach;
         // predictive baseline: value streams pass through non-updating frames unchanged
         if constexpr(vr_predictive_active) {
            _vr_secondary_slot(depth) = _vr_secondary_slot(depth + 1);
         }
         return {std::move(action_value_map), tail_prob};
      }
   }

   // fetch (or create) this infostate's node without cloning the infostate for
   // every visit: heterogeneous value lookup finds existing entries directly.
   // (The infostate object itself advances along the single trajectory via the
   // inplace observation fold above, exactly like before.)
   const auto& live_infostate_sptr = infostates.at(active_player);
   auto infostate_and_data_iter = m_infonode.find(*live_infostate_sptr);
   bool success = false;
   if(infostate_and_data_iter == m_infonode.end()) {
      std::tie(infostate_and_data_iter, success) = _infonodes().emplace(
         sptr< info_state_type >{utils::clone_any_way(live_infostate_sptr)}, infostate_data_type{}
      );
   }
   const auto& infostate = infostate_and_data_iter->first;
   auto& infonode_data = infostate_and_data_iter->second;
   if(success) {
      // success means we have indeed emplaced a new data node, instead of simply fetching an
      // existing one. We thus need to fill it with the legal actions at this node.
      infonode_data.emplace(_env().actions(active_player, state));
   }

   const auto& actions = infonode_data.actions();
   auto& action_policy = this->template fetch_policy< PolicyLabel::current >(*infostate, actions);

   // apply one round of regret matching on the current policy before using it. MCCFR only
   // updates the policy once you revisit it, as it is a lazy update schedule. As such, one would
   // need to update all infostates after the last iteration to ensure that the policy is fully
   // up-to-date

   // apply one round of regret matching on the current policy before using it via the
   // minimizer's recommendation step
   m_regret_minimizer.recommend(infonode_data.data(), action_policy, _iteration());

   auto [sampled_action, action_sampling_prob, action_policy_prob] = _sample_action(
      active_player, player_to_update, actions, action_policy, *infostate
   );

   // snapshot the path bookkeeping BEFORE mutating it: the deeper traversal
   // works on the mutated values while the post-traversal updates below need
   // their NODE-ENTRY values back; observation buffers and infostates flow
   // permanently downward (single linear trajectory). Only the acting
   // player's entries are mutated below, so scalar save/restore suffices
   // (formerly full-table copies).
   // NOTE: no reference is held across the recursion -- deeper frames may
   // insert new keys into the compacted table (storage shifts); the entry is
   // re-resolved via operator[] after the recursive call returns.
   double& actor_reach_entry = reach_probability.get()[active_player];
   const double saved_actor_reach = actor_reach_entry;
   [[maybe_unused]] double saved_actor_weight = 0.;
   if constexpr(config.weighting == MCCFRWeightingMode::lazy) {
      saved_actor_weight = weights.get()[active_player];
   }

   // scale the acting player's reach entry / lazy weight in place
   actor_reach_entry *= action_policy_prob;

   if constexpr(config.weighting == MCCFRWeightingMode::lazy) {
      auto& active_weight = weights.get()[active_player];
      active_weight = active_weight * action_policy_prob
                      + infonode_data.data()
                           .extras
                           .pending_avg_accumulator[infonode_data.data().index_of(sampled_action)];
   }

   // ---- B8 (probing) pre-edge snapshot -------------------------------------
   // Gibson et al. (AAAI 2012), Algorithm 1: at every visited infoset of the
   // UPDATING player each non-sampled action is probed with one on-policy
   // rollout. The live observation buffers / infostates flow permanently
   // downward along the single trajectory, so the PRE-edge state needed by the
   // sibling-edge probes must be captured before the trajectory fold advances
   // them past the sampled action.
   [[maybe_unused]] bool probing_engaged = false;
   [[maybe_unused]] size_t probing_sampled_idx = 0;
   [[maybe_unused]] std::vector< double > probed_action_values{};
   if constexpr(probing_active) {
      probing_sampled_idx = infonode_data.data().index_of(sampled_action);
      // probes fire only at infosets of the UPDATING player (paper Algorithm 1:
      // "each non-terminal history h with P(h) = i reached"); alternating
      // updates are statically enforced, so player_to_update is always set
      probing_engaged =
         actions.size() > 1 and active_player == player_to_update.value_or(Player::chance);
      if(probing_engaged) {
         // deep-copy helper: rollout containers must not alias the live ones
         auto clone_infostate_map = [](const auto& src) {
            auto out = src;
            for(auto& entry : std::views::values(out)) {
               entry = std::make_shared< info_state_type >(*entry);
            }
            return out;
         };
         // B-stream adaptation: the live traversal containers are seat-indexed
         // tables; the rollout walk (declared on classic player hashmaps)
         // materializes them through the cold-path view adapters before the
         // deep clone -- rollouts mutate only these local copies.
         auto probe_observation_buffer = raw_observation_buffer_view(observation_buffer);
         auto probe_infostates = clone_infostate_map(raw_infostate_view(infostates));
         probed_action_values.assign(actions.size(), 0.);
         const double probe_divisor = sample_probability.get();
         for(const auto [probe_idx, probe_action] : std::views::enumerate(actions)) {
            if(probe_idx == probing_sampled_idx) {
               continue;
            }
            // fresh per-probe container copies: rollouts mutate their locals
            auto rollout_observation_buffer = probe_observation_buffer;
            auto rollout_infostates = clone_infostate_map(probe_infostates);
            // arena slots strictly below this frame are safe to reuse: every
            // _arena_state call reconstructs its slot IN PLACE from 'source',
            // so neither this frame's slot nor the later main recursion can be
            // corrupted by probe rollouts sharing the deeper slots
            world_state_type& probe_state = _arena_state(depth + 1, state);
            _env().transition(probe_state, probe_action);
            next_infostate_and_obs_buffers_inplace(
               _env(),
               rollout_observation_buffer,
               rollout_infostates,
               state,
               probe_action,
               probe_state
            );
            probed_action_values[probe_idx] =
               _probe_rollout(
                  probe_state,
                  depth + 1,
                  rollout_observation_buffer,
                  rollout_infostates,
                  *player_to_update
               )
               / probe_divisor;
         }
      }
   }

   // advance along the single trajectory via the arena slot of the next depth
   world_state_type& next_state = _arena_state(depth + 1, state);
   _env().transition(next_state, sampled_action);

   next_infostate_and_obs_buffers_seated(
      _env(), observation_buffer, infostates, state, sampled_action, next_state
   );

   // rolling world-state-edge identity for the ESCHER history-value store: the
   // parent's trajectory hash combined with the sampled action's registry index
   // uniquely names the edge (h --a*--> h a*) without requiring any hashability
   // of the action/world-state types themselves
   size_t child_path_hash = path_hash;
   if constexpr(vr_history_active) {
      common::hash_combine(child_path_hash, infonode_data.data().index_of(sampled_action));
   }

   auto [action_value_map, tail_prob] = _traverse(
      player_to_update,
      next_state,
      depth + 1,
      reach_probability,
      observation_buffer,
      infostates,
      Probability{sample_probability.get() * action_sampling_prob},
      weights,
      child_path_hash
   );
   reach_probability.get()[active_player] = saved_actor_reach;
   if constexpr(config.weighting == MCCFRWeightingMode::lazy) {
      weights.get()[active_player] = saved_actor_weight;
   }

   // ---- B8 (probing): lift the sampled branch onto the per-unit scale ------
   // The child recursion returns u(z)/(q_prefix * xi*) on the trajectory's
   // importance scale; multiplying back by xi* (and by the residual policy
   // tail pi(h a*, z)) yields an estimate unbiased for v(h a*)/q_prefix -- the
   // SAME scale the probe rollouts were lifted to above. All |A| entries of
   // the probed counterfactual value vector are thereby unbiased per-visit
   // estimates of v(h_I a)/q_prefix (Proposition 1 of Gibson et al., AAAI
   // 2012), so the regret increments accumulated from them target, in
   // expectation, the TRUE per-iteration counterfactual regret vector.
   if constexpr(probing_active) {
      if(probing_engaged) {
         probed_action_values[probing_sampled_idx] =
            action_value_map.get().at(active_player) * tail_prob.get() * action_sampling_prob;
      }
   }

   // ---- VR-MCCFR quantities (Schmid et al., AAAI 2019, eqs (7)-(11)) and ----
   // ---- their ESCHER history-value / importance-sampling-free variants   ----
   // Applied at the updating player's infosets only (the configuration is
   // restricted to alternating updates, see _sanity_check_config): these are
   // exactly the nodes where this implementation materializes the updater's
   // own value stream.
   [[maybe_unused]] double vr_sampled_value = 0.;  // eq (9) sampled branch
   [[maybe_unused]] double vr_cf_weight = 0.;  // eq (11) factor pi_-i(h)/q(h)
   [[maybe_unused]] bool vr_active = false;
   [[maybe_unused]] size_t vr_sampled_idx = 0;
   [[maybe_unused]] double vr_sampled_baseline_snapshot = 0.;
   if constexpr(vr_mode != VarianceReductionMode::none) {
      vr_active = active_player == player_to_update.value();
   }
   // baseline snapshot/regression accessors of this node's edges, uniformly
   // indexed by registry position across both storage backends (per-infostate
   // action table for VR-MCCFR; engine-side per-history-edge store for ESCHER).
   [[maybe_unused]] auto vr_baseline_at = [&](size_t idx) -> double {
      if constexpr(vr_mode == VarianceReductionMode::action_baseline) {
         return infonode_data.data().vr_extras.baseline[idx];
      } else if constexpr(vr_mode == VarianceReductionMode::history_value) {
         size_t key = path_hash;
         common::hash_combine(key, idx);
         const auto found = m_vr_history_store.edge_values.find(key);
         return found == m_vr_history_store.edge_values.end() ? 0. : found->second;
      } else {
         return 0.;
      }
   };
   [[maybe_unused]] auto vr_regress_sampled_edge = [&](size_t s_idx, double target) {
      if constexpr(vr_mode == VarianceReductionMode::action_baseline) {
         infonode_data.data().vr_extras.baseline[s_idx] = target;
      } else if constexpr(vr_mode == VarianceReductionMode::history_value) {
         size_t key = path_hash;
         common::hash_combine(key, s_idx);
         m_vr_history_store.edge_values[key] = target;
      } else {
         // mode none: no baseline storage exists; never invoked
         (void) s_idx;
         (void) target;
      }
   };
   if constexpr(vr_mode != VarianceReductionMode::none) {
      if(vr_active) {
         vr_sampled_idx = infonode_data.data().index_of(sampled_action);
         // incoming bootstrapped value u(h a*|z): the raw terminal reward at
         // the deepest of this player's infosets, else the eq-(10) mixture
         // propagated from the child infoset
         double& stream = action_value_map.get().at(active_player);
         // ESCHER fixed-policy sampling drops ALL importance corrections (the
         // sampling rule is iteration-invariant, so no unbiasedness factor is
         // needed; visit frequencies supply Theorem 1's positive w(s) scaling).
         // The static config check guarantees eschew_is => history_value mode.
         const bool eschew_is = config.updater_sampling == UpdaterSamplingMode::fixed_uniform;
         const double deviation_factor = eschew_is ? 1. : 1. / action_sampling_prob;

         vr_sampled_baseline_snapshot = vr_baseline_at(vr_sampled_idx);
         // eq (9), sampled branch: b̂ + (u(h a*|z) - b̂)/ξ(h,a*) (or without the
         // 1/ξ under ESCHER fixed-policy sampling). Dividing the deviation by
         // the ACTUAL sampling probability of a* is what keeps the estimator
         // exact for ANY positive sampling rule: in expectation the b̂ term
         // cancels between the sampled branch and the off-trajectory branch
         // (both enter with weight ξ·(1/ξ) resp. (1-ξ) against the same
         // baseline).
         vr_sampled_value = vr_sampled_baseline_snapshot
                            + (stream - vr_sampled_baseline_snapshot) * deviation_factor;
         // eq (10): propagate the σ_t-weighted mixture of the sampled-branch
         // value and the off-trajectory baselines up to the parent infoset
         // (bootstrapped propagation; chance nodes pass values unchanged)
         double off_trajectory_mass = 0.;
         for(auto idx : std::views::iota(size_t{0}, actions.size())) {
            if(! (actions[idx] == sampled_action)) {
               off_trajectory_mass += action_policy[actions[idx]] * vr_baseline_at(idx);
            }
         }
         stream = action_policy[sampled_action] * vr_sampled_value + off_trajectory_mass;
         // eq (11) prefactor: counterfactual reach over prefix sampling
         // probability q(h) (everything sampled above this infoset). ESCHER
         // fixed-policy sampling omits it entirely.
         vr_cf_weight = eschew_is ? 1.
                                  : cf_reach_probability(active_player, reach_probability.get())
                                       / sample_probability.get();
         // NOTE: the baselines are intentionally not mutated here -- the regret
         // update below must read the same baseline snapshot that produced
         // vr_sampled_value (paper order: values -> regrets -> regression).
      }
   }

   auto active_weight_param = [&] {
      if constexpr(config.weighting == MCCFRWeightingMode::lazy) {
         return Weight{weights.get()[active_player]};
      } else {
         return utils::empty{};
      }
   };

   if constexpr(config.update_mode == UpdateMode::simultaneous) {
      _update_regrets(
         reach_probability,
         active_player,
         infonode_data,
         sampled_action,
         Probability{action_policy_prob},
         StateValue{action_value_map.get()[active_player]},
         tail_prob
      );

      _update_average_policy(
         *infostate,
         infonode_data,
         action_policy,
         Probability{reach_probability.get()[active_player]},
         sample_probability,
         sampled_action,
         std::invoke(active_weight_param)
      );

   } else {
      static_assert(
         config.update_mode == UpdateMode::alternating,
         "The update mode has to be either alternating or simultaneous."
      );
      // in alternating updates we update the regret only for the player_to_update and the
      // strategy only if the current player is the next one in line to traverse the tree and
      // update
      if(active_player == *player_to_update) {
         if constexpr(probing_active) {
            // B8: probing replaces the vanilla importance-weighted update at
            // every visited updater infoset with the probed counterfactual
            // value vector (paper Algorithm 1 lines 40-44). VR baselines are
            // statically incompatible (see probing_supported).
            if(probing_engaged) {
               _update_regrets_probing(
                  reach_probability,
                  active_player,
                  infonode_data,
                  actions,
                  action_policy,
                  probed_action_values
               );
            } else {
               _update_regrets(
                  reach_probability,
                  active_player,
                  infonode_data,
                  sampled_action,
                  Probability{action_policy_prob},
                  StateValue{action_value_map.get()[active_player]},
                  tail_prob
               );
            }
         } else if constexpr(vr_mode != VarianceReductionMode::none) {
            // paper step (d): regret accumulation against the untouched
            // baseline snapshot
            _vr_accumulate_regrets(
               infonode_data,
               actions,
               action_policy,
               vr_sampled_idx,
               vr_cf_weight,
               vr_sampled_value,
               vr_baseline_at
            );
            if constexpr(config.baseline_update_rule == BaselineUpdateRule::running_mean) {
               // paper step (e): running-mean regression of the sampled edge's
               // baseline onto its corrected estimate
               _vr_regress_running_mean(
                  vr_sampled_idx,
                  vr_sampled_baseline_snapshot,
                  vr_sampled_value,
                  vr_regress_sampled_edge
               );
            } else {
               // predictive baseline (Davis, Schmid, Bowling, ICML 2020,
               // eq (8)): the sampled edge's baseline regresses onto the
               // trajectory value RE-EVALUATED under the next strategy.
               // sigma^{t+1} is available because this infoset's regret table
               // was just updated (post-order traversal: every deeper
               // updater-infoset has been processed as well).
               std::decay_t< decltype(action_policy) > next_policy{};
               m_regret_minimizer.recommend(infonode_data.data(), next_policy, _iteration());
               const bool eschew_is = config.updater_sampling == UpdaterSamplingMode::fixed_uniform;
               const double deviation_factor = eschew_is ? 1. : 1. / action_sampling_prob;
               // S2: the sigma^{t+1}-weighted continuation value of the sampled
               // edge (raw terminal reward at the deepest updater infoset)
               const double secondary_in = _vr_secondary_slot(depth + 1);
               const double snapshot = vr_sampled_baseline_snapshot;
               const double s2_corrected = snapshot + (secondary_in - snapshot) * deviation_factor;
               double off_trajectory_mass_next = 0.;
               for(auto idx : std::views::iota(size_t{0}, actions.size())) {
                  if(! (actions[idx] == sampled_action)) {
                     off_trajectory_mass_next += next_policy[actions[idx]] * vr_baseline_at(idx);
                  }
               }
               // outgoing second stream: node value under sigma^{t+1}
               // (= eq (8)'s ub(h | sigma^{t+1}, z) consumed by the parent)
               const double second_stream = next_policy[sampled_action] * s2_corrected
                                            + off_trajectory_mass_next;
               // eq (8), beta-damped: b <- b + beta*(S2 - b); beta = 1 recovers
               // the paper's direct replacement
               const double predictive_target = snapshot
                                                + config.baseline_update_rate
                                                     * (secondary_in - snapshot);
               vr_regress_sampled_edge(vr_sampled_idx, predictive_target);
               _vr_secondary_slot(depth) = second_stream;
            }
         } else {
            _update_regrets(
               reach_probability,
               active_player,
               infonode_data,
               sampled_action,
               Probability{action_policy_prob},
               StateValue{action_value_map.get()[active_player]},
               tail_prob
            );
         }
      } else if(active_player == _preview_next_player_to_update()) {
         // the check in this if statement collapses to a simple true in the 2-player case
         _update_average_policy(
            *infostate,
            infonode_data,
            action_policy,
            Probability{reach_probability.get()[active_player]},
            sample_probability,
            sampled_action,
            std::invoke(active_weight_param)
         );
      }
   }

   // predictive baseline side channel: every frame publishes its outgoing
   // next-strategy value stream at its own depth slot; non-updating player
   // frames pass their child's stream through unchanged (updating frames wrote
   // their second mixture above, terminal frames seed the raw reward)
   if constexpr(vr_predictive_active) {
      if(not vr_active) {
         _vr_secondary_slot(depth) = _vr_secondary_slot(depth + 1);
      }
   }

   // ---- B8 (probing): root-side value diagnostic ---------------------------
   // Every engaged updater frame folds its OWN subtree in expectation via the
   // sigma-mixture of the probed counterfactual value vector -- the paper's
   // v_hat_i(sigma, I) = sum_a sigma(I,a) v_hat(a)/q_i(I) aggregation. Because
   // the traversal MUST keep returning vanilla-flow (value, tail) pairs --
   // ancestor updates derive their sampled-branch estimates from them, and any
   // rescaling here would break the per-unit-prefix consistency documented in
   // rm::ProbingSamplingRule -- the folded mixture is published through the
   // 'm_probe_root_value' diagnostic slot instead of the return channel.
   // Post-order unwinding means the LAST write per iteration comes from the
   // SHALLOWEST engaged frame, i.e. the one closest to the root.
   if constexpr(probing_active) {
      if(probing_engaged) {
         m_probe_root_value = std::ranges::fold_left(
            std::views::iota(size_t{0}, actions.size()) | std::views::transform([&](size_t idx) {
               return action_policy[actions[idx]] * probed_action_values[idx];
            }),
            0.,
            std::plus{}
         );
      }
   }

   return std::pair{std::move(action_value_map), Probability{tail_prob.get() * action_policy_prob}};
}

template <
   MCCFRConfig config,
   typename Env,
   typename Policy,
   typename AveragePolicy,
   typename SamplingRule >
auto MCCFR< config, Env, Policy, AveragePolicy, SamplingRule >::_terminal_value(
   world_state_type& state,
   std::optional< Player > player_to_update,
   Probability sample_probability,
   size_t depth
)
   requires(config.algorithm == MCCFRAlgorithmMode::outcome_sampling)
{
   // every terminal frame seeds the predictive side channel with the raw
   // updater value: the deepest updating infoset's eq-(9) correction and the
   // predictive regression both consume it unchanged
   if constexpr(vr_predictive_active) {
      _vr_secondary_slot(depth) = std::invoke([&] {
         if constexpr(config.update_mode == UpdateMode::alternating) {
            return _env().reward(player_to_update.value(), state);
         } else {
            return collect_rewards(_env(), state).at(player_to_update.value());
         }
      });
   }
   if constexpr(vr_mode != VarianceReductionMode::none) {
      // VR-MCCFR seeds the trajectory with the RAW terminal reward: the
      // importance weights are applied later, once per sampled action
      // (eq 9's 1/ξ(h,a*)) and once per updated infoset (eq 11's 1/q(h)),
      // so dividing by the full-path sampling probability here would double-
      // count them.
      if constexpr(config.update_mode == UpdateMode::alternating) {
         return std::pair{
            StateValueMap{std::unordered_map< Player, double >{
               {player_to_update.value(), _env().reward(player_to_update.value(), state)}}},
            Probability{1.}};
      } else if constexpr(config.update_mode == UpdateMode::simultaneous) {
         return std::pair{StateValueMap{collect_rewards(_env(), state)}, Probability{1.}};
      } else {
         static_assert(
            common::always_false_v< Env >, "Update Mode not one of alternating or simultaneous"
         );
      }
   } else if constexpr(config.update_mode == UpdateMode::alternating) {
      return std::pair{
         StateValueMap{std::unordered_map< Player, double >{
            {player_to_update.value(),
             _env().reward(player_to_update.value(), state) / sample_probability.get()}}},
         Probability{1.}};
   } else if constexpr(config.update_mode == UpdateMode::simultaneous) {
      return std::pair{
         StateValueMap{[&] {
            auto rewards_map = collect_rewards(_env(), state);
            for(double& reward : rewards_map.values()) {
               reward /= sample_probability.get();
            }
            return rewards_map;
         }()},
         Probability{1.}};
   } else {
      static_assert(
         common::always_false_v< Env >, "Update Mode not one of alternating or simultaneous"
      );
   }
}

template <
   MCCFRConfig config,
   typename Env,
   typename Policy,
   typename AveragePolicy,
   typename SamplingRule >
template < typename BaselineAt >
void MCCFR< config, Env, Policy, AveragePolicy, SamplingRule >::_vr_accumulate_regrets(
   infostate_data_type& infostate_data,
   const std::vector< action_type >& actions,
   const auto& action_policy,
   size_t sampled_idx,
   double cf_weight,
   double sampled_value,
   BaselineAt&& baseline_at
) const
   requires(config.algorithm == MCCFRAlgorithmMode::outcome_sampling)
{
   // Paper step (d): accumulate R(I,a) += v̂ᵇ(I,a) - v̂ᵇ(I) for EVERY action,
   // where (eq 11) v̂ᵇ(I,a) = pi_-i(h)/q(h) * ûᵇ(h,a), the sampled action's
   // value coming from eq (9)'s sampled branch and every off-trajectory
   // action valued by its baseline. The baselines are read through the
   // 'baseline_at' accessor -- the same snapshot that produced 'sampled_value'
   // (no mutation has happened yet).
   auto action_value_of = [&](auto idx) {
      return idx == sampled_idx ? sampled_value : baseline_at(idx);
   };
   double node_value = 0.;
   for(auto idx : std::views::iota(size_t{0}, actions.size())) {
      node_value += action_policy[actions[idx]] * action_value_of(idx);
   }
   node_value *= cf_weight;
   for(auto idx : std::views::iota(size_t{0}, actions.size())) {
      // fold through the minimizer's 'observe' (bit-for-bit identical to the
      // historical direct table write for plain regret matching)
      m_regret_minimizer.observe(
         infostate_data.data(), actions[idx], cf_weight * action_value_of(idx) - node_value
      );
   }
}

template <
   MCCFRConfig config,
   typename Env,
   typename Policy,
   typename AveragePolicy,
   typename SamplingRule >
void MCCFR< config, Env, Policy, AveragePolicy, SamplingRule >::_vr_regress_running_mean(
   size_t sampled_idx,
   double snapshot,
   double target,
   auto&& regress_sampled_edge
) const
   requires(config.algorithm == MCCFRAlgorithmMode::outcome_sampling)
{
   // paper step (e): regress the sampled edge's baseline onto its
   // baseline-corrected estimate b̂ <- b̂ + β(v̂ - b̂). The write goes through
   // the caller-provided sink so both the per-infostate table and the ESCHER
   // history-value store share this rule.
   regress_sampled_edge(sampled_idx, snapshot + config.baseline_update_rate * (target - snapshot));
}

template <
   MCCFRConfig config,
   typename Env,
   typename Policy,
   typename AveragePolicy,
   typename SamplingRule >
void MCCFR< config, Env, Policy, AveragePolicy, SamplingRule >::_update_regrets(
   const ReachProbabilityMap& reach_probability,  // = pi(z[I])
   Player active_player,
   infostate_data_type& infostate_data,  // = -->r(I) and A(I)
   const action_type& sampled_action,  // = 'a', the sampled action
   Probability sampled_action_policy_prob,  // = sigma(I, a) for the sampled action
   StateValue action_value,  // = u(z[I]a)
   Probability tail_prob  // = pi(z[I]a, z)
) const
   requires(config.algorithm == MCCFRAlgorithmMode::outcome_sampling)
{
   auto cf_value_weight = action_value.get()
                           * cf_reach_probability(active_player, reach_probability.get());
   for(const auto& action : infostate_data.actions()) {
      // compute the estimated counterfactual regret increment and fold it into
      // the minimizer's tables. Routing through 'observe' keeps the plain-RM
      // arithmetic bit-for-bit identical to the historical direct table write,
      // while the predictive kernels (MCCFR+) apply their clip-at-fold +
      // instantaneous-prediction-buffer bookkeeping here.
      const double increment = [&] {
         if(action == sampled_action) {
            // note that tail_prob = pi(z[I]a, z)
            // the probability pi(z[I]a, z) - pi(z[I], z) can also be expressed as
            // pi(z[I]a, z) * (1 - sigma(I, a)), since
            //    pi(h, z) = pi(z) / pi(h)   and    pi(ha) = pi(h) * sigma(I[h], a)
            // --> pi(ha, z) - pi(h, z) = pi(z) / (pi(h) * sigma(I, a)) - pi(z) / pi(h)
            //                          = pi(z) / (pi(h) * sigma(I, a)) * ( 1 - sigma(I, a))
            //                          = pi(ha, z) * ( 1 - sigma(I, a))
            return cf_value_weight * tail_prob.get() * (1. - sampled_action_policy_prob.get());
         } else {
            // we are returning here the formula: -W * pi(z[I], z)
            return -cf_value_weight * tail_prob.get() * sampled_action_policy_prob.get();
         }
      }();
      m_regret_minimizer.observe(infostate_data.data(), action, increment);
   }
}

template <
   MCCFRConfig config,
   typename Env,
   typename Policy,
   typename AveragePolicy,
   typename SamplingRule >
void MCCFR< config, Env, Policy, AveragePolicy, SamplingRule >::_update_regrets_probing(
   const ReachProbabilityMap& reach_probability,
   Player active_player,
   infostate_data_type& infostate_data,  // -->r(I) and A(I)
   const std::vector< action_type >& actions,
   const auto& action_policy,  // = sigma(I, .)
   const std::vector< double >& probed_action_values  // = v_hat(a), registry-indexed
) const
   requires(config.algorithm == MCCFRAlgorithmMode::outcome_sampling)
{
   // Gibson et al. (AAAI 2012), Algorithm 1 lines 40-44: with the probed
   // counterfactual value vector available for EVERY action, the regret
   // increment is the plain CFR form applied to the estimates:
   //    r(I,a) += pi_-i(h_I) * (v_hat(a) - sum_a' sigma(I,a') v_hat(a'))
   // Every v_hat entry is unbiased for v(h_I a)/q_prefix on the COMMON
   // 1/sample_probability scale, so in expectation this accumulates the TRUE
   // per-iteration counterfactual regret vector R(I,.) -- no visit-frequency
   // rescaling and no per-coordinate projection as under vanilla outcome
   // sampling.
   double node_value = 0.;
   for(auto idx : std::views::iota(size_t{0}, actions.size())) {
      node_value += action_policy[actions[idx]] * probed_action_values[idx];
   }
   const double cf_reach = cf_reach_probability(active_player, reach_probability.get());
   for(auto idx : std::views::iota(size_t{0}, actions.size())) {
      infostate_data.regret(actions[idx]) += cf_reach * (probed_action_values[idx] - node_value);
   }
}

template <
   MCCFRConfig config,
   typename Env,
   typename Policy,
   typename AveragePolicy,
   typename SamplingRule >
auto MCCFR< config, Env, Policy, AveragePolicy, SamplingRule >::_probe_rollout(
   world_state_type& state,
   size_t depth,
   player_hashmap< std::vector< std::pair< observation_type, observation_type > > >&
      observation_buffer,
   player_hashmap< sptr< info_state_type > >& infostates,
   Player updating_player
) -> double
   requires(config.algorithm == MCCFRAlgorithmMode::outcome_sampling)
{
   // Gibson et al. (AAAI 2012), Algorithm 1 'Probe': descend from h along a
   // single on-policy trajectory -- chance from the true distribution, all
   // players from the current strategy sigma^t -- and return u_i(z). Pure
   // value recursion: no nodes are created and no tables are mutated.
   if(_env().is_terminal(state)) {
      return _env().reward(updating_player, state);
   }

   Player active_player = _env().active_player(state);

   if constexpr(not concepts::deterministic_fosg< env_type >) {
      if(active_player == Player::chance) {
         auto chosen_outcome = _sample_outcome< false >(state);
         world_state_type& next_state = _arena_state(depth + 1, state);
         _env().transition(next_state, chosen_outcome);
         next_infostate_and_obs_buffers_inplace(
            _env(), observation_buffer, infostates, state, chosen_outcome, next_state
         );
         return _probe_rollout(next_state, depth + 1, observation_buffer, infostates, updating_player);
      }
   }

   const auto& live_infostate_sptr = infostates.at(active_player);
   const auto actions = _env().actions(active_player, state);
   // READ-ONLY current-policy lookup: rollout-only histories must not
   // materialize new table entries (fetch_policy would insert them), so
   // infostates never recommended before fall back to uniform play here
   const auto& player_policy_table = this->_policy().at(active_player);
   const auto policy_entry_iter = player_policy_table.find(*live_infostate_sptr);
   const bool has_policy_entry = policy_entry_iter != player_policy_table.end();
   const auto& chosen_action =
      common::choose(
         actions,
         [&](const action_type& act) {
            return has_policy_entry ? policy_entry_iter->second.at(act)
                                    : 1. / static_cast< double >(actions.size());
         },
         m_rng
      );

   world_state_type& next_state = _arena_state(depth + 1, state);
   _env().transition(next_state, chosen_action);
   next_infostate_and_obs_buffers_inplace(
      _env(), observation_buffer, infostates, state, chosen_action, next_state
   );
   return _probe_rollout(next_state, depth + 1, observation_buffer, infostates, updating_player);
}

template <
   MCCFRConfig config,
   typename Env,
   typename Policy,
   typename AveragePolicy,
   typename SamplingRule >
void MCCFR< config, Env, Policy, AveragePolicy, SamplingRule >::_update_average_policy(
   const info_state_type& infostate,
   infostate_data_type& infonode_data,
   const auto& current_policy,
   Probability reach_prob,
   Probability sample_prob,
   const action_type& sampled_action,
   ConditionalWeight weight
)
   requires(config.algorithm == MCCFRAlgorithmMode::outcome_sampling)
{
   auto& avg_policy = this->template fetch_policy< false >(infostate, infonode_data.actions());

   if constexpr(config.weighting == MCCFRWeightingMode::lazy) {
      for(auto idx : std::views::iota(size_t{0}, infonode_data.actions().size())) {
         const auto& action = infonode_data.actions()[idx];
         auto policy_incr = (weight.get() + reach_prob.get()) * current_policy[action];
         avg_policy[action] += policy_incr;
         if(action == sampled_action) [[unlikely]] {
            infonode_data.data().extras.pending_avg_accumulator[idx] = 0.;
         } else [[likely]] {
            infonode_data.data().extras.pending_avg_accumulator[idx] += policy_incr;
         }
      }
   }

   if constexpr(config.weighting == MCCFRWeightingMode::optimistic) {
      auto& infostate_last_visit = infonode_data.data().extras.last_visit_iteration;
      auto current_iter = _iteration();
      // we add + 1 to the current iter counter, since the iterations start counting at 0
      auto last_visit_difference = static_cast< double >(1 + current_iter - infostate_last_visit);
      for(const action_type& action : infonode_data.actions()) {
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

template <
   MCCFRConfig config,
   typename Env,
   typename Policy,
   typename AveragePolicy,
   typename SamplingRule >
auto MCCFR< config, Env, Policy, AveragePolicy, SamplingRule >::_sample_action_on_policy(
   const std::vector< action_type >& actions,
   auto& action_policy
)
{
   return common::choose(
      actions, [&](const auto& act) { return action_policy[act]; }, m_rng
   );
}

template <
   MCCFRConfig config,
   typename Env,
   typename Policy,
   typename AveragePolicy,
   typename SamplingRule >
auto MCCFR< config, Env, Policy, AveragePolicy, SamplingRule >::_sample_action(
   Player active_player,
   std::optional< Player > player_to_update,
   const std::vector< action_type >& actions,
   auto& action_policy,
   const info_state_type& infostate
)
{
   // we first define the sampling schemes:
   // 1. Sampling directly from policy calls the policy map as many times as there are options to
   // choose from and returns the sampled action, its policy probability, and its policy
   // probability again (for API consistency)
   auto on_policy_sampling = [&] {
      // in the non-epsilon case we simply use the player's policy to sample the next move
      // from. Thus, in this case, the action's sample probability and action's policy
      // probability are the same, i.e. action_sample_prob = action_policy_prob in the return
      // value
      const auto& chosen_action = _sample_action_on_policy(actions, action_policy);
      auto action_prob = action_policy[chosen_action];
      return std::tuple{chosen_action, action_prob, action_prob};
   };

   // 2. Epsilon-On-Policy sampling with respect to the policy map executes two steps: first, it
   // decides whether we sample uniformly from the actions or not. If so, it executes a separate
   // branch for uniform sampling. Alternatively it reverts to sampling procedure 1. and
   // adapts the sampling likelihood for the chosen sample.
   // This samples values according to the policy:
   //    epsilon * uniform(A(I)) + (1 - epsilon) * policy(I)
   auto epsilon_on_policy_sampling = [&] {
      double uniform_prob = 1. / static_cast< double >(actions.size());
      if(m_uniform_01_dist(m_rng) < m_epsilon) {
         // with probability epsilon we do exploration, i.e. uniform sampling, over all actions
         // available. This is a tiny speedup over querying the actual policy map for the
         // epsilon-on-policy enhanced likelihoods
         const auto& chosen_action = common::choose(actions, m_rng);
         return std::tuple{
            chosen_action,
            m_epsilon * uniform_prob + (1 - m_epsilon) * action_policy[chosen_action],
            action_policy[chosen_action]};
      } else {
         // if we don't explore, then we simply sample according to the policy.
         // BUT: Since in theory we have done epsilon-on-policy exploration, yet merely in two
         // separate steps, we need to adapt the returned sampling probability to the
         // epsilon-on-policy probability of the sampled action
         const auto& [chosen_action, _, action_prob] = on_policy_sampling();
         return std::tuple{
            std::move(chosen_action),
            m_epsilon * uniform_prob + (1 - m_epsilon) * action_prob,
            action_prob};
      }
   };

   // B6: an injectable custom sampling rule (ESCHER/bandit agents) takes
   // precedence whenever the config selects it. The rule reports the action
   // and its SAMPLING probability; the policy probability is derived here so
   // downstream importance weights stay unchanged.
   // B7: average-strategy rules (ASS, Gibson et al., NIPS 2012) are a tagged
   // special case reading the ACCUMULATED AVERAGE strategy table instead of
   // the current policy (protocol change documented in sampling_rules.hpp).
   if constexpr(config.exploration == MCCFRExplorationMode::custom_sampling_policy) {
      if constexpr(average_strategy_sampling_rule< SamplingRule >) {
         // placement mirrors the epsilon-on-policy scheme: the rule acts at the
         // updating player's infosets (every actual player's under simultaneous
         // updates); opponent infosets stay on-policy over the CURRENT strategy.
         if(_epsilon_mixed_sampling_active(player_to_update, active_player)) {
            auto& average_action_policy = this->template fetch_policy< PolicyLabel::average >(
               infostate, actions
            );
            const auto [chosen_action, sample_prob] = m_sampling_rule(
               m_rng,
               actions,
               [&average_action_policy](const action_type& act) {
                  return average_action_policy[act];
               }
            );
            return std::tuple{chosen_action, sample_prob, action_policy[chosen_action]};
         }
         return on_policy_sampling();
      } else {
         // untagged rules replace the epsilon-mixture wherever that mixture
         // would apply; pure on-policy sampling applies elsewhere. This makes an
         // injected EpsilonOnPolicySamplingRule draw-for-draw identical to the
         // built-in epsilon_on_policy exploration.
         if(_epsilon_mixed_sampling_active(player_to_update, active_player)) {
            const auto [chosen_action, sample_prob] = m_sampling_rule(
               m_rng, actions, [&](const action_type& act) { return action_policy[act]; }
            );
            return std::tuple{chosen_action, sample_prob, action_policy[chosen_action]};
         }
         return on_policy_sampling();
      }
   }

   // ESCHER (McAleer et al., ICLR 2023, sec. 3): the UPDATING player samples
   // from a FIXED uniform distribution over legal actions -- iteration-
   // invariant by construction, which is what lets the VR machinery drop all
   // importance corrections. Opponents keep sampling on-policy from their
   // current strategy. The returned policy probability remains the current
   // strategy's so reach tracking / average-policy updates are unaffected;
   // the sampling probability (uniform) is only consumed by the importance
   // corrections that fixed_uniform mode disables.
   if constexpr(config.updater_sampling == UpdaterSamplingMode::fixed_uniform) {
      if(player_to_update.has_value() and active_player == *player_to_update) {
         const auto& chosen_action = common::choose(actions, m_rng);
         const double uniform_prob = 1. / static_cast< double >(actions.size());
         return std::tuple{chosen_action, uniform_prob, action_policy[chosen_action]};
      }
   }

   // here we now decide what sampling procedure is exactly executed. It depends on the MCCFR
   // config given and then on the specific algorithm's sampling scheme
   if constexpr(config.algorithm == MCCFRAlgorithmMode::outcome_sampling) {
      if(_epsilon_mixed_sampling_active(player_to_update, active_player)) {
         // if we do simultaneous updates we need to explore for each player that we update!
         return epsilon_on_policy_sampling();
      } else {
         return on_policy_sampling();
      }
   } else {
      // currently, for all other algorithms we always sample according to the policy
      return on_policy_sampling();
   }
}

template <
   MCCFRConfig config,
   typename Env,
   typename Policy,
   typename AveragePolicy,
   typename SamplingRule >
template < bool return_likelihood >
auto MCCFR< config, Env, Policy, AveragePolicy, SamplingRule >::_sample_outcome(
   const world_state_type& state
)
{
   auto chance_actions = _env().chance_actions(state);
   // linear-scan probability table over the (few) legal outcomes of a chance
   // node: replaces the former per-chance-visit unordered_map built through
   // ranges::to. The weights are materialized here in chance_actions order --
   // exactly the order in which common::choose queries them below -- so the
   // discrete distribution (and hence the consumed RNG stream) is identical.
   std::vector< std::pair< chance_outcome_type, double > > chance_probabilities;
   chance_probabilities.reserve(chance_actions.size());
   for(const auto& outcome : chance_actions) {
      chance_probabilities.emplace_back(outcome, _env().chance_probability(state, outcome));
   }
   auto probability_of = [&chance_probabilities](const chance_outcome_type& outcome) -> double {
      const auto found = std::ranges::find_if(chance_probabilities, [&](const auto& outcome_prob) {
         return outcome_prob.first == outcome;
      });
      assert(found != chance_probabilities.end());
      return found->second;
   };
   auto& chosen_outcome = common::choose(
      chance_actions, [&](const auto& outcome) { return probability_of(outcome); }, m_rng
   );

   if constexpr(return_likelihood) {
      double chance_prob = probability_of(chosen_outcome);
      return std::tuple{std::move(chosen_outcome), chance_prob};
   } else {
      return chosen_outcome;
   }
}

template <
   MCCFRConfig config,
   typename Env,
   typename Policy,
   typename AveragePolicy,
   typename SamplingRule >
template < bool return_likelihood >
auto MCCFR< config, Env, Policy, AveragePolicy, SamplingRule >::_sample_outcome_pcs(
   const world_state_type& state
)
{
   auto chance_actions = _env().chance_actions(state);
   if(chance_actions.empty()) {
      throw std::logic_error("PCS chance resolution invoked on a state without legal outcomes.");
   }
   // the classification of the EVENT uses its first legal outcome as
   // representative; every provided game-side trait classifies per event, not
   // per individual outcome identity.
   if(concepts::has::public_chance_event(_env(), state, chance_actions.front())) {
      // public chance event: sample from the distribution exactly like vanilla OS
      return _sample_outcome< return_likelihood >(state);
   }
   // PRIVATE chance event (single-trajectory PCS): resolve deterministically to
   // the FIRST legal outcome without consuming RNG. The importance-weight
   // correction is threaded through the sample-probability accumulator by the
   // caller via the returned TRUE probability of the chosen outcome; reach
   // bookkeeping is untouched. See rm::PublicChanceSamplingRule for what this
   // deviates from Gibson's multi-view formulation.
   const auto& chosen_outcome = chance_actions.front();
   const double chance_prob = _env().chance_probability(state, chosen_outcome);
   if constexpr(return_likelihood) {
      return std::tuple{chosen_outcome, chance_prob};
   } else {
      return chosen_outcome;
   }
}

////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// External-Sampling MCCFR ////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

template <
   MCCFRConfig config,
   typename Env,
   typename Policy,
   typename AveragePolicy,
   typename SamplingRule >
StateValue MCCFR< config, Env, Policy, AveragePolicy, SamplingRule >::_traverse(
   Player player_to_update,
   world_state_type& state,
   size_t depth,
   ObservationbufferMap& observation_buffer,
   InfostateSptrMap& infostates,
   delayed_update_set& infostates_to_update
)
   // clang-format off
   requires(
      config.algorithm == MCCFRAlgorithmMode::external_sampling
      or (
         config.algorithm == MCCFRAlgorithmMode::pure_cfr
         and config.update_mode == UpdateMode::alternating
      )
   )
// clang-format on
{
   Player active_player = _env().active_player(state);

   if(_env().is_terminal(state)) {
      return StateValue{_env().reward(player_to_update, state)};
   }

   // now we check first if we even need to consider a chance player, as the env could be
   // simply deterministic. In that case we might need to traverse the chance player's actions
   // or an active player's actions
   if constexpr(not concepts::deterministic_fosg< env_type >) {
      if(active_player == Player::chance) {
         auto chosen_outcome = _sample_outcome< false >(state);

         world_state_type& next_state = _arena_state(depth + 1, state);
         _env().transition(next_state, chosen_outcome);

         next_infostate_and_obs_buffers_seated(
            _env(), observation_buffer, infostates, state, chosen_outcome, next_state
         );

         // single sampled outcome: buffers/infostates flow permanently downward
         return _traverse(
            player_to_update,
            next_state,
            depth + 1,
            observation_buffer,
            infostates,
            infostates_to_update
         );
      }
   }

   // fetch (or create) this infostate's node without cloning the infostate on
   // every visit: heterogeneous value lookup finds existing entries directly
   const auto& live_infostate_sptr = infostates.at(active_player);
   auto infostate_and_data_iter = m_infonode.find(*live_infostate_sptr);
   bool success = false;
   if(infostate_and_data_iter == m_infonode.end()) {
      std::tie(infostate_and_data_iter, success) = _infonodes().emplace(
         sptr< info_state_type >{utils::clone_any_way(live_infostate_sptr)}, infostate_data_type{}
      );
   }
   const auto& infostate = infostate_and_data_iter->first;
   auto& infonode_data = infostate_and_data_iter->second;
   if(success) {
      // success means we have indeed emplaced a new data node, instead of simply fetching an
      // existing one. We thus need to fill it with the legal actions at this node.
      infonode_data.emplace(_env().actions(active_player, state));
   }
   if constexpr(config.algorithm == MCCFRAlgorithmMode::pure_cfr) {
      infostates_to_update.emplace(infostate.get(), &infonode_data);
   } else {
      // for external sampling we can simply minimize upon traversal
      _invoke_regret_minimizer(common::deref(infostate), infonode_data);
   }
   const auto& actions = infonode_data.actions();
   auto& action_policy = this->template fetch_policy< PolicyLabel::current >(*infostate, actions);

   auto traverse_for_action_value = [&](const auto& action) {
      // advance the arena slot of the next depth: in-place reconstruction + transition
      world_state_type& next_state = _arena_state(depth + 1, state);
      _env().transition(next_state, action);

      // fold this edge's observations into the live seat-indexed containers
      // exactly like the inplace helper would -- and restore everything after
      // the recursion so sibling edges start from identical pre-edge container
      // states.
      auto& obs_table = observation_buffer;
      const auto public_obs = _env().public_observation(state, action, next_state);
      const Player next_active_player = _env().active_player(next_state);
      // null pointer stands in for the former contains()==false
      const auto infostate_slot = next_active_player == Player::chance
                                     ? nullptr
                                     : infostates.find(next_active_player);
      const bool flushes = infostate_slot != nullptr;
      sptr< info_state_type > saved_flush_target{};
      sptr< info_state_type > child_infostate{};
      if(flushes) {
         // advance a CLONE of the next-active player's infostate so the live
         // entry can be restored verbatim after the recursion; the flush
         // target's buffer is SWAPPED into this frame's reusable scratch slot
         // instead of being heap-copied
         saved_flush_target = *infostate_slot;
         child_infostate = std::make_shared< info_state_type >(*saved_flush_target);
         auto& obs_scratch = this->_obs_scratch_slot(depth);
         // drop dead residue a previous sibling edge left in the scratch slot
         // so that the flush target's live buffer is EMPTY during the recursion
         // below (the swap keeps capacities on both sides)
         obs_scratch.clear();
         obs_scratch.swap(obs_table[next_active_player]);
      }
      // fixed-size STACK bookkeeping bounded by max_player_seats: sizes of
      // exactly the buffers this edge appends to (replaces the former heap
      // vector<pair<Player,size_t>>)
      std::array< size_t, max_player_seats > saved_buffer_sizes{};
      std::array< size_t, max_player_seats > saved_seats{};
      size_t saved_size_count = 0;
      for(auto player : _env().players(next_state)) {
         if(player == Player::chance) {
            continue;
         }
         if(not (flushes and player == next_active_player)) {
            const auto seat_idx = static_cast< size_t >(player);
            saved_seats[saved_size_count] = seat_idx;
            saved_buffer_sizes[seat_idx] = obs_table[player].size();
            ++saved_size_count;
         }
      }
      for(auto player : _env().players(next_state)) {
         if(player == Player::chance) {
            continue;
         }
         if(flushes and player == next_active_player) {
            // drain the pre-edge history (held by the scratch slot after the
            // swap) into the clone and clear the scratch for reuse; the live
            // buffer stays empty during the recursion below
            auto& scratch_history = this->_obs_scratch_slot(depth);
            // NOTE: the scratch slot now OWNS the pre-edge buffer contents
            // (acquired via the swap above); they are the restoration source
            // for sibling edges. Drain them by COPY -- moving out or clearing
            // here would destroy the saved state.
            for(auto& obs : scratch_history) {
               ::nor::detail::update_infostate(child_infostate, obs.first, obs.second);
            }
            ::nor::detail::update_infostate(
               child_infostate,
               public_obs,
               _env().private_observation(player, state, action, next_state)
            );
         } else {
            obs_table[player].emplace_back(
               public_obs, _env().private_observation(player, state, action, next_state)
            );
         }
      }

      if(flushes) {
         *infostate_slot = std::move(child_infostate);
      }

      double value = _traverse(
                        player_to_update,
                        next_state,
                        depth + 1,
                        observation_buffer,
                        infostates,
                        infostates_to_update
      )
                        .get();

      if(flushes) {
         *infostate_slot = std::move(saved_flush_target);
         this->_obs_scratch_slot(depth).swap(obs_table[next_active_player]);
      }
      for(auto idx : std::views::iota(size_t{0}, saved_size_count)) {
         const auto seat_idx = saved_seats[idx];
         obs_table[static_cast< Player >(seat_idx)].resize(saved_buffer_sizes[seat_idx]);
      }
      return value;
   };

   auto sample_or_fetch_action = [&]() -> decltype(auto) {
      if constexpr(config.algorithm == MCCFRAlgorithmMode::pure_cfr) {
         auto& sampled_action_opt = infonode_data.data().extras.sampled_action;
         return sampled_action_opt.has_value()
                   ? *sampled_action_opt
                   : sampled_action_opt.emplace(_sample_action_on_policy(actions, action_policy));
      } else {
         return _sample_action_on_policy(actions, action_policy);
      }
   };

   if(active_player == player_to_update) {
      // for the traversing player we explore all actions possible

      // the first round of action iteration we will traverse the tree further to find all action
      // values from this node and compute the state value of the current node.
      // insertion-ordered linear-scan table replaces the former unordered_map
      // (one bucket array + node allocation per legal action, per infoset
      // visit); lookups stay O(#actions).
      std::vector< std::pair< action_type, double > > value_estimates;
      value_estimates.reserve(actions.size());
      // mirrors unordered_map::emplace's keep-the-existing-entry semantics and
      // returns a reference to the STORED estimate
      auto emplace_estimate = [&](const action_type& action, double estimate) -> double& {
         const auto found = std::ranges::find_if(value_estimates, [&](const auto& entry) {
            return entry.first == action;
         });
         if(found != value_estimates.end()) {
            return found->second;
         }
         value_estimates.emplace_back(action, estimate);
         return value_estimates.back().second;
      };
      auto estimate_of = [&](const action_type& action) -> const double& {
         const auto found = std::ranges::find_if(value_estimates, [&](const auto& entry) {
            return entry.first == action;
         });
         assert(found != value_estimates.end());
         return found->second;
      };

      auto state_value_estimate = std::invoke([&] {
         if constexpr(config.algorithm == MCCFRAlgorithmMode::external_sampling) {
            return std::ranges::fold_left(
               actions | std::views::transform([&](const auto& action) {
                  return emplace_estimate(action, traverse_for_action_value(action))
                         * action_policy[action];
               }),
               double(0.),
               std::plus{}
            );
         } else {
            // pure cfr samples a designated action first as the pure strategy action at this
            // infoset, collects the value of each action and then updates (in another iteration)
            // the actions with their value difference to the sampled action's value.
            for(const auto& action : actions) {
               emplace_estimate(action, traverse_for_action_value(action));
            }
            return estimate_of(sample_or_fetch_action());
         }
      });
      // in the second round of action iteration we update the regret of each action through the
      // previously found action and state values
      for(const auto& action : actions) {
         infonode_data.regret(action) += estimate_of(action) - state_value_estimate;
      }

      return StateValue{state_value_estimate};
   } else {
      // for the non-traversing player we sample a single action and continue;
      auto&& sampled_action = sample_or_fetch_action();

      if(active_player == _preview_next_player_to_update()) {
         // this update scheme represents the 'simple' update plan mentioned in open_spiel. We
         // are updating the policy if the active player is the next player to be updated in the
         // update cycle. Updates the average policy with the current policy
         auto& average_action_policy = this->template fetch_policy< PolicyLabel::average >(
            *infostate, actions
         );
         if constexpr(config.algorithm == MCCFRAlgorithmMode::pure_cfr) {
            // we do not need to update the other actions since we sampled first a pure strategy
            // and then sampled from said strategy (other action sampling prob is thus 0)
            average_action_policy[sampled_action] += 1;
         } else {
            // external sampling updates all entries by the current policy
            for(const auto& action : actions) {
               average_action_policy[action] += action_policy[action];
            }
         }
      }
      return StateValue{traverse_for_action_value(sampled_action)};
   }
}

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////// Chance-Sampling MCCFR & Pure CFR Sim. Updating //////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

template <
   MCCFRConfig config,
   typename Env,
   typename Policy,
   typename AveragePolicy,
   typename SamplingRule >
StateValueMap MCCFR< config, Env, Policy, AveragePolicy, SamplingRule >::_traverse(
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
      )
// clang-format on
{
   if(_env().is_terminal(curr_worldstate)) {
      return StateValueMap{collect_rewards(_env(), curr_worldstate)};
   }

   if constexpr(config.algorithm != MCCFRAlgorithmMode::pure_cfr and config.pruning_mode == CFRPruningMode::partial) {
      if(_partial_pruning_condition(player_to_update, reach_probability)) {
         // if the entire subtree is pruned then the values that could be found are all 0. for
         // each player
         return StateValueMap{std::invoke([&] {
            StateValueMap::UnderlyingType map;
            for(auto player : _env().players(curr_worldstate) | utils::is_actual_player_pred) {
               map[player] = 0.;
            }
            return map;
         })};
      }
   }

   Player active_player = _env().active_player(curr_worldstate);
   // the state's value for each player. To be filled by the action traversal functions.
   StateValueMap state_value{};
   // each action's value for each player. To be filled by the action traversal functions.
   // REUSED STORAGE: this depth's arena slot replaces the former fresh
   // unordered_map per visit (see rm/action_value_table.hpp); DFS never
   // interleaves same-depth frames, so clearing on entry/exit keeps successive
   // visitors of a depth isolated without any content save/restore.
   if(action_value_arena.size() <= depth) {
      action_value_arena.resize(depth + 1);
   }
   auto& action_value = action_value_arena[depth];
   action_value.clear();
   // traverse all child states from this state. The constexpr check for determinism in the env
   // allows deterministic envs to not provide certain functions that are only needed in the
   // stochastic case.
   if constexpr(concepts::stochastic_env< env_type >) {
      if(active_player == Player::chance) {
         auto [chosen_outcome, _] = _sample_outcome(curr_worldstate);
         world_state_type& next_state = _arena_state(depth + 1, curr_worldstate);
         _env().transition(next_state, chosen_outcome);

         next_infostate_and_obs_buffers_seated(
            _env(), observation_buffer, infostates, curr_worldstate, chosen_outcome, next_state
         );

         // single sampled outcome: containers flow permanently downward
         return _traverse(
            player_to_update,
            next_state,
            depth + 1,
            reach_probability,
            observation_buffer,
            infostates,
            infostates_to_update,
            action_value_arena
         );
      }
   }
   const auto& live_infostate_sptr = infostates.at(active_player);
   auto infostate_and_data_iter = m_infonode.find(*live_infostate_sptr);
   bool success = false;
   if(infostate_and_data_iter == m_infonode.end()) {
      std::tie(infostate_and_data_iter, success) = _infonodes().emplace(
         sptr< info_state_type >{utils::clone_any_way(live_infostate_sptr)}, infostate_data_type{}
      );
   }
   const auto& infostate = infostate_and_data_iter->first;
   auto& infonode_data = infostate_and_data_iter->second;
   infostates_to_update.emplace(infostate.get(), &infonode_data);
   if(success) {
      // success means we have indeed emplaced a new data node, instead of simply fetching an
      // existing one.
      // We thus need to fill it with the legal actions at this node.
      infonode_data.emplace(_env().actions(active_player, curr_worldstate));
   }
   const auto& actions = infonode_data.actions();
   auto& curr_action_policy = this->template fetch_policy< PolicyLabel::current >(
      *infostate, actions
   );
   auto& avg_action_policy = this->template fetch_policy< PolicyLabel::average >(
      *infostate, actions
   );

   for(const action_type& action : actions) {
      auto action_prob = curr_action_policy[action];

      // advance the arena slot of the next depth: in-place reconstruction + transition
      world_state_type& next_wstate = _arena_state(depth + 1, curr_worldstate);
      _env().transition(next_wstate, action);
      // fold this edge's observations into the live seat-indexed containers
      // exactly like the inplace helper would -- restoring everything after the
      // recursion so sibling edges start from identical pre-edge container states.
      auto& obs_table = observation_buffer;
      const auto public_obs = _env().public_observation(curr_worldstate, action, next_wstate);
      const Player next_active_player = _env().active_player(next_wstate);
      // null pointer stands in for the former contains()==false
      const auto infostate_slot = next_active_player == Player::chance
                                     ? nullptr
                                     : infostates.find(next_active_player);
      const bool flushes = infostate_slot != nullptr;
      sptr< info_state_type > saved_flush_target{};
      sptr< info_state_type > child_infostate{};
      if(flushes) {
         saved_flush_target = *infostate_slot;
         child_infostate = std::make_shared< info_state_type >(*saved_flush_target);
         // flush target's buffer swapped into this frame's reusable scratch slot
         auto& obs_scratch = this->_obs_scratch_slot(depth);
         // drop dead residue a previous sibling edge left in the scratch slot
         // (see the chance_sampling traversal for the full rationale)
         obs_scratch.clear();
         obs_scratch.swap(obs_table[next_active_player]);
      }
      // fixed-size STACK bookkeeping bounded by max_player_seats (replaces the
      // former heap vector<pair<Player,size_t>>)
      std::array< size_t, max_player_seats > saved_buffer_sizes{};
      std::array< size_t, max_player_seats > saved_seats{};
      size_t saved_size_count = 0;
      for(auto player : _env().players(next_wstate)) {
         if(player == Player::chance) {
            continue;
         }
         if(not (flushes and player == next_active_player)) {
            const auto seat_idx = static_cast< size_t >(player);
            saved_seats[saved_size_count] = seat_idx;
            saved_buffer_sizes[seat_idx] = obs_table[player].size();
            ++saved_size_count;
         }
      }
      for(auto player : _env().players(next_wstate)) {
         if(player == Player::chance) {
            continue;
         }
         if(flushes and player == next_active_player) {
            auto& scratch_history = this->_obs_scratch_slot(depth);
            // NOTE: the scratch slot now OWNS the pre-edge buffer contents
            // (acquired via the swap above); they are the restoration source
            // for sibling edges. Drain them by COPY -- moving out or clearing
            // here would destroy the saved state.
            for(auto& obs : scratch_history) {
               ::nor::detail::update_infostate(child_infostate, obs.first, obs.second);
            }
            ::nor::detail::update_infostate(
               child_infostate,
               public_obs,
               _env().private_observation(player, curr_worldstate, action, next_wstate)
            );
         } else {
            obs_table[player].emplace_back(
               public_obs, _env().private_observation(player, curr_worldstate, action, next_wstate)
            );
         }
      }
      if(flushes) {
         *infostate_slot = std::move(child_infostate);
      }

      // scale the acting player's reach entry in place (restored post-recursion
      // via operator[] re-lookup -- deeper inserts may shift the table storage)
      double& actor_reach_entry = reach_probability.get()[active_player];
      const double saved_actor_reach = actor_reach_entry;
      actor_reach_entry *= action_prob;

      StateValueMap child_rewards_map = _traverse(
         player_to_update,
         next_wstate,
         depth + 1,
         reach_probability,
         observation_buffer,
         infostates,
         infostates_to_update,
         action_value_arena
      );

      if(flushes) {
         *infostate_slot = std::move(saved_flush_target);
         this->_obs_scratch_slot(depth).swap(obs_table[next_active_player]);
      }
      for(auto idx : std::views::iota(size_t{0}, saved_size_count)) {
         const auto seat_idx = saved_seats[idx];
         obs_table[static_cast< Player >(seat_idx)].resize(saved_buffer_sizes[seat_idx]);
      }
      reach_probability.get()[active_player] = saved_actor_reach;

      if constexpr(config.algorithm == MCCFRAlgorithmMode::chance_sampling) {
         // add the child state's value to the respective player's value table, multiplied by the
         // policies likelihood of playing this action
         for(auto [player, child_value] : child_rewards_map.get()) {
            state_value.get()[player] += action_prob * child_value;
         }
      }
      detail::emplace_action_value(action_value, action, std::move(child_rewards_map));
   }
   if constexpr(config.algorithm == MCCFRAlgorithmMode::pure_cfr) {
      // in the pure-cfr case we only need to emplace the value of the sampled action
      auto& sampled_action_opt = infonode_data.data().extras.sampled_action;
      if(not sampled_action_opt.has_value()) {
         // emplace sampled action for the pure strategy at this infostate if not already done
         sampled_action_opt = _sample_action_on_policy(actions, curr_action_policy);
      }
      for(auto [player, child_value] :
          detail::find_action_value(action_value, *sampled_action_opt).get()) {
         state_value.get().emplace(player, child_value);
      }
   }
   // we can only update our regrets and policies if we are traversing with the current
   // policy, since the average policy is not to be changed directly (but through averaging up
   // all current policies)
   if constexpr(config.update_mode == UpdateMode::alternating) {
      // in alternating updates, we only update the regret and strategy if the current
      // player is the chosen player to update.
      if(active_player == player_to_update.value()) {
         update_regret_and_policy(
            *infostate,
            reach_probability,
            state_value,
            action_value,
            avg_action_policy,
            curr_action_policy
         );
      }
   } else {
      // if we do simultaneous updates, then we always update the regret and strategy
      // values of the node's active player.
      update_regret_and_policy(
         *infostate,
         reach_probability,
         state_value,
         action_value,
         avg_action_policy,
         curr_action_policy
      );
   }
   // release the slot for the next visitor of this depth (the child value maps
   // were consumed above; clearing keeps the slot's heap capacity for reuse)
   action_value.clear();

   return state_value;
}

template <
   MCCFRConfig config,
   typename Env,
   typename Policy,
   typename AveragePolicy,
   typename SamplingRule >
void MCCFR< config, Env, Policy, AveragePolicy, SamplingRule >::update_regret_and_policy(
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
   )
{
   auto& istate_data = _infonode(infostate);

   auto player = infostate.player();
   double cf_reach_prob = rm::cf_reach_probability(player, reach_probability.get());
   [[maybe_unused]] double player_reach_prob = reach_probability.get().at(player);
   double player_state_value = state_value.get().at(player);

   for(const auto& [action_variant, action_value] : action_value_map) {
      // we only call this function with action values from a non-chance player, so we can safely
      // assume that the action is of action_type
      const auto& action = std::get< 0 >(action_variant);
      // update the cumulative regret according to the formula:
      // let I be the infostate, p be the player, r the cumulative regret
      //    r = \sum_a counterfactual_reach_prob_{p}(I) * (value_{p}(I-->a) - value_{p}(I))
      if(cf_reach_prob > 0.) {
         // this if statement effectively introduces partial pruning. But this is such a slight
         // modification (and gain, if any) that it is to be included in all variants of CFR
         istate_data.regret(action) += cf_reach_prob
                                       * (action_value.get().at(player) - player_state_value);
      }
      if constexpr(config.algorithm == MCCFRAlgorithmMode::chance_sampling) {
         // update the cumulative policy according to the formula:
         // let
         //    'I' be the infostate,
         //    'p' be the player,
         //    'a' be the chosen action,
         //    'sigma^t' the current policy
         //  -->  avg_sigma^{t+1}(I) = \sum_a reach_prob_{p}(I) * sigma^t(I, a)
         avg_action_policy[action] += player_reach_prob * curr_action_policy[action];
      }
   }
   if constexpr(config.algorithm == MCCFRAlgorithmMode::pure_cfr) {
      const auto& sampled_action = *(istate_data.data().extras.sampled_action);
      // For Pure CFR we really increment only the sampled action's a' average policy, because
      // the remaining increments are all 0 avg_sigma^{t+1}(I) = 1 if a == a' else 0
      avg_action_policy[sampled_action] += 1;
   }
}

}  // namespace nor::rm
