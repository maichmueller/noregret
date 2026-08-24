#pragma once

#include <spdlog/spdlog.h>
#include <algorithm>

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
      not(config.variance_reduced_baselines and config.update_mode != UpdateMode::alternating),
      "VR-MCCFR baselines are currently restricted to alternating updates. Under "
      "simultaneous updates the baseline-corrected low-variance increments make both "
      "players' current policies chase each other within a single trajectory, and "
      "average-strategy convergence stalls (empirically established on Kuhn poker for "
      "epsilon-on-policy exploration >= 0.3 across baseline rates beta in {0, 0.01, "
      "0.03, 0.1, 1}: exploitability plateaus near 0.11-0.23 instead of descending). "
      "Alternating updates damp the chase and converge fast."
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
};

template <
   MCCFRConfig config,
   typename Env,
   typename Policy,
   typename AveragePolicy,
   typename SamplingRule >
auto MCCFR< config, Env, Policy, AveragePolicy, SamplingRule >::iterate(size_t n_iters)
{
   std::vector< std::unordered_map< Player, double > > root_values_per_iteration;
   root_values_per_iteration.reserve(n_iters);
   for([[maybe_unused]] auto _ : std::views::iota(size_t(0), n_iters)) {
      SPDLOG_DEBUG("Iteration number: {}", _iteration());
      std::optional< Player > player_to_update = std::nullopt;
      if constexpr(config.update_mode == UpdateMode::alternating) {
         player_to_update = _cycle_player_to_update();
      }
      root_values_per_iteration.emplace_back(std::invoke([&] {
         if constexpr(config.algorithm == MCCFRAlgorithmMode::outcome_sampling) {
            return _iterate(player_to_update).first.get().to_hashmap();
         }
         if constexpr(  // clang-format off
            (config.algorithm == MCCFRAlgorithmMode::chance_sampling)
            or (
               config.algorithm == MCCFRAlgorithmMode::pure_cfr
               and config.update_mode == UpdateMode::simultaneous
            )  // clang-format on
         ) {
            return _iterate(player_to_update).get().to_hashmap();
         }
         if constexpr(  // clang-format off
            (config.algorithm == MCCFRAlgorithmMode::external_sampling)
            or (
               config.algorithm == MCCFRAlgorithmMode::pure_cfr
               and config.update_mode == UpdateMode::alternating
            )  // clang-format on
         ) {
            return std::unordered_map< Player, double >{
               {*player_to_update, _iterate(*player_to_update).get()}
            };
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
auto MCCFR< config, Env, Policy, AveragePolicy, SamplingRule >::iterate(std::optional< Player > player_to_update)
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
auto MCCFR< config, Env, Policy, AveragePolicy, SamplingRule >::_iterate(std::optional< Player > player_to_update)
{
   auto players = _env().players(*_root_state_uptr());
   auto init_infostates = [&] {
      std::unordered_map< Player, sptr< info_state_type > > infostates;
      for(auto player : players | utils::is_actual_player_filter) {
         infostates.emplace(player, std::make_shared< info_state_type >(player));
      }
      return InfostateSptrMap{std::move(infostates)};
   };
   auto init_reach_probs = [&] {
      std::unordered_map< Player, double > rp_map;
      for(auto player : players) {
         rp_map.emplace(player, 1.);
      }
      return ReachProbabilityMap{std::move(rp_map)};
   };
   auto init_obs_buffer = [&] {
      std::unordered_map< Player, std::vector< std::pair< observation_type, observation_type > > >
         obs_map;
      for(auto player : players | utils::is_actual_player_filter) {
         obs_map[player];
      }
      return ObservationbufferMap{std::move(obs_map)};
   };

   // the traversal containers are created once per iteration here and then
   // shared down the recursion by reference; world states live in the
   // depth-indexed arena (slot 0 holds the root copy).
   world_state_type& root_arena_state = _arena_state(0, *_root_state_uptr());

   if constexpr(config.algorithm == MCCFRAlgorithmMode::outcome_sampling) {
      auto weights = std::invoke([&] {
         if constexpr(config.weighting == MCCFRWeightingMode::lazy) {
            std::unordered_map< Player, double > w;
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
      return _traverse(
         player_to_update, root_arena_state, /*depth=*/0, reach_probs, obs_buffer, infostates,
         Probability{1.}, weights
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
      auto values = _traverse(
         player_to_update, root_arena_state, /*depth=*/0, reach_probs, obs_buffer, infostates,
         update_set
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
            or update_set.find({infostate_sptr.get(), &data}) != update_set.end()
         ) {
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
   auto& current_policy =
      this->template fetch_policy< PolicyLabel::current >(infostate, data.actions());
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
std::pair< StateValueMap, Probability > MCCFR< config, Env, Policy, AveragePolicy, SamplingRule >::_traverse(
   std::optional< Player > player_to_update,
   world_state_type& state,
   size_t depth,
   ReachProbabilityMap& reach_probability,
   ObservationbufferMap& observation_buffer,
   InfostateSptrMap& infostates,
   Probability sample_probability,
   ConditionalWeightMap& weights
)
   requires(config.algorithm == MCCFRAlgorithmMode::outcome_sampling)
{
   if(_env().is_terminal(state)) {
      return _terminal_value(state, player_to_update, sample_probability);
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
         double& chance_reach_entry = reach_probability.get()[Player::chance];
         const double saved_chance_reach = chance_reach_entry;
         chance_reach_entry *= chance_prob;

         world_state_type& next_state = _arena_state(depth + 1, state);
         _env().transition(next_state, chosen_outcome);

         next_infostate_and_obs_buffers_inplace(
            _env(), observation_buffer.get(), infostates.get(), state, chosen_outcome, next_state
         );

         auto [action_value_map, tail_prob] = _traverse(
            player_to_update,
            next_state,
            depth + 1,
            reach_probability,
            observation_buffer,
            infostates,
            Probability{sample_probability.get() * chance_prob},
            weights
         );
         chance_reach_entry = saved_chance_reach;
         return {std::move(action_value_map), tail_prob};
      }
   }

   // fetch (or create) this infostate's node without cloning the infostate for
   // every visit: heterogeneous value lookup finds existing entries directly.
   // (The infostate object itself advances along the single trajectory via the
   // inplace observation fold above, exactly like before.)
   const auto& live_infostate_sptr = infostates.get().at(active_player);
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
   // permanently downward (single linear trajectory)
   const auto saved_reach_probs = reach_probability.get();
   [[maybe_unused]] WeightMap saved_weights{{}};
   if constexpr(config.weighting == MCCFRWeightingMode::lazy) {
      saved_weights.get() = weights.get();
   }

   // scale the acting player's reach entry / lazy weight in place
   double& actor_reach_entry = reach_probability.get()[active_player];
   actor_reach_entry *= action_policy_prob;

   if constexpr(config.weighting == MCCFRWeightingMode::lazy) {
      auto& active_weight = weights.get()[active_player];
      active_weight = active_weight * action_policy_prob
                      + infonode_data.data()
                           .extras.pending_avg_accumulator[infonode_data.data().index_of(
                              sampled_action
                           )];
   }

   // advance along the single trajectory via the arena slot of the next depth
   world_state_type& next_state = _arena_state(depth + 1, state);
   _env().transition(next_state, sampled_action);

   next_infostate_and_obs_buffers_inplace(
      _env(), observation_buffer.get(), infostates.get(), state, sampled_action, next_state
   );

   auto [action_value_map, tail_prob] = _traverse(
      player_to_update,
      next_state,
      depth + 1,
      reach_probability,
      observation_buffer,
      infostates,
      Probability{sample_probability.get() * action_sampling_prob},
      weights
   );
   reach_probability.get() = std::move(saved_reach_probs);
   if constexpr(config.weighting == MCCFRWeightingMode::lazy) {
      weights.get() = std::move(saved_weights.get());
   }

   // ---- VR-MCCFR quantities (Schmid et al., AAAI 2019, eqs (7)-(11)) ----
   // Applied at the updating player's infosets only (the configuration is
   // restricted to alternating updates, see _sanity_check_config): these are
   // exactly the nodes where this implementation materializes the updater's
   // own value stream.
   [[maybe_unused]] double vr_sampled_value = 0.;  // eq (9) sampled branch
   [[maybe_unused]] double vr_cf_weight = 0.;      // eq (11) factor pi_-i(h)/q(h)
   [[maybe_unused]] bool vr_active = false;
   if constexpr(config.variance_reduced_baselines) {
      vr_active = active_player == player_to_update.value();
   }
   if constexpr(config.variance_reduced_baselines) {
      if(vr_active) {
         auto& baseline_table = infonode_data.data().vr_extras.baseline;
         // incoming bootstrapped value u(h a*|z): the raw terminal reward at
         // the deepest of this player's infosets, else the eq-(10) mixture
         // propagated from the child infoset
          double& stream = action_value_map.get().at(active_player);
          const size_t sampled_idx = infonode_data.data().index_of(sampled_action);
          const double sampled_baseline = baseline_table[sampled_idx];
         // eq (9), sampled branch: b̂(I,a*) + (u(h a*|z) - b̂(I,a*))/ξ(h,a*).
         // Dividing the deviation by the ACTUAL sampling probability of a*
         // (the ε-on-policy mixture where exploration applies) is what keeps
         // the estimator exact for ANY positive sampling rule: in expectation
         // the b̂(I,a*) term cancels between the sampled branch and the
         // off-trajectory branch (both enter with weight ξ·(1/ξ) resp.
         // (1-ξ) against the same baseline).
         vr_sampled_value =
            sampled_baseline + (stream - sampled_baseline) / action_sampling_prob;
         // eq (10): propagate the σ_t-weighted mixture of the sampled-branch
         // value and the off-trajectory baselines up to the parent infoset
         // (bootstrapped propagation; chance nodes pass values unchanged)
          double off_trajectory_mass = 0.;
          for(auto idx : std::views::iota(size_t{0}, actions.size())) {
             if(!(actions[idx] == sampled_action)) {
                off_trajectory_mass += action_policy[actions[idx]] * baseline_table[idx];
             }
          }
         stream = action_policy[sampled_action] * vr_sampled_value + off_trajectory_mass;
         // eq (11) prefactor: counterfactual reach over prefix sampling
         // probability q(h) (everything sampled above this infoset)
         vr_cf_weight = cf_reach_probability(active_player, reach_probability.get())
                        / sample_probability.get();
         // NOTE: the baseline table is intentionally not mutated yet -- the
         // regret update below must read the same baseline snapshot that
         // produced vr_sampled_value (paper order: values -> regrets ->
         // baseline regression).
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
         if constexpr(config.variance_reduced_baselines) {
            _update_regrets_variance_reduced(
               infonode_data,
               actions,
               action_policy,
               sampled_action,
               vr_cf_weight,
               vr_sampled_value
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
   Probability sample_probability
)
   requires(config.algorithm == MCCFRAlgorithmMode::outcome_sampling)
{
   if constexpr(config.variance_reduced_baselines) {
      // VR-MCCFR seeds the trajectory with the RAW terminal reward: the
      // importance weights are applied later, once per sampled action
      // (eq 9's 1/ξ(h,a*)) and once per updated infoset (eq 11's 1/q(h)),
      // so dividing by the full-path sampling probability here would double-
      // count them.
      if constexpr(config.update_mode == UpdateMode::alternating) {
         return std::pair{
            StateValueMap{std::unordered_map< Player, double >{
               {player_to_update.value(), _env().reward(player_to_update.value(), state)}
            }},
            Probability{1.}
         };
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
             _env().reward(player_to_update.value(), state) / sample_probability.get()}
         }},
         Probability{1.}
      };
   } else if constexpr(config.update_mode == UpdateMode::simultaneous) {
      return std::pair{
         StateValueMap{[&] {
            auto rewards_map = collect_rewards(_env(), state);
            for(auto& [player, reward] : rewards_map) {
               reward /= sample_probability.get();
            }
            return rewards_map;
         }()},
         Probability{1.}
      };
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
void MCCFR< config, Env, Policy, AveragePolicy, SamplingRule >::_update_regrets_variance_reduced(
   infostate_data_type& infostate_data,
   const std::vector< action_type >& actions,
   const auto& action_policy,
   const action_type& sampled_action,
   double cf_weight,
   double sampled_value
) const
   requires(config.algorithm == MCCFRAlgorithmMode::outcome_sampling)
{
   // Paper step (d): accumulate R(I,a) += v̂ᵇ(I,a) - v̂ᵇ(I) for EVERY action,
   // where (eq 11) v̂ᵇ(I,a) = pi_-i(h)/q(h) * ûᵇ(h,a), the sampled action's
   // value coming from eq (9)'s sampled branch and every off-trajectory
   // action valued by its baseline. The baseline snapshot read here is the
   // same one that produced 'sampled_value' (no mutation has happened yet).
    auto& baseline_table = infostate_data.data().vr_extras.baseline;
    const size_t sampled_idx = infostate_data.data().index_of(sampled_action);
    auto action_value_of = [&](auto idx, const action_type& action) {
       return action == sampled_action ? sampled_value : baseline_table[idx];
    };
    double node_value = 0.;
    for(auto idx : std::views::iota(size_t{0}, actions.size())) {
       node_value += action_policy[actions[idx]] * action_value_of(idx, actions[idx]);
    }
    node_value *= cf_weight;
    for(auto idx : std::views::iota(size_t{0}, actions.size())) {
       infostate_data.regret(actions[idx]) +=
          cf_weight * action_value_of(idx, actions[idx]) - node_value;
    }
    // paper step (e): regress the sampled action's baseline onto its
    // baseline-corrected estimate b̂(I,a*) <- b̂(I,a*) + β(v̂ - b̂(I,a*))
    auto& sampled_baseline = baseline_table[sampled_idx];
    sampled_baseline += config.baseline_update_rate * (sampled_value - sampled_baseline);
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
      // compute the estimated counterfactual regret and add it to the cumulative regret table
      infostate_data.regret(action) += [&] {
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
   }
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
   return common::choose(actions, [&](const auto& act) { return action_policy[act]; }, m_rng);
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
            action_policy[chosen_action]
         };
      } else {
         // if we don't explore, then we simply sample according to the policy.
         // BUT: Since in theory we have done epsilon-on-policy exploration, yet merely in two
         // separate steps, we need to adapt the returned sampling probability to the
         // epsilon-on-policy probability of the sampled action
         const auto& [chosen_action, _, action_prob] = on_policy_sampling();
         return std::tuple{
            std::move(chosen_action),
            m_epsilon * uniform_prob + (1 - m_epsilon) * action_prob,
            action_prob
         };
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
            auto& average_action_policy =
               this->template fetch_policy< PolicyLabel::average >(infostate, actions);
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
            const auto [chosen_action, sample_prob] =
               m_sampling_rule(m_rng, actions, [&](const action_type& act) { return action_policy[act]; });
            return std::tuple{chosen_action, sample_prob, action_policy[chosen_action]};
         }
         return on_policy_sampling();
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
auto MCCFR< config, Env, Policy, AveragePolicy, SamplingRule >::_sample_outcome(const world_state_type& state)
{
   auto chance_actions = _env().chance_actions(state);
   auto chance_probabilities = std::ranges::to< std::unordered_map< chance_outcome_type, double > >(
      chance_actions | std::views::transform([this, &state](const auto& outcome) {
         return std::pair{outcome, _env().chance_probability(state, outcome)};
      })
   );
   auto& chosen_outcome = common::choose(
      chance_actions, [&](const auto& outcome) { return chance_probabilities[outcome]; }, m_rng
   );

   if constexpr(return_likelihood) {
      double chance_prob = chance_probabilities[chosen_outcome];
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

         next_infostate_and_obs_buffers_inplace(
            _env(), observation_buffer.get(), infostates.get(), state, chosen_outcome, next_state
         );

         // single sampled outcome: buffers/infostates flow permanently downward
         return _traverse(
            player_to_update, next_state, depth + 1, observation_buffer, infostates, infostates_to_update
         );
      }
   }

   // fetch (or create) this infostate's node without cloning the infostate on
   // every visit: heterogeneous value lookup finds existing entries directly
   const auto& live_infostate_sptr = infostates.get().at(active_player);
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

      // fold this edge's observations into the live containers exactly like the
      // inplace helper would -- and restore everything after the recursion so
      // sibling edges start from identical pre-edge container states.
      auto& obs_buffer = observation_buffer.get();
      const auto public_obs = _env().public_observation(state, action, next_state);
      const Player next_active_player = _env().active_player(next_state);
      const bool flushes =
         next_active_player != Player::chance and infostates.get().contains(next_active_player);
      std::optional< std::vector< std::pair< observation_type, observation_type > > >
         saved_flush_target_buffer{};
      sptr< info_state_type > saved_flush_target{};
      sptr< info_state_type > child_infostate{};
      if(flushes) {
         // advance a CLONE of the next-active player's infostate so the live
         // entry can be restored verbatim after the recursion
         saved_flush_target = infostates.get().at(next_active_player);
         child_infostate = std::make_shared< info_state_type >(*saved_flush_target);
         saved_flush_target_buffer = obs_buffer.at(next_active_player);
      }
      std::vector< std::pair< Player, size_t > > saved_buffer_sizes;
      for(const auto& [player, buffer] : obs_buffer) {
         if(not flushes or player != next_active_player) {
            saved_buffer_sizes.emplace_back(player, buffer.size());
         }
      }
      for(auto player : _env().players(next_state)) {
         if(player == Player::chance) {
            continue;
         }
         if(flushes and player == next_active_player) {
            auto& obs_history = obs_buffer[player];
            for(auto& obs : obs_history) {
               ::nor::detail::update_infostate(
                  child_infostate, std::move(obs.first), std::move(obs.second)
               );
            }
            obs_history.clear();
            ::nor::detail::update_infostate(
               child_infostate,
               public_obs,
               _env().private_observation(player, state, action, next_state)
            );
         } else {
            obs_buffer[player].emplace_back(
               public_obs, _env().private_observation(player, state, action, next_state)
            );
         }
      }

      if(flushes) {
         infostates.get().at(next_active_player) = std::move(child_infostate);
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
         infostates.get().at(next_active_player) = std::move(saved_flush_target);
         obs_buffer.at(next_active_player) = std::move(*saved_flush_target_buffer);
      }
      for(const auto& [player, size] : saved_buffer_sizes) {
         obs_buffer.at(player).resize(size);
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
      // values from this node and compute the state value of the current node
      std::unordered_map< action_type, double > value_estimates;
      value_estimates.reserve(actions.size());

      auto state_value_estimate = std::invoke([&] {
         if constexpr(config.algorithm == MCCFRAlgorithmMode::external_sampling) {
            return std::ranges::fold_left(
               actions | std::views::transform([&](const auto& action) {
                  return value_estimates.emplace(action, traverse_for_action_value(action))
                            .first->second
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
               value_estimates.emplace(action, traverse_for_action_value(action));
            }
            return value_estimates[sample_or_fetch_action()];
         }
      });
      // in the second round of action iteration we update the regret of each action through the
      // previously found action and state values
      for(const auto& action : actions) {
         infonode_data.regret(action) += value_estimates[action] - state_value_estimate;
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
   delayed_update_set& infostates_to_update
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
   std::unordered_map< action_variant_type, StateValueMap > action_value;
   // traverse all child states from this state. The constexpr check for determinism in the env
   // allows deterministic envs to not provide certain functions that are only needed in the
   // stochastic case.
   if constexpr(concepts::stochastic_env< env_type >) {
      if(active_player == Player::chance) {
         auto [chosen_outcome, _] = _sample_outcome(curr_worldstate);
world_state_type& next_state = _arena_state(depth + 1, curr_worldstate);
          _env().transition(next_state, chosen_outcome);

          next_infostate_and_obs_buffers_inplace(
             _env(), observation_buffer.get(), infostates.get(), curr_worldstate, chosen_outcome, next_state
          );

          // single sampled outcome: containers flow permanently downward
          return _traverse(
             player_to_update,
             next_state,
             depth + 1,
             reach_probability,
             observation_buffer,
             infostates,
             infostates_to_update
          );
      }
   }
   const auto& live_infostate_sptr = infostates.get().at(active_player);
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
   auto& curr_action_policy = this->template fetch_policy< PolicyLabel::current >(*infostate, actions);
   auto& avg_action_policy = this->template fetch_policy< PolicyLabel::average >(*infostate, actions);

   for(const action_type& action : actions) {
      auto action_prob = curr_action_policy[action];

      auto child_reach_prob = reach_probability.get();
      child_reach_prob[active_player] *= action_prob;

      // advance the arena slot of the next depth: in-place reconstruction + transition
      world_state_type& next_wstate = _arena_state(depth + 1, curr_worldstate);
      _env().transition(next_wstate, action);
      // fold this edge's observations into the live containers exactly like the
      // inplace helper would -- restoring everything after the recursion so
      // sibling edges start from identical pre-edge container states.
      auto& obs_buffer = observation_buffer.get();
      const auto public_obs = _env().public_observation(curr_worldstate, action, next_wstate);
      const Player next_active_player = _env().active_player(next_wstate);
      const bool flushes =
         next_active_player != Player::chance and infostates.get().contains(next_active_player);
      sptr< info_state_type > saved_flush_target{};
      sptr< info_state_type > child_infostate{};
      std::optional< std::vector< std::pair< observation_type, observation_type > > >
         saved_flush_target_buffer{};
      if(flushes) {
         saved_flush_target = infostates.get().at(next_active_player);
         child_infostate = std::make_shared< info_state_type >(*saved_flush_target);
         saved_flush_target_buffer = obs_buffer.at(next_active_player);
      }
      std::vector< std::pair< Player, size_t > > saved_buffer_sizes;
      for(const auto& [player, buffer] : obs_buffer) {
         if(not flushes or player != next_active_player) {
            saved_buffer_sizes.emplace_back(player, buffer.size());
         }
      }
      for(auto player : _env().players(next_wstate)) {
         if(player == Player::chance) {
            continue;
         }
         if(flushes and player == next_active_player) {
            auto& obs_history = obs_buffer[player];
            for(auto& obs : obs_history) {
               ::nor::detail::update_infostate(child_infostate, std::move(obs.first), std::move(obs.second));
            }
            obs_history.clear();
            ::nor::detail::update_infostate(
               child_infostate,
               public_obs,
               _env().private_observation(player, curr_worldstate, action, next_wstate)
            );
         } else {
            obs_buffer[player].emplace_back(
               public_obs, _env().private_observation(player, curr_worldstate, action, next_wstate)
            );
         }
      }
      if(flushes) {
         infostates.get().at(next_active_player) = std::move(child_infostate);
      }

      // scale the acting player's reach entry in place (restored post-recursion)
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
         infostates_to_update
      );

      if(flushes) {
         infostates.get().at(next_active_player) = std::move(saved_flush_target);
         obs_buffer.at(next_active_player) = std::move(*saved_flush_target_buffer);
      }
      for(const auto& [player, size] : saved_buffer_sizes) {
         obs_buffer.at(player).resize(size);
      }
      actor_reach_entry = saved_actor_reach;

      if constexpr(config.algorithm == MCCFRAlgorithmMode::chance_sampling) {
         // add the child state's value to the respective player's value table, multiplied by the
         // policies likelihood of playing this action
         for(auto [player, child_value] : child_rewards_map.get()) {
            state_value.get()[player] += action_prob * child_value;
         }
      }
      action_value.emplace(action, std::move(child_rewards_map));
   }
   if constexpr(config.algorithm == MCCFRAlgorithmMode::pure_cfr) {
      // in the pure-cfr case we only need to emplace the value of the sampled action
      auto& sampled_action_opt = infonode_data.data().extras.sampled_action;
      if(not sampled_action_opt.has_value()) {
         // emplace sampled action for the pure strategy at this infostate if not already done
         sampled_action_opt = _sample_action_on_policy(actions, curr_action_policy);
      }
      for(auto [player, child_value] : action_value.at(*sampled_action_opt).get()) {
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
   const std::unordered_map< action_variant_type, StateValueMap >& action_value_map,
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
