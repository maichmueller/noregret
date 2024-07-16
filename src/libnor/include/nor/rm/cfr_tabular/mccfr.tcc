#pragma once

#include "mccfr.hpp"

namespace nor::rm {

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
constexpr void MCCFR< config, Env, Policy, AveragePolicy >::_sanity_check_config()
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
};

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
auto MCCFR< config, Env, Policy, AveragePolicy >::iterate(size_t n_iters)
{
   std::vector< std::unordered_map< Player, double > > root_values_per_iteration;
   root_values_per_iteration.reserve(n_iters);
   for([[maybe_unused]] auto _ : ranges::views::iota(size_t(0), n_iters)) {
      SPDLOG_DEBUG("Iteration number: {}", _iteration());
      std::optional< Player > player_to_update = std::nullopt;
      if constexpr(config.update_mode == UpdateMode::alternating) {
         player_to_update = _cycle_player_to_update();
      }
      root_values_per_iteration.emplace_back(std::invoke([&] {
         if constexpr(config.algorithm == MCCFRAlgorithmMode::outcome_sampling) {
            return _iterate(player_to_update).first.get();
         }
         if constexpr(  // clang-format off
            (config.algorithm == MCCFRAlgorithmMode::chance_sampling)
            or (
               config.algorithm == MCCFRAlgorithmMode::pure_cfr
               and config.update_mode == UpdateMode::simultaneous
            )  // clang-format on
         ) {
            return _iterate(player_to_update).get();
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

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
auto MCCFR< config, Env, Policy, AveragePolicy >::iterate(std::optional< Player > player_to_update)
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

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
auto MCCFR< config, Env, Policy, AveragePolicy >::_iterate(std::optional< Player > player_to_update)
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

   if constexpr(config.algorithm == MCCFRAlgorithmMode::outcome_sampling) {
      // in outcome-sampling we only have a single trajectory to traverse in the tree. Hence, we
      // can maintain the lifetime of that world state in this upstream function call and merely
      // pass in the state as reference
      auto init_world_state = utils::static_unique_ptr_downcast< world_state_type >(
         utils::clone_any_way(_root_state_uptr())
      );

      return _traverse(
         player_to_update,
         *init_world_state,
         init_reach_probs(),
         init_obs_buffer(),
         init_infostates(),
         Probability{1.},
         std::invoke([&] {
            if constexpr(config.weighting == MCCFRWeightingMode::lazy) {
               std::unordered_map< Player, double > weights;
               for(auto player : players | utils::is_actual_player_filter) {
                  weights.emplace(player, 0.);
               }
               return WeightMap{std::move(weights)};
            } else {
               return utils::empty{};
            }
         })
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
      auto value = _traverse(
         player_to_update.value(),
         utils::static_unique_ptr_downcast< world_state_type >(
            utils::clone_any_way(_root_state_uptr())
         ),
         init_obs_buffer(),
         init_infostates(),
         update_set
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
      auto values = _traverse(
         player_to_update,
         utils::static_unique_ptr_downcast< world_state_type >(
            utils::clone_any_way(_root_state_uptr())
         ),
         init_reach_probs(),
         init_obs_buffer(),
         init_infostates(),
         update_set
      );
      _initiate_regret_minimization(update_set);
      update_set.clear();
      return values;
   }
}

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
void MCCFR< config, Env, Policy, AveragePolicy >::_initiate_regret_minimization(
   const delayed_update_set& update_set
)
{
   // here we now invoke the actual regret minimization procedure for each infostate individually
   auto execution_policy = std::execution::par_unseq;

   auto node_view = std::invoke([&] {
      if constexpr(config.algorithm == MCCFRAlgorithmMode::pure_cfr) {
         return ranges::views::all(_infonodes());
      } else {
         return ranges::views::all(update_set);
      };
   });
   std::for_each(
      execution_policy,
      node_view.begin(),
      node_view.end(),
      [&](auto& infostate_ptr_data) {
         auto& [infostate_ptr, data] = infostate_ptr_data;
         if constexpr(config.algorithm == MCCFRAlgorithmMode::pure_cfr) {
            // reset the sampled plan per information state
            data.template storage_element< 1 >().reset();
         }
         if(not (  // for all algos but alternating pure-cfr we always update the given range
               config.algorithm == MCCFRAlgorithmMode::pure_cfr
               and config.update_mode == UpdateMode::alternating
            )
            or update_set.contains(infostate_ptr)  // for alternating pure-cfr we have to check if
                                                   // this infostate was meant to be updated as well
         ) {
            _invoke_regret_minimizer(common::deref(infostate_ptr), data);
         }
      }
   );
}
template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
void MCCFR< config, Env, Policy, AveragePolicy >::_invoke_regret_minimizer(
   const info_state_type& infostate,
   infostate_data_type& data
)
{
   m_regret_minimizer(
      this->template fetch_policy< PolicyLabel::current >(infostate, data.actions()),
      data.regret(),
      // we provide the accessor to get the underlying referenced action, as the infodata
      // stores only reference wrappers to the actions
      [](const action_type& action) { return std::cref(action); }
   );
}

////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// Outcome-Sampling MCCFR /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
std::pair< StateValueMap, Probability > MCCFR< config, Env, Policy, AveragePolicy >::_traverse(
   std::optional< Player > player_to_update,
   world_state_type& state,
   ReachProbabilityMap reach_probability,
   ObservationbufferMap observation_buffer,
   InfostateSptrMap infostates,
   Probability sample_probability,
   ConditionalWeightMap weights
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
         auto [chosen_outcome, chance_prob] = _sample_outcome(state);

         reach_probability.get()[Player::chance] *= chance_prob;

         auto state_before_transition = utils::static_unique_ptr_downcast< world_state_type >(
            utils::clone_any_way(state)
         );
         _env().transition(state, chosen_outcome);

         next_infostate_and_obs_buffers_inplace(
            _env(),
            observation_buffer.get(),
            infostates.get(),
            *state_before_transition,
            chosen_outcome,
            state
         );

         return _traverse(
            player_to_update,
            state,
            std::move(reach_probability),
            std::move(observation_buffer),
            std::move(infostates),
            Probability{sample_probability.get() * chance_prob},
            std::move(weights)
         );
      }
   }

   // we have to clone the infostate to ensure that it is not written to upon further traversal
   // (we need this state after traversal to update policy and regrets)
   auto [infostate_and_data_iter, success] = _infonodes().try_emplace(
      utils::clone_any_way(infostates.get().at(active_player)), infostate_data_type{}
   );
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

   m_regret_minimizer(
      action_policy,
      infonode_data.regret(),
      // we provide the accessor to get the underlying referenced action, as the infodata
      // stores only reference wrappers to the actions
      [](const action_type& action) { return std::cref(action); }
   );

   auto [sampled_action, action_sampling_prob, action_policy_prob] = _sample_action(
      active_player, player_to_update, actions, action_policy
   );

   auto next_reach_prob = reach_probability.get();
   next_reach_prob[active_player] *= action_policy_prob;

   auto next_weights = weights;
   if constexpr(config.weighting == MCCFRWeightingMode::lazy) {
      auto& active_weight = next_weights.get()[active_player];
      active_weight = active_weight * action_policy_prob
                      + infonode_data.template storage_element< 1 >(std::cref(sampled_action));
   }
   auto state_before_transition = utils::static_unique_ptr_downcast< world_state_type >(
      utils::clone_any_way(state)
   );

   _env().transition(state, sampled_action);

   next_infostate_and_obs_buffers_inplace(
      _env(),
      observation_buffer.get(),
      infostates.get(),
      *state_before_transition,
      sampled_action,
      state
   );

   auto [action_value_map, tail_prob] = _traverse(
      player_to_update,
      state,
      ReachProbabilityMap{std::move(next_reach_prob)},
      std::move(observation_buffer),
      std::move(infostates),
      Probability{sample_probability.get() * action_sampling_prob},
      std::move(next_weights)
   );

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
         _update_regrets(
            reach_probability,
            active_player,
            infonode_data,
            sampled_action,
            Probability{action_policy_prob},
            StateValue{action_value_map.get()[active_player]},
            tail_prob
         );
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

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
auto MCCFR< config, Env, Policy, AveragePolicy >::_terminal_value(
   world_state_type& state,
   std::optional< Player > player_to_update,
   Probability sample_probability
)
   requires(config.algorithm == MCCFRAlgorithmMode::outcome_sampling)
{
   if constexpr(config.update_mode == UpdateMode::alternating) {
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

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
void MCCFR< config, Env, Policy, AveragePolicy >::_update_regrets(
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

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
void MCCFR< config, Env, Policy, AveragePolicy >::_update_average_policy(
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
      for(const action_type& action : infonode_data.actions()) {
         auto policy_incr = (weight.get() + reach_prob.get()) * current_policy[action];
         avg_policy[action] += policy_incr;
         if(action == sampled_action) [[unlikely]] {
            infonode_data.template storage_element< 1 >(std::cref(action)) = 0.;
         } else [[likely]] {
            infonode_data.template storage_element< 1 >(std::cref(action)) += policy_incr;
         }
      }
   }

   if constexpr(config.weighting == MCCFRWeightingMode::optimistic) {
      auto& infostate_last_visit = infonode_data.template storage_element< 1 >();
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

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
auto MCCFR< config, Env, Policy, AveragePolicy >::_sample_action_on_policy(
   const std::vector< action_type >& actions,
   auto& action_policy
)
{
   return common::choose(actions, [&](const auto& act) { return action_policy[act]; }, m_rng);
}

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
auto MCCFR< config, Env, Policy, AveragePolicy >::_sample_action(
   Player active_player,
   std::optional< Player > player_to_update,
   const std::vector< action_type >& actions,
   auto& action_policy
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

   // here we now decide what sampling procedure is exactly executed. It depends on the MCCFR
   // config given and then on the specific algorithm's sampling scheme
   if constexpr(config.algorithm == MCCFRAlgorithmMode::outcome_sampling) {
      if((config.update_mode == UpdateMode::simultaneous
          or active_player == player_to_update.value_or(Player::chance))) {
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

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
template < bool return_likelihood >
auto MCCFR< config, Env, Policy, AveragePolicy >::_sample_outcome(const world_state_type& state)
{
   auto chance_actions = _env().chance_actions(state);
   auto chance_probabilities = ranges::to< std::unordered_map< chance_outcome_type, double > >(
      chance_actions | ranges::views::transform([this, &state](const auto& outcome) {
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

////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// External-Sampling MCCFR ////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
StateValue MCCFR< config, Env, Policy, AveragePolicy >::_traverse(
   Player player_to_update,
   uptr< world_state_type > state,
   ObservationbufferMap observation_buffer,
   InfostateSptrMap infostates,
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

         auto state_before_transition = utils::static_unique_ptr_downcast< world_state_type >(
            utils::clone_any_way(*state)
         );
         _env().transition(*state, chosen_outcome);

         next_infostate_and_obs_buffers_inplace(
            _env(),
            observation_buffer.get(),
            infostates.get(),
            *state_before_transition,
            chosen_outcome,
            *state
         );

         return _traverse(
            player_to_update,
            std::move(state),
            std::move(observation_buffer),
            std::move(infostates),
            infostates_to_update
         );
      }
   }

   auto [infostate_and_data_iter, success] = _infonodes().try_emplace(
      utils::clone_any_way(infostates.get().at(active_player)), infostate_data_type{}
   );
   const auto& infostate = infostate_and_data_iter->first;
   auto& infonode_data = infostate_and_data_iter->second;
   if(success) {
      // success means we have indeed emplaced a new data node, instead of simply fetching an
      // existing one. We thus need to fill it with the legal actions at this node.
      infonode_data.emplace(_env().actions(active_player, *state));
   }
   if constexpr(config.algorithm == MCCFRAlgorithmMode::pure_cfr) {
      infostates_to_update.emplace(std::tuple{infostate.get(), std::ref(infonode_data)});
   } else {
      // for external sampling we can simply minimize upon traversal
      _invoke_regret_minimizer(common::deref(infostate), infonode_data);
   }
   const auto& actions = infonode_data.actions();
   auto& action_policy = this->template fetch_policy< PolicyLabel::current >(*infostate, actions);

   auto traverse_for_action_value = [&](const auto& action, bool inplace = false) {
      auto next_state = child_state(_env(), *state, action);

      auto [next_observation_buffer, next_infostates] = std::invoke([&] {
         if(inplace) {
            next_infostate_and_obs_buffers_inplace(
               _env(), observation_buffer.get(), infostates.get(), *state, action, *next_state
            );
            return std::tuple{std::move(observation_buffer.get()), std::move(infostates.get())};
         } else {
            return next_infostate_and_obs_buffers(
               _env(), observation_buffer.get(), infostates.get(), *state, action, *next_state
            );
         }
      });

      return _traverse(
                player_to_update,
                std::move(next_state),
                ObservationbufferMap{std::move(next_observation_buffer)},
                InfostateSptrMap{std::move(next_infostates)},
                infostates_to_update
      )
         .get();
   };

   auto sample_or_fetch_action = [&]() -> decltype(auto) {
      if constexpr(config.algorithm == MCCFRAlgorithmMode::pure_cfr) {
         auto& sampled_action_opt = infonode_data.template storage_element< 1 >();
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
            return ranges::accumulate(
               actions | ranges::views::transform([&](const auto& action) {
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
      return StateValue{traverse_for_action_value(sampled_action, true)};
   }
}

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////// Chance-Sampling MCCFR & Pure CFR Sim. Updating //////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
StateValueMap MCCFR< config, Env, Policy, AveragePolicy >::_traverse(
   std::optional< Player > player_to_update,
   uptr< world_state_type > curr_worldstate,
   ReachProbabilityMap reach_probability,
   ObservationbufferMap observation_buffer,
   InfostateSptrMap infostates,
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
   if(_env().is_terminal(*curr_worldstate)) {
      return StateValueMap{collect_rewards(_env(), *curr_worldstate)};
   }

   if constexpr(config.algorithm != MCCFRAlgorithmMode::pure_cfr and config.pruning_mode == CFRPruningMode::partial) {
      if(_partial_pruning_condition(player_to_update, reach_probability)) {
         // if the entire subtree is pruned then the values that could be found are all 0. for
         // each player
         return StateValueMap{std::invoke([&] {
            StateValueMap::UnderlyingType map;
            for(auto player : _env().players(*curr_worldstate) | utils::is_actual_player_pred) {
               map[player] = 0.;
            }
            return map;
         })};
      }
   }

   Player active_player = _env().active_player(*curr_worldstate);
   // the state's value for each player. To be filled by the action traversal functions.
   StateValueMap state_value{{}};
   // each action's value for each player. To be filled by the action traversal functions.
   std::unordered_map< action_variant_type, StateValueMap > action_value;
   // traverse all child states from this state. The constexpr check for determinism in the env
   // allows deterministic envs to not provide certain functions that are only needed in the
   // stochastic case.
   if constexpr(concepts::stochastic_env< env_type >) {
      if(active_player == Player::chance) {
         auto [chosen_outcome, _] = _sample_outcome(*curr_worldstate);

         auto next_state = utils::static_unique_ptr_downcast< world_state_type >(
            utils::clone_any_way(curr_worldstate)
         );
         _env().transition(*next_state, chosen_outcome);

         next_infostate_and_obs_buffers_inplace(
            _env(),
            observation_buffer.get(),
            infostates.get(),
            *curr_worldstate,
            chosen_outcome,
            *next_state
         );

         return _traverse(
            player_to_update,
            std::move(next_state),
            std::move(reach_probability),
            std::move(observation_buffer),
            std::move(infostates),
            infostates_to_update
         );
      }
   }
   auto [infostate_and_data_iter, success] = _infonodes().try_emplace(
      utils::clone_any_way(infostates.get().at(active_player)), infostate_data_type{}
   );
   const auto& infostate = infostate_and_data_iter->first;
   auto& infonode_data = infostate_and_data_iter->second;
   infostates_to_update.emplace(std::tuple{infostate.get(), std::ref(infonode_data)});
   if(success) {
      // success means we have indeed emplaced a new data node, instead of simply fetching an
      // existing one.
      // We thus need to fill it with the legal actions at this node.
      infonode_data.emplace(_env().actions(active_player, *curr_worldstate));
   }
   const auto& actions = infonode_data.actions();
   auto& curr_action_policy = this->template fetch_policy< PolicyLabel::current >(*infostate, actions);
   auto& avg_action_policy = this->template fetch_policy< PolicyLabel::average >(*infostate, actions);

   for(const action_type& action : actions) {
      auto action_prob = curr_action_policy[action];

      auto child_reach_prob = reach_probability.get();
      child_reach_prob[active_player] *= action_prob;

      uptr< world_state_type > next_wstate_uptr = child_state(_env(), *curr_worldstate, action);
      auto [child_observation_buffer, child_infostate_map] = next_infostate_and_obs_buffers(
         _env(),
         observation_buffer.get(),
         infostates.get(),
         *curr_worldstate,
         action,
         *next_wstate_uptr
      );

      StateValueMap child_rewards_map = _traverse(
         player_to_update,
         std::move(next_wstate_uptr),
         ReachProbabilityMap{std::move(child_reach_prob)},
         ObservationbufferMap{std::move(child_observation_buffer)},
         InfostateSptrMap{std::move(child_infostate_map)},
         infostates_to_update
      );

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
      auto& sampled_action_opt = infonode_data.template storage_element< 1 >();
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

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
void MCCFR< config, Env, Policy, AveragePolicy >::update_regret_and_policy(
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
      const auto& sampled_action = *(istate_data.template storage_element< 1 >());
      // For Pure CFR we really increment only the sampled action's a' average policy, because
      // the remaining increments are all 0 avg_sigma^{t+1}(I) = 1 if a == a' else 0
      avg_action_policy[sampled_action] += 1;
   }
}

}  // namespace nor::rm
