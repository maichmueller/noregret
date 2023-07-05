#pragma once

#include "cfr.hpp"

namespace nor::rm {

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
auto VanillaCFR< config, Env, Policy, AveragePolicy >::iterate(size_t n_iters)
{
   std::vector< player_hash_map< double > > root_values_per_iteration;
   root_values_per_iteration.reserve(n_iters);
   for([[maybe_unused]] auto _ : ranges::views::iota(size_t(0), n_iters)) {
      LOGD2("Iteration number: ", _iteration());
      StateValueMap value = std::invoke([&] {
         if constexpr(config.update_mode == UpdateMode::alternating) {
            auto player_to_update = _cycle_player_to_update();
            if(_iteration() < _env().players(*_root_state()).size() - 1) [[unlikely]] {
               return _iterate< true >(player_to_update);
            } else [[likely]] {
               return _iterate< false >(player_to_update);
            }
         } else {
            if(_iteration() == 0) [[unlikely]] {
               return _iterate< true >(std::nullopt);
            } else [[likely]] {
               return _iterate< false >(std::nullopt);
            }
         }
      });
      root_values_per_iteration.emplace_back(std::move(value.get()));
      _iteration()++;
   }
   return root_values_per_iteration;
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
auto VanillaCFR< config, Env, Policy, AveragePolicy >::iterate(
   std::optional< Player > player_to_update
)
   requires(config.update_mode == UpdateMode::alternating)
{
   LOGD2("Iteration number: ", _iteration());
   // run the iteration
   StateValueMap values = [&] {
      if(_iteration() < _env().players(*_root_state()).size() - 1)
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
   std::optional< Player > player_to_update
)
{
   auto root_players = _env().players(root_state());
   auto root_game_value = _traverse< initializing_run, use_current_policy >(
      player_to_update,
      _root_state().copy(),
      std::invoke([&] {
         ReachProbabilityMap rp_map{{}};
         for(auto player : root_players) {
            rp_map.get().emplace(player, 1.);
         }
         return rp_map;
      }),
      std::invoke([&] {
         ObservationbufferMap obs_map{{}};
         for(auto player : root_players | is_actual_player_filter) {
            obs_map.get().emplace(
               player,
               std::vector< std::pair<
                  ObservationHolder< observation_type >,
                  ObservationHolder< observation_type > > >{}
            );
         }
         return ObservationbufferMap{std::move(obs_map)};
      }),
      std::invoke([&] {
         SharedInfostateMap infostates{{}};
         for(auto player : root_players | is_actual_player_filter) {
            infostates.get().emplace(player, std::make_shared< info_state_type >(player));
         }
         return infostates;
      })
   );

   if constexpr(use_current_policy) {
      _initiate_regret_minimization(player_to_update);
   }
   return root_game_value;
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
void VanillaCFR< config, Env, Policy, AveragePolicy >::_initiate_regret_minimization(
   const std::optional< Player >& player_to_update
)
{
   auto policy_weight = [&]([[maybe_unused]] infostate_data_type& data_node) {
      if constexpr(common::isin(
                      config.weighting_mode,
                      {CFRWeightingMode::linear, CFRWeightingMode::discounted}
                   )) {
         // weighting by an iteration dependant factor multiplies the current iteration t as
         // t^gamma onto the update INCREMENT. The numerically more stable approach, however, is
         // to multiply the ACCUMULATED strategy with (t/(t+1))^gamma, as the risk of reaching
         // numerical ceilings is reduced. This is mathematically equivalent (e.g. see Noam
         // Brown's PhD thesis "Equilibrium Finding for Large Adversarial Imperfect-Information
         // Games").

         // normalization factor from the papers is irrelevant, as it is absorbed by the
         // normalization constant of each action policy afterwards.
         // add + 1 to the iteration count to account for the correct factor of iteration t
         // (iteration 0 (numerically) is iteration 1 (logically), as in the theoretical work)
         auto t = double(_iteration() + 1);
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
         // Note: we are not incrementing the iteration + 1 to the logical equivalent as for the
         // policy weight above for mere empirical resasons: In test cases convergence was faster
         // this way and the mixing of different iteration weights is negligible in the limit.
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
   auto node_view = std::invoke([&] {
      if constexpr(config.update_mode == UpdateMode::alternating) {
         return _infonodes()
                | ranges::views::filter(
                   [update_player = *player_to_update](const auto& infostate_ptr_data) {
                      return std::get< 0 >(infostate_ptr_data)->player() == update_player;
                   }
                );
      } else {
         return ranges::views::all(_infonodes());
      }
   });

   std::for_each(
      std::execution::par_unseq,
      node_view.begin(),
      node_view.end(),
      [&](auto& infostate_ptr_data) {
         auto& [infostate_ptr, data] = infostate_ptr_data;
         _invoke_regret_minimizer(*infostate_ptr, data, policy_weight(data), regret_weights(data));
      }
   );
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
void VanillaCFR< config, Env, Policy, AveragePolicy >::_invoke_regret_minimizer(
   const InfostateHolder< info_state_type >& infostate,
   infostate_data_type& istate_data,
   [[maybe_unused]] double policy_weight,
   [[maybe_unused]] const auto& regret_weights
)
{
   // since we are reusing this variable a few times we alias it here
   auto& current_policy = fetch_policy< PolicyLabel::current >(infostate, istate_data.actions());

   // Discounted CFR only:
   // we first multiply the accumulated regret by the correct weight as per discount setting
   auto& regret_table = istate_data.regret();
   if constexpr(config.weighting_mode == CFRWeightingMode::discounted) {
      static_assert(
         std::is_same_v< std::array< double, 2 >, std::remove_cvref_t< decltype(regret_weights) > >,
         "Expected a regret weight array of length 2."
      );
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
   if constexpr(config.pruning_mode == CFRPruningMode::regret_based //
                and config.regret_minimizing_mode == RegretMinimizingMode::regret_matching_plus) {
      m_regret_minimizer(
         current_policy,
         regret_table,
         [](const ActionHolder< action_type >& action) { return std::ref(action.get()); },
         istate_data.template storage_element< 1 >()
      );
   } else {
      m_regret_minimizer(
         current_policy,
         regret_table,
         [](const ActionHolder< action_type >& action) { return std::ref(action.get()); }
      );
   }

   // now we update the current accumulated policy by the iteration factor, again as per
   // discount setting.
   if constexpr(common::isin(
                   config.weighting_mode, {CFRWeightingMode::linear, CFRWeightingMode::discounted}
                )) {
      // we are expecting to be given the right weight for the configuration here
      for(auto& policy_prob : fetch_policy< PolicyLabel::average >(infostate, istate_data.actions())
                                 | ranges::views::values) {
         policy_prob *= policy_weight;
      }
   }
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
void VanillaCFR< config, Env, Policy, AveragePolicy >::_invoke_regret_minimizer(
   const InfostateHolder< info_state_type >& infostate,
   infostate_data_type& istate_data,
   auto&&...
)
   requires(config.weighting_mode == CFRWeightingMode::exponential)
{
   auto& instant_regret_table = istate_data.template storage_element< 1 >();
   auto exp_l1_weights = std::invoke([&] {
      // we need to reset this infostate data node's accumulated weights to prepare empty
      // buffers for the next iteration
      std::unordered_map<
         std::reference_wrapper< const action_type >,
         double,
         common::ref_wrapper_hasher< const action_type >,
         common::ref_wrapper_comparator< const action_type > >
         l1;

      // instant regret table holds action_ref -> instant_regret entries in a table
      // instant_regret is r(I, a), not r(h, a)
      double average_instant_regret = ranges::accumulate(
                                         instant_regret_table | ranges::views::values,
                                         double(0.),
                                         std::plus{}
                                      )
                                      / double(instant_regret_table.size());

      ranges::for_each(instant_regret_table, [&](auto& actionref_to_instant_regret) {
         auto& [action_ref, instant_regret] = actionref_to_instant_regret;
         // instant_regret is r(I, a), not R(I, a), which would be the cumulative regret
         l1[action_ref] = std::exp(instant_regret - average_instant_regret);
      });
      return l1;
   });
   // exponential cfr requires weighting the cumulative regret by the L1 factor to EACH (I, a)
   // pair. Yet L1, which is actually L1(I, a), is only known after the entire tree has been
   // traversed and thus can't be done during the traversal. Hence, we need to update our
   // cumulative regret by the correct weight now here upon iteration over all infostates
   auto& curr_policy = fetch_policy< PolicyLabel::current >(infostate, istate_data.actions());
   auto& regret_table = istate_data.regret();
   for(auto& [action, cumul_regret] : regret_table) {
      auto action_ref = std::cref(action.get());
      auto& instant_regret = instant_regret_table[action_ref];
      LOGD2("Action", action);
      LOGD2("L1 weight", exp_l1_weights[action_ref]);
      LOGD2("Instant regret", instant_regret);
      LOGD2("All instant regret", ranges::views::values(instant_regret_table));
      LOGD2("Cumul regret before", ranges::views::values(istate_data.regret()));

      if(instant_regret >= 0) {
         LOGD2("Cumul regret update (>=0)", exp_l1_weights[action_ref] * instant_regret);
         cumul_regret += exp_l1_weights[action_ref] * instant_regret;
      } else {
         LOGD2(
            "Cumul regret update (<0)",
            exp_l1_weights[action_ref] * m_expcfr_params.beta(instant_regret, _iteration())
         );
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
       fetch_policy< PolicyLabel::average >(infostate, istate_data.actions())) {
      double reach_prob = istate_data.template storage_element< 2 >();
      double l1_weight = exp_l1_weights[std::cref(action.get())];
      // this is the cumulative enumerator update
      LOGD2("Cumul Policy Reach prob", reach_prob);
      LOGD2("Cumul Policy L1 Weight prob", l1_weight);
      LOGD2("Cumul Policy Current Policy", curr_policy[action]);
      LOGD2("Cumul Policy numerator update", l1_weight * reach_prob * curr_policy[action]);
      avg_policy_prob += l1_weight * reach_prob * curr_policy[action];
      // this is the cumulative denominator update
      LOGD2("Cumul Policy denominator update", l1_weight * reach_prob);
      istate_data.template storage_element< 3 >(std::cref(action.get())) += l1_weight * reach_prob;
   }

   LOGD2("Cumul Policy after", ranges::views::values(istate_data.template storage_element< 3 >()));
   // here we now perform the actual regret minimizing update step as we update the current
   // policy through a regret matching algorithm. The specific algorihtm is determined by the
   // config we entered to the class

   // we provide the accessor lambda to get the underlying referenced action, as the infodata
   // stores only reference wrappers to the actions
   m_regret_minimizer(curr_policy, regret_table, [](const ActionHolder< action_type >& action) {
      return std::ref(action.get());
   });
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
template < bool initialize_infonodes, bool use_current_policy >
StateValueMap VanillaCFR< config, Env, Policy, AveragePolicy >::_traverse(
   std::optional< Player > player_to_update,
   WorldstateHolder< world_state_type > state,
   ReachProbabilityMap reach_probability_map,
   ObservationbufferMap observation_buffer,
   SharedInfostateMap infostates
)
{
   if(_env().is_terminal(state)) {
      return StateValueMap{collect_rewards(_env(), state)};
   }

   if constexpr(config.pruning_mode == CFRPruningMode::partial) {
      if(_partial_pruning_condition(player_to_update, reach_probability_map)) {
         // if the entire subtree is pruned then the values that could be found are all 0. for
         // each player
         return StateValueMap{std::invoke([&] {
            StateValueMap::UnderlyingType map;
            for(auto player : _env().players(state) | is_actual_player_pred) {
               map[player] = 0.;
            }
            return map;
         })};
      }
   }

   Player active_player = _env().active_player(state);
   // the state's value for each player. To be filled by the action traversal functions.
   StateValueMap state_value{{}};
   // each action's value for each player. To be filled by the action traversal functions.
   std::unordered_map< action_variant_type, StateValueMap > action_value;
   // traverse all child states from this state. The constexpr check for determinism in the env
   // allows deterministic envs to not provide certain functions that are only needed in the
   // stochastic case.
   if constexpr(concepts::stochastic_env< env_type >) {
      if(active_player == Player::chance) {
         _traverse_chance_actions< initialize_infonodes, use_current_policy >(
            player_to_update,
            active_player,
            std::move(state),
            reach_probability_map,
            std::move(observation_buffer),
            std::move(infostates),
            state_value,
            action_value
         );
         // if this is a chance node then we don't need to update any regret or average policy
         // after the traversal
         return state_value;
      }
   }

   InfostateHolder< info_state_type, std::true_type > this_infostate = infostates.get().at(
      active_player
   );

   _traverse_player_actions< initialize_infonodes, use_current_policy >(
      player_to_update,
      active_player,
      std::move(state),
      reach_probability_map,
      std::move(observation_buffer),
      std::move(infostates),
      state_value,
      action_value
   );

   if constexpr(use_current_policy) {
      // we can only update our regrets and policies if we are traversing with the current
      // policy, since the average policy is not to be changed directly (but through averaging up
      // all current policies)
      if constexpr(config.update_mode == UpdateMode::alternating) {
         // in alternating updates, we only update the regret and strategy if the current
         // player is the chosen player to update.
         if(active_player == player_to_update.value()) {
            update_regret_and_policy(
               *this_infostate, reach_probability_map, state_value, action_value
            );
         }
      } else {
         // if we do simultaenous updates, then we always update the regret and strategy
         // values of the node's active player.
         update_regret_and_policy(
            *this_infostate, reach_probability_map, state_value, action_value
         );
      }
   }
   return StateValueMap{std::move(state_value)};
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
template < bool initialize_infonodes, bool use_current_policy >
void VanillaCFR< config, Env, Policy, AveragePolicy >::_traverse_player_actions(
   std::optional< Player > player_to_update,
   Player active_player,
   WorldstateHolder< world_state_type > state,
   const ReachProbabilityMap& reach_probability,
   const ObservationbufferMap& observation_buffer,
   SharedInfostateMap infostate_map,
   StateValueMap& state_value,
   std::unordered_map< action_variant_type, StateValueMap >& action_value
)
{
   const auto& this_infostate = infostate_map.get().at(active_player);
   if constexpr(initialize_infonodes) {
      _infonodes().emplace(
         this_infostate, infostate_data_type{_env().actions(active_player, state)}
      );
   }
   const auto& actions = _infonode(this_infostate).actions();
   auto& action_policy = fetch_policy< use_current_policy >(this_infostate, actions);
   double normalizing_factor = std::invoke([&] {
      if constexpr(not use_current_policy) {
         // we try to normalize only for the average policy, since iterations with the current
         // policy are for the express purpose of updating the average strategy. As such, we
         // should not intervene to change these values, as that may alter the values incorrectly
         double normalization = ranges::accumulate(
            action_policy | ranges::views::values, double(0.), std::plus{}
         );
         if(std::abs(normalization) < 1e-20) {
            throw std::invalid_argument(
               "Average policy likelihoods accumulate to 0. Such values cannot be normalized."
            );
         }
         return normalization;
      } else
         return 1.;
   });
   for(const action_type& action : actions) {
      auto action_prob = action_policy[action] / normalizing_factor;

      auto child_reach_prob = reach_probability.get();
      child_reach_prob[active_player] *= action_prob;

      auto next_wstate = child_state(_env(), state, action);
      auto [child_observation_buffer, child_infostate_map] = next_infostate_and_obs_buffers(
         _env(), observation_buffer.get(), infostate_map.get(), state, action, next_wstate
      );

      StateValueMap child_rewards_map = _traverse< initialize_infonodes, use_current_policy >(
         player_to_update,
         std::move(next_wstate),
         ReachProbabilityMap{std::move(child_reach_prob)},
         ObservationbufferMap{std::move(child_observation_buffer)},
         SharedInfostateMap{std::move(child_infostate_map)}
      );
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
   WorldstateHolder< world_state_type > state,
   const ReachProbabilityMap& reach_probability,
   const ObservationbufferMap& observation_buffer,
   SharedInfostateMap infostate_map,
   StateValueMap& state_value,
   std::unordered_map< action_variant_type, StateValueMap >& action_value
)
{
   for(auto&& outcome : _env().chance_actions(state)) {
      auto next_wstate = child_state(_env(), state, outcome);

      auto child_reach_prob = reach_probability.get();
      auto outcome_prob = _env().chance_probability(*state, outcome);
      child_reach_prob[active_player] *= outcome_prob;

      auto [child_observation_buffer, child_infostate_map] = next_infostate_and_obs_buffers(
         _env(), observation_buffer.get(), infostate_map.get(), state, outcome, next_wstate
      );

      StateValueMap child_rewards_map = _traverse< initialize_infonodes, use_current_policy >(
         player_to_update,
         std::move(next_wstate),
         ReachProbabilityMap{std::move(child_reach_prob)},
         ObservationbufferMap{std::move(child_observation_buffer)},
         SharedInfostateMap{std::move(child_infostate_map)}
      );
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
   const InfostateHolder< info_state_type >& infostate,
   const ReachProbabilityMap& reach_probability,
   const StateValueMap& state_value,
   const std::unordered_map< action_variant_type, StateValueMap >& action_value_map
)
{
   auto& istate_data = _infonode(infostate);
   const auto& actions = istate_data.actions();
   auto& curr_action_policy = fetch_policy< PolicyLabel::current >(infostate, actions);
   auto& avg_action_policy = [&]() -> auto& {
      if constexpr(config.weighting_mode != CFRWeightingMode::exponential) {
         return fetch_policy< PolicyLabel::average >(infostate, actions);
      } else {
         // this value will be ignored, so we can simply return anything that is cheap to fetch
         // (it has to be an l-value so can't return an r-value like simply 0
         return _env();
      }
   }();
   auto player = infostate.player();
   double cf_reach_prob = rm::cf_reach_probability(player, reach_probability.get());
   double player_reach_prob = reach_probability.get().at(player);
   double player_state_value = state_value.get().at(player);

   for(const auto& [action_variant, action_value] : action_value_map) {
      // we only call this function with action values from a non-chance player, so we can safely
      // assume that the action is of action_type
      const auto& action = std::get< 0 >(action_variant);
      // update the cumulative regret according to the formula:
      // let I be the infostate, p be the player, r the cumulative regret
      //    r = \sum_a counterfactual_reach_prob_{p}(I) * (value_{p}(I-->a) - value_{p}(I))
      if constexpr(config.weighting_mode != CFRWeightingMode::exponential) {
         if(cf_reach_prob > 0) {
            // this if statement effectively introduces partial pruning. But this is such a
            // slight modification (and gain, if any) that it is to be included in all variants
            // of CFR
            //
            // all other cfr variants currently implemented need the average regret update at
            // history update time
            istate_data.regret(action) += cf_reach_prob
                                          * (action_value.get().at(player) - player_state_value);
         }
      } else {
         // for the exponential cfr method we need to remember these regret increments of
         // iteration t, until the end of iteration t, and apply them once we have computed the
         // L1 weights (i.e. at infostate update time, not history update time). After iteration
         // t ends we have to delete them again, so that this is only a memory of the current
         // iteration! Each history h that passed through infostate I will increment here the
         // instantaneous regret values r(h,a), in order to accumulate r(I, a) = sum_h r(h, a)

         // We emplace the action first into the cumul regret map (if not already there) to
         // receive the action key's reference back. This reference is then going to be
         // ref-wrapped and added to the instant temp regret map. We avoid copying the action
         // this way. For games with elaborate action types, this may be worth the extra runtime,
         // for small action values probably not. We do however save some memory since ref
         // wrappers are merely as big as one pointer.
         auto [iter, _] = istate_data.regret().try_emplace(action, 0.);
         // if we emplaced merely std::cref(action) here then we would have silent segfaults that
         // lead to erroenuous memory storing
         istate_data.template storage_element< 1 >(std::cref(iter->first)
         ) += cf_reach_prob * (action_value.get().at(player) - player_state_value);
      }
      if constexpr(config.weighting_mode != CFRWeightingMode::exponential) {
         // update the cumulative policy according to the formula:
         // let
         //    'I' be the infostate,
         //    'p' be the player,
         //    'a' be the chosen action,
         //    'sigma^t' the current policy
         //  -->  avg_sigma^{t+1} = \sum_a reach_prob_{p}(I) * sigma^t(I, a)
         avg_action_policy[action] += player_reach_prob * curr_action_policy[action];
         // For exponential CFR we update the average policy after the tree traversal
      }
   }
   if constexpr(config.weighting_mode == CFRWeightingMode::exponential) {
      // For exponential CFR we need to store the reach probability of the active player until
      // the end of the iteration
      istate_data.template storage_element< 2 >() = player_reach_prob;
   }
}

namespace detail {
/// @brief a verification of the correctness of the chosen configuration
template < CFRConfig config >
consteval bool sanity_check_cfr_config()
{
   if constexpr(
      (config.weighting_mode == CFRWeightingMode::exponential)
      and (config.pruning_mode == CFRPruningMode::regret_based)
      and (config.regret_minimizing_mode == RegretMinimizingMode::regret_matching_plus)
   ) {
      // there is currently no theoretic work on combining these methods and the update rule
      // for the cumulative regret in both clash with different approaches (exp weighting
      // wants e^L1 weighted updates) while regret-based-pruning with CFR+ wants to replace
      // the cumulative regret with r(I,a) only if r(I,a) > 0 and R^T(I,a) < 0, otherwise do a
      // normal cumulative regret update (i.e. R^t+1(I,a) = R^t(I,a) + r(I,a))
      return false;
   }
   return true;
}

}  // namespace detail

}  // namespace nor::rm
