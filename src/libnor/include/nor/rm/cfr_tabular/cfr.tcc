#pragma once

#include <spdlog/spdlog.h>
#include <algorithm>
#include <limits>

#include "cfr.hpp"

namespace nor::rm {

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
auto VanillaCFR< config, Env, Policy, AveragePolicy >::iterate(size_t n_iters)
{
   std::vector< player_hashmap< double > > root_values_per_iteration;
   root_values_per_iteration.reserve(n_iters);
   for([[maybe_unused]] auto _ : std::views::iota(size_t(0), n_iters)) {
      SPDLOG_DEBUG("Iteration number: {}", _iteration());
      StateValueMap value = std::invoke([&] {
         if constexpr(config.update_mode == UpdateMode::alternating) {
            auto player_to_update = _cycle_player_to_update();
            if(_iteration() < _env().players(*_root_state_uptr()).size() - 1) [[unlikely]] {
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
      root_values_per_iteration.emplace_back(std::move(value.get().to_hashmap()));
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
   SPDLOG_DEBUG("Iteration number: {}", _iteration());
   // run the iteration
   StateValueMap values = [&] {
      if(_iteration() < _env().players(*_root_state_uptr()).size() - 1)
         return _iterate< true >(_cycle_player_to_update(player_to_update));
      else
         return _iterate< false >(_cycle_player_to_update(player_to_update));
   }();
   // and increment our iteration counter
   _iteration()++;
   return std::vector{std::move(values.get().to_hashmap())};
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
template < bool initializing_run, bool use_current_policy >
auto VanillaCFR< config, Env, Policy, AveragePolicy >::_iterate(
   std::optional< Player > player_to_update
)
{
   auto root_players = _env().players(root_state());

   // the traversal containers are created once per iteration here (cheap) and
   // then mutated in place down the recursion with explicit save/restore at
   // every recursion boundary; no per-edge container copies happen anymore.
   auto rp_map = std::invoke([&] {
      ReachProbabilityMap rp{{}};
      for(auto player : root_players) {
         rp.get().emplace(player, 1.);
      }
      return rp;
   });
   auto obs_map = std::invoke([&] {
      ObservationbufferMap obs{{}};
      for(auto player : root_players | utils::is_actual_player_filter) {
         obs.get().emplace(
            player, std::vector< std::pair< observation_type, observation_type > >{}
         );
      }
      return obs;
   });
   auto infostates = std::invoke([&] {
      InfostateSptrMap istates{{}};
      for(auto player : root_players | utils::is_actual_player_filter) {
         istates.get().emplace(player, std::make_shared< info_state_type >(player));
      }
      return istates;
   });

   // copy-assign the root state into arena slot 0 (reused across iterations)
   world_state_type& root_arena_state = _arena_state(0, *_root_state_uptr());

   auto root_game_value = _traverse< initializing_run, use_current_policy >(
      player_to_update,
      root_arena_state,
      /*depth=*/0,
      rp_map,
      obs_map,
      infostates
   );

   if constexpr(use_current_policy) {
      _initiate_regret_minimization(player_to_update);
   }
   return root_game_value;
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
auto VanillaCFR< config, Env, Policy, AveragePolicy >::_arena_state(
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

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
void VanillaCFR< config, Env, Policy, AveragePolicy >::_initiate_regret_minimization(
   const std::optional< Player >& player_to_update
)
{
   // here we now invoke the actual regret minimization procedure for each infostate individually.
   // The sweep is intentionally serial: per-infostate workloads are tiny and a parallel
   // schedule would render float summation orders (and hence table contents)
   // non-deterministic. See the determinism note in rm_utils.hpp.
   auto node_view = std::invoke([&] {
      if constexpr(config.update_mode == UpdateMode::alternating) {
         return _infonodes()
                | std::views::filter(
                   [update_player = *player_to_update](const auto& infostate_ptr_data) {
                      return std::get< 0 >(infostate_ptr_data)->player() == update_player;
                   }
                );
      } else {
         return std::views::all(_infonodes());
      }
   });

   std::for_each(node_view.begin(), node_view.end(), [&](auto& infostate_ptr_data) {
      auto& [infostate_ptr, data] = infostate_ptr_data;
      _invoke_regret_minimizer(*infostate_ptr);
   });
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
void VanillaCFR< config, Env, Policy, AveragePolicy >::_invoke_regret_minimizer(
   const info_state_type& infostate
)
{
   auto& istate_data = _infonode(infostate);

   if constexpr(config.weighting_mode == CFRWeightingMode::exponential) {
      // exponential cfr defers all cumulative updates to the end of an
      // iteration: the L1-weighted regret increments are applied here, the
      // average policy numerator/denominator pair is updated and finally the
      // current policy is refreshed through regular regret matching.
      auto& current_policy =
         this->template fetch_policy< PolicyLabel::current >(infostate, istate_data.actions());
      auto& average_policy =
         this->template fetch_policy< PolicyLabel::average >(infostate, istate_data.actions());
      m_regret_minimizer.finalize_iteration(
         istate_data.data(),
         current_policy,
         average_policy,
         _iteration(),
         [this](double instant_regret, size_t iteration) {
            return m_expcfr_params.beta(instant_regret, iteration);
         }
      );
   } else {
      auto& current_policy =
         this->template fetch_policy< PolicyLabel::current >(infostate, istate_data.actions());
      // derive the recommendation from the (possibly weighted) stored regret
      m_regret_minimizer.recommend(istate_data.data(), current_policy, _iteration());

      // scale the accumulated average policy by this iteration's weight.
      // B3: with 'weight_by_cycle' the gamma exponentiation indexes by the
      // cycle number (iteration / num_players) instead of the raw iteration.
      if constexpr(config.weighting_mode == CFRWeightingMode::discounted) {
         const size_t weight_index =
            m_regret_minimizer.discounted_parameters().weight_by_cycle ? this->cycle()
                                                                          : _iteration();
         const double policy_weight = m_regret_minimizer.policy_weight(weight_index);
         for(auto& policy_prob :
             this->template fetch_policy< PolicyLabel::average >(infostate, istate_data.actions())
                | std::views::values) {
            policy_prob *= policy_weight;
         }
      }
   }
   if constexpr(
      config.pruning_mode == CFRPruningMode::regret_based
      or config.pruning_mode == CFRPruningMode::dynamic_thresholding
   ) {
      // the regret tables are now in their post-iteration state (RM+RBP has folded its
      // instantaneous buffer via the replace-if-positive rule); scan for entries negative
      // enough to open a pruning window for the NEXT iterations
      _arm_pruning_windows(infostate, istate_data.data());
   }
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
template < bool initialize_infonodes, bool use_current_policy >
StateValueMap VanillaCFR< config, Env, Policy, AveragePolicy >::_traverse(
   std::optional< Player > player_to_update,
   world_state_type& state,
   size_t depth,
   ReachProbabilityMap& reach_probability,
   ObservationbufferMap& observation_buffer,
   InfostateSptrMap& infostates
)
{
    if(_env().is_terminal(state)) {
       // NOTE: pass the ROOT participant set explicitly. The default of
       // collect_rewards derives the player set from env.players(terminal
       // state), which excludes players that have folded out of poker-like
       // games. Downstream, update_regret_and_policy looks up every actual
       // player in each action's value map and would throw out_of_range for
       // such subtrees. Environments must support reward() for all initial
       // participants (e.g. leduc pays folded players their sunk stakes).
       return StateValueMap{
          collect_rewards(_env(), state, _env().players(root_state()))};
    }

   if constexpr(config.pruning_mode == CFRPruningMode::partial) {
      if(_partial_pruning_condition(player_to_update, reach_probability)) {
         // if the entire subtree is pruned then the values that could be found are all 0. for
         // each player
         return StateValueMap{std::invoke([&] {
            StateValueMap::UnderlyingType map;
            for(auto player : _env().players(state) | utils::is_actual_player_pred) {
               map[player] = 0.;
            }
            return map;
         })};
      }
   }

   Player active_player = _env().active_player(state);
   // the state's value for each player. To be filled by the action traversal functions.
   StateValueMap state_value{};
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
            state,
            depth,
            reach_probability,
            observation_buffer,
            infostates,
            state_value,
            action_value
         );
         // if this is a chance node then we don't need to update any regret or average policy
         // after the traversal
         return state_value;
      }
   }

   sptr< info_state_type > this_infostate = infostates.get().at(active_player);

   _traverse_player_actions< initialize_infonodes, use_current_policy >(
      player_to_update,
      active_player,
      state,
      depth,
      reach_probability,
      observation_buffer,
      infostates,
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
            update_regret_and_policy(*this_infostate, reach_probability, state_value, action_value);
         }
      } else {
         // if we do simultaenous updates, then we always update the regret and strategy
         // values of the node's active player.
         update_regret_and_policy(*this_infostate, reach_probability, state_value, action_value);
      }
   }
   return StateValueMap{std::move(state_value)};
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
template < bool initialize_infonodes, bool use_current_policy >
void VanillaCFR< config, Env, Policy, AveragePolicy >::_traverse_player_actions(
   std::optional< Player > player_to_update,
   Player active_player,
   world_state_type& state,
   size_t depth,
   ReachProbabilityMap& reach_probability,
   ObservationbufferMap& observation_buffer,
   InfostateSptrMap& infostates,
   StateValueMap& state_value,
   std::unordered_map< action_variant_type, StateValueMap >& action_value
)
{
   const auto& this_infostate = infostates.get().at(active_player);
   if constexpr(initialize_infonodes) {
      _infonodes().emplace(
         this_infostate, infostate_data_type{_env().actions(active_player, state)}
      );
   }
   const auto& actions = _infonode(this_infostate).actions();
   auto& action_policy = this->template fetch_policy< use_current_policy >(
      *this_infostate, actions
   );
   double normalizing_factor = std::invoke([&] {
      if constexpr(not use_current_policy) {
         // we try to normalize only for the average policy, since iterations with the current
         // policy are for the express purpose of updating the average strategy. As such, we
         // should not intervene to change these values, as that may alter the values incorrectly
         double normalization =
            std::ranges::fold_left(action_policy | std::views::values, double(0.), std::plus{});
         if(std::abs(normalization) < 1e-20) {
            throw std::invalid_argument(
               "Average policy likelihoods accumulate to 0. Such values cannot be normalized."
            );
         }
         return normalization;
      } else
         return 1.;
   });
   // deferred per-visit records of pruned edges traversed in THIS visit; their value
   // contributions are pushed once state_value below is fully aggregated (the buffer update
   // needs the final v(I), which includes all sibling actions)
   [[maybe_unused]] std::vector< std::pair< size_t, double > > pending_window_visits{};
   for(auto [action_idx, action] : std::views::enumerate(actions)) {
      if constexpr(
         config.pruning_mode == CFRPruningMode::regret_based
         or config.pruning_mode == CFRPruningMode::dynamic_thresholding
      ) {
         if constexpr(use_current_policy) {
            if(_rbp_gate(
                  player_to_update,
                  active_player,
                  _infonode(this_infostate).data(),
                  action_idx,
                  action,
                  infostates,
                  observation_buffer,
                  state,
                  depth
               ))
            {
               // the pruned action's current probability is exactly zero, so its contribution
               // to v(I) and its average-strategy increment would both vanish anyway; skipping
               // the recursion leaves state_value and action_value untouched for this action
               pending_window_visits.emplace_back(
                  action_idx, rm::cf_reach_probability(active_player, reach_probability.get())
               );
               continue;
            }
         }
      }
      auto action_prob = action_policy[action] / normalizing_factor;

      // ---- save/restore bookkeeping for the recursion boundary --------------
      // reach probability: only the acting player's entry is scaled
      double& player_reach_entry = reach_probability.get()[active_player];
      const double saved_reach_prob = player_reach_entry;
      // scale in place instead of copying the whole reach map per edge
      player_reach_entry *= action_prob;

      // advance the arena slot of the next depth: copy-assign + transition
      // (no allocation after the first visit to this depth)
      world_state_type& next_wstate = _arena_state(depth + 1, state);
      _env().transition(next_wstate, action);

      // fold the transition's observations into the buffers / the cloned
      // infostate exactly like next_infostate_and_obs_buffers_inplace would --
      // but on the live containers, with restoration after the recursion.
      // NOTE: a transition may leave the chance player in charge (multi-draw
      // deals); like the inplace helper we never buffer/flush FOR chance and
      // never touch the infostate table then.
      auto& obs_buffer = observation_buffer.get();
      const auto public_obs = _env().public_observation(state, action, next_wstate);
      const Player next_active_player = _env().active_player(next_wstate);
      // the infostate that advances across this edge belongs to the player who
      // is active in the CHILD state (exactly like the inplace helper): when
      // that player takes over, his buffered observations are flushed into a
      // clone of his infostate while every other player's sptr stays untouched
      const auto infostate_entry_it = infostates.get().find(next_active_player);
      const bool flushes =
         next_active_player != Player::chance and infostate_entry_it != infostates.get().end();

      sptr< info_state_type > saved_infostate{};
      auto child_infostate = flushes ?
                                std::make_shared< info_state_type >(*infostate_entry_it->second) :
                                nullptr;
      std::optional< std::vector< std::pair< observation_type, observation_type > > >
         saved_flush_target_buffer{};
      if(flushes) {
         // remember the pre-edge infostate and buffer so they can be restored
         // verbatim once the recursion returns
         saved_infostate = infostate_entry_it->second;
         // snapshot of the flush-target's buffer (it is drained into the clone)
         saved_flush_target_buffer = obs_buffer.at(next_active_player);
      }
      // sizes of all non-flush-target buffers (they receive appended pairs)
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
               ::nor::detail::update_infostate(
                  child_infostate, std::move(obs.first), std::move(obs.second)
               );
            }
            obs_history.clear();
            ::nor::detail::update_infostate(
               child_infostate,
               public_obs,
               _env().private_observation(player, state, action, next_wstate)
            );
         } else {
            obs_buffer[player].emplace_back(
               public_obs, _env().private_observation(player, state, action, next_wstate)
            );
         }
      }
      if(flushes) {
         infostate_entry_it->second = std::move(child_infostate);
      }

      StateValueMap child_rewards_map = _traverse< initialize_infonodes, use_current_policy >(
         player_to_update,
         next_wstate,
         depth + 1,
         reach_probability,
         observation_buffer,
         infostates
      );

      // ---- restore everything the recursion mutated -------------------------
      if(flushes) {
         infostate_entry_it->second = std::move(saved_infostate);
         obs_buffer.at(next_active_player) = std::move(*saved_flush_target_buffer);
      }
      for(const auto& [player, size] : saved_buffer_sizes) {
         obs_buffer.at(player).resize(size);
      }
      player_reach_entry = saved_reach_prob;

      // add the child state's value to the respective player's value table, multiplied by the
      // policies likelihood of playing this action
      for(auto [player, child_value] : child_rewards_map.get()) {
         state_value.get()[player] += action_prob * child_value;
      }
      action_value.emplace(action, std::move(child_rewards_map));
   }

   if constexpr(
      config.pruning_mode == CFRPruningMode::regret_based
      or config.pruning_mode == CFRPruningMode::dynamic_thresholding
   ) {
      if(not pending_window_visits.empty()) {
         auto& node_data = _infonode(this_infostate).data();
         const double v_I = state_value.get().at(active_player);
         for(auto [action_idx, cf_reach] : pending_window_visits) {
            _push_window_visit(node_data, active_player, action_idx, cf_reach, v_I);
         }
      }
   }
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
template < bool initialize_infonodes, bool use_current_policy >
void VanillaCFR< config, Env, Policy, AveragePolicy >::_traverse_chance_actions(
   std::optional< Player > player_to_update,
   Player active_player,
   world_state_type& state,
   size_t depth,
   ReachProbabilityMap& reach_probability,
   ObservationbufferMap& observation_buffer,
   InfostateSptrMap& infostates,
   StateValueMap& state_value,
   std::unordered_map< action_variant_type, StateValueMap >& action_value
)
{
   for(auto&& outcome : _env().chance_actions(state)) {
      auto outcome_prob = _env().chance_probability(state, outcome);

      // NOTE: environments differ in whether 'Player::chance' is part of
      // env.players(root_state) (kuhn: yes, leduc: no). Default-initializing
      // the entry via operator[] here would multiply 0 by the outcome
      // probability and permanently zero out the counterfactual reach of every
      // descendant infostate, silently disabling all regret updates. Seed the
      // entry with the outcome probability when absent. Save/restore keeps the
      // caller's map intact across sibling edges without per-edge map copies.
      auto [chance_reach_entry, inserted] =
         reach_probability.get().try_emplace(active_player, outcome_prob);
      const double saved_chance_reach = chance_reach_entry->second;
      if(not inserted) {
         chance_reach_entry->second *= outcome_prob;
      }

      world_state_type& next_wstate = _arena_state(depth + 1, state);
      _env().transition(next_wstate, outcome);

      // chance transitions fold into the buffers/infostates exactly like the
      // inplace helper: observations are buffered for every actual player and a
      // flush only happens once an ACTUAL player becomes active (multi-draw
      // deals chain several chance edges with buffering only).
      auto& obs_buffer = observation_buffer.get();
      const auto public_obs = _env().public_observation(state, outcome, next_wstate);
      const Player next_active_player = _env().active_player(next_wstate);
      const auto infostate_entry_it = infostates.get().find(next_active_player);
      const bool flushes =
         next_active_player != Player::chance and infostate_entry_it != infostates.get().end();

      sptr< info_state_type > saved_infostate{};
      auto child_infostate = flushes ?
                                std::make_shared< info_state_type >(*infostate_entry_it->second) :
                                nullptr;
      std::optional< std::vector< std::pair< observation_type, observation_type > > >
         saved_flush_target_buffer{};
      if(flushes) {
         saved_infostate = infostate_entry_it->second;
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
               ::nor::detail::update_infostate(
                  child_infostate, std::move(obs.first), std::move(obs.second)
               );
            }
            obs_history.clear();
            ::nor::detail::update_infostate(
               child_infostate,
               public_obs,
               _env().private_observation(player, state, outcome, next_wstate)
            );
         } else {
            obs_buffer[player].emplace_back(
               public_obs, _env().private_observation(player, state, outcome, next_wstate)
            );
         }
      }
      if(flushes) {
         infostate_entry_it->second = std::move(child_infostate);
      }

      StateValueMap child_rewards_map = _traverse< initialize_infonodes, use_current_policy >(
         player_to_update,
         next_wstate,
         depth + 1,
         reach_probability,
         observation_buffer,
         infostates
      );

      if(flushes) {
         infostate_entry_it->second = std::move(saved_infostate);
         obs_buffer.at(next_active_player) = std::move(*saved_flush_target_buffer);
      }
      for(const auto& [player, size] : saved_buffer_sizes) {
         obs_buffer.at(player).resize(size);
      }
      if(inserted) {
         reach_probability.get().erase(active_player);
      } else {
         chance_reach_entry->second = saved_chance_reach;
      }

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
   const info_state_type& infostate,
   const ReachProbabilityMap& reach_probability,
   const StateValueMap& state_value,
   const std::unordered_map< action_variant_type, StateValueMap >& action_value_map
)
{
   auto& istate_data = _infonode(infostate);
   const auto& actions = istate_data.actions();
   auto& curr_action_policy = this->template fetch_policy< PolicyLabel::current >(
      infostate, actions
   );
   auto& avg_action_policy = [&]() -> auto& {
      if constexpr(config.weighting_mode != CFRWeightingMode::exponential) {
         return this->template fetch_policy< PolicyLabel::average >(infostate, actions);
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
            m_regret_minimizer.observe(
               istate_data.data(),
               action,
               cf_reach_prob * (action_value.get().at(player) - player_state_value)
            );
         }
      } else {
         // for the exponential cfr method we need to remember these regret increments of
         // iteration t, until the end of iteration t, and apply them once we have computed the
         // L1 weights (i.e. at infostate update time, not history update time). After iteration
         // t ends we have to delete them again, so that this is only a memory of the current
         // iteration! Each history h that passed through infostate I will increment here the
         // instantaneous regret values r(h,a), in order to accumulate r(I, a) = sum_h r(h, a)
         m_regret_minimizer.observe(
            istate_data.data(),
            action,
            cf_reach_prob * (action_value.get().at(player) - player_state_value)
         );
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
      istate_data.data().reach_prob_snapshot = player_reach_prob;
   }
}
namespace detail {

/// @brief a verification of the correctness of the chosen configuration
template < CFRConfig config >
consteval bool sanity_check_cfr_config()
{
   if constexpr(
      config.regret_minimizing_mode == RegretMinimizingMode::predictive_regret_matching_plus
      or config.regret_minimizing_mode
         == RegretMinimizingMode::sap_predictive_regret_matching_plus
      or config.regret_minimizing_mode == RegretMinimizingMode::ap_predictive_regret_matching_plus
      or config.regret_minimizing_mode
         == RegretMinimizingMode::p2p_predictive_regret_matching_plus
      or config.regret_minimizing_mode
         == RegretMinimizingMode::smooth_predictive_regret_matching_plus
      or config.regret_minimizing_mode
         == RegretMinimizingMode::stable_predictive_regret_matching_plus
      or config.regret_minimizing_mode == RegretMinimizingMode::discounted_regret_matching_plus
      or config.regret_minimizing_mode
         == RegretMinimizingMode::discounted_predictive_regret_matching_plus
   ) {
      // the predictive regret minimizers (PCFR+/SAPCFR+/APCFR+/P2PCFR+ and the
      // Smooth/Stable-PRM+ robustifications) pair the strategy
      // snapshot of the previous recommendation with the instantaneous regrets
      // of exactly one full iteration. This pairing is only well-defined for
      // alternating updates over unpruned traversals (in simultaneous updates
      // and under pruning the rho/sigma_snap correspondence breaks). The DCFR+/PDCFR+
      // kernels (arXiv:2404.13891) share the same one-phase-per-recommend contract
      // (their deferred fold consumes exactly one fully observed instantaneous
      // regret vector per recommend) and the paper prescribes alternating updates.
      // The discounted weighting mode is REQUIRED as carrier of the gamma-side
      // average-policy accumulation; its own alpha/beta regret discounts are
      // compiled out by the minimizer selection for these modes
      if constexpr(
         config.update_mode != UpdateMode::alternating
         or config.pruning_mode != CFRPruningMode::none
         or config.weighting_mode != CFRWeightingMode::discounted
      ) {
         return false;
      }
   }
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
   if constexpr(config.pruning_mode == CFRPruningMode::regret_based) {
      // Regret-based pruning is proven for alternating updates whose local recommendations are
      // regret-matching based with UNIFORM average-strategy accumulation (Brown & Sandholm,
      // NIPS 2015, secs. 3-4.2; their sec. 4.2 even flags linear averaging as noticeably noisier
      // under RBP). The discounted/linear/exponential families re-weight exactly the regret
      // increments a pruned window buffers and were not analyzed; simultaneous updates traverse
      // every player in one pass so skipped windows would have to be folded per-player.
      // CHOICE: statically reject instead of implementing an unanalyzed safe variant.
      if constexpr(
         config.regret_minimizing_mode != RegretMinimizingMode::regret_matching_plus
         or config.update_mode != UpdateMode::alternating
         or config.weighting_mode != CFRWeightingMode::uniform
      ) {
         return false;
      }
   }
   // dynamic_thresholding composes with any base minimizer recommending via regret matching on
   // its cumulative table (plain RM, RM+ via DiscountedCFR carrier, ExponentialCFR) -- which is
   // what the selection in minimizers.hpp produces; it also keeps working under simultaneous
   // updates since thresholding only reshapes recommendations. The predictive/discounted-kernel
   // modes are already rejected above because they require pruning_mode == none altogether.
   return true;
}

}  // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////
////////////////////// regret-based pruning / dynamic thresholding engine /////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

/// resolves the registered action at 'action_idx' (gate/fold helpers speak in table indices)
template < typename NodeData >
decltype(auto) action_of_index(NodeData& node_data, size_t action_idx)
{
   return node_data.registry.actions[action_idx];
}

/// depth-first enumeration of the full game tree tracking min/max terminal reward per player.
/// The one-shot fallback for environments that do not support the B4 payoff-bounds trait
/// (concepts::has::supports_payoff_bounds); intended for the small bed games RBP targets.
template < typename Env, typename State >
void probe_payoff_bounds_tree(
   const Env& env,
   State& state,
   const std::vector< Player >& players,
   player_hashmap< pruning::PayoffBound >& bounds
)
{
   if(env.is_terminal(state)) {
      for(auto [player, value] : collect_rewards(env, state, players)) {
         auto& bound = bounds.at(player);
         bound.lower = std::min(bound.lower, value);
         bound.upper = std::max(bound.upper, value);
      }
      return;
   }
   Player active = env.active_player(state);
   if constexpr(concepts::stochastic_env< Env >) {
      if(active == Player::chance) {
         for(auto&& outcome : env.chance_actions(state)) {
            State child{state};
            env.transition(child, outcome);
            probe_payoff_bounds_tree(env, child, players, bounds);
         }
         return;
      }
   }
   for(const auto& action : env.actions(active, state)) {
      State child{state};
      env.transition(child, action);
      probe_payoff_bounds_tree(env, child, players, bounds);
   }
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
pruning::PayoffBound VanillaCFR< config, Env, Policy, AveragePolicy >::_payoff_bound(Player player)
{
   if constexpr(
      config.pruning_mode == CFRPruningMode::regret_based
      or config.pruning_mode == CFRPruningMode::dynamic_thresholding
   ) {
      if(m_payoff_bounds.empty()) {
         auto root_players = _env().players(root_state());
         if constexpr(concepts::has::supports_payoff_bounds< env_type >) {
            // B4 trait contract: the environment reports its own per-player bounds
            for(auto p : root_players) {
               if(p == Player::chance) {
                  continue;
               }
               auto [lo, hi] = _env().payoff_bounds(p);
               m_payoff_bounds.emplace(p, pruning::PayoffBound{.lower = lo, .upper = hi});
            }
         } else {
            // fallback: symmetric max-|reward| style bounds probed from the tree exactly once.
            // Seeding with 0/0 only ever WIDENS the observed range, which shrinks windows and
            // is therefore sound for pruning.
            for(auto p : root_players) {
               if(p != Player::chance) {
                  m_payoff_bounds.emplace(p, pruning::PayoffBound{});
               }
            }
            auto& root_ref = *_root_state_uptr();
            using world_state_type = auto_world_state_type< Env >;
            probe_payoff_bounds_tree(_env(), root_ref, root_players, m_payoff_bounds);
         }
      }
      return m_payoff_bounds.at(player);
   } else {
      return pruning::PayoffBound{};
   }
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
template < typename NodeData >
void VanillaCFR< config, Env, Policy, AveragePolicy >::_arm_pruning_windows(
   const info_state_type& infostate,
   NodeData& node_data
)
{
   auto& rbp = node_data.rbp;
   const double bound_range = _payoff_bound(infostate.player()).range();
   // conservative per-iteration regret growth bound r^t(I,a) <= increment_bound; exponential
   // weighting inflates by e^range because its L1 factors rescale increments (pruning.hpp)
   const double inc_bound =
      pruning::window_increment_bound(config.weighting_mode, bound_range);
   const size_t t0 = _iteration() + 1;
   for(auto idx : std::views::iota(size_t{0}, node_data.regret.size())) {
      if(rbp.pruned_until[idx] != 0) {
         continue;
      }
      const double R = node_data.regret[idx];
      // Appendix-B minimum-skip filter (NIPS'15): only open a window when even the WORST-CASE
      // window length floor(|R| / inc-bound) clears the configured minimum
      if(R >= -config.rbp_min_skip_iterations * inc_bound) {
         continue;
      }
      const size_t m = pruning::theorem1_window(R, inc_bound);
      rbp.pruned_until[idx] = t0 + m;
      rbp.last_pruned_iteration[idx] = t0;
      rbp.pessimistic_regret[idx] = R;
      rbp.br_regret_buffer[idx] = 0.;
      rbp.cached_br_value[idx] = 0.;
      rbp.visits_since_refresh[idx] = 0;
      ++m_pruning_stats.windows_armed;
   }
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
template < typename NodeData >
bool VanillaCFR< config, Env, Policy, AveragePolicy >::_rbp_gate(
   std::optional< Player > player_to_update,
   Player active_player,
   NodeData& node_data,
   size_t action_idx,
   const action_type& /*action*/,
   InfostateSptrMap& infostates,
   ObservationbufferMap& observation_buffer,
   const world_state_type& state,
   size_t depth
)
{
   auto& rbp = node_data.rbp;
   if(rbp.pruned_until[action_idx] == 0) {
      return false;
   }
   if(not player_to_update.has_value() or active_player != player_to_update.value()) {
      // windows only gate the OWNING player's own update traversals: opponents passing through
      // (I,a) still need the subtree for their average-strategy updates (their counterfactual
      // regret updates below (I,a) are zero-reach anyway)
      return false;
   }
   const size_t current_iteration = _iteration();
   if(current_iteration < rbp.pruned_until[action_idx]) {
      auto& since_refresh = rbp.visits_since_refresh[action_idx];
      if(since_refresh % config.rbp_br_refresh_period == 0) {
         // periodic best-response traversal of D(h,a) against the opponents' AVERAGE strategies
         rbp.cached_br_value[action_idx] = std::invoke([&] {
            player_hashmap< sptr< info_state_type > > ist_copy = infostates.get();
            player_hashmap< std::vector< std::pair< observation_type, observation_type > > >
               obs_copy = observation_buffer.get();
            world_state_type& child =
               _br_advance(action_of_index(node_data, action_idx), state, depth, ist_copy, obs_copy);
            return _br_expectimax(
               active_player,
               child,
               depth + 1,
               std::move(ist_copy),
               std::move(obs_copy)
            );
         });
         ++m_pruning_stats.br_refreshes;
      }
      ++since_refresh;
      ++m_pruning_stats.skipped_edge_visits;
      return true;
   }
   // deadline expired: fold the buffered best-response regret and resume normal traversal
   _rbp_fold(node_data, action_idx, action_of_index(node_data, action_idx));
   return false;
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
template < typename NodeData >
void VanillaCFR< config, Env, Policy, AveragePolicy >::_push_window_visit(
   NodeData& node_data,
   Player active_player,
   size_t action_idx,
   double cf_reach_prob,
   double state_value_for_player
)
{
   auto& rbp = node_data.rbp;
   // eq-(9) pessimistic tracker: assume the pruned action delivered its maximal payoff U(I,a);
   // this upper-bounds the true regret evolution because v(I->a) <= U(I,a) pointwise
   const double u_upper = _payoff_bound(active_player).upper;
   rbp.pessimistic_regret[action_idx] += cf_reach_prob * (u_upper - state_value_for_player);
   // best-response buffer: the missing true increments pi_{-i}(v(I->a) - v(I)) are replaced by
   // pi_{-i}(v_BR - v(I)); folded into the regret table at unfold time
   rbp.br_regret_buffer[action_idx] +=
      cf_reach_prob * (rbp.cached_br_value[action_idx] - state_value_for_player);
   if(pruning::pessimistic_unfold_required(rbp.pessimistic_regret[action_idx])) {
      _rbp_fold(node_data, action_idx, node_data.registry.actions[action_idx]);
   }
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
template < typename NodeData >
void VanillaCFR< config, Env, Policy, AveragePolicy >::_rbp_fold(
   NodeData& node_data,
   size_t action_idx,
   const action_type& action
)
{
   auto& rbp = node_data.rbp;
   if(rbp.br_regret_buffer[action_idx] != 0.) {
      // "update the regrets to match this": announce the window's best response as having been
      // played on every skipped iteration; observe routes into the minimizer's own update rule
      // (RM+RBP's replace-if-positive fold happens at its next recommend)
      m_regret_minimizer.observe(node_data, action, rbp.br_regret_buffer[action_idx]);
   }
   rbp.pruned_until[action_idx] = 0;
   rbp.br_regret_buffer[action_idx] = 0.;
   rbp.pessimistic_regret[action_idx] = 0.;
   rbp.cached_br_value[action_idx] = 0.;
   rbp.visits_since_refresh[action_idx] = 0;
   ++m_pruning_stats.window_folds;
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
std::vector< double >
VanillaCFR< config, Env, Policy, AveragePolicy >::_normalized_average_policy(
   const info_state_type& infostate,
   const std::vector< action_type >& actions
)
{
   std::vector< double > probs;
   probs.reserve(actions.size());
   double sum = 0.;
   if constexpr(config.weighting_mode == CFRWeightingMode::exponential) {
      // exponential weighting keeps an unnormalized numerator plus a separate denominator
      // table; normalize against the denominators exactly like average_policy() does
      const auto& node_data = _infonode(infostate).data();
      auto& raw = this->template fetch_policy< PolicyLabel::average >(infostate, actions);
      for(const auto& action : actions) {
         // NOTE: 'actions' here comes from the environment and need not share the infostate
         // registry's registration order -- resolve each action's table slot explicitly
         const double denominator =
            std::max(node_data.avg_policy_denominator[node_data.index_of(action)], 1e-20);
         const double p = raw[action] / denominator;
         probs.push_back(p);
         sum += p;
      }
   } else {
      auto& raw = this->template fetch_policy< PolicyLabel::average >(infostate, actions);
      for(const auto& action : actions) {
         sum += raw[action];
      }
      for(const auto& action : actions) {
         probs.push_back(raw[action]);
      }
   }
   if(sum < 1e-20) {
      // never-visited infostate: fall back to uniform
      std::ranges::fill(probs, 1. / static_cast< double >(actions.size()));
      return probs;
   }
   for(double& p : probs) {
      p /= sum;
   }
   return probs;
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
template < typename ActionOrOutcome >
auto VanillaCFR< config, Env, Policy, AveragePolicy >::_br_advance(
   const ActionOrOutcome& action_or_outcome,
   const world_state_type& state,
   size_t depth,
   player_hashmap< sptr< info_state_type > >& infostates,
   player_hashmap< std::vector< std::pair< observation_type, observation_type > > >&
      observation_buffers
) -> world_state_type&
{
   // NOTE on arena safety: this walk only ever touches arena slots strictly DEEPER than the
   // gating traversal frame's depth, and that frame holds no live references to those slots
   world_state_type& next_wstate = _arena_state(depth + 1, state);
   _env().transition(next_wstate, action_or_outcome);

   const auto public_obs = _env().public_observation(state, action_or_outcome, next_wstate);
   const Player next_active_player = _env().active_player(next_wstate);
   if(next_active_player == Player::chance) {
      return next_wstate;
   }
   auto infostate_entry_it = infostates.find(next_active_player);
   if(infostate_entry_it == infostates.end()) {
      infostate_entry_it =
         infostates.emplace(next_active_player, std::make_shared< info_state_type >()).first;
   }
   auto child_infostate = std::make_shared< info_state_type >(*infostate_entry_it->second);
   auto& obs_history = observation_buffers[next_active_player];
   for(auto& obs : obs_history) {
      ::nor::detail::update_infostate(child_infostate, std::move(obs.first), std::move(obs.second));
   }
   obs_history.clear();
   ::nor::detail::update_infostate(
      child_infostate,
      public_obs,
      _env().private_observation(next_active_player, state, action_or_outcome, next_wstate)
   );
   infostate_entry_it->second = std::move(child_infostate);
   for(auto player : _env().players(next_wstate)) {
      if(player == Player::chance or player == next_active_player) {
         continue;
      }
      observation_buffers[player].emplace_back(
         public_obs, _env().private_observation(player, state, action_or_outcome, next_wstate)
      );
   }
   return next_wstate;
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
double VanillaCFR< config, Env, Policy, AveragePolicy >::_br_expectimax(
   Player br_player,
   world_state_type& state,
   size_t depth,
   player_hashmap< sptr< info_state_type > > infostates,
   player_hashmap< std::vector< std::pair< observation_type, observation_type > > >
      observation_buffers
)
{
   if(_env().is_terminal(state)) {
      // pass the ROOT participant set explicitly (see _traverse's terminal note);
      // collect_rewards returns a plain player->value map
      return collect_rewards(_env(), state, _env().players(root_state())).at(br_player);
   }
   const Player active_player = _env().active_player(state);

   if constexpr(concepts::stochastic_env< env_type >) {
      if(active_player == Player::chance) {
         double value = 0.;
         for(auto&& outcome : _env().chance_actions(state)) {
            const double prob = _env().chance_probability(state, outcome);
            if(prob <= 0.) {
               continue;
            }
            auto ist_child = infostates;
            auto obs_child = observation_buffers;
            world_state_type& next =
               _br_advance(outcome, state, depth, ist_child, obs_child);
            value += prob
                     * _br_expectimax(
                        br_player, next, depth + 1, std::move(ist_child), std::move(obs_child)
                     );
         }
         return value;
      }
   }

   const auto& actions = _env().actions(active_player, state);
   if(active_player == br_player) {
      // best response: greedy max over the subtree values
      double best = std::numeric_limits< double >::lowest();
      for(const auto& action : actions) {
         auto ist_child = infostates;
         auto obs_child = observation_buffers;
         world_state_type& next = _br_advance(action, state, depth, ist_child, obs_child);
         best = std::max(
            best,
            _br_expectimax(br_player, next, depth + 1, std::move(ist_child), std::move(obs_child))
         );
      }
      return best;
   }

   // opponent: expectation under their CURRENT AVERAGE strategy (fetch_policy<average>)
   const auto& active_infostate = *infostates.at(active_player);
   const std::vector< double > avg_probs = _normalized_average_policy(active_infostate, actions);
   double value = 0.;
   for(auto [idx, action] : std::views::enumerate(actions)) {
      if(avg_probs[idx] <= 0.) {
         continue;
      }
      auto ist_child = infostates;
      auto obs_child = observation_buffers;
      world_state_type& next = _br_advance(action, state, depth, ist_child, obs_child);
      value += avg_probs[idx]
               * _br_expectimax(
                  br_player, next, depth + 1, std::move(ist_child), std::move(obs_child)
               );
   }
   return value;
}

}  // namespace nor::rm
