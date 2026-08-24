#pragma once

#include <spdlog/spdlog.h>
#include <algorithm>

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
   for(const action_type& action : actions) {
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
   return true;
}

}  // namespace detail

}  // namespace nor::rm
