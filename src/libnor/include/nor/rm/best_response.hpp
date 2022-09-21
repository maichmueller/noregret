
#ifndef NOR_BEST_RESPONSE_HPP
#define NOR_BEST_RESPONSE_HPP

#include "nor/concepts.hpp"
#include "rm_utils.hpp"

namespace nor::rm {

namespace detail {
template <
   concepts::fosg Env,
   concepts::world_state Worldstate = typename fosg_auto_traits< Env >::world_state_type,
   concepts::info_state Infostate = typename fosg_auto_traits< Env >::info_state_type,
   concepts::info_state Action = typename fosg_auto_traits< Env >::action_type,
   concepts::observation Observation = typename fosg_auto_traits< Env >::observation_type,
   typename OutStatePolicy,
   typename InStatePolicy >
std::vector< double > _best_response(
   const std::vector< Player >& best_responders,
   Env& env,
   uptr< Worldstate > wstate,
   std::unordered_map< Player, Infostate > infostates,
   std::unordered_map< Player, std::vector< Observation > > observation_buffer,
   std::unordered_map< Player, Probability > opp_reach_probs,
   std::unordered_map< Player, const InStatePolicy* > player_policies,
   std::unordered_map< Player, OutStatePolicy >& br_policies)
{
   if(env.is_terminal(*wstate)) {
      std::vector< double > values{};
      values.reserve(best_responders.size());
      for(auto responder : best_responders) {
         values.emplace_back(responder, env.reward(responder, *wstate));
      }
      return values;
   }
   if constexpr(concepts::stochastic_fosg< Env, Worldstate, Action >) {
      if(env.active_player(*wstate) == Player::chance) {
         // has to be cast specifically to size_t to avoid an implicit conversion to double
         // and thus call the right constructor.
         std::vector< double > values(size_t(best_responders.size()), 0.);
         for(auto outcome : env.chance_outcomes(*wstate)) {
            double outcome_prob = env.chance_probability(*wstate, outcome);
            auto next_state = child_state(env, wstate, outcome);

            auto child_opp_reach_probs = opp_reach_probs;
            for(auto responder : best_responders) {
               child_opp_reach_probs[responder].get() *= outcome_prob;
            }

            auto [child_observation_buffer, child_infostate_map] = fill_infostate_and_obs_buffers(
               env, observation_buffer, infostates, outcome, *next_state);

            auto child_values = _best_response(
               best_responders,
               env,
               std::move(next_state),
               child_infostate_map,
               child_observation_buffer,
               child_opp_reach_probs,
               player_policies,
               br_policies);

            for(size_t i = 0; i < best_responders.size(); i++) {
               values[i] += outcome_prob * child_values[responder];
            }
         }
      }
   }
   if(ranges::any_of(best_responders, [&, active_player = env.active_player(*wstate)](Player p) {
         return p == active_player;
      })) {
   } else {
   }
}

}  // namespace detail

template <
   concepts::fosg Env,
   concepts::world_state Worldstate = typename fosg_auto_traits< Env >::world_state_type,
   concepts::info_state Infostate = typename fosg_auto_traits< Env >::info_state_type,
   concepts::info_state Action = typename fosg_auto_traits< Env >::action_type,
   typename OutStatePolicy = TabularPolicy<
      Infostate,
      Action,
      UniformPolicy< Infostate, Action >,
      std::unordered_map< Infostate, Action > >,
   typename InStatePolicy = TabularPolicy<
      Infostate,
      Action,
      UniformPolicy< Infostate, Action >,
      std::unordered_map< Infostate, Action > > >
   requires concepts::state_policy<
               OutStatePolicy,
               Infostate,
               Action > and std::is_default_constructible_v< OutStatePolicy >
OutStatePolicy best_response(
   std::vector< Player > best_responders,
   Env& env,
   uptr< Worldstate > wstate,
   std::unordered_map< Player, const InStatePolicy* > player_policies,
   std::unordered_map< Player, Infostate > infostates = {})
{
   // default init the best response policies. It is to be filled by the internal function for best
   // response calculation.
   std::unordered_map< Player, OutStatePolicy > br_policies{};
   for(auto player : best_responders) {
      infostates.emplace(player, OutStatePolicy{});
   }
   if(infostates.empty()) {
      // if the infostates are all empty then we will be filling them with empty ones, assuming this
      // is the root of the game.
      for(auto player : env.players(wstate)) {
         infostates.emplace(player, Infostate{player});
      }
   }
   detail::_best_response(env, wstate, std::move(infostates), player_policies, br_policies);
   return br_policies;
}

}  // namespace nor::rm

#endif  // NOR_BEST_RESPONSE_HPP
