
#ifndef NOR_KUHN_ENV_HPP
#define NOR_KUHN_ENV_HPP

#include <range/v3/all.hpp>
#include <string>
#include <vector>

#include "common/common.hpp"
#include "kuhn_poker/kuhn_poker.hpp"
#include "nor/fosg_states.hpp"
#include "nor/fosg_traits.hpp"
#include "nor/game_defs.hpp"

namespace nor::games::kuhn {

using namespace ::kuhn;

inline auto to_kuhn_player(const nor::Player& player)
{
   return static_cast< kuhn::Player >(player);
}
inline auto to_nor_player(const kuhn::Player& player)
{
   return static_cast< nor::Player >(player);
}

using Observation = std::string;

std::string observation(
   const State& state,
   std::optional< Player > observing_player = std::nullopt);

class PublicState: public DefaultPublicstate< PublicState, Observation > {
   using base = DefaultPublicstate< PublicState, Observation >;
   using base::base;
};
class InfoState: public nor::DefaultInfostate< InfoState, Observation > {
   using base = DefaultInfostate< InfoState, Observation >;
   using base::base;
};

class Environment {
  public:
   // nor fosg typedefs
   using world_state_type = State;
   using info_state_type = InfoState;
   using public_state_type = PublicState;
   using action_type = Action;
   using chance_outcome_type = Card;
   using observation_type = Observation;
   // nor fosg traits
   static constexpr size_t max_player_count() { return 2; }
   static constexpr size_t player_count() { return 2; }
   static constexpr TurnDynamic turn_dynamic() { return TurnDynamic::sequential; }
   static constexpr Stochasticity stochasticity() { return Stochasticity::choice; }

   Environment() = default;

   std::vector< action_type > actions(Player, const world_state_type& wstate) const
   {
      return wstate.actions();
   }
   std::vector< chance_outcome_type > chance_actions(const world_state_type& wstate) const
   {
      return wstate.chance_actions();
   }

   double chance_probability(
      const std::pair< world_state_type, chance_outcome_type >& wstate_action_pair) const
   {
      return wstate_action_pair.first.chance_probability(wstate_action_pair.second);
   }
   //   std::vector< action_type > actions(const info_state_type& istate) const;
   static inline std::vector< Player > players()
   {
      return {Player::chance, Player::alex, Player::bob};
   }
   Player active_player(const world_state_type& wstate) const;
   static bool is_terminal(world_state_type& wstate);
   static double reward(Player player, world_state_type& wstate);
   void transition(world_state_type& worldstate, const action_type& action) const;
   void transition(world_state_type& worldstate, const chance_outcome_type& action) const;
   observation_type private_observation(Player player, const world_state_type& wstate) const;
   observation_type private_observation(Player player, const action_type& action) const;
   observation_type private_observation(Player player, const chance_outcome_type& action) const;
   observation_type public_observation(const world_state_type& wstate) const;
   observation_type public_observation(const action_type& action) const;
   observation_type public_observation(const chance_outcome_type& action) const;
};

}  // namespace nor::games::kuhn

namespace nor {

template <>
struct fosg_traits< games::kuhn::InfoState > {
   using observation_type = nor::games::kuhn::Observation;
};

template <>
struct fosg_traits< games::kuhn::Environment > {
   using world_state_type = nor::games::kuhn::State;
   using info_state_type = nor::games::kuhn::InfoState;
   using public_state_type = nor::games::kuhn::PublicState;
   using action_type = nor::games::kuhn::Action;
   using chance_outcome_type = nor::games::kuhn::Card;
   using observation_type = nor::games::kuhn::Observation;

   static constexpr size_t max_player_count = 2;
   static constexpr TurnDynamic turn_dynamic = TurnDynamic::sequential;
   static constexpr Stochasticity stochasticity = Stochasticity::choice;
};

}  // namespace nor

namespace std {
template < typename StateType >
requires common::is_any_v< StateType, nor::games::kuhn::PublicState, nor::games::kuhn::InfoState >
struct hash< StateType > {
   size_t operator()(const StateType& state) const noexcept { return state.hash(); }
};

}  // namespace std

#endif  // NOR_KUHN_ENV_HPP
