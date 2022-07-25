
#ifndef NOR_RPS_ENV_HPP
#define NOR_RPS_ENV_HPP

#include "common/common.hpp"
#include "kuhn_poker/kuhn_poker.hpp"
#include "nor/fosg_states.hpp"
#include "nor/fosg_traits.hpp"
#include "nor/game_defs.hpp"
#include "rock_paper_scissors/rock_paper_scissors.hpp"

namespace nor::games::rps {

using namespace ::rps;

inline auto to_player(const rps::Team& player)
{
   return static_cast< nor::Player >(player);
}
inline auto to_team(const nor::Player& player)
{
   return static_cast< rps::Team >(player);
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
   using observation_type = Observation;
   // nor fosg traits
   static constexpr size_t max_player_count() { return 2; }
   static constexpr size_t player_count() { return 2; }
   static constexpr TurnDynamic turn_dynamic() { return TurnDynamic::sequential; }
   static constexpr Stochasticity stochasticity() { return Stochasticity::deterministic; }

   Environment() = default;

   std::vector< action_type > actions(Player player, const world_state_type&) const
   {
      std::vector< action_type > valid_actions;
      valid_actions.reserve(3);
      for(auto hand : {Hand::paper, Hand::rock, Hand::scissors}) {
         valid_actions.emplace_back(Action{to_team(player), hand});
      }
      return valid_actions;
   }

   std::vector<
      PlayerInformedType< std::optional< std::variant< std::monostate, action_type > > > >
   history(Player, const world_state_type& wstate) const;

   std::vector< PlayerInformedType< std::variant< std::monostate, action_type > > >
   history_full(const world_state_type& wstate) const;

   static inline std::vector< Player > players(const world_state_type&)
   {
      return {Player::alex, Player::bob};
   }
   Player active_player(const world_state_type& wstate) const;
   static bool is_terminal(world_state_type& wstate);
   static constexpr bool is_competing(const world_state_type&, Player) { return true; }
   static double reward(Player player, world_state_type& wstate);
   void transition(world_state_type& worldstate, const action_type& action) const;
   observation_type private_observation(Player player, const world_state_type& wstate) const;
   observation_type private_observation(Player player, const action_type& action) const;
   observation_type public_observation(const world_state_type&) const { return {}; }
   observation_type public_observation(const action_type&) const { return {}; }
   observation_type tiny_repr(const world_state_type& wstate) const;
};

}  // namespace nor::games::rps

namespace nor {
template <>
struct fosg_traits< games::rps::InfoState > {
   using observation_type = nor::games::rps::Observation;
};

template <>
struct fosg_traits< games::rps::Environment > {
   using world_state_type = nor::games::rps::State;
   using info_state_type = nor::games::rps::InfoState;
   using public_state_type = nor::games::rps::PublicState;
   using action_type = nor::games::rps::Action;
   using observation_type = nor::games::rps::Observation;

   static constexpr size_t max_player_count = 2;
   static constexpr TurnDynamic turn_dynamic = TurnDynamic::sequential;
   static constexpr Stochasticity stochasticity = Stochasticity::deterministic;
};

}  // namespace nor

namespace std {
template < typename StateType >
   requires common::is_any_v< StateType, nor::games::rps::PublicState, nor::games::rps::InfoState >
struct hash< StateType > {
   size_t operator()(const StateType& state) const noexcept { return state.hash(); }
};

}  // namespace std

#endif  // NOR_RPS_ENV_HPP
