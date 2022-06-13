
#ifndef NOR_STRATEGO_ENV_HPP
#define NOR_STRATEGO_ENV_HPP

#include <range/v3/all.hpp>
#include <string>
#include <vector>

#include "nor/fosg_states.hpp"
#include "nor/fosg_traits.hpp"
#include "nor/game_defs.hpp"
#include "stratego/Game.hpp"

namespace nor::games::stratego {

using namespace ::stratego;

inline auto to_team(const Player& player)
{
   return Team(static_cast< size_t >(player));
}
inline auto to_player(const Team& team)
{
   return Player(static_cast< size_t >(team));
}

using Observation = std::string;

std::string observation(
   const State& state,
   std::optional< Player > observing_player = std::nullopt);


class Publicstate: public DefaultPublicstate< Publicstate, Observation > {
   using base = DefaultPublicstate< Publicstate, Observation >;
   using base::base;
};
class Infostate: public nor::DefaultInfostate< Infostate, Observation > {
   using base = DefaultInfostate< Infostate, Observation >;
   using base::base;
};

class Environment {
  public:
   // nor fosg typedefs
   using world_state_type = State;
   using info_state_type = Infostate;
   using public_state_type = Publicstate;
   using action_type = Action;
   using observation_type = Observation;
   // nor fosg traits
   static constexpr size_t max_player_count() { return 2; }
   static constexpr size_t player_count() { return 2; }
   static constexpr TurnDynamic turn_dynamic() { return TurnDynamic::sequential; }
   static constexpr Stochasticity stochasticity() { return Stochasticity::deterministic; }

   explicit Environment(uptr< Logic >&& logic);

   std::vector< action_type > actions(Player player, const world_state_type& wstate) const;
   //   std::vector< action_type > actions(const info_state_type& istate) const;
   static inline std::vector< Player > players() { return {Player::alex, Player::bob}; }
   Player active_player(const world_state_type& wstate) const;
   void reset(world_state_type& wstate) const;
   static bool is_terminal(world_state_type& wstate);
   static double reward(Player player, world_state_type& wstate);
   void transition(world_state_type& worldstate, const action_type& action) const;
   observation_type private_observation(Player player, const world_state_type& wstate) const;
   observation_type private_observation(Player player, const action_type& action) const;
   observation_type public_observation(const world_state_type& wstate) const;
   observation_type public_observation(const action_type& action) const;

  private:
   uptr< Logic > m_logic;

   static double _status_to_reward(Status status, Player player);
};

}  // namespace nor::games::stratego

namespace nor {

template <>
struct fosg_traits< games::stratego::Infostate > {
   using observation_type = nor::games::stratego::Observation;
};

template <>
struct fosg_traits< games::stratego::Environment > {
   using world_state_type = nor::games::stratego::State;
   using info_state_type = nor::games::stratego::Infostate;
   using public_state_type = nor::games::stratego::Publicstate;
   using action_type = nor::games::stratego::Action;
   using observation_type = nor::games::stratego::Observation;

   static constexpr size_t max_player_count = 2;
   static constexpr TurnDynamic turn_dynamic = TurnDynamic::sequential;
   static constexpr Stochasticity stochasticity = Stochasticity::deterministic;
};

}  // namespace nor

namespace std {

template < typename StateType >
requires common::is_any_v<
   StateType,
   nor::games::stratego::Publicstate,
   nor::games::stratego::Infostate > struct hash< StateType > {
   size_t operator()(const StateType& state) const noexcept { return state.hash(); }
};

}  // namespace std

#endif  // NOR_STRATEGO_ENV_HPP
