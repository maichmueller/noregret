
#ifndef NOR_ENV_RPS_HPP
#define NOR_ENV_RPS_HPP

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

std::string
observation(const State& state, std::optional< Player > observing_player = std::nullopt);

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
   using chance_outcome_type = std::monostate;
   using action_variant_type = action_variant_type_generator_t< action_type, chance_outcome_type >;
   // nor fosg traits
   static constexpr size_t max_player_count() { return 2; }
   static constexpr size_t player_count() { return 2; }
   static constexpr bool serialized() { return true; }
   static constexpr bool unrolled() { return true; }
   static constexpr Stochasticity stochasticity() { return Stochasticity::deterministic; }

  public:
   Environment() = default;

   std::vector< action_type > actions(Player, const world_state_type&) const
   {
      return std::vector{Action::paper, Action::rock, Action::scissors};
   }

   std::vector< PlayerInformedType< std::optional< action_variant_type > > >
   private_history(Player player, const world_state_type& wstate) const;

   std::vector< PlayerInformedType< std::optional< action_variant_type > > > public_history(
      const world_state_type& wstate
   ) const;

   std::vector< PlayerInformedType< action_variant_type > > open_history(
      const world_state_type& wstate
   ) const;

   static inline std::vector< Player > players(const world_state_type&)
   {
      return {Player::alex, Player::bob};
   }
   Player active_player(const world_state_type& wstate) const;
   static bool is_terminal(const world_state_type& wstate);
   static constexpr bool is_partaking(const world_state_type&, Player) { return true; }
   static double reward(Player player, const world_state_type& wstate);
   void transition(world_state_type& worldstate, const action_type& action) const;

   observation_type private_observation(
      Player observer,
      const world_state_type& wstate,
      const action_type& action,
      const world_state_type& next_wstate
   ) const;

   observation_type public_observation(
      const world_state_type& wstate,
      const action_type& action,
      const world_state_type& next_wstate
   ) const;

   observation_type tiny_repr(const world_state_type& wstate) const;
};

}  // namespace nor::games::rps

namespace nor {
template <>
struct fosg_traits< games::rps::Infostate > {
   using observation_type = nor::games::rps::Observation;
};

template <>
struct fosg_traits< games::rps::Environment > {
   using world_state_type = nor::games::rps::State;
   using info_state_type = nor::games::rps::Infostate;
   using public_state_type = nor::games::rps::Publicstate;
   using action_type = nor::games::rps::Action;
   using observation_type = nor::games::rps::Observation;
};

}  // namespace nor

namespace std {
template < typename StateType >
   requires common::is_any_v< StateType, nor::games::rps::Publicstate, nor::games::rps::Infostate >
struct hash< StateType > {
   size_t operator()(const StateType& state) const noexcept { return state.hash(); }
};

}  // namespace std

#endif  // NOR_ENV_RPS_HPP
