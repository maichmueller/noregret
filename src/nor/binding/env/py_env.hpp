
#ifndef NOR_PY_ENV_HPP
#define NOR_PY_ENV_HPP

#include <pybind11/pybind11.h>

#include <range/v3/all.hpp>
#include <string>
#include <vector>

#include "nor/fosg_states.hpp"
#include "nor/fosg_traits.hpp"
#include "nor/game_defs.hpp"
#include "py_action.hpp"
#include "py_chance_outcome.hpp"
#include "py_infostate.hpp"
#include "py_observation.hpp"
#include "py_publicstate.hpp"
#include "py_worldstate.hpp"

namespace py = pybind11;

namespace pynor {

class Environment {
  public:
   // nor fosg typedefs
   using world_state_type = Worldstate;
   using info_state_type = Infostate;
   using public_state_type = Publicstate;
   using action_type = Action;
   using observation_type = Observation;
   using chance_outcome_type = ChanceOutcome;
   // nor fosg traits
   static constexpr size_t max_player_count() { return std::dynamic_extent; }
   static constexpr size_t player_count() { return std::dynamic_extent; }
   static constexpr nor::TurnDynamic turn_dynamic() { return nor::TurnDynamic::sequential; }
   static constexpr nor::Stochasticity stochasticity() { return nor::Stochasticity::choice; }

   Environment() = default;

   virtual ~Environment() = default;

   virtual std::vector< action_type > actions(
      nor::Player player,
      const world_state_type& wstate) = 0;

   virtual std::vector<
      nor::PlayerInformedType< std::optional< std::variant< std::monostate, action_type > > > >
   private_history(nor::Player, const world_state_type& wstate) = 0;

   virtual std::vector< nor::PlayerInformedType< std::variant< std::monostate, action_type > > >
   public_history(const world_state_type& wstate) = 0;

   virtual std::vector< nor::PlayerInformedType< std::variant< std::monostate, action_type > > >
   omniscient_history(const world_state_type& wstate) = 0;

   virtual std::vector< nor::Player > players(const world_state_type&) = 0;

   virtual nor::Player active_player(const world_state_type& wstate) = 0;
   virtual void reset(world_state_type& wstate) = 0;
   virtual bool is_terminal(world_state_type& wstate) = 0;
   virtual bool is_partaking(const world_state_type&, nor::Player) = 0;
   virtual double reward(nor::Player player, world_state_type& wstate) = 0;
   virtual void transition(world_state_type& worldstate, const action_type& action) = 0;
   virtual observation_type private_observation(
      nor::Player player,
      const world_state_type& wstate) = 0;
   virtual observation_type private_observation(nor::Player player, const action_type& action) = 0;
   virtual observation_type public_observation(const world_state_type& wstate) = 0;
   virtual observation_type public_observation(const action_type& action) = 0;
};

}  // namespace pynor

namespace nor {

template <>
struct fosg_traits< pynor::Infostate > {
   using observation_type = pynor::Observation;
};

template <>
struct fosg_traits< pynor::Environment > {
   using world_state_type = pynor::Worldstate;
   using info_state_type = pynor::Infostate;
   using public_state_type = pynor::Publicstate;
   using action_type = pynor::Action;
   using observation_type = pynor::Observation;

   static constexpr size_t max_player_count = 2;
   static constexpr TurnDynamic turn_dynamic = TurnDynamic::sequential;
   static constexpr Stochasticity stochasticity = Stochasticity::deterministic;
};

}  // namespace nor

namespace std {

template < typename StateType >
   requires common::is_any_v< StateType, pynor::Publicstate, pynor::Infostate >
struct hash< StateType > {
   size_t operator()(const StateType& state) const noexcept { return state.__hash__(); }
};

}  // namespace std

#endif  // NOR_PY_ENV_HPP
