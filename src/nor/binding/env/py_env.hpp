
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
#include "py_observation.hpp"
#include "py_infostate.hpp"
#include "py_publicstate.hpp"
#include "py_worldstate.hpp"


namespace py = pybind11;

namespace nor {


class Action {
   virtual size_t hash() const = 0;
   virtual bool operator==(const Action&) = 0;
};

class Observation {};
class State {};
class Infostate {};
class Publicstate {};
class ChanceOutcome {};

class PyAction {};
class PyObservation {};
class PyState {};
class PyInfostate {};
class PyPublicstate {};
class PyChanceOutcome {};

class Environment {
  public:
   // nor fosg typedefs
   using world_state_type = PyState;
   using info_state_type = PyInfostate;
   using public_state_type = PyPublicstate;
   using action_type = PyAction;
   using observation_type = PyObservation;
   using chance_outcome_type = PyChanceOutcome;
   // nor fosg traits
   static constexpr size_t max_player_count() { return std::dynamic_extent; }
   static constexpr size_t player_count() { return std::dynamic_extent; }
   static constexpr TurnDynamic turn_dynamic() { return TurnDynamic::sequential; }
   static constexpr Stochasticity stochasticity() { return Stochasticity::choice; }

   Environment() = default;

   virtual ~Environment() = default;

   virtual std::vector< action_type > actions(Player player, const world_state_type& wstate) = 0;

   virtual std::vector<
      PlayerInformedType< std::optional< std::variant< std::monostate, action_type > > > >
   history(Player, const world_state_type& wstate) = 0;

   virtual std::vector< PlayerInformedType< std::variant< std::monostate, action_type > > >
   history_full(const world_state_type& wstate) = 0;

   virtual std::vector< Player > players(const world_state_type&) = 0;

   virtual Player active_player(const world_state_type& wstate) = 0;
   virtual void reset(world_state_type& wstate) = 0;
   virtual bool is_terminal(world_state_type& wstate) = 0;
   virtual bool is_competing(const world_state_type&, Player) = 0;
   virtual double reward(Player player, world_state_type& wstate) = 0;
   virtual void transition(world_state_type& worldstate, const action_type& action) = 0;
   virtual observation_type private_observation(Player player, const world_state_type& wstate) = 0;
   virtual observation_type private_observation(Player player, const action_type& action) = 0;
   virtual observation_type public_observation(const world_state_type& wstate) = 0;
   virtual observation_type public_observation(const action_type& action) = 0;
};

template <>
struct fosg_traits< PyInfostate > {
   using observation_type = PyObservation;
};

template <>
struct fosg_traits< PyEnvironment > {
   using world_state_type = PyState;
   using info_state_type = PyInfostate;
   using public_state_type = PyPublicstate;
   using action_type = PyAction;
   using observation_type = PyObservation;

   static constexpr size_t max_player_count = 2;
   static constexpr TurnDynamic turn_dynamic = TurnDynamic::sequential;
   static constexpr Stochasticity stochasticity = Stochasticity::deterministic;
};

}  // namespace nor

namespace std {

template < typename StateType >
   requires common::is_any_v< StateType, nor::PyPublicstate, nor::PyInfostate >
struct hash< StateType > {
   size_t operator()(const StateType& state) const noexcept { return state.hash(); }
};

}  // namespace std

#endif  // NOR_PY_ENV_HPP
