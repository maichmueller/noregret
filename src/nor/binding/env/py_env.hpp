
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

#ifndef NOR_SINGLE_ARG
   #define NOR_SINGLE_ARG(...) __VA_ARGS__
#endif

#ifndef NOR_VirtualBaseMethod
   #define NOR_VirtualBaseMethod(func, return_arg, ...) \
      virtual return_arg func(__VA_ARGS__)              \
      {                                                 \
         throw NotImplementedError("'" #func "'");      \
      }
#endif

#ifndef NOR_VirtualBaseMethodConst
   #define NOR_VirtualBaseMethodConst(func, return_arg, ...) \
      virtual return_arg func(__VA_ARGS__) const             \
      {                                                      \
         throw NotImplementedError(#func);                   \
      }
#endif

namespace pynor {

namespace py = ::pybind11;

struct NotImplementedError: public std::exception {
   NotImplementedError(std::string_view sv) : m_msg(_make_msg(sv)) {}
   const char* what() const noexcept override { return m_msg.c_str(); }

  private:
   std::string m_msg;

   std::string _make_msg(std::string_view sv)
   {
      std::stringstream ss;
      ss << "'" << sv << "'"
         << " is not implemented.";
      return ss.str();
   }
};

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
   virtual size_t max_player_count() { return std::dynamic_extent; }
   virtual size_t player_count() { return std::dynamic_extent; }
   virtual nor::Stochasticity stochasticity() = 0;
   virtual bool serialized() = 0;
   virtual bool unrolled() = 0;

   Environment() = default;

   virtual ~Environment() = default;

   NOR_VirtualBaseMethodConst(
      actions,
      NOR_SINGLE_ARG(std::vector< action_type >),
      nor::Player player,
      const world_state_type& wstate
   );

   NOR_VirtualBaseMethodConst(
      private_history,
      NOR_SINGLE_ARG(std::vector< nor::PlayerInformedType<
                        std::optional< std::variant< std::monostate, action_type > > > >),
      const world_state_type& wstate
   );

   NOR_VirtualBaseMethodConst(
      public_history,
      NOR_SINGLE_ARG(std::vector<
                     nor::PlayerInformedType< std::variant< std::monostate, action_type > > >),
      const world_state_type& wstate
   );

   NOR_VirtualBaseMethodConst(
      omniscient_history,
      NOR_SINGLE_ARG(std::vector<
                     nor::PlayerInformedType< std::variant< std::monostate, action_type > > >),
      const world_state_type& wstate
   );

   NOR_VirtualBaseMethodConst(
      players,
      NOR_SINGLE_ARG(std::vector< nor::Player >),
      const world_state_type& wstate
   );

   NOR_VirtualBaseMethodConst(active_player, nor::Player, const world_state_type& wstate);

   NOR_VirtualBaseMethod(reset, void, const world_state_type& wstate);
   NOR_VirtualBaseMethod(is_terminal, bool, world_state_type& wstate);
   NOR_VirtualBaseMethod(is_partaking, bool, const world_state_type& wstate, nor::Player player);
   NOR_VirtualBaseMethod(reward, double, nor::Player player, world_state_type& wstate);
   NOR_VirtualBaseMethod(transition, void, world_state_type& wstate, const action_type& action);
   NOR_VirtualBaseMethod(
      private_observation,
      observation_type,
      nor::Player player,
      const world_state_type& wstate
   );
   NOR_VirtualBaseMethod(
      private_observation,
      observation_type,
      nor::Player player,
      const action_type& action
   );
   NOR_VirtualBaseMethod(public_observation, observation_type, const world_state_type& wstate);
   NOR_VirtualBaseMethod(public_observation, observation_type, const action_type& action);
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
