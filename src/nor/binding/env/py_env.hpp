
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

class DynamicEnvironment {
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
   NOR_VirtualBaseMethodConst(stochasticity, nor::Stochasticity);
   NOR_VirtualBaseMethodConst(serialized, bool);
   NOR_VirtualBaseMethodConst(unrolled, bool);

   DynamicEnvironment() = default;

   virtual ~DynamicEnvironment() = default;

   NOR_VirtualBaseMethodConst(
      actions,
      NOR_SINGLE_ARG(std::vector< uptr< action_type > >),
      nor::Player player,
      const world_state_type& wstate
   );

   NOR_VirtualBaseMethodConst(
      chance_actions,
      NOR_SINGLE_ARG(std::vector< uptr< chance_outcome_type > >),
      const world_state_type& wstate
   );

   NOR_VirtualBaseMethodConst(
      chance_probability,
      double,
      const world_state_type& wstate,
      const chance_outcome_type& outcome
   );

   NOR_VirtualBaseMethodConst(
      private_history,
      NOR_SINGLE_ARG(std::vector< nor::PlayerInformedType<
                        std::optional< std::variant< chance_outcome_type, action_type > > > >),
      nor::Player player,
      const world_state_type& wstate
   );

   NOR_VirtualBaseMethodConst(
      public_history,
      NOR_SINGLE_ARG(std::vector<
                     nor::PlayerInformedType< std::variant< chance_outcome_type, action_type > > >),
      const world_state_type& wstate
   );

   NOR_VirtualBaseMethodConst(
      open_history,
      NOR_SINGLE_ARG(std::vector<
                     nor::PlayerInformedType< std::variant< chance_outcome_type, action_type > > >),
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

   NOR_VirtualBaseMethod(
      transition,
      void,
      world_state_type& world_state,
      const action_type& action
   );
   NOR_VirtualBaseMethod(
      transition,
      void,
      world_state_type& world_state,
      const chance_outcome_type& action
   );

   NOR_VirtualBaseMethod(
      private_observation,
      observation_type,
      nor::Player player,
      const world_state_type& wstate,
      const action_type& action,
      const world_state_type& next_wstate
   );
   NOR_VirtualBaseMethod(
      public_observation,
      observation_type,
      const world_state_type& wstate,
      const action_type& action,
      const world_state_type& next_wstate
   );

   NOR_VirtualBaseMethod(
      private_observation,
      observation_type,
      nor::Player player,
      const world_state_type& wstate,
      const chance_outcome_type& chance_outcome,
      const world_state_type& next_wstate
   );
   NOR_VirtualBaseMethod(
      public_observation,
      observation_type,
      const world_state_type& wstate,
      const chance_outcome_type& chance_outcome,
      const world_state_type& next_wstate
   );
};

}  // namespace pynor

namespace nor {

template <>
struct fosg_traits< pynor::Infostate > {
   using observation_type = pynor::Observation;
};

template <>
struct fosg_traits< pynor::DynamicEnvironment > {
   using world_state_type = pynor::Worldstate;
   using info_state_type = pynor::Infostate;
   using public_state_type = pynor::Publicstate;
   using action_type = pynor::Action;
   using observation_type = pynor::Observation;
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
