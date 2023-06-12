
#ifndef NOR_ENV_POLYMORPHIC_HPP
#define NOR_ENV_POLYMORPHIC_HPP

#include <range/v3/all.hpp>
#include <span>
#include <string>
#include <vector>

#include "nor/fosg_states.hpp"
#include "nor/fosg_traits.hpp"
#include "nor/game_defs.hpp"

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

namespace nor::games::polymorph {

struct NotImplementedError: public std::exception {
   NotImplementedError(std::string_view msg) : m_msg(_make_msg(msg)) {}
   [[nodiscard]] const char* what() const noexcept override { return m_msg.c_str(); }

  private:
   std::string m_msg;

   static std::string _make_msg(std::string_view msg)
   {
      std::stringstream ss;
      ss << "'" << msg << "' is not implemented.";
      return ss.str();
   }
};

/// the current concept requirement on the c++ side is for the action to be hashable, so that it can
/// be used in a hash map. This requires us to have a hash function and equality function to check
/// upon collision
struct Action {
   virtual ~Action() = default;

   NOR_VirtualBaseMethodConst(hash, size_t, );
   NOR_VirtualBaseMethodConst(operator==, bool, const Action&);
};

/// the current concept requirement on the c++ side is for the chance outcome to be hashable, so
/// that it can be used in a hash map. This requires us to have a hash function and equality
/// function to check upon collision
struct ChanceOutcome {
   virtual ~ChanceOutcome() = 0;

   NOR_VirtualBaseMethodConst(hash, size_t);
   NOR_VirtualBaseMethodConst(operator==, bool, const ChanceOutcome&);
};

/// the current concept requirement on the c++ side is for an observation to be hashable, so that it
/// can be used in a hash map. This requires us to have a hash function and equality function to
/// check upon collision
struct Observation {
   virtual ~Observation() = default;

   NOR_VirtualBaseMethodConst(hash, size_t);
   NOR_VirtualBaseMethodConst(operator==, bool, const Observation&);
};

/// the current concept requirements on the c++ side for a infostate are:
/// 1. hashable.
/// 2. is sized
/// 3. copy constructible
/// 4. equality comparable
/// 5. append method
/// 6. __getitem__ operator
/// 7. player method
struct Infostate {
   virtual ~Infostate() = default;

   /// non-const methods
   NOR_VirtualBaseMethod(update, Infostate&, const Observation&, const Observation&);
   /// const methods
   NOR_VirtualBaseMethodConst(hash, size_t);
   NOR_VirtualBaseMethodConst(operator==, bool, const Infostate&);
   NOR_VirtualBaseMethodConst(size, size_t);
   NOR_VirtualBaseMethodConst(operator[], const Observation&, size_t);
   NOR_VirtualBaseMethodConst(player, nor::Player);
};

/// the current concept requirements on the c++ side for a publicstate are:
/// 1. hashable.
/// 2. is sized
/// 3. copy constructible
/// 4. equality comparable
/// 5. append method
/// 6. __getitem__ operator
struct Publicstate {
   virtual ~Publicstate() = default;

   /// non-const methods
   NOR_VirtualBaseMethod(update, Publicstate&, const Observation&);
   /// const methods
   NOR_VirtualBaseMethodConst(hash, size_t);
   NOR_VirtualBaseMethodConst(operator==, bool, const Publicstate&);
   NOR_VirtualBaseMethodConst(size, size_t);
   NOR_VirtualBaseMethodConst(operator[], const Observation&, size_t);
};

/// the current concept requirements on the c++ side for a worldstate are:
/// 1. move constructible
struct Worldstate {
   virtual ~Worldstate() = default;
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
   NOR_VirtualBaseMethodConst(stochasticity, nor::Stochasticity);
   NOR_VirtualBaseMethodConst(serialized, bool);
   NOR_VirtualBaseMethodConst(unrolled, bool);

   Environment() = default;

   virtual ~Environment() = default;

   NOR_VirtualBaseMethodConst(
      actions,
      NOR_SINGLE_ARG(std::vector< ActionWrapper< action_type > >),
      nor::Player /*player*/,
      const world_state_type& /*wstate*/
   );

   NOR_VirtualBaseMethodConst(
      chance_actions,
      NOR_SINGLE_ARG(std::vector< ActionWrapper< chance_outcome_type > >),
      const world_state_type& /*wstate*/
   );

   NOR_VirtualBaseMethodConst(
      chance_probability,
      double,
      const world_state_type& /*wstate*/,
      const chance_outcome_type& /*outcome*/
   );

   NOR_VirtualBaseMethodConst(
      private_history,
      NOR_SINGLE_ARG(std::vector< nor::PlayerInformedType< std::optional< std::variant<
                        ChanceOutcomeWrapper< chance_outcome_type >,
                        ActionWrapper< action_type > > > > >),
      nor::Player /*player*/,
      const world_state_type& /*wstate*/
   );

   NOR_VirtualBaseMethodConst(
      public_history,
      NOR_SINGLE_ARG(std::vector< nor::PlayerInformedType< std::variant<
                        ChanceOutcomeWrapper< chance_outcome_type >,
                        ActionWrapper< action_type > > > >),
      const world_state_type& /*wstate*/
   );

   NOR_VirtualBaseMethodConst(
      open_history,
      NOR_SINGLE_ARG(std::vector< nor::PlayerInformedType< std::variant<
                        ChanceOutcomeWrapper< chance_outcome_type >,
                        ActionWrapper< action_type > > > >),
      const world_state_type& /*wstate*/
   );

   NOR_VirtualBaseMethodConst(
      players,
      NOR_SINGLE_ARG(std::vector< nor::Player >),
      const world_state_type& /*wstate*/
   );

   NOR_VirtualBaseMethodConst(active_player, nor::Player, const world_state_type& /*wstate*/);

   NOR_VirtualBaseMethod(reset, void, const world_state_type& /*wstate*/);
   NOR_VirtualBaseMethod(is_terminal, bool, world_state_type& /*wstate*/);
   NOR_VirtualBaseMethod(
      is_partaking,
      bool,
      const world_state_type& /*wstate*/,
      nor::Player /*player*/
   );
   NOR_VirtualBaseMethod(reward, double, nor::Player /*player*/, world_state_type& /*wstate*/);
   NOR_VirtualBaseMethod(
      transition,
      void,
      world_state_type& /*world_state*/,
      const action_type& /*action*/
   );
   NOR_VirtualBaseMethod(
      transition,
      void,
      world_state_type& /*world_state*/,
      const chance_outcome_type& /*action*/
   );

   NOR_VirtualBaseMethod(
      private_observation,
      ObservationWrapper< observation_type >,
      nor::Player /*player*/,
      const world_state_type& /*wstate*/,
      const action_type& /*action*/,
      const world_state_type& /*next_wstate*/
   );
   NOR_VirtualBaseMethod(
      public_observation,
      ObservationWrapper< observation_type >,
      const world_state_type& /*wstate*/,
      const action_type& /*action*/,
      const world_state_type& /*next_wstate*/
   );

   NOR_VirtualBaseMethod(
      private_observation,
      ObservationWrapper< observation_type >,
      nor::Player /*player*/,
      const world_state_type& /*wstate*/,
      const chance_outcome_type& /*chance_outcome*/,
      const world_state_type& /*next_wstate*/
   );
   NOR_VirtualBaseMethod(
      public_observation,
      ObservationWrapper< observation_type >,
      const world_state_type& /*wstate*/,
      const chance_outcome_type& /*chance_outcome*/,
      const world_state_type& /*next_wstate*/
   );
};

}  // namespace nor::games::polymorph

namespace nor {

template <>
struct fosg_traits< games::polymorph::Infostate > {
   using observation_type = nor::games::polymorph::Observation;
};

template <>
struct fosg_traits< nor::games::polymorph::Environment > {
   using world_state_type = nor::games::polymorph::Worldstate;
   using info_state_type = nor::games::polymorph::Infostate;
   using public_state_type = nor::games::polymorph::Publicstate;
   using action_type = nor::games::polymorph::Action;
   using observation_type = nor::games::polymorph::Observation;
};

}  // namespace nor

namespace std {

template < typename StateType >
   requires common::
      is_any_v< StateType, nor::games::polymorph::Publicstate, nor::games::polymorph::Infostate >
   struct hash< StateType > {
   size_t operator()(const StateType& state) const noexcept { return state.hash(); }
};

}  // namespace std

#endif  // NOR_ENV_POLYMORPHIC_HPP
