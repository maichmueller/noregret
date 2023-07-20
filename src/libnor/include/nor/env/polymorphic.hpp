
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
   Action() = default;
   virtual ~Action() = default;
   virtual uptr< Action > clone() const { return std::make_unique< Action >(); }
   NOR_VirtualBaseMethodConst(hash, size_t, );
   NOR_VirtualBaseMethodConst(operator==, bool, const Action&);
};

/// the current concept requirement on the c++ side is for the chance outcome to be hashable, so
/// that it can be used in a hash map. This requires us to have a hash function and equality
/// function to check upon collision
struct ChanceOutcome {
   ChanceOutcome() = default;
   virtual ~ChanceOutcome() = default;
   virtual uptr< ChanceOutcome > clone() const { return std::make_unique< ChanceOutcome >(); }
   NOR_VirtualBaseMethodConst(hash, size_t);
   NOR_VirtualBaseMethodConst(operator==, bool, const ChanceOutcome&);
};

/// the current concept requirement on the c++ side is for an observation to be hashable, so that it
/// can be used in a hash map. This requires us to have a hash function and equality function to
/// check upon collision
struct Observation {
   Observation() = default;
   virtual ~Observation() = default;
   virtual uptr< Observation > clone() const { return std::make_unique< Observation >(); }
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
   using observation_type = Observation;

   Infostate(Player player) : m_player(player) {}
   virtual ~Infostate() = default;

   virtual uptr< Infostate > clone() const { return std::make_unique< Infostate >(player()); }
   /// non-const methods
   NOR_VirtualBaseMethod(
      update, //
      void,  //
      const observation_type&,  //
      const observation_type&
      );
   /// const methods
   Player player() const { return m_player; }
   NOR_VirtualBaseMethodConst(hash, size_t);
   NOR_VirtualBaseMethodConst(operator==, bool, const Infostate&);
   NOR_VirtualBaseMethodConst(size, size_t);
   NOR_VirtualBaseMethodConst(
      operator[],
      NOR_SINGLE_ARG(const std::pair<
                     ObservationHolder< observation_type >,
                     ObservationHolder< observation_type > >&),
      size_t
   );
   NOR_VirtualBaseMethodConst(
      latest,
      NOR_SINGLE_ARG(const std::pair<
                     ObservationHolder< observation_type >,
                     ObservationHolder< observation_type > >&)
   );

  private:
   Player m_player;
};

/// the current concept requirements on the c++ side for a publicstate are:
/// 1. hashable.
/// 2. is sized
/// 3. copy constructible
/// 4. equality comparable
/// 5. append method
/// 6. __getitem__ operator
struct Publicstate {
   using observation_type = Observation;
   Publicstate() = default;
   virtual ~Publicstate() = default;

   virtual uptr< Publicstate > clone() const { return std::make_unique< Publicstate >(); }

   /// non-const methods
   NOR_VirtualBaseMethod(update, void, const observation_type&);
   /// const methods
   NOR_VirtualBaseMethodConst(hash, size_t);
   NOR_VirtualBaseMethodConst(operator==, bool, const Publicstate&);
   NOR_VirtualBaseMethodConst(size, size_t);
   NOR_VirtualBaseMethodConst(operator[], const ObservationHolder< observation_type >&, size_t);
   NOR_VirtualBaseMethodConst(latest, const ObservationHolder< observation_type >&);
};

/// the current concept requirements on the c++ side for a worldstate are:
/// 1. move constructible
struct Worldstate {
   Worldstate() = default;
   virtual ~Worldstate() = default;
   virtual uptr< Worldstate > clone() const { return std::make_unique< Worldstate >(); }
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

  private:
   using action_holder = ActionHolder< action_type >;
   using chance_outcome_holder = ChanceOutcomeHolder< chance_outcome_type >;
   using observation_holder = ObservationHolder< observation_type >;
   using info_state_holder = InfostateHolder< info_state_type >;
   using public_state_holder = PublicstateHolder< public_state_type >;
   using world_state_holder = WorldstateHolder< world_state_type >;

  public:
   Environment() = default;

   virtual ~Environment() = default;

   // nor fosg traits
   virtual size_t max_player_count() { return std::dynamic_extent; }
   virtual size_t player_count() { return std::dynamic_extent; }
   NOR_VirtualBaseMethodConst(stochasticity, nor::Stochasticity);
   NOR_VirtualBaseMethodConst(serialized, bool);
   NOR_VirtualBaseMethodConst(unrolled, bool);

   // API

   NOR_VirtualBaseMethodConst(
      actions,
      NOR_SINGLE_ARG(std::vector< action_holder >),
      nor::Player /*player*/,
      const world_state_type& /*wstate*/
   );

   NOR_VirtualBaseMethodConst(
      chance_actions,
      NOR_SINGLE_ARG(std::vector< chance_outcome_holder >),
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
      NOR_SINGLE_ARG(std::vector< nor::PlayerInformedType<
                        std::optional< std::variant< chance_outcome_holder, action_holder > > > >),
      nor::Player /*player*/,
      const world_state_type& /*wstate*/
   );

   NOR_VirtualBaseMethodConst(
      public_history,
      NOR_SINGLE_ARG(std::vector< nor::PlayerInformedType<
                        std::variant< chance_outcome_holder, action_holder > > >),
      const world_state_type& /*wstate*/
   );

   NOR_VirtualBaseMethodConst(
      open_history,
      NOR_SINGLE_ARG(std::vector< nor::PlayerInformedType<
                        std::variant< chance_outcome_holder, action_holder > > >),
      const world_state_type& /*wstate*/
   );

   NOR_VirtualBaseMethodConst(
      players,
      NOR_SINGLE_ARG(std::vector< nor::Player >),
      const world_state_type& /*wstate*/
   );

   NOR_VirtualBaseMethodConst(active_player, nor::Player, const world_state_type& /*wstate*/);

   NOR_VirtualBaseMethodConst(is_terminal, bool, const world_state_type& /*wstate*/);
   NOR_VirtualBaseMethodConst(
      is_partaking,
      bool,
      const world_state_type& /*wstate*/,
      nor::Player /*player*/
   );
   NOR_VirtualBaseMethodConst(
      reward,
      double,
      nor::Player /*player*/,
      const world_state_type& /*wstate*/
   );
   NOR_VirtualBaseMethodConst(
      rewards,
      double,
      std::vector< nor::Player > /*players*/,
      const world_state_type& /*wstate*/
   );
   NOR_VirtualBaseMethodConst(
      transition,
      void,
      world_state_type& /*world_state*/,
      const action_type& /*action*/
   );
   NOR_VirtualBaseMethodConst(
      transition,
      void,
      world_state_type& /*world_state*/,
      const chance_outcome_type& /*action*/
   );

   NOR_VirtualBaseMethodConst(
      private_observation,
      observation_holder,
      nor::Player /*player*/,
      const world_state_type& /*wstate*/,
      const action_type& /*action*/,
      const world_state_type& /*next_wstate*/
   );
   NOR_VirtualBaseMethodConst(
      public_observation,
      observation_holder,
      const world_state_type& /*wstate*/,
      const action_type& /*action*/,
      const world_state_type& /*next_wstate*/
   );

   NOR_VirtualBaseMethodConst(
      private_observation,
      observation_holder,
      nor::Player /*player*/,
      const world_state_type& /*wstate*/,
      const chance_outcome_type& /*chance_outcome*/,
      const world_state_type& /*next_wstate*/
   );
   NOR_VirtualBaseMethodConst(
      public_observation,
      observation_holder,
      const world_state_type& /*wstate*/,
      const chance_outcome_type& /*chance_outcome*/,
      const world_state_type& /*next_wstate*/
   );
};

namespace detail {

template < typename T, typename Base, bool owning >
struct ActionOutcomeObsView: public Base {
   explicit ActionOutcomeObsView(std::conditional_t< owning, T, T& > arg)
       : obj(std::invoke([&]() -> decltype(auto) {
            if constexpr(owning) {
               return std::move(arg);
            } else {
               return &arg;
            }
         }))
   {
   }

   [[nodiscard]] uptr< Base > clone() const override { return common::deref(obj).clone(); }
   [[nodiscard]] size_t hash() const override { return common::deref(obj).hash(); }
   bool operator==(const Base& act) const override { return common::deref(obj).operator==(act); }

  private:
   std::conditional_t< owning, T, T* > obj;
};

template < typename T, typename Base, bool owning >
struct WorldstateView: public Base {
   explicit WorldstateView(std::conditional_t< owning, T, T& > arg)
       : obj(std::invoke([&]() -> decltype(auto) {
            if constexpr(owning) {
               return std::move(arg);
            } else {
               return &arg;
            }
         }))
   {
   }

   [[nodiscard]] uptr< Base > clone() const override { return common::deref(obj).clone(); }

  private:
   std::conditional_t< owning, T, T* > obj;
};

}  // namespace detail

class TypeErasedAction: public Action {
  public:
   template < typename ActionType >
      requires concepts::action< std::remove_cvref_t< ActionType > >
   explicit TypeErasedAction(ActionType&& obj)
       : action(std::invoke(
          []< typename T >(T&& arg) {
             using T_raw = std::remove_reference_t< T >;
             using view_type = detail::
                ActionOutcomeObsView< T_raw, Action, std::is_lvalue_reference_v< T > >;
             return std::make_shared< view_type >(std::forward< T >(arg));
          },
          std::forward< ActionType >(obj)
       ))
   {
   }

  private:
   sptr< Action > action;
};

class TypeErasedOutcome: public ChanceOutcome {
  public:
   template < typename OutcomeType >
      requires concepts::chance_outcome< std::remove_cvref_t< OutcomeType > >
   explicit TypeErasedOutcome(OutcomeType&& obj)
       : outcome(std::invoke(
          []< typename T >(T&& arg) {
             using T_raw = std::remove_reference_t< T >;
             using view_type = detail::
                ActionOutcomeObsView< T_raw, ChanceOutcome, std::is_lvalue_reference_v< T > >;
             return std::make_shared< view_type >(std::forward< T >(arg));
          },
          std::forward< OutcomeType >(obj)
       ))
   {
   }

  private:
   sptr< ChanceOutcome > outcome;
};

class TypeErasedObservation: public Observation {
  public:
   template < typename ObservationType >
      requires concepts::observation< std::remove_cvref_t< ObservationType > >
   explicit TypeErasedObservation(ObservationType&& obj)
       : observation(std::invoke(
          []< typename T >(T&& arg) {
             using T_raw = std::remove_reference_t< T >;
             using view_type = detail::
                ActionOutcomeObsView< T_raw, Observation, std::is_lvalue_reference_v< T > >;
             return std::make_shared< view_type >(std::forward< T >(arg));
          },
          std::forward< ObservationType >(obj)
       ))
   {
   }

  private:
   sptr< Observation > observation;
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

template < typename Type >
   requires common::is_any_v<
      Type,
      nor::games::polymorph::Publicstate,
      nor::games::polymorph::Infostate,
      nor::games::polymorph::Action,
      nor::games::polymorph::ChanceOutcome,
      nor::games::polymorph::Observation >
struct hash< Type > {
   size_t operator()(const Type& t) const noexcept { return t.hash(); }
};

}  // namespace std

#endif  // NOR_ENV_POLYMORPHIC_HPP
