#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <mutex>
#include <new>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

// The binding sees only the value-oriented runtime contract. catalog.hpp, and with it every
// concrete game and solver instantiation, stays inside the compiled partitions.
#include "runtime/runtime.hpp"

namespace nb = nanobind;
namespace rt = nor::binding::runtime;
using namespace nb::literals;

namespace noregret_binding {

///////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// reflected enum plumbing //////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

template < typename Enum >
[[nodiscard]] std::string enum_name(Enum value)
{
   const auto name = nor::meta::enum_name(value);
   return name.empty() ? std::to_string(static_cast< std::underlying_type_t< Enum > >(value))
                       : std::string(name);
}

/// Accept either the bound enumeration object or its reflected name, so Python callers never have
/// to spell out an enumerator table this module did not write by hand either.
template < typename Enum >
[[nodiscard]] Enum enum_from_python(nb::handle value, const char* description)
{
   if(nb::isinstance< nb::str >(value)) {
      const auto name = nb::cast< std::string >(value);
      if(const auto result = nor::meta::enum_from_name< Enum >(name); result)
         return *result;
      throw nb::value_error((std::string("unknown ") + description + " name: " + name).c_str());
   }
   Enum result{};
   if(nb::try_cast< Enum >(value, result))
      return result;
   throw nb::type_error((std::string("expected a ") + description + " or its name").c_str());
}

///////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// errors ///////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

[[nodiscard]] std::string capability_message(const rt::CapabilityError& error)
{
   std::ostringstream out;
   out << error.message << " [code=" << enum_name(error.code);
   if(static_cast< uint16_t >(error.game) != 0)
      out << ", game=" << enum_name(error.game);
   if(static_cast< uint16_t >(error.solver) != 0)
      out << ", solver=" << enum_name(error.solver);
   if(static_cast< uint16_t >(error.profile) != 0)
      out << ", profile=" << enum_name(error.profile);
   out << ']';
   return out.str();
}

/// Carries the full CapabilityError so the translator can attach its context to the Python object
/// instead of flattening everything into a message string.
class BindingCapabilityError final: public std::exception {
  public:
   explicit BindingCapabilityError(rt::CapabilityError error)
       : m_error(std::move(error)), m_message(capability_message(m_error))
   {
   }

   [[nodiscard]] const char* what() const noexcept final { return m_message.c_str(); }
   [[nodiscard]] const rt::CapabilityError& error() const noexcept { return m_error; }

  private:
   rt::CapabilityError m_error;
   std::string m_message;
};

class StaleViewError final: public std::logic_error {
  public:
   explicit StaleViewError(const char* message) : std::logic_error(message) {}
};

class ClosedSessionError final: public std::logic_error {
  public:
   ClosedSessionError() : std::logic_error("the solver session has been closed") {}
};

class ReentrantSessionError final: public std::logic_error {
  public:
   ReentrantSessionError()
       : std::logic_error(
          "a session operation was reentered from inside the same session; provider callbacks "
          "and policy inspection may not call back into their own session"
       )
   {
   }
};

template < typename T >
[[nodiscard]] T unwrap(rt::Result< T > result)
{
   if(not result)
      throw BindingCapabilityError(std::move(result.error()));
   return std::move(*result);
}

inline void unwrap(rt::Result< void > result)
{
   if(not result)
      throw BindingCapabilityError(std::move(result.error()));
}

///////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// domain values ////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

template < rt::ValueKind Kind >
struct dynamic_value_for;

template <>
struct dynamic_value_for< rt::ValueKind::action > {
   using type = rt::DynamicAction;
};
template <>
struct dynamic_value_for< rt::ValueKind::chance_outcome > {
   using type = rt::DynamicChanceOutcome;
};
template <>
struct dynamic_value_for< rt::ValueKind::observation > {
   using type = rt::DynamicObservation;
};
template <>
struct dynamic_value_for< rt::ValueKind::info_state > {
   using type = rt::DynamicInfoState;
};
template <>
struct dynamic_value_for< rt::ValueKind::public_state > {
   using type = rt::DynamicPublicState;
};
template <>
struct dynamic_value_for< rt::ValueKind::world_state > {
   using type = rt::DynamicWorldState;
};

template < rt::ValueKind Kind >
using dynamic_value_t = typename dynamic_value_for< Kind >::type;

template < rt::ValueKind Kind >
inline constexpr bool is_named_kind = Kind == rt::ValueKind::action
                                      or Kind == rt::ValueKind::chance_outcome
                                      or Kind == rt::ValueKind::observation;

/**
 * @brief The one Python-facing type per domain kind.
 *
 * A value is either a copy of a concrete C++ value that a compiled game produced (erased once, at
 * the runtime boundary) or a value a Python provider authored. Both are exposed identically:
 * to_string() and to_tensor() answer None when the underlying value has no such representation,
 * and no representation is ever invented for one that does not.
 */
template < rt::ValueKind Kind >
class PythonValue {
  public:
   using dynamic_type = dynamic_value_t< Kind >;

   PythonValue()
   {
      if constexpr(Kind == rt::ValueKind::info_state or Kind == rt::ValueKind::public_state or Kind == rt::ValueKind::world_state) {
         m_value.template emplace< dynamic_type >();
      } else {
         m_value.template emplace< rt::ErasedValue >();
      }
   }

   [[nodiscard]] static PythonValue named(
      std::string type_name,
      std::string identity,
      std::optional< std::string > text = std::nullopt,
      std::optional< rt::TensorData > tensor = std::nullopt
   )
   {
      PythonValue result;
      if constexpr(is_named_kind< Kind >) {
         result.m_value = dynamic_type::named(
            std::move(type_name), std::move(identity), std::move(text), std::move(tensor)
         );
      } else if constexpr(Kind == rt::ValueKind::world_state) {
         result.m_value = dynamic_type{rt::DynamicWorldValue::named(
            std::move(type_name), std::move(identity), std::move(text), std::move(tensor)
         )};
      } else {
         throw nb::type_error("state histories are built with their own API, not from a name");
      }
      return result;
   }

   [[nodiscard]] static PythonValue with_player(nor::Player player)
   {
      PythonValue result;
      if constexpr(Kind == rt::ValueKind::info_state) {
         result.m_value = dynamic_type{player};
      } else {
         (void) player;
         throw nb::type_error("only InfoState is constructed from a player");
      }
      return result;
   }

   /**
    * @brief Adopt a value that crossed the erased runtime boundary.
    *
    * A dynamic game's solver stores the very values its provider authored, so when one comes back
    * it is unwrapped rather than left erased. Otherwise the same action would look different
    * depending on which side of the boundary handed it to Python, and its identity -- the only
    * thing a provider can compare on -- would read as absent.
    */
   [[nodiscard]] static PythonValue from_static(rt::ErasedValue value)
   {
      if(value.valid() and value.kind() != Kind)
         throw nb::type_error("erased value has the wrong domain kind");
      PythonValue result;
      if(const auto* authored = value.template get_if< dynamic_type >()) {
         result.m_value = *authored;
      } else {
         result.m_value = std::move(value);
      }
      return result;
   }

   [[nodiscard]] static PythonValue from_dynamic(dynamic_type value)
   {
      PythonValue result;
      result.m_value = std::move(value);
      return result;
   }

   [[nodiscard]] static constexpr rt::ValueKind kind() noexcept { return Kind; }

   [[nodiscard]] bool is_dynamic() const noexcept
   {
      return std::holds_alternative< dynamic_type >(m_value);
   }

   [[nodiscard]] bool valid() const noexcept
   {
      if(const auto* value = std::get_if< rt::ErasedValue >(&m_value))
         return value->valid();
      if constexpr(Kind == rt::ValueKind::info_state or Kind == rt::ValueKind::public_state) {
         return true;
      } else if constexpr(Kind == rt::ValueKind::world_state) {
         return std::get< dynamic_type >(m_value).value().valid();
      } else {
         return std::get< dynamic_type >(m_value).valid();
      }
   }

   [[nodiscard]] std::string type_name() const
   {
      if(const auto* value = std::get_if< rt::ErasedValue >(&m_value))
         return std::string(value->type_name());
      if constexpr(Kind == rt::ValueKind::world_state) {
         return std::string(std::get< dynamic_type >(m_value).value().type_name());
      } else if constexpr(is_named_kind< Kind >) {
         return std::string(std::get< dynamic_type >(m_value).type_name());
      } else {
         return {};
      }
   }

   [[nodiscard]] std::optional< std::string > identity() const
   {
      if(std::holds_alternative< rt::ErasedValue >(m_value))
         return std::nullopt;
      if constexpr(Kind == rt::ValueKind::world_state) {
         return std::string(std::get< dynamic_type >(m_value).value().identity());
      } else if constexpr(is_named_kind< Kind >) {
         return std::string(std::get< dynamic_type >(m_value).identity());
      } else {
         return std::nullopt;
      }
   }

   [[nodiscard]] std::optional< std::string > to_string() const
   {
      if(const auto* value = std::get_if< rt::ErasedValue >(&m_value))
         return value->to_string();
      if constexpr(Kind == rt::ValueKind::world_state) {
         return std::get< dynamic_type >(m_value).value().to_string();
      } else if constexpr(is_named_kind< Kind >) {
         return std::get< dynamic_type >(m_value).to_string();
      } else {
         return std::nullopt;
      }
   }

   [[nodiscard]] std::optional< rt::TensorData > to_tensor() const
   {
      if(const auto* value = std::get_if< rt::ErasedValue >(&m_value))
         return value->to_tensor();
      if constexpr(Kind == rt::ValueKind::world_state) {
         return std::get< dynamic_type >(m_value).value().to_tensor();
      } else if constexpr(is_named_kind< Kind >) {
         return std::get< dynamic_type >(m_value).to_tensor();
      } else {
         return std::nullopt;
      }
   }

   [[nodiscard]] rt::ValueCapabilities capabilities() const
   {
      if(const auto* value = std::get_if< rt::ErasedValue >(&m_value)) {
         if(const auto* metadata = value->metadata())
            return metadata->capabilities;
         return {};
      }
      return rt::ValueCapabilities{
         .to_string = to_string().has_value(), .to_tensor = to_tensor().has_value()};
   }

   [[nodiscard]] size_t hash() const
   {
      if(const auto* value = std::get_if< rt::ErasedValue >(&m_value))
         return value->hash();
      return std::get< dynamic_type >(m_value).hash();
   }

   friend bool operator==(const PythonValue& left, const PythonValue& right)
   {
      if(left.m_value.index() != right.m_value.index())
         return false;
      if(const auto* left_static = std::get_if< rt::ErasedValue >(&left.m_value))
         return *left_static == std::get< rt::ErasedValue >(right.m_value);
      return std::get< dynamic_type >(left.m_value) == std::get< dynamic_type >(right.m_value);
   }

   /// The value in the erased runtime representation, re-erasing a provider-authored value. A
   /// lookup key therefore works the same whether it came from the solver or from Python.
   [[nodiscard]] rt::ErasedValue as_erased() const
   {
      if(const auto* value = std::get_if< rt::ErasedValue >(&m_value))
         return *value;
      return rt::ErasedValue::make< Kind >(std::get< dynamic_type >(m_value));
   }

   [[nodiscard]] const dynamic_type& dynamic() const
   {
      const auto* value = std::get_if< dynamic_type >(&m_value);
      if(value == nullptr)
         throw nb::type_error("a compiled game value cannot be used as a provider value");
      return *value;
   }

   [[nodiscard]] dynamic_type& mutable_dynamic()
   {
      auto* value = std::get_if< dynamic_type >(&m_value);
      if(value == nullptr)
         throw nb::type_error("a compiled game value cannot be used as a provider value");
      return *value;
   }

   // -- history-shaped kinds -----------------------------------------------------------------

   void update_info(
      const PythonValue< rt::ValueKind::observation >& public_observation,
      const PythonValue< rt::ValueKind::observation >& private_observation
   )
   {
      if constexpr(Kind == rt::ValueKind::info_state) {
         mutable_dynamic().update(public_observation.dynamic(), private_observation.dynamic());
      } else {
         (void) public_observation;
         (void) private_observation;
         throw nb::type_error("update(public, private) is only defined for InfoState");
      }
   }

   void update_public(const PythonValue< rt::ValueKind::observation >& observation)
   {
      if constexpr(Kind == rt::ValueKind::public_state) {
         mutable_dynamic().update(observation.dynamic());
      } else {
         (void) observation;
         throw nb::type_error("update(observation) is only defined for PublicState");
      }
   }

   void set_world(const PythonValue< rt::ValueKind::world_state >& value)
   {
      if constexpr(Kind == rt::ValueKind::world_state) {
         m_value = value.dynamic();
      } else {
         (void) value;
         throw nb::type_error("set(WorldState) is only defined for WorldState");
      }
   }

   [[nodiscard]] nor::Player player() const
   {
      if constexpr(Kind == rt::ValueKind::info_state) {
         return dynamic().player();
      } else {
         throw nb::type_error("player is only defined for InfoState");
      }
   }

   [[nodiscard]] size_t size() const
   {
      if constexpr(Kind == rt::ValueKind::info_state or Kind == rt::ValueKind::public_state) {
         return dynamic().size();
      } else {
         throw nb::type_error("a history length is only defined for InfoState and PublicState");
      }
   }

   [[nodiscard]] PythonValue< rt::ValueKind::observation > observation_at(size_t index) const
   {
      if constexpr(Kind == rt::ValueKind::public_state) {
         return PythonValue< rt::ValueKind::observation >::from_dynamic(dynamic()[index]);
      } else {
         (void) index;
         throw nb::type_error("indexed observations are only defined for PublicState");
      }
   }

   [[nodiscard]] std::
      pair< PythonValue< rt::ValueKind::observation >, PythonValue< rt::ValueKind::observation > >
      info_at(size_t index) const
   {
      if constexpr(Kind == rt::ValueKind::info_state) {
         const auto& [public_observation, private_observation] = dynamic()[index];
         return {
            PythonValue< rt::ValueKind::observation >::from_dynamic(public_observation),
            PythonValue< rt::ValueKind::observation >::from_dynamic(private_observation)};
      } else {
         (void) index;
         throw nb::type_error("indexed observation pairs are only defined for InfoState");
      }
   }

   [[nodiscard]] PythonValue< rt::ValueKind::observation > latest_observation() const
   {
      if constexpr(Kind == rt::ValueKind::public_state) {
         return PythonValue< rt::ValueKind::observation >::from_dynamic(dynamic().latest());
      } else {
         throw nb::type_error("latest() is only defined for PublicState");
      }
   }

   [[nodiscard]] std::
      pair< PythonValue< rt::ValueKind::observation >, PythonValue< rt::ValueKind::observation > >
      latest_info() const
   {
      if constexpr(Kind == rt::ValueKind::info_state) {
         const auto& [public_observation, private_observation] = dynamic().latest();
         return {
            PythonValue< rt::ValueKind::observation >::from_dynamic(public_observation),
            PythonValue< rt::ValueKind::observation >::from_dynamic(private_observation)};
      } else {
         throw nb::type_error("latest() is only defined for InfoState");
      }
   }

  private:
   std::variant< rt::ErasedValue, dynamic_type > m_value;
};

using PythonAction = PythonValue< rt::ValueKind::action >;
using PythonChanceOutcome = PythonValue< rt::ValueKind::chance_outcome >;
using PythonObservation = PythonValue< rt::ValueKind::observation >;
using PythonInfoState = PythonValue< rt::ValueKind::info_state >;
using PythonPublicState = PythonValue< rt::ValueKind::public_state >;
using PythonWorldState = PythonValue< rt::ValueKind::world_state >;

///////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// dynamic provider trampoline //////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

template < typename Value >
[[nodiscard]] std::vector< typename Value::dynamic_type > dynamic_vector(nb::handle value)
{
   std::vector< typename Value::dynamic_type > result;
   for(const nb::handle item : nb::borrow< nb::iterable >(value)) {
      const auto entry = nb::cast< Value >(item);
      if(not entry.is_dynamic())
         throw nb::value_error("a provider must return values it authored itself");
      result.push_back(entry.dynamic());
   }
   return result;
}

[[nodiscard]] rt::DynamicWorldState dynamic_world(nb::handle value)
{
   const auto state = nb::cast< PythonWorldState >(value);
   if(not state.is_dynamic())
      throw nb::value_error("a provider must return a WorldState it authored itself");
   return state.dynamic();
}

[[nodiscard]] rt::DynamicObservation dynamic_observation(nb::handle value)
{
   const auto observation = nb::cast< PythonObservation >(value);
   if(not observation.is_dynamic())
      throw nb::value_error("a provider must return an Observation it authored itself");
   return observation.dynamic();
}

/// The trampoline's ticket acquires the GIL and raises for a missing pure override on its own.
[[nodiscard]] nb::object call_override(
   const nb::detail::trampoline< 24 >& trampoline,
   const char* name,
   bool pure,
   auto&&... args
)
{
   nb::detail::ticket ticket(trampoline, name, pure);
   if(not ticket.key.is_valid()) {
      if(pure)
         throw std::runtime_error(std::string("the provider must implement ") + name);
      return nb::none();
   }
   return trampoline.base().attr(ticket.key)(std::forward< decltype(args) >(args)...);
}

/**
 * @brief Bridges a Python provider onto the one dynamic-provider interface.
 *
 * This class performs no validation of its own: every value returned here passes through
 * DynamicEnvironment, which is the single place that decides what a solver is allowed to see.
 */
class PyDynamicEnvironmentProvider final: public rt::DynamicEnvironmentProvider {
  public:
   NB_TRAMPOLINE(rt::DynamicEnvironmentProvider, 24);

   [[nodiscard]] size_t max_player_count() const final
   {
      return nb::cast< size_t >(call_override(nb_trampoline, "max_player_count", true));
   }
   [[nodiscard]] size_t player_count() const final
   {
      return nb::cast< size_t >(call_override(nb_trampoline, "player_count", true));
   }
   [[nodiscard]] nor::Stochasticity stochasticity() const final
   {
      return nb::cast< nor::Stochasticity >(call_override(nb_trampoline, "stochasticity", true));
   }
   [[nodiscard]] bool serialized() const final
   {
      const auto result = call_override(nb_trampoline, "serialized", false);
      return result.is_none() ? true : nb::cast< bool >(result);
   }
   [[nodiscard]] bool unrolled() const final
   {
      const auto result = call_override(nb_trampoline, "unrolled", false);
      return result.is_none() ? true : nb::cast< bool >(result);
   }
   [[nodiscard]] rt::DynamicWorldState initial_world_state() const final
   {
      return dynamic_world(call_override(nb_trampoline, "initial_world_state", true));
   }
   [[nodiscard]] std::vector< rt::DynamicAction >
   actions(nor::Player player, const rt::DynamicWorldState& state) const final
   {
      return dynamic_vector< PythonAction >(call_override(
         nb_trampoline, "actions", true, player, PythonWorldState::from_dynamic(state)
      ));
   }
   [[nodiscard]] std::vector< nor::Player > players(const rt::DynamicWorldState& state) const final
   {
      std::vector< nor::Player > result;
      const auto returned = call_override(
         nb_trampoline, "players", true, PythonWorldState::from_dynamic(state)
      );
      for(const nb::handle item : nb::borrow< nb::iterable >(returned))
         result.push_back(nb::cast< nor::Player >(item));
      return result;
   }
   [[nodiscard]] nor::Player active_player(const rt::DynamicWorldState& state) const final
   {
      return nb::cast< nor::Player >(
         call_override(nb_trampoline, "active_player", true, PythonWorldState::from_dynamic(state))
      );
   }
   [[nodiscard]] bool is_terminal(const rt::DynamicWorldState& state) const final
   {
      return nb::cast< bool >(
         call_override(nb_trampoline, "is_terminal", true, PythonWorldState::from_dynamic(state))
      );
   }
   [[nodiscard]] bool is_partaking(const rt::DynamicWorldState& state, nor::Player player)
      const final
   {
      const auto result = call_override(
         nb_trampoline, "is_partaking", false, PythonWorldState::from_dynamic(state), player
      );
      return result.is_none() ? true : nb::cast< bool >(result);
   }
   [[nodiscard]] double reward(nor::Player player, const rt::DynamicWorldState& state) const final
   {
      return nb::cast< double >(
         call_override(nb_trampoline, "reward", true, player, PythonWorldState::from_dynamic(state))
      );
   }

   void transition(rt::DynamicWorldState& state, const rt::DynamicAction& action) const final
   {
      state = dynamic_world(call_override(
         nb_trampoline,
         "transition_action",
         true,
         PythonWorldState::from_dynamic(state),
         PythonAction::from_dynamic(action)
      ));
   }

   [[nodiscard]] rt::DynamicObservation private_observation(
      nor::Player player,
      const rt::DynamicWorldState& state,
      const rt::DynamicAction& action,
      const rt::DynamicWorldState& next_state
   ) const final
   {
      return dynamic_observation(call_override(
         nb_trampoline,
         "private_observation_action",
         true,
         player,
         PythonWorldState::from_dynamic(state),
         PythonAction::from_dynamic(action),
         PythonWorldState::from_dynamic(next_state)
      ));
   }

   [[nodiscard]] rt::DynamicObservation public_observation(
      const rt::DynamicWorldState& state,
      const rt::DynamicAction& action,
      const rt::DynamicWorldState& next_state
   ) const final
   {
      return dynamic_observation(call_override(
         nb_trampoline,
         "public_observation_action",
         true,
         PythonWorldState::from_dynamic(state),
         PythonAction::from_dynamic(action),
         PythonWorldState::from_dynamic(next_state)
      ));
   }

   [[nodiscard]] std::vector< rt::DynamicChanceOutcome > chance_actions(
      const rt::DynamicWorldState& state
   ) const final
   {
      const auto result = call_override(
         nb_trampoline, "chance_actions", false, PythonWorldState::from_dynamic(state)
      );
      return result.is_none() ? std::vector< rt::DynamicChanceOutcome >{}
                              : dynamic_vector< PythonChanceOutcome >(result);
   }

   [[nodiscard]] double chance_probability(
      const rt::DynamicWorldState& state,
      const rt::DynamicChanceOutcome& outcome
   ) const final
   {
      const auto result = call_override(
         nb_trampoline,
         "chance_probability",
         false,
         PythonWorldState::from_dynamic(state),
         PythonChanceOutcome::from_dynamic(outcome)
      );
      if(result.is_none())
         return rt::DynamicEnvironmentProvider::chance_probability(state, outcome);
      return nb::cast< double >(result);
   }

   void transition(rt::DynamicWorldState& state, const rt::DynamicChanceOutcome& outcome)
      const final
   {
      const auto result = call_override(
         nb_trampoline,
         "transition_chance",
         false,
         PythonWorldState::from_dynamic(state),
         PythonChanceOutcome::from_dynamic(outcome)
      );
      if(result.is_none()) {
         rt::DynamicEnvironmentProvider::transition(state, outcome);
         return;
      }
      state = dynamic_world(result);
   }

   [[nodiscard]] rt::DynamicObservation private_observation(
      nor::Player player,
      const rt::DynamicWorldState& state,
      const rt::DynamicChanceOutcome& outcome,
      const rt::DynamicWorldState& next_state
   ) const final
   {
      const auto result = call_override(
         nb_trampoline,
         "private_observation_chance",
         false,
         player,
         PythonWorldState::from_dynamic(state),
         PythonChanceOutcome::from_dynamic(outcome),
         PythonWorldState::from_dynamic(next_state)
      );
      return result.is_none() ? rt::DynamicEnvironmentProvider::private_observation(
                player, state, outcome, next_state
             )
                              : dynamic_observation(result);
   }

   [[nodiscard]] rt::DynamicObservation public_observation(
      const rt::DynamicWorldState& state,
      const rt::DynamicChanceOutcome& outcome,
      const rt::DynamicWorldState& next_state
   ) const final
   {
      const auto result = call_override(
         nb_trampoline,
         "public_observation_chance",
         false,
         PythonWorldState::from_dynamic(state),
         PythonChanceOutcome::from_dynamic(outcome),
         PythonWorldState::from_dynamic(next_state)
      );
      return result.is_none()
                ? rt::DynamicEnvironmentProvider::public_observation(state, outcome, next_state)
                : dynamic_observation(result);
   }
};

///////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// session synchronization //////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief The single synchronization state a session and every wrapper derived from it share.
 *
 * The mutex serializes operations on one session; the epoch marks every wrapper handed out before
 * a mutation as stale. Wrappers hold only a weak reference to this object, so a policy handle can
 * never keep a solver alive past the Python session object that owns it.
 */
struct SessionState {
   SessionState(rt::SolverSession session, bool dynamic_game)
       : solver(std::move(session)), dynamic(dynamic_game)
   {
   }

   ~SessionState()
   {
      const std::lock_guard lock(mutex);
      closed = true;
      solver = rt::SolverSession{};
   }

   std::mutex mutex;
   rt::SolverSession solver;
   /// A dynamic session runs Python provider code inside the solver, so it keeps the GIL.
   bool dynamic = false;
   size_t epoch = 0;
   bool closed = false;
};

/// Sessions currently executing an operation on this thread, innermost last.
thread_local std::vector< SessionState* > active_sessions;

/**
 * @brief RAII entry into one session operation.
 *
 * Ordering matters here. Reentrancy is rejected before anything blocks, because a reentrant call
 * on the same thread would otherwise deadlock on a mutex its own frame already holds. Neither kind
 * of session waits for that mutex while holding the GIL; the difference is what happens afterwards.
 * A static session keeps the GIL released for the whole operation, because nothing inside a
 * compiled game's traversal touches Python. A dynamic session takes the GIL back, because its
 * environment calls into the Python provider on this very thread.
 *
 * Every read of shared session state happens under this guard. That is why even the advisory
 * `valid` queries on the policy wrappers go through it rather than peeking at the epoch.
 */
class SessionOperation {
  public:
   template < typename Predicate >
   SessionOperation(
      std::shared_ptr< SessionState > state,
      bool mutation,
      std::optional< size_t > expected_epoch,
      Predicate&& still_valid
   )
       : m_state(std::move(state)), m_mutation(mutation)
   {
      if(not m_state or m_state->closed or not m_state->solver)
         throw ClosedSessionError{};
      if(std::ranges::find(active_sessions, m_state.get()) != active_sessions.end())
         throw ReentrantSessionError{};

      if(m_state->dynamic) {
         // A dynamic operation must run holding the GIL, but it must not *wait* holding it: the
         // thread that owns the session is running Python provider code, and the interpreter
         // hands the GIL around while it does. Blocking for the mutex with the GIL in hand would
         // stall that thread on the GIL while it still owns the mutex, and neither would move
         // again. Drop the GIL for the wait only, then take it back for the operation itself.
         {
            const nb::gil_scoped_release release;
            m_lock = std::unique_lock< std::mutex >(m_state->mutex);
         }
      } else {
         // Nothing inside a compiled game's traversal touches Python, so the whole operation runs
         // without the GIL.
         m_release.emplace();
         m_lock = std::unique_lock< std::mutex >(m_state->mutex);
      }
      if(m_state->closed or not m_state->solver)
         throw ClosedSessionError{};
      if(expected_epoch.has_value() and m_state->epoch != *expected_epoch) {
         throw StaleViewError{
            "this policy handle was taken before the session advanced; take a new one"};
      }
      if(not std::forward< Predicate >(still_valid)()) {
         throw StaleViewError{"this policy handle no longer refers to live solver storage"};
      }
      active_sessions.push_back(m_state.get());
      m_registered = true;
   }

   SessionOperation(std::shared_ptr< SessionState > state, bool mutation)
       : SessionOperation(std::move(state), mutation, std::nullopt, [] { return true; })
   {
   }

   SessionOperation(const SessionOperation&) = delete;
   SessionOperation& operator=(const SessionOperation&) = delete;
   SessionOperation(SessionOperation&&) = delete;
   SessionOperation& operator=(SessionOperation&&) = delete;

   ~SessionOperation()
   {
      if(m_registered and not active_sessions.empty() and active_sessions.back() == m_state.get()) {
         active_sessions.pop_back();
      }
   }

   [[nodiscard]] rt::SolverSession& solver() noexcept { return m_state->solver; }
   [[nodiscard]] const rt::SolverSession& solver() const noexcept { return m_state->solver; }
   [[nodiscard]] size_t epoch() const noexcept { return m_state->epoch; }

   /// Marks every previously handed out wrapper stale. Called after a mutation succeeds.
   void changed() noexcept
   {
      if(m_mutation)
         ++m_state->epoch;
   }

  private:
   std::shared_ptr< SessionState > m_state;
   bool m_mutation = false;
   bool m_registered = false;
   // Declaration order is the destruction contract: the lock is released first, then the GIL is
   // reacquired, so an exception leaves the interpreter in the state nanobind expects.
   std::optional< nb::gil_scoped_release > m_release;
   std::unique_lock< std::mutex > m_lock;
};

///////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// policy wrappers //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

class PythonPolicyRow {
  public:
   PythonPolicyRow(std::weak_ptr< SessionState > state, rt::PolicyNodeView row, size_t epoch)
       : m_state(std::move(state)), m_row(std::move(row)), m_epoch(epoch)
   {
   }

   [[nodiscard]] bool valid() const
   {
      try {
         const auto operation = lock();
         return true;
      } catch(const ClosedSessionError&) {
         return false;
      } catch(const StaleViewError&) {
         return false;
      }
   }
   [[nodiscard]] rt::PolicyViewKind kind() const noexcept { return m_row.kind(); }
   [[nodiscard]] size_t generation() const noexcept { return m_row.generation(); }

   [[nodiscard]] nor::Player player() const
   {
      auto operation = lock();
      return m_row.player();
   }
   [[nodiscard]] PythonInfoState info_state() const
   {
      auto operation = lock();
      return PythonInfoState::from_static(m_row.info_state());
   }
   [[nodiscard]] size_t size() const
   {
      auto operation = lock();
      return m_row.size();
   }
   [[nodiscard]] PythonAction action_at(size_t index) const
   {
      auto operation = lock();
      return PythonAction::from_static(m_row.action_at(index));
   }
   [[nodiscard]] double value_at(size_t index) const
   {
      auto operation = lock();
      return m_row.value_at(index);
   }
   [[nodiscard]] std::optional< double > find(const PythonAction& action) const
   {
      auto operation = lock();
      return m_row.find(action.as_erased());
   }
   [[nodiscard]] bool contains(const PythonAction& action) const
   {
      auto operation = lock();
      return m_row.contains(action.as_erased());
   }
   [[nodiscard]] double at(const PythonAction& action) const
   {
      auto result = find(action);
      if(not result)
         throw nb::key_error("this action is not present in the policy row");
      return *result;
   }
   [[nodiscard]] std::vector< rt::PolicyActionEntry > entries() const
   {
      auto operation = lock();
      return m_row.to_entries();
   }
   [[nodiscard]] rt::TensorData tensor() const
   {
      auto operation = lock();
      return m_row.to_tensor();
   }

  private:
   [[nodiscard]] SessionOperation lock() const
   {
      return SessionOperation{m_state.lock(), false, m_epoch, [this] { return m_row.valid(); }};
   }

   std::weak_ptr< SessionState > m_state;
   rt::PolicyNodeView m_row;
   size_t m_epoch = 0;
};

/**
 * @brief A borrowed policy of one session.
 *
 * No visitor callback is exposed. A Python callback running inside the solver's node traversal is
 * exactly the hazard the C++ side guards against, and there is nothing a visitor offers here that
 * focused lookup plus the explicit to_entries() bulk copy does not.
 */
class PythonPolicy {
  public:
   PythonPolicy(
      std::weak_ptr< SessionState > state,
      rt::PolicyLookup lookup,
      rt::PolicyViewKind kind,
      size_t epoch
   )
       : m_state(std::move(state)), m_lookup(std::move(lookup)), m_kind(kind), m_epoch(epoch)
   {
   }

   /// Advisory: true when a lookup taken right now would succeed. It is answered under the same
   /// guard every other read uses, so a reentrant query reports the reentrancy rather than a stale
   /// snapshot.
   [[nodiscard]] bool valid() const
   {
      try {
         const auto operation = lock();
         return true;
      } catch(const ClosedSessionError&) {
         return false;
      } catch(const StaleViewError&) {
         return false;
      }
   }
   [[nodiscard]] rt::PolicyViewKind kind() const noexcept { return m_kind; }
   [[nodiscard]] size_t generation() const noexcept { return m_lookup.generation(); }

   [[nodiscard]] std::optional< PythonPolicyRow > find(const PythonInfoState& info_state) const
   {
      auto operation = lock();
      auto row = m_lookup.find(info_state.as_erased());
      if(not row)
         return std::nullopt;
      return PythonPolicyRow{m_state, std::move(*row), m_epoch};
   }

   [[nodiscard]] PythonPolicyRow at(const PythonInfoState& info_state) const
   {
      auto result = find(info_state);
      if(not result)
         throw nb::key_error("this information state is not present in the policy");
      return std::move(*result);
   }

   /// The explicit bulk copy. Row order is unspecified; see the C++ contract in types.hpp.
   [[nodiscard]] std::vector< rt::PolicyEntry > entries() const
   {
      auto operation = lock();
      return m_lookup.to_entries();
   }

  private:
   [[nodiscard]] SessionOperation lock() const
   {
      return SessionOperation{m_state.lock(), false, m_epoch, [this] { return m_lookup.valid(); }};
   }

   std::weak_ptr< SessionState > m_state;
   rt::PolicyLookup m_lookup;
   rt::PolicyViewKind m_kind = rt::PolicyViewKind::current;
   size_t m_epoch = 0;
};

///////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// session and game /////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

class PythonSession {
  public:
   explicit PythonSession(std::shared_ptr< SessionState > state) : m_state(std::move(state)) {}

   [[nodiscard]] rt::IterationResult iterate()
   {
      SessionOperation operation(m_state, true);
      auto result = unwrap(operation.solver().iterate());
      operation.changed();
      return result;
   }
   void advance(size_t iterations)
   {
      SessionOperation operation(m_state, true);
      unwrap(operation.solver().advance(iterations));
      if(iterations != 0)
         operation.changed();
   }
   [[nodiscard]] std::optional< rt::IterationResult > advance_last(size_t iterations)
   {
      SessionOperation operation(m_state, true);
      auto result = unwrap(operation.solver().advance_last(iterations));
      if(iterations != 0)
         operation.changed();
      return result;
   }
   [[nodiscard]] rt::TraceResult trace(size_t iterations, size_t every)
   {
      SessionOperation operation(m_state, true);
      auto result = unwrap(operation.solver().trace(iterations, every));
      if(iterations != 0)
         operation.changed();
      return result;
   }
   [[nodiscard]] rt::SessionStats stats() const
   {
      SessionOperation operation(m_state, false);
      return unwrap(operation.solver().stats());
   }
   [[nodiscard]] PythonPolicy policy(rt::PolicyViewKind kind) const
   {
      SessionOperation operation(m_state, false);
      auto lookup = unwrap(operation.solver().policy_lookup(kind));
      return PythonPolicy{m_state, std::move(lookup), kind, operation.epoch()};
   }
   [[nodiscard]] bool dynamic() const noexcept { return m_state and m_state->dynamic; }
   [[nodiscard]] bool closed() const noexcept { return not m_state or m_state->closed; }

  private:
   std::shared_ptr< SessionState > m_state;
};

/// The one Python-facing game abstraction. A compiled game and a pure-Python provider differ only
/// in which handle this holds; both produce the same Session type and the same solver families.
class PythonGame {
  public:
   explicit PythonGame(rt::GameHandle game) : m_game(std::move(game)) {}
   explicit PythonGame(rt::DynamicGameHandle game) : m_game(std::move(game)) {}

   [[nodiscard]] bool is_dynamic() const noexcept
   {
      return std::holds_alternative< rt::DynamicGameHandle >(m_game);
   }
   [[nodiscard]] rt::GameId id() const noexcept
   {
      return std::visit([](const auto& game) { return game.game_id(); }, m_game);
   }
   [[nodiscard]] std::string name() const { return enum_name(id()); }
   [[nodiscard]] rt::GameSpec spec() const
   {
      return std::visit([](const auto& game) { return game.spec(); }, m_game);
   }
   /// A dynamic game's roster is fixed by admission, so its bounds coincide.
   [[nodiscard]] std::pair< size_t, size_t > player_bounds() const
   {
      if(const auto* dynamic_game = std::get_if< rt::DynamicGameHandle >(&m_game)) {
         const auto count = dynamic_game->admission()->player_count;
         return {count, count};
      }
      const auto* descriptor = rt::find_game(id());
      if(descriptor == nullptr)
         return {0, 0};
      return {descriptor->min_players, descriptor->max_players};
   }
   [[nodiscard]] nor::Stochasticity stochasticity() const
   {
      if(const auto* dynamic_game = std::get_if< rt::DynamicGameHandle >(&m_game))
         return dynamic_game->admission()->stochasticity;
      const auto* descriptor = rt::find_game(id());
      return descriptor == nullptr ? nor::Stochasticity::deterministic : descriptor->stochasticity;
   }
   [[nodiscard]] nb::list capabilities() const
   {
      nb::list result;
      if(is_dynamic()) {
         for(const auto& descriptor : rt::dynamic_capabilities())
            result.append(descriptor);
      } else {
         for(const auto& descriptor : rt::capabilities_for(id()))
            result.append(descriptor);
      }
      return result;
   }

   [[nodiscard]] PythonSession make_session(
      nb::handle solver_or_profile,
      nb::handle profile,
      double epsilon,
      uint64_t seed
   ) const
   {
      rt::SolverId solver{};
      rt::ProfileId selected_profile{};
      if(profile.is_none()) {
         selected_profile = enum_from_python< rt::ProfileId >(solver_or_profile, "ProfileId");
         const auto* descriptor = rt::find_profile(selected_profile);
         if(descriptor == nullptr)
            throw nb::value_error("unknown solver profile");
         solver = descriptor->solver;
      } else {
         solver = enum_from_python< rt::SolverId >(solver_or_profile, "SolverId");
         selected_profile = enum_from_python< rt::ProfileId >(profile, "ProfileId");
      }

      const rt::SessionOptions options{.epsilon = epsilon, .seed = seed};
      rt::Result< rt::SolverSession > result = std::visit(
         [&](const auto& game) -> rt::Result< rt::SolverSession > {
            using game_type = std::remove_cvref_t< decltype(game) >;
            if constexpr(std::is_same_v< game_type, rt::GameHandle >) {
               return rt::make_session(game, solver, selected_profile, options);
            } else {
               return game.make_session(solver, selected_profile, options);
            }
         },
         m_game
      );
      auto session = unwrap(std::move(result));
      return PythonSession{std::make_shared< SessionState >(std::move(session), is_dynamic())};
   }

  private:
   std::variant< rt::GameHandle, rt::DynamicGameHandle > m_game;
};

///////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// spec plumbing ////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

[[nodiscard]] rt::SpecValue spec_value(nb::handle value)
{
   if(nb::isinstance< nb::bool_ >(value))
      return nb::cast< bool >(value);
   if(nb::isinstance< nb::int_ >(value)) {
      // Try the signed spelling first so negative values retain the existing failed-conversion
      // representation for unsigned fields. A Python int can exceed LLONG_MAX while still being
      // a valid uint64_t, so the unsigned conversion must be a separate fallback rather than an
      // implicit narrowing through long long.
      long long signed_value = 0;
      if(nb::try_cast< long long >(value, signed_value)) {
         if(signed_value < 0)
            return static_cast< double >(signed_value);
         return static_cast< uint64_t >(signed_value);
      }
      uint64_t unsigned_value = 0;
      if(nb::try_cast< uint64_t >(value, unsigned_value))
         return unsigned_value;
      throw nb::type_error("an integer GameSpec field value must fit in uint64_t");
   }
   if(nb::isinstance< nb::float_ >(value))
      return nb::cast< double >(value);
   throw nb::type_error("a GameSpec field value must be a bool, int or float");
}

[[nodiscard]] rt::GameFieldId field_from_python(nb::handle value)
{
   return enum_from_python< rt::GameFieldId >(value, "GameFieldId");
}

void apply_kwargs(rt::GameSpec& spec, const nb::kwargs& fields)
{
   for(const auto item : fields) {
      spec.set(field_from_python(item.first), spec_value(item.second));
   }
}

class PythonGameSpec {
  public:
   explicit PythonGameSpec(rt::GameSpec spec) : m_spec(std::move(spec)) {}

   [[nodiscard]] rt::GameId game_id() const noexcept { return m_spec.game_id(); }
   [[nodiscard]] std::vector< rt::SpecField > fields() const { return m_spec.fields(); }
   [[nodiscard]] bool contains(nb::handle field) const
   {
      return m_spec.contains(field_from_python(field));
   }
   [[nodiscard]] std::optional< rt::SpecValue > find(nb::handle field) const
   {
      if(const auto* value = m_spec.find(field_from_python(field)))
         return *value;
      return std::nullopt;
   }
   void set(nb::handle field, nb::handle value)
   {
      m_spec.set(field_from_python(field), spec_value(value));
   }
   [[nodiscard]] const rt::GameSpec& spec() const noexcept { return m_spec; }

  private:
   rt::GameSpec m_spec;
};

void construct_static_game(PythonGame* self, nb::handle game, const nb::kwargs& fields)
{
   const auto id = enum_from_python< rt::GameId >(game, "GameId");
   auto spec = rt::GameSpec::defaults(id);
   apply_kwargs(spec, fields);
   new(self) PythonGame(unwrap(rt::make_game(spec)));
}

void construct_dynamic_game(
   PythonGame* self,
   std::shared_ptr< rt::DynamicEnvironmentProvider > provider,
   const nb::kwargs& fields
)
{
   if(not provider)
      throw nb::value_error("a dynamic Game requires a provider");
   auto spec = rt::GameSpec{rt::GameId::dynamic};
   apply_kwargs(spec, fields);
   new(self) PythonGame(unwrap(rt::make_dynamic_game(std::move(spec), std::move(provider))));
}

///////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// registration helpers /////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

PyObject* capability_exception_type = nullptr;

/// Registered after nb::exception<> so that it runs first and can attach structured context.
void translate_capability(const std::exception_ptr& pointer, void* payload)
{
   try {
      std::rethrow_exception(pointer);
   } catch(const BindingCapabilityError& error) {
      auto* type = static_cast< PyObject* >(payload);
      nb::object instance = nb::steal(PyObject_CallFunction(type, "s", error.what()));
      if(not instance.is_valid())
         return;
      const auto& context = error.error();
      // An error raised before a solver or profile was selected leaves those fields at zero, which
      // is not an enumerator. Casting one would raise from inside the translator itself and the
      // caller would see that failure instead of the real error.
      const auto optional_enum = [](auto value) -> nb::object {
         if(static_cast< uint16_t >(value) == 0)
            return nb::none();
         return nb::cast(value);
      };
      nb::setattr(instance, "code", nb::cast(context.code));
      nb::setattr(instance, "game", optional_enum(context.game));
      nb::setattr(instance, "solver", optional_enum(context.solver));
      nb::setattr(instance, "profile", optional_enum(context.profile));
      PyErr_SetObject(type, instance.ptr());
   }
}

template < typename Enum >
void bind_enum(nb::module_& module, const char* name)
{
   auto enumeration = nb::enum_< Enum >(module, name);
   // The enumerator table comes from reflection; this module never repeats an enumerator name.
   for(const auto entry : nor::meta::detail::make_enum_entries< Enum >())
      enumeration.value(std::string(entry.name).c_str(), entry.value);
}

template < rt::ValueKind Kind >
void bind_domain_value(nb::module_& module, const char* name)
{
   using value_type = PythonValue< Kind >;
   auto value = nb::class_< value_type >(module, name);
   value.def(nb::init<>());

   if constexpr(is_named_kind< Kind > or Kind == rt::ValueKind::world_state) {
      const auto make_named = [](std::string type_name,
                                 std::string identity,
                                 std::optional< std::string > text,
                                 std::optional< rt::TensorData > tensor) {
         return value_type::named(
            std::move(type_name), std::move(identity), std::move(text), std::move(tensor)
         );
      };
      value.def(
         "__init__",
         [make_named](
            value_type* self,
            std::string type_name,
            std::string identity,
            std::optional< std::string > text,
            std::optional< rt::TensorData > tensor
         ) {
            new(self) value_type(make_named(
               std::move(type_name), std::move(identity), std::move(text), std::move(tensor)
            ));
         },
         "type_name"_a,
         "identity"_a,
         "text"_a = nb::none(),
         "tensor"_a = nb::none()
      );
      value.def_static(
         "named",
         make_named,
         "type_name"_a,
         "identity"_a,
         "text"_a = nb::none(),
         "tensor"_a = nb::none()
      );
   }
   if constexpr(Kind == rt::ValueKind::info_state) {
      value.def(
         "__init__",
         [](value_type* self, nor::Player player) {
            new(self) value_type(value_type::with_player(player));
         },
         "player"_a
      );
   }

   value.def_prop_ro("valid", &value_type::valid)
      .def_prop_ro("is_dynamic", &value_type::is_dynamic)
      .def_prop_ro("kind", [](const value_type&) { return Kind; })
      .def_prop_ro("type_name", &value_type::type_name)
      .def_prop_ro("identity", &value_type::identity)
      .def_prop_ro("capabilities", &value_type::capabilities)
      .def("to_string", &value_type::to_string)
      .def("to_tensor", &value_type::to_tensor)
      .def("__bool__", &value_type::valid)
      .def("__hash__", &value_type::hash)
      .def(
         "__eq__",
         [](const value_type& left, nb::handle right) -> nb::object {
            value_type other;
            if(not nb::try_cast< value_type >(right, other))
               return nb::borrow(Py_NotImplemented);
            return nb::cast(left == other);
         }
      )
      .def("__repr__", [name](const value_type& self) {
         auto text = self.to_string();
         const auto identity = self.identity();
         return std::string(name) + "(" + self.type_name() + ", "
                + (identity ? *identity : (text ? *text : std::string{"<opaque>"})) + ")";
      });

   if constexpr(Kind == rt::ValueKind::info_state) {
      value.def_prop_ro("player", &value_type::player)
         .def("update", &value_type::update_info, "public_observation"_a, "private_observation"_a)
         .def("__len__", &value_type::size)
         .def("__getitem__", &value_type::info_at, "index"_a)
         .def("latest", &value_type::latest_info);
   } else if constexpr(Kind == rt::ValueKind::public_state) {
      value.def("update", &value_type::update_public, "observation"_a)
         .def("__len__", &value_type::size)
         .def("__getitem__", &value_type::observation_at, "index"_a)
         .def("latest", &value_type::latest_observation);
   } else if constexpr(Kind == rt::ValueKind::world_state) {
      value.def("set", &value_type::set_world, "state"_a);
   }
}

}  // namespace noregret_binding

NB_MODULE(_noregret, module)
{
   using namespace noregret_binding;

   module.doc(
   ) = "Compiled no-regret solver bindings. Compiled games select one registered game/solver "
       "profile and then run entirely in statically dispatched C++; a pure-Python game runs the "
       "same compiled solvers through the dynamic provider boundary.";

   bind_enum< nor::Player >(module, "Player");
   bind_enum< nor::Stochasticity >(module, "Stochasticity");
   bind_enum< rt::GameId >(module, "GameId");
   bind_enum< rt::SolverId >(module, "SolverId");
   bind_enum< rt::ProfileId >(module, "ProfileId");
   bind_enum< rt::GameFieldId >(module, "GameFieldId");
   bind_enum< rt::SpecKind >(module, "SpecKind");
   bind_enum< rt::ValueKind >(module, "ValueKind");
   bind_enum< rt::PolicyViewKind >(module, "PolicyKind");
   bind_enum< rt::CapabilityErrorCode >(module, "CapabilityErrorCode");

   nb::class_< rt::TensorData >(module, "TensorData")
      .def(nb::init<>())
      .def(
         "__init__",
         [](rt::TensorData* self, std::vector< double > values, std::vector< size_t > shape) {
            new(self) rt::TensorData{std::move(values), std::move(shape)};
         },
         "values"_a,
         "shape"_a
      )
      .def_rw("values", &rt::TensorData::values)
      .def_rw("shape", &rt::TensorData::shape)
      .def("__eq__", [](const rt::TensorData& left, nb::handle right) -> nb::object {
         rt::TensorData other;
         if(not nb::try_cast< rt::TensorData >(right, other))
            return nb::borrow(Py_NotImplemented);
         return nb::cast(left == other);
      });

   nb::class_< rt::ValueCapabilities >(module, "ValueCapabilities")
      .def(nb::init<>())
      .def_rw("to_string", &rt::ValueCapabilities::to_string)
      .def_rw("to_tensor", &rt::ValueCapabilities::to_tensor);

   bind_domain_value< rt::ValueKind::action >(module, "Action");
   bind_domain_value< rt::ValueKind::chance_outcome >(module, "ChanceOutcome");
   bind_domain_value< rt::ValueKind::observation >(module, "Observation");
   bind_domain_value< rt::ValueKind::info_state >(module, "InfoState");
   bind_domain_value< rt::ValueKind::public_state >(module, "PublicState");
   bind_domain_value< rt::ValueKind::world_state >(module, "WorldState");

   nb::class_< rt::SpecField >(module, "SpecField")
      .def_ro("id", &rt::SpecField::id)
      .def_prop_ro("value", [](const rt::SpecField& field) { return field.value; })
      .def("__repr__", [](const rt::SpecField& field) {
         return "SpecField(" + enum_name(field.id) + ")";
      });

   nb::class_< PythonGameSpec >(module, "GameSpec")
      .def(
         "__init__",
         [](PythonGameSpec* self, nb::handle game, const nb::kwargs& fields) {
            auto spec = rt::GameSpec::defaults(enum_from_python< rt::GameId >(game, "GameId"));
            apply_kwargs(spec, fields);
            new(self) PythonGameSpec(std::move(spec));
         },
         "game"_a,
         "fields"_a
      )
      .def_prop_ro("game", &PythonGameSpec::game_id)
      .def_prop_ro("fields", &PythonGameSpec::fields)
      .def("contains", &PythonGameSpec::contains, "field"_a)
      .def("find", &PythonGameSpec::find, "field"_a)
      .def("set", &PythonGameSpec::set, "field"_a, "value"_a)
      .def("__repr__", [](const PythonGameSpec& spec) {
         return "GameSpec(" + enum_name(spec.game_id()) + ")";
      });

   nb::class_< rt::FieldDescriptor >(module, "FieldDescriptor")
      .def_ro("id", &rt::FieldDescriptor::id)
      .def_prop_ro("name", [](const rt::FieldDescriptor& value) { return std::string(value.name); })
      .def_ro("kind", &rt::FieldDescriptor::kind)
      .def_prop_ro("default_value", [](const rt::FieldDescriptor& value) {
         return value.default_value;
      });

   nb::class_< rt::GameDescriptor >(module, "GameDescriptor")
      .def_ro("id", &rt::GameDescriptor::id)
      .def_prop_ro("name", [](const rt::GameDescriptor& value) { return std::string(value.name); })
      .def_ro("min_players", &rt::GameDescriptor::min_players)
      .def_ro("max_players", &rt::GameDescriptor::max_players)
      .def_ro("stochasticity", &rt::GameDescriptor::stochasticity)
      .def_prop_ro(
         "fields",
         [](const rt::GameDescriptor& value) {
            return std::vector< rt::FieldDescriptor >{value.fields.begin(), value.fields.end()};
         }
      )
      .def("__repr__", [](const rt::GameDescriptor& value) {
         return "GameDescriptor(" + std::string(value.name) + ")";
      });

   nb::class_< rt::SolverDescriptor >(module, "SolverDescriptor")
      .def_ro("id", &rt::SolverDescriptor::id)
      .def_prop_ro("name", [](const rt::SolverDescriptor& value) {
         return std::string(value.name);
      });

   nb::class_< rt::ProfileDescriptor >(module, "ProfileDescriptor")
      .def_ro("id", &rt::ProfileDescriptor::id)
      .def_ro("solver", &rt::ProfileDescriptor::solver)
      .def_prop_ro("name", [](const rt::ProfileDescriptor& value) {
         return std::string(value.name);
      });

   nb::class_< rt::CapabilityDescriptor >(module, "CapabilityDescriptor")
      .def_ro("game", &rt::CapabilityDescriptor::game)
      .def_ro("solver", &rt::CapabilityDescriptor::solver)
      .def_ro("profile", &rt::CapabilityDescriptor::profile)
      .def("__repr__", [](const rt::CapabilityDescriptor& value) {
         return "CapabilityDescriptor(" + enum_name(value.game) + ", " + enum_name(value.profile)
                + ")";
      });

   nb::class_< rt::DynamicCapabilityDescriptor >(module, "DynamicCapabilityDescriptor")
      .def_prop_ro(
         "game", [](const rt::DynamicCapabilityDescriptor&) { return rt::GameId::dynamic; }
      )
      .def_ro("solver", &rt::DynamicCapabilityDescriptor::solver)
      .def_ro("profile", &rt::DynamicCapabilityDescriptor::profile)
      .def_prop_ro(
         "name",
         [](const rt::DynamicCapabilityDescriptor& value) { return std::string(value.name); }
      )
      .def("__repr__", [](const rt::DynamicCapabilityDescriptor& value) {
         return "DynamicCapabilityDescriptor(" + enum_name(value.profile) + ")";
      });

   nb::class_< rt::RootValue >(module, "RootValue")
      .def_ro("player", &rt::RootValue::player)
      .def_ro("value", &rt::RootValue::value)
      .def("__repr__", [](const rt::RootValue& value) {
         return "RootValue(" + enum_name(value.player) + ", " + std::to_string(value.value) + ")";
      });

   nb::class_< rt::IterationResult >(module, "IterationResult")
      .def_ro("iteration", &rt::IterationResult::iteration)
      .def_ro("root_values", &rt::IterationResult::root_values)
      .def("__len__", [](const rt::IterationResult& value) { return value.root_values.size(); });

   nb::class_< rt::TraceResult >(module, "TraceResult")
      .def_ro("first_iteration", &rt::TraceResult::first_iteration)
      .def_ro("last_iteration", &rt::TraceResult::last_iteration)
      .def_ro("iterations", &rt::TraceResult::iterations)
      .def("__len__", &rt::TraceResult::size)
      .def("__getitem__", [](const rt::TraceResult& value, size_t index) {
         if(index >= value.size())
            throw nb::index_error("trace index out of range");
         return value[index];
      });

   nb::class_< rt::SessionStats >(module, "SessionStats")
      .def_ro("game", &rt::SessionStats::game)
      .def_ro("solver", &rt::SessionStats::solver)
      .def_ro("profile", &rt::SessionStats::profile)
      .def_ro("iteration", &rt::SessionStats::iteration)
      .def_ro("cycle", &rt::SessionStats::cycle)
      .def_ro("player_count", &rt::SessionStats::player_count)
      .def_ro("current_policy_entries", &rt::SessionStats::current_policy_entries)
      .def_ro("average_policy_entries", &rt::SessionStats::average_policy_entries);

   nb::exception< BindingCapabilityError > capability_error(module, "CapabilityError");
   capability_exception_type = capability_error.ptr();
   // Translators are consulted most-recently-registered first, so this one wins over the plain
   // message-only translator that nb::exception installed above.
   nb::register_exception_translator(translate_capability, capability_exception_type);
   nb::exception< StaleViewError > stale_error(module, "StaleViewError");
   nb::exception< ClosedSessionError > closed_error(module, "ClosedSessionError");
   nb::exception< ReentrantSessionError > reentrant_error(module, "ReentrantSessionError");

   nb::class_< PythonPolicyRow >(module, "PolicyRow")
      .def_prop_ro("valid", &PythonPolicyRow::valid)
      .def_prop_ro("kind", &PythonPolicyRow::kind)
      .def_prop_ro("generation", &PythonPolicyRow::generation)
      .def_prop_ro("player", &PythonPolicyRow::player)
      .def_prop_ro("info_state", &PythonPolicyRow::info_state)
      .def_prop_ro("size", &PythonPolicyRow::size)
      .def("__len__", &PythonPolicyRow::size)
      .def("action_at", &PythonPolicyRow::action_at, "index"_a)
      .def("value_at", &PythonPolicyRow::value_at, "index"_a)
      .def("find", &PythonPolicyRow::find, "action"_a)
      .def("contains", &PythonPolicyRow::contains, "action"_a)
      .def("__contains__", &PythonPolicyRow::contains, "action"_a)
      .def("at", &PythonPolicyRow::at, "action"_a)
      .def("__getitem__", &PythonPolicyRow::at, "action"_a)
      .def(
         "to_entries",
         [](const PythonPolicyRow& row) {
            nb::list result;
            for(const auto& entry : row.entries()) {
               result.append(
                  nb::make_tuple(PythonAction::from_static(entry.action), entry.probability)
               );
            }
            return result;
         },
         "Explicitly copy this row as a list of (Action, probability) pairs, in the node's "
         "deterministic action order."
      )
      .def("to_tensor", &PythonPolicyRow::tensor);

   nb::class_< PythonPolicy >(module, "Policy")
      .def_prop_ro("valid", &PythonPolicy::valid)
      .def_prop_ro("kind", &PythonPolicy::kind)
      .def_prop_ro("generation", &PythonPolicy::generation)
      .def("find", &PythonPolicy::find, "info_state"_a)
      .def("at", &PythonPolicy::at, "info_state"_a)
      .def("__getitem__", &PythonPolicy::at, "info_state"_a)
      .def(
         "to_entries",
         [](const PythonPolicy& policy) {
            nb::list result;
            for(const auto& entry : policy.entries()) {
               nb::list actions;
               for(const auto& action : entry.actions) {
                  actions.append(
                     nb::make_tuple(PythonAction::from_static(action.action), action.probability)
                  );
               }
               result.append(nb::make_tuple(
                  entry.player, PythonInfoState::from_static(entry.info_state), actions
               ));
            }
            return result;
         },
         "Explicitly copy the whole policy as a list of (player, InfoState, actions) rows. The "
         "ROW ORDER IS UNSPECIFIED: it follows the solver's hash-map order and may change between "
         "builds and runs. Sort the result if a stable presentation is needed."
      );

   nb::class_< PythonSession >(module, "Session")
      .def("iterate", &PythonSession::iterate, "Run exactly one iteration and return its result.")
      .def("advance", &PythonSession::advance, "n"_a, "Run n iterations, collecting nothing.")
      .def(
         "advance_last",
         &PythonSession::advance_last,
         "n"_a,
         "Run n iterations and return only the last result, or None when n is zero."
      )
      .def(
         "trace",
         &PythonSession::trace,
         "n"_a,
         "every"_a = 1,
         "Run n iterations and collect the result after every `every`-th one."
      )
      .def("stats", &PythonSession::stats)
      .def("policy", &PythonSession::policy, "kind"_a = rt::PolicyViewKind::current)
      .def_prop_ro("dynamic", &PythonSession::dynamic)
      .def_prop_ro("closed", &PythonSession::closed);

   nb::class_< rt::DynamicEnvironmentProvider, PyDynamicEnvironmentProvider >(
      module, "DynamicEnvironmentProvider"
   )
      .def(nb::init<>())
      .def("max_player_count", &rt::DynamicEnvironmentProvider::max_player_count)
      .def("player_count", &rt::DynamicEnvironmentProvider::player_count)
      .def("stochasticity", &rt::DynamicEnvironmentProvider::stochasticity)
      .def("serialized", &rt::DynamicEnvironmentProvider::serialized)
      .def("unrolled", &rt::DynamicEnvironmentProvider::unrolled)
      .def(
         "initial_world_state",
         [](const rt::DynamicEnvironmentProvider& self) {
            return PythonWorldState::from_dynamic(self.initial_world_state());
         }
      )
      .def(
         "actions",
         [](const rt::DynamicEnvironmentProvider& self,
            nor::Player player,
            const PythonWorldState& state) {
            nb::list result;
            for(const auto& action : self.actions(player, state.dynamic()))
               result.append(PythonAction::from_dynamic(action));
            return result;
         },
         "player"_a,
         "state"_a
      )
      .def(
         "players",
         [](const rt::DynamicEnvironmentProvider& self, const PythonWorldState& state) {
            return self.players(state.dynamic());
         },
         "state"_a
      )
      .def(
         "active_player",
         [](const rt::DynamicEnvironmentProvider& self, const PythonWorldState& state) {
            return self.active_player(state.dynamic());
         },
         "state"_a
      )
      .def(
         "is_terminal",
         [](const rt::DynamicEnvironmentProvider& self, const PythonWorldState& state) {
            return self.is_terminal(state.dynamic());
         },
         "state"_a
      )
      .def(
         "is_partaking",
         [](const rt::DynamicEnvironmentProvider& self,
            const PythonWorldState& state,
            nor::Player player) { return self.is_partaking(state.dynamic(), player); },
         "state"_a,
         "player"_a
      )
      .def(
         "reward",
         [](const rt::DynamicEnvironmentProvider& self,
            nor::Player player,
            const PythonWorldState& state) { return self.reward(player, state.dynamic()); },
         "player"_a,
         "state"_a
      )
      .def(
         "transition_action",
         [](const rt::DynamicEnvironmentProvider& self,
            PythonWorldState state,
            const PythonAction& action) {
            self.transition(state.mutable_dynamic(), action.dynamic());
            return state;
         },
         "state"_a,
         "action"_a,
         "Return the successor world state. The argument is a copy; providers return a new state."
      )
      .def(
         "transition_chance",
         [](const rt::DynamicEnvironmentProvider& self,
            PythonWorldState state,
            const PythonChanceOutcome& outcome) {
            self.transition(state.mutable_dynamic(), outcome.dynamic());
            return state;
         },
         "state"_a,
         "outcome"_a
      )
      .def(
         "private_observation_action",
         [](const rt::DynamicEnvironmentProvider& self,
            nor::Player player,
            const PythonWorldState& state,
            const PythonAction& action,
            const PythonWorldState& next_state) {
            return PythonObservation::from_dynamic(self.private_observation(
               player, state.dynamic(), action.dynamic(), next_state.dynamic()
            ));
         },
         "player"_a,
         "state"_a,
         "action"_a,
         "next_state"_a
      )
      .def(
         "public_observation_action",
         [](const rt::DynamicEnvironmentProvider& self,
            const PythonWorldState& state,
            const PythonAction& action,
            const PythonWorldState& next_state) {
            return PythonObservation::from_dynamic(
               self.public_observation(state.dynamic(), action.dynamic(), next_state.dynamic())
            );
         },
         "state"_a,
         "action"_a,
         "next_state"_a
      )
      .def(
         "chance_actions",
         [](const rt::DynamicEnvironmentProvider& self, const PythonWorldState& state) {
            nb::list result;
            for(const auto& outcome : self.chance_actions(state.dynamic()))
               result.append(PythonChanceOutcome::from_dynamic(outcome));
            return result;
         },
         "state"_a
      )
      .def(
         "chance_probability",
         [](const rt::DynamicEnvironmentProvider& self,
            const PythonWorldState& state,
            const PythonChanceOutcome& outcome) {
            return self.chance_probability(state.dynamic(), outcome.dynamic());
         },
         "state"_a,
         "outcome"_a
      )
      .def(
         "private_observation_chance",
         [](const rt::DynamicEnvironmentProvider& self,
            nor::Player player,
            const PythonWorldState& state,
            const PythonChanceOutcome& outcome,
            const PythonWorldState& next_state) {
            return PythonObservation::from_dynamic(self.private_observation(
               player, state.dynamic(), outcome.dynamic(), next_state.dynamic()
            ));
         },
         "player"_a,
         "state"_a,
         "outcome"_a,
         "next_state"_a
      )
      .def(
         "public_observation_chance",
         [](const rt::DynamicEnvironmentProvider& self,
            const PythonWorldState& state,
            const PythonChanceOutcome& outcome,
            const PythonWorldState& next_state) {
            return PythonObservation::from_dynamic(
               self.public_observation(state.dynamic(), outcome.dynamic(), next_state.dynamic())
            );
         },
         "state"_a,
         "outcome"_a,
         "next_state"_a
      );

   nb::class_< PythonGame >(module, "Game")
      // The provider overload is declared first because the static one accepts an untyped handle
      // (a GameId or its reflected name) and would otherwise swallow a provider argument.
      .def("__init__", &construct_dynamic_game, "provider"_a, "fields"_a)
      .def("__init__", &construct_static_game, "game"_a, "fields"_a)
      .def_static(
         "from_spec",
         [](const PythonGameSpec& spec) { return PythonGame{unwrap(rt::make_game(spec.spec()))}; },
         "spec"_a
      )
      .def_prop_ro("id", &PythonGame::id)
      .def_prop_ro("name", &PythonGame::name)
      .def_prop_ro("is_dynamic", &PythonGame::is_dynamic)
      .def_prop_ro("min_players", [](const PythonGame& game) { return game.player_bounds().first; })
      .def_prop_ro(
         "max_players", [](const PythonGame& game) { return game.player_bounds().second; }
      )
      .def_prop_ro("stochasticity", &PythonGame::stochasticity)
      .def_prop_ro("spec", [](const PythonGame& game) { return PythonGameSpec{game.spec()}; })
      .def_prop_ro("capabilities", &PythonGame::capabilities)
      .def(
         "make_session",
         &PythonGame::make_session,
         "solver_or_profile"_a,
         "profile"_a = nb::none(),
         "epsilon"_a = 0.6,
         "seed"_a = 0
      )
      .def("__repr__", [](const PythonGame& game) { return "Game(" + game.name() + ")"; });

   module.def("games", [] {
      return std::vector< rt::GameDescriptor >{rt::games().begin(), rt::games().end()};
   });
   module.def("solvers", [] {
      return std::vector< rt::SolverDescriptor >{rt::solvers().begin(), rt::solvers().end()};
   });
   module.def("profiles", [] {
      return std::vector< rt::ProfileDescriptor >{rt::profiles().begin(), rt::profiles().end()};
   });
   module.def("capabilities", [] {
      return std::vector< rt::CapabilityDescriptor >{
         rt::capabilities().begin(), rt::capabilities().end()};
   });
   module.def("dynamic_capabilities", [] {
      return std::vector< rt::DynamicCapabilityDescriptor >{
         rt::dynamic_capabilities().begin(), rt::dynamic_capabilities().end()};
   });
   module.def(
      "capabilities_for",
      [](nb::handle game) {
         return rt::capabilities_for(enum_from_python< rt::GameId >(game, "GameId"));
      },
      "game"_a
   );
   module.def(
      "profiles_for",
      [](nb::handle solver) {
         return rt::profiles_for(enum_from_python< rt::SolverId >(solver, "SolverId"));
      },
      "solver"_a
   );
}
