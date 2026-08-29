#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <mutex>
#include <new>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "runtime/runtime.hpp"

namespace nb = nanobind;
namespace rt = nor::binding::runtime;
using namespace nb::literals;

namespace noregret_binding {

template < typename Enum >
[[nodiscard]] std::string enum_name(Enum value)
{
   const auto name = nor::meta::enum_name(value);
   return name.empty() ? std::to_string(static_cast< std::underlying_type_t< Enum > >(value))
                       : std::string(name);
}

template < typename Enum >
[[nodiscard]] Enum enum_from_python(nb::handle value, const char* description)
{
   if(nb::isinstance< nb::str >(value)) {
      const auto name = nb::cast< std::string >(value);
      if(const auto result = nor::meta::enum_from_name< Enum >(name); result)
         return *result;
      throw nb::value_error((std::string("unknown ") + description + " name: " + name).c_str());
   }
   try {
      return nb::cast< Enum >(value);
   } catch(const nb::cast_error&) {
      throw nb::type_error((std::string("expected ") + description + " enum or name").c_str());
   }
}

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
   StaleViewError() : std::logic_error("policy view is stale; obtain a new view after mutation") {}
};

class ReentrantSessionError final: public std::logic_error {
  public:
   ReentrantSessionError() : std::logic_error("same-session reentrant operation is not permitted")
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
class PythonValue {
  public:
   using dynamic_type = dynamic_value_t< Kind >;

   PythonValue()
   {
      if constexpr(Kind == rt::ValueKind::info_state or Kind == rt::ValueKind::public_state or Kind == rt::ValueKind::world_state)
         m_value.template emplace< dynamic_type >();
      else
         m_value.template emplace< rt::ErasedValue >();
   }

   PythonValue(
      std::string type_name,
      std::string identity,
      std::optional< std::string > text,
      std::optional< rt::TensorData > tensor
   )
   {
      if constexpr(Kind == rt::ValueKind::action or Kind == rt::ValueKind::chance_outcome or Kind == rt::ValueKind::observation) {
         m_value = dynamic_type::named(
            std::move(type_name), std::move(identity), std::move(text), std::move(tensor)
         );
      } else if constexpr(Kind == rt::ValueKind::world_state) {
         m_value = dynamic_type{rt::DynamicWorldValue::named(
            std::move(type_name), std::move(identity), std::move(text), std::move(tensor)
         )};
      } else {
         throw nb::type_error("state values are constructed with their state-specific API");
      }
   }

   explicit PythonValue(nor::Player player)
   {
      if constexpr(Kind == rt::ValueKind::info_state)
         m_value = dynamic_type{player};
      else {
         (void) player;
         throw nb::type_error("only InfoState accepts a player constructor");
      }
   }

   [[nodiscard]] static PythonValue named(
      std::string type_name,
      std::string identity,
      std::optional< std::string > text = std::nullopt,
      std::optional< rt::TensorData > tensor = std::nullopt
   )
   {
      return PythonValue{
         std::move(type_name), std::move(identity), std::move(text), std::move(tensor)};
   }

   [[nodiscard]] static PythonValue from_static(rt::ErasedValue value)
   {
      if(value.valid() and value.kind() != Kind)
         throw std::invalid_argument("erased value has the wrong domain kind");
      PythonValue result;
      result.m_value = std::move(value);
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
      if constexpr(Kind == rt::ValueKind::info_state or Kind == rt::ValueKind::public_state)
         return true;
      else if constexpr(Kind == rt::ValueKind::world_state)
         return std::get< dynamic_type >(m_value).value().valid();
      else
         return std::get< dynamic_type >(m_value).valid();
   }
   [[nodiscard]] explicit operator bool() const noexcept { return valid(); }

   [[nodiscard]] std::string type_name() const
   {
      if(const auto* value = std::get_if< rt::ErasedValue >(&m_value))
         return std::string(value->type_name());
      if constexpr(Kind == rt::ValueKind::world_state)
         return std::string(std::get< dynamic_type >(m_value).value().type_name());
      if constexpr(Kind == rt::ValueKind::action or Kind == rt::ValueKind::chance_outcome or Kind == rt::ValueKind::observation)
         return std::string(std::get< dynamic_type >(m_value).type_name());
      return {};
   }
   [[nodiscard]] std::string name() const { return type_name(); }

   [[nodiscard]] std::optional< std::string > identity() const
   {
      if(std::holds_alternative< rt::ErasedValue >(m_value))
         return std::nullopt;
      if constexpr(Kind == rt::ValueKind::world_state)
         return std::string(std::get< dynamic_type >(m_value).value().identity());
      if constexpr(Kind == rt::ValueKind::action or Kind == rt::ValueKind::chance_outcome or Kind == rt::ValueKind::observation)
         return std::string(std::get< dynamic_type >(m_value).identity());
      return std::nullopt;
   }

   [[nodiscard]] std::optional< std::string > to_string() const
   {
      if(const auto* value = std::get_if< rt::ErasedValue >(&m_value))
         return value->to_string();
      if constexpr(Kind == rt::ValueKind::world_state)
         return std::get< dynamic_type >(m_value).value().to_string();
      if constexpr(Kind == rt::ValueKind::action or Kind == rt::ValueKind::chance_outcome or Kind == rt::ValueKind::observation)
         return std::get< dynamic_type >(m_value).to_string();
      return std::nullopt;
   }

   [[nodiscard]] std::optional< rt::TensorData > to_tensor() const
   {
      if(const auto* value = std::get_if< rt::ErasedValue >(&m_value))
         return value->to_tensor();
      if constexpr(Kind == rt::ValueKind::world_state)
         return std::get< dynamic_type >(m_value).value().to_tensor();
      if constexpr(Kind == rt::ValueKind::action or Kind == rt::ValueKind::chance_outcome or Kind == rt::ValueKind::observation)
         return std::get< dynamic_type >(m_value).to_tensor();
      return std::nullopt;
   }

   [[nodiscard]] rt::ValueCapabilities capabilities() const
   {
      if(const auto* value = std::get_if< rt::ErasedValue >(&m_value)) {
         if(const auto* metadata = value->metadata())
            return metadata->capabilities;
         return {};
      }
      if constexpr(Kind == rt::ValueKind::world_state) {
         const auto& value = std::get< dynamic_type >(m_value).value();
         return {
            .to_string = value.to_string().has_value(), .to_tensor = value.to_tensor().has_value()};
      } else if constexpr(Kind == rt::ValueKind::action or Kind == rt::ValueKind::chance_outcome or Kind == rt::ValueKind::observation) {
         const auto& value = std::get< dynamic_type >(m_value);
         return {
            .to_string = value.to_string().has_value(), .to_tensor = value.to_tensor().has_value()};
      } else {
         return {};
      }
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

   [[nodiscard]] const rt::ErasedValue& erased() const
   {
      const auto* value = std::get_if< rt::ErasedValue >(&m_value);
      if(value == nullptr)
         throw std::invalid_argument("dynamic value cannot be used as a static erased value");
      return *value;
   }

   [[nodiscard]] const dynamic_type& dynamic() const
   {
      const auto* value = std::get_if< dynamic_type >(&m_value);
      if(value == nullptr)
         throw std::invalid_argument("static value cannot be used as a dynamic provider value");
      return *value;
   }

   void update(
      const PythonValue< rt::ValueKind::observation >& public_observation,
      const PythonValue< rt::ValueKind::observation >& private_observation
   )
   {
      if constexpr(Kind == rt::ValueKind::info_state) {
         if(not is_dynamic())
            throw std::invalid_argument("static InfoState values are immutable");
         std::get< dynamic_type >(m_value).update(
            public_observation.dynamic(), private_observation.dynamic()
         );
      } else {
         (void) public_observation;
         (void) private_observation;
         throw nb::type_error("update(public, private) is only valid for InfoState");
      }
   }

   void update(const PythonValue< rt::ValueKind::observation >& observation)
   {
      if constexpr(Kind == rt::ValueKind::public_state) {
         if(not is_dynamic())
            throw std::invalid_argument("static PublicState values are immutable");
         std::get< dynamic_type >(m_value).update(observation.dynamic());
      } else {
         (void) observation;
         throw nb::type_error("update(observation) is only valid for PublicState");
      }
   }

   void set(const PythonValue< rt::ValueKind::world_state >& value)
   {
      if constexpr(Kind == rt::ValueKind::world_state) {
         if(not value.is_dynamic())
            throw std::invalid_argument("WorldState.set requires a dynamic WorldState");
         m_value = value.dynamic();
      } else {
         (void) value;
         throw nb::type_error("set(WorldState) is only valid for WorldState");
      }
   }

   [[nodiscard]] nor::Player player() const
   {
      if constexpr(Kind == rt::ValueKind::info_state) {
         if(not is_dynamic())
            throw std::invalid_argument("static InfoState does not expose a player");
         return std::get< dynamic_type >(m_value).player();
      } else {
         throw nb::type_error("player is only valid for InfoState");
      }
   }

   [[nodiscard]] size_t size() const
   {
      if constexpr(Kind == rt::ValueKind::info_state or Kind == rt::ValueKind::public_state) {
         if(not is_dynamic())
            throw std::invalid_argument("static state history is not exposed");
         return std::get< dynamic_type >(m_value).size();
      } else {
         throw nb::type_error("size is only valid for state histories");
      }
   }

   [[nodiscard]] PythonValue< rt::ValueKind::observation > observation_at(size_t index) const
   {
      if constexpr(Kind == rt::ValueKind::public_state)
         return PythonValue< rt::ValueKind::observation >::from_dynamic(
            std::get< dynamic_type >(m_value)[index]
         );
      else {
         (void) index;
         throw nb::type_error("indexed observations are only valid for PublicState");
      }
   }

   [[nodiscard]] std::
      pair< PythonValue< rt::ValueKind::observation >, PythonValue< rt::ValueKind::observation > >
      info_at(size_t index) const
   {
      if constexpr(Kind == rt::ValueKind::info_state) {
         const auto& [public_observation, private_observation] = std::get< dynamic_type >(m_value
         )[index];
         return {
            PythonValue< rt::ValueKind::observation >::from_dynamic(public_observation),
            PythonValue< rt::ValueKind::observation >::from_dynamic(private_observation)};
      } else {
         (void) index;
         throw nb::type_error("indexed pairs are only valid for InfoState");
      }
   }

   [[nodiscard]] PythonValue< rt::ValueKind::observation > latest_observation() const
   {
      if constexpr(Kind == rt::ValueKind::public_state)
         return PythonValue< rt::ValueKind::observation >::from_dynamic(
            std::get< dynamic_type >(m_value).latest()
         );
      else
         throw nb::type_error("latest is only valid for PublicState");
   }

   [[nodiscard]] std::
      pair< PythonValue< rt::ValueKind::observation >, PythonValue< rt::ValueKind::observation > >
      latest_info() const
   {
      if constexpr(Kind == rt::ValueKind::info_state) {
         const auto& [public_observation, private_observation] = std::get< dynamic_type >(m_value)
                                                                    .latest();
         return {
            PythonValue< rt::ValueKind::observation >::from_dynamic(public_observation),
            PythonValue< rt::ValueKind::observation >::from_dynamic(private_observation)};
      } else {
         throw nb::type_error("latest is only valid for InfoState");
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

template < typename Value >
[[nodiscard]] std::vector< Value > python_vector(nb::handle value)
{
   std::vector< Value > result;
   for(const nb::handle item : nb::borrow< nb::iterable >(value))
      result.push_back(nb::cast< Value >(item));
   return result;
}

template < typename Value >
[[nodiscard]] std::vector< typename Value::dynamic_type > dynamic_vector(nb::handle value)
{
   std::vector< typename Value::dynamic_type > result;
   for(const auto& item : python_vector< Value >(value)) {
      if(not item.is_dynamic())
         throw nb::value_error("dynamic providers must return owned dynamic domain values");
      result.push_back(item.dynamic());
   }
   return result;
}

[[nodiscard]] rt::DynamicWorldState dynamic_world(nb::handle value)
{
   const auto state = nb::cast< PythonWorldState >(value);
   if(not state.is_dynamic())
      throw nb::value_error("dynamic provider must return a dynamic WorldState");
   return state.dynamic();
}

[[nodiscard]] rt::DynamicObservation dynamic_observation(nb::handle value)
{
   const auto observation = nb::cast< PythonObservation >(value);
   if(not observation.is_dynamic())
      throw nb::value_error("dynamic provider must return a dynamic Observation");
   return observation.dynamic();
}

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
         throw std::runtime_error(std::string("Python provider must implement ") + name);
      return nb::none();
   }
   return trampoline.base().attr(ticket.key)(std::forward< decltype(args) >(args)...);
}

class PyDynamicEnvironmentProvider final: public rt::DynamicEnvironmentProvider {
  public:
   NB_TRAMPOLINE(rt::DynamicEnvironmentProvider, 24);

   [[nodiscard]] size_t max_player_count() const final
   {
      nb::gil_scoped_acquire acquire;
      return nb::cast< size_t >(call_override(nb_trampoline, "max_player_count", true));
   }
   [[nodiscard]] size_t player_count() const final
   {
      nb::gil_scoped_acquire acquire;
      return nb::cast< size_t >(call_override(nb_trampoline, "player_count", true));
   }
   [[nodiscard]] nor::Stochasticity stochasticity() const final
   {
      nb::gil_scoped_acquire acquire;
      return nb::cast< nor::Stochasticity >(call_override(nb_trampoline, "stochasticity", true));
   }
   [[nodiscard]] bool serialized() const final
   {
      nb::gil_scoped_acquire acquire;
      const auto result = call_override(nb_trampoline, "serialized", false);
      return result.is_none() ? true : nb::cast< bool >(result);
   }
   [[nodiscard]] bool unrolled() const final
   {
      nb::gil_scoped_acquire acquire;
      const auto result = call_override(nb_trampoline, "unrolled", false);
      return result.is_none() ? true : nb::cast< bool >(result);
   }
   [[nodiscard]] rt::DynamicWorldState initial_world_state() const final
   {
      nb::gil_scoped_acquire acquire;
      return dynamic_world(call_override(nb_trampoline, "initial_world_state", true));
   }
   [[nodiscard]] std::vector< rt::DynamicAction >
   actions(nor::Player player, const rt::DynamicWorldState& state) const final
   {
      nb::gil_scoped_acquire acquire;
      return dynamic_vector< PythonAction >(call_override(
         nb_trampoline, "actions", true, player, PythonWorldState::from_dynamic(state)
      ));
   }
   [[nodiscard]] std::vector< nor::Player > players(const rt::DynamicWorldState& state) const final
   {
      nb::gil_scoped_acquire acquire;
      return python_vector< nor::Player >(
         call_override(nb_trampoline, "players", true, PythonWorldState::from_dynamic(state))
      );
   }
   [[nodiscard]] nor::Player active_player(const rt::DynamicWorldState& state) const final
   {
      nb::gil_scoped_acquire acquire;
      return nb::cast< nor::Player >(
         call_override(nb_trampoline, "active_player", true, PythonWorldState::from_dynamic(state))
      );
   }
   [[nodiscard]] bool is_terminal(const rt::DynamicWorldState& state) const final
   {
      nb::gil_scoped_acquire acquire;
      return nb::cast< bool >(
         call_override(nb_trampoline, "is_terminal", true, PythonWorldState::from_dynamic(state))
      );
   }
   [[nodiscard]] bool is_partaking(const rt::DynamicWorldState& state, nor::Player player)
      const final
   {
      nb::gil_scoped_acquire acquire;
      return nb::cast< bool >(call_override(
         nb_trampoline, "is_partaking", true, PythonWorldState::from_dynamic(state), player
      ));
   }
   [[nodiscard]] double reward(nor::Player player, const rt::DynamicWorldState& state) const final
   {
      nb::gil_scoped_acquire acquire;
      return nb::cast< double >(
         call_override(nb_trampoline, "reward", true, player, PythonWorldState::from_dynamic(state))
      );
   }

   void transition(rt::DynamicWorldState& state, const rt::DynamicAction& action) const final
   {
      nb::gil_scoped_acquire acquire;
      auto result = call_override(
         nb_trampoline,
         "transition_action",
         true,
         PythonWorldState::from_dynamic(state),
         PythonAction::from_dynamic(action)
      );
      apply_transition_result(state, result);
   }

   [[nodiscard]] rt::DynamicObservation private_observation(
      nor::Player player,
      const rt::DynamicWorldState& state,
      const rt::DynamicAction& action,
      const rt::DynamicWorldState& next_state
   ) const final
   {
      nb::gil_scoped_acquire acquire;
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
      nb::gil_scoped_acquire acquire;
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
      nb::gil_scoped_acquire acquire;
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
      nb::gil_scoped_acquire acquire;
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
      nb::gil_scoped_acquire acquire;
      const auto result = call_override(
         nb_trampoline,
         "transition_chance",
         false,
         PythonWorldState::from_dynamic(state),
         PythonChanceOutcome::from_dynamic(outcome)
      );
      if(result.is_none())
         return rt::DynamicEnvironmentProvider::transition(state, outcome);
      apply_transition_result(state, result);
   }
   [[nodiscard]] rt::DynamicObservation private_observation(
      nor::Player player,
      const rt::DynamicWorldState& state,
      const rt::DynamicChanceOutcome& outcome,
      const rt::DynamicWorldState& next_state
   ) const final
   {
      nb::gil_scoped_acquire acquire;
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
      nb::gil_scoped_acquire acquire;
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

  private:
   static void apply_transition_result(rt::DynamicWorldState& state, const nb::object& result)
   {
      if(not result.is_none())
         state = dynamic_world(result);
   }
};

struct SessionState {
   explicit SessionState(rt::SolverSession session, bool dynamic_game)
       : solver(std::move(session)), dynamic(dynamic_game)
   {
   }

   ~SessionState()
   {
      std::lock_guard lock(mutex);
      closed = true;
      solver = rt::SolverSession{};
   }

   mutable std::mutex mutex;
   rt::SolverSession solver;
   bool dynamic = false;
   size_t epoch = 0;
   bool closed = false;
};

thread_local std::vector< SessionState* > active_sessions;

class SessionOperation {
  public:
   SessionOperation(std::shared_ptr< SessionState > state, bool mutation)
       : m_state(std::move(state)), m_mutation(mutation)
   {
      if(not m_state or m_state->closed or not m_state->solver)
         throw std::runtime_error("session is closed");
      if(std::ranges::find(active_sessions, m_state.get()) != active_sessions.end())
         throw ReentrantSessionError{};

      // Static work releases the GIL before it can block on the binding-owned mutex. Dynamic work
      // keeps it so provider callbacks can execute safely.
      if(not m_state->dynamic)
         m_release.emplace();
      m_lock = std::unique_lock< std::mutex >(m_state->mutex);
      if(m_state->closed or not m_state->solver)
         throw std::runtime_error("session is closed");
      active_sessions.push_back(m_state.get());
   }

   ~SessionOperation()
   {
      if(not active_sessions.empty() and active_sessions.back() == m_state.get())
         active_sessions.pop_back();
   }

   [[nodiscard]] rt::SolverSession& solver() noexcept { return m_state->solver; }
   [[nodiscard]] const std::shared_ptr< SessionState >& state() const noexcept { return m_state; }
   void changed() noexcept
   {
      if(m_mutation)
         ++m_state->epoch;
   }
   void require_epoch(size_t epoch) const
   {
      if(m_state->closed or m_state->epoch != epoch)
         throw StaleViewError{};
   }

  private:
   std::shared_ptr< SessionState > m_state;
   bool m_mutation = false;
   std::optional< nb::gil_scoped_release > m_release;
   std::unique_lock< std::mutex > m_lock;
};

class PythonPolicyRow;

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

   [[nodiscard]] bool valid() const
   {
      auto operation = lock();
      return m_lookup.valid();
   }
   [[nodiscard]] rt::PolicyViewKind kind() const noexcept { return m_kind; }
   [[nodiscard]] size_t generation() const
   {
      auto operation = lock();
      return m_lookup.generation();
   }
   [[nodiscard]] std::optional< PythonPolicyRow > find(const PythonInfoState& info_state) const;
   [[nodiscard]] PythonPolicyRow at(const PythonInfoState& info_state) const;
   [[nodiscard]] std::vector< rt::PolicyEntry > entries() const
   {
      auto operation = lock();
      return m_lookup.to_entries();
   }

  private:
   [[nodiscard]] SessionOperation lock() const
   {
      SessionOperation operation(m_state.lock(), false);
      operation.require_epoch(m_epoch);
      if(not m_lookup.valid())
         throw StaleViewError{};
      return operation;
   }

   std::weak_ptr< SessionState > m_state;
   rt::PolicyLookup m_lookup;
   rt::PolicyViewKind m_kind = rt::PolicyViewKind::current;
   size_t m_epoch = 0;
};

class PythonPolicyRow {
  public:
   PythonPolicyRow(std::weak_ptr< SessionState > state, rt::PolicyNodeView row, size_t epoch)
       : m_state(std::move(state)), m_row(std::move(row)), m_epoch(epoch)
   {
   }

   [[nodiscard]] bool valid() const
   {
      auto operation = lock();
      return m_row.valid();
   }
   [[nodiscard]] rt::PolicyViewKind kind() const
   {
      auto operation = lock();
      return m_row.kind();
   }
   [[nodiscard]] size_t generation() const
   {
      auto operation = lock();
      return m_row.generation();
   }
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
      if(not action.is_dynamic())
         return m_row.find(action.erased());
      return std::nullopt;
   }
   [[nodiscard]] bool contains(const PythonAction& action) const
   {
      auto operation = lock();
      return not action.is_dynamic() and m_row.contains(action.erased());
   }
   [[nodiscard]] double at(const PythonAction& action) const
   {
      auto operation = lock();
      if(action.is_dynamic())
         throw std::out_of_range("policy row does not contain this dynamic action");
      return m_row.at(action.erased());
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
      SessionOperation operation(m_state.lock(), false);
      operation.require_epoch(m_epoch);
      if(not m_row.valid())
         throw StaleViewError{};
      return operation;
   }

   std::weak_ptr< SessionState > m_state;
   rt::PolicyNodeView m_row;
   size_t m_epoch = 0;
};

std::optional< PythonPolicyRow > PythonPolicy::find(const PythonInfoState& info_state) const
{
   auto operation = lock();
   if(info_state.is_dynamic())
      return std::nullopt;
   auto row = m_lookup.find(info_state.erased());
   if(not row)
      return std::nullopt;
   return PythonPolicyRow{m_state, std::move(*row), m_epoch};
}

PythonPolicyRow PythonPolicy::at(const PythonInfoState& info_state) const
{
   auto result = find(info_state);
   if(not result)
      throw std::out_of_range("information state is not present in this policy");
   return std::move(*result);
}

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
      operation.changed();
   }
   [[nodiscard]] std::optional< rt::IterationResult > advance_last(size_t iterations)
   {
      SessionOperation operation(m_state, true);
      auto result = unwrap(operation.solver().advance_last(iterations));
      operation.changed();
      return result;
   }
   [[nodiscard]] rt::TraceResult trace(size_t iterations, size_t every)
   {
      SessionOperation operation(m_state, true);
      auto result = unwrap(operation.solver().trace(iterations, every));
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
      return PythonPolicy{m_state, std::move(lookup), kind, operation.state()->epoch};
   }
   [[nodiscard]] bool closed() const noexcept
   {
      const auto state = m_state;
      return not state or state->closed;
   }

  private:
   std::shared_ptr< SessionState > m_state;
};

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
   [[nodiscard]] std::string name() const
   {
      if(is_dynamic())
         return "dynamic";
      return enum_name(id());
   }
   [[nodiscard]] rt::GameSpec spec() const
   {
      return std::visit([](const auto& game) { return game.spec(); }, m_game);
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

   [[nodiscard]] std::shared_ptr< PythonSession > make_session(
      nb::object solver_or_profile,
      nb::object profile,
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
            throw nb::value_error("unknown profile");
         solver = descriptor->solver;
      } else {
         solver = enum_from_python< rt::SolverId >(solver_or_profile, "SolverId");
         selected_profile = enum_from_python< rt::ProfileId >(profile, "ProfileId");
      }

      const rt::SessionOptions options{.epsilon = epsilon, .seed = seed};
      rt::Result< rt::SolverSession > result = std::visit(
         [&](const auto& game) -> rt::Result< rt::SolverSession > {
            using game_type = std::remove_cvref_t< decltype(game) >;
            if constexpr(std::is_same_v< game_type, rt::GameHandle >)
               return rt::make_session(game, solver, selected_profile, options);
            else
               return game.make_session(solver, selected_profile, options);
         },
         m_game
      );
      auto session = unwrap(std::move(result));
      return std::make_shared< PythonSession >(
         std::make_shared< SessionState >(std::move(session), is_dynamic())
      );
   }

  private:
   std::variant< rt::GameHandle, rt::DynamicGameHandle > m_game;
};

[[nodiscard]] rt::SpecValue spec_value(nb::handle value)
{
   if(PyBool_Check(value.ptr()))
      return nb::cast< bool >(value);
   if(PyLong_Check(value.ptr())) {
      const auto signed_value = PyLong_AsLongLong(value.ptr());
      if(signed_value == -1 and PyErr_Occurred()) {
         PyErr_Clear();
         throw nb::value_error("integer GameSpec fields must fit in a signed 64-bit value");
      }
      if(signed_value < 0)
         return static_cast< double >(signed_value);
      return static_cast< uint64_t >(signed_value);
   }
   if(PyFloat_Check(value.ptr()))
      return nb::cast< double >(value);
   throw nb::type_error("GameSpec fields must be bool, int, or float");
}

template < typename Enum >
[[nodiscard]] Enum field_enum(nb::handle value, const char* description)
{
   return enum_from_python< Enum >(value, description);
}

[[nodiscard]] rt::GameFieldId field_from_python(nb::handle value)
{
   return field_enum< rt::GameFieldId >(value, "GameFieldId");
}

void apply_kwargs(rt::GameSpec& spec, const nb::kwargs& fields)
{
   for(const nb::handle item : fields.items()) {
      const auto pair = nb::cast< nb::tuple >(item);
      spec.set(field_from_python(pair[0]), spec_value(pair[1]));
   }
}

[[nodiscard]] rt::GameSpec static_spec(rt::GameId id, const nb::kwargs& fields)
{
   auto spec = rt::GameSpec::defaults(id);
   apply_kwargs(spec, fields);
   return spec;
}

void construct_static_game(PythonGame* self, rt::GameId id, const nb::kwargs& fields)
{
   new(self) PythonGame(unwrap(rt::make_game(static_spec(id, fields))));
}

void construct_named_game(PythonGame* self, const std::string& name, const nb::kwargs& fields)
{
   const auto id = enum_from_python< rt::GameId >(nb::str(name), "GameId");
   construct_static_game(self, id, fields);
}

void construct_dynamic_game(
   PythonGame* self,
   const std::shared_ptr< PyDynamicEnvironmentProvider >& provider,
   const nb::kwargs& fields
)
{
   if(not provider)
      throw nb::value_error("dynamic Game requires a provider");
   auto spec = rt::GameSpec{rt::GameId::dynamic};
   apply_kwargs(spec, fields);
   std::shared_ptr< const rt::DynamicEnvironmentProvider > const_provider = provider;
   new(self) PythonGame(unwrap(rt::make_dynamic_game(std::move(spec), std::move(const_provider))));
}

class PythonGameSpec {
  public:
   explicit PythonGameSpec(rt::GameId id) : m_spec(id) {}
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
   PythonGameSpec& set(nb::handle field, nb::handle value)
   {
      m_spec.set(field_from_python(field), spec_value(value));
      return *this;
   }
   [[nodiscard]] rt::GameSpec copy() const { return m_spec; }

  private:
   rt::GameSpec m_spec;
};

class PythonCatalog {
  public:
   [[nodiscard]] std::vector< rt::GameDescriptor > games() const
   {
      return {rt::games().begin(), rt::games().end()};
   }
   [[nodiscard]] std::vector< rt::SolverDescriptor > solvers() const
   {
      return {rt::solvers().begin(), rt::solvers().end()};
   }
   [[nodiscard]] std::vector< rt::ProfileDescriptor > profiles() const
   {
      return {rt::profiles().begin(), rt::profiles().end()};
   }
   [[nodiscard]] std::vector< rt::CapabilityDescriptor > combinations() const
   {
      return {rt::capabilities().begin(), rt::capabilities().end()};
   }
};

PyObject* capability_exception_type = nullptr;

void translate_capability(const std::exception_ptr& pointer, void* payload)
{
   try {
      std::rethrow_exception(pointer);
   } catch(const BindingCapabilityError& error) {
      auto* type = static_cast< PyObject* >(payload);
      auto* instance = PyObject_CallFunction(type, "s", error.what());
      if(instance == nullptr)
         return;
      const auto& context = error.error();
      nb::setattr(nb::handle(instance), "code", nb::cast(context.code));
      nb::setattr(nb::handle(instance), "game", nb::cast(context.game));
      nb::setattr(nb::handle(instance), "solver", nb::cast(context.solver));
      nb::setattr(nb::handle(instance), "profile", nb::cast(context.profile));
      PyErr_SetObject(type, instance);
      Py_DECREF(instance);
   }
}

template < typename Enum >
void bind_enum(nb::module_& module, const char* name)
{
   auto enumeration = nb::enum_< Enum >(module, name);
   for(const auto entry : nor::meta::detail::make_enum_entries< Enum >())
      enumeration.value(std::string(entry.name).c_str(), entry.value);
}

template < rt::ValueKind Kind >
void bind_domain_value(nb::module_& module, const char* name)
{
   using value_type = PythonValue< Kind >;
   auto value = nb::class_< value_type >(module, name);
   value.def(nb::init<>());
   if constexpr(Kind == rt::ValueKind::action or Kind == rt::ValueKind::chance_outcome or Kind == rt::ValueKind::observation or Kind == rt::ValueKind::world_state) {
      value.def(
         nb::init<
            std::string,
            std::string,
            std::optional< std::string >,
            std::optional< rt::TensorData > >(),
         "type_name"_a,
         "identity"_a,
         "text"_a = std::nullopt,
         "tensor"_a = std::nullopt
      );
      value.def_static(
         "named",
         [](std::string type_name,
            std::string identity,
            std::optional< std::string > text,
            std::optional< rt::TensorData > tensor) {
            return value_type::named(
               std::move(type_name), std::move(identity), std::move(text), std::move(tensor)
            );
         },
         "type_name"_a,
         "identity"_a,
         "text"_a = std::nullopt,
         "tensor"_a = std::nullopt
      );
   }
   if constexpr(Kind == rt::ValueKind::info_state)
      value.def(nb::init< nor::Player >(), "player"_a = nor::Player::unknown);
   value.def_prop_ro("valid", &value_type::valid)
      .def_prop_ro("is_dynamic", &value_type::is_dynamic)
      .def_prop_ro("kind", &value_type::kind)
      .def_prop_ro("type_name", &value_type::type_name)
      .def_prop_ro("type", &value_type::type_name)
      .def_prop_ro("name", &value_type::name)
      .def_prop_ro("identity", &value_type::identity)
      .def("to_string", &value_type::to_string)
      .def("to_tensor", &value_type::to_tensor)
      .def_prop_ro("capabilities", &value_type::capabilities)
      .def("__bool__", &value_type::valid)
      .def("__hash__", &value_type::hash)
      .def("__eq__", [](const value_type& left, const value_type& right) { return left == right; });

   if constexpr(Kind == rt::ValueKind::info_state) {
      value.def_prop_ro("player", &value_type::player)
         .def("update", &value_type::update, "public_observation"_a, "private_observation"_a)
         .def("__len__", &value_type::size)
         .def("__getitem__", &value_type::info_at)
         .def("latest", &value_type::latest_info);
   } else if constexpr(Kind == rt::ValueKind::public_state) {
      value.def("update", &value_type::update, "observation"_a)
         .def("__len__", &value_type::size)
         .def("__getitem__", &value_type::observation_at)
         .def("latest", &value_type::latest_observation);
   } else if constexpr(Kind == rt::ValueKind::world_state) {
      value.def("set", &value_type::set);
   }
}

}  // namespace noregret_binding

NB_MODULE(_noregret, module)
{
   using namespace noregret_binding;

   bind_enum< nor::Player >(module, "Player");
   bind_enum< nor::Stochasticity >(module, "Stochasticity");
   bind_enum< rt::GameId >(module, "GameId");
   bind_enum< rt::SolverId >(module, "SolverId");
   bind_enum< rt::ProfileId >(module, "ProfileId");
   bind_enum< rt::GameFieldId >(module, "GameFieldId");
   bind_enum< rt::SpecKind >(module, "SpecKind");
   bind_enum< rt::ValueKind >(module, "ValueKind");
   bind_enum< rt::PolicyViewKind >(module, "PolicyViewKind");
   bind_enum< rt::CapabilityErrorCode >(module, "CapabilityErrorCode");

   nb::class_< rt::TensorData >(module, "TensorData")
      .def(nb::init<>())
      .def_rw("values", &rt::TensorData::values)
      .def_rw("shape", &rt::TensorData::shape);
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
   module.attr("DynamicAction") = module.attr("Action");
   module.attr("DynamicChanceOutcome") = module.attr("ChanceOutcome");
   module.attr("DynamicObservation") = module.attr("Observation");
   module.attr("DynamicInfoState") = module.attr("InfoState");
   module.attr("DynamicPublicState") = module.attr("PublicState");
   module.attr("DynamicWorldState") = module.attr("WorldState");

   nb::class_< rt::SpecField >(module, "SpecField")
      .def(nb::init<>())
      .def_rw("id", &rt::SpecField::id)
      .def_prop_ro("value", [](const rt::SpecField& field) { return field.value; });
   nb::class_< PythonGameSpec >(module, "GameSpec")
      .def(nb::init< rt::GameId >(), "game"_a)
      .def_prop_ro("game", &PythonGameSpec::game_id)
      .def_prop_ro("game_id", &PythonGameSpec::game_id)
      .def_prop_ro("fields", &PythonGameSpec::fields)
      .def("contains", &PythonGameSpec::contains, "field"_a)
      .def("find", &PythonGameSpec::find, "field"_a)
      .def("set", &PythonGameSpec::set, "field"_a, "value"_a, nb::rv_policy::reference_internal)
      .def("copy", &PythonGameSpec::copy)
      .def_static("defaults", [](rt::GameId id) {
         return PythonGameSpec{rt::GameSpec::defaults(id)};
      });

   nb::class_< rt::FieldDescriptor >(module, "FieldDescriptor")
      .def_prop_ro("id", [](const rt::FieldDescriptor& value) { return value.id; })
      .def_prop_ro("name", [](const rt::FieldDescriptor& value) { return std::string(value.name); })
      .def_prop_ro("kind", [](const rt::FieldDescriptor& value) { return value.kind; })
      .def_prop_ro("default_value", [](const rt::FieldDescriptor& value) {
         return value.default_value;
      });
   nb::class_< rt::GameDescriptor >(module, "GameDescriptor")
      .def_prop_ro("id", &rt::GameDescriptor::id)
      .def_prop_ro("name", [](const rt::GameDescriptor& value) { return std::string(value.name); })
      .def_prop_ro("min_players", &rt::GameDescriptor::min_players)
      .def_prop_ro("max_players", &rt::GameDescriptor::max_players)
      .def_prop_ro("stochasticity", &rt::GameDescriptor::stochasticity)
      .def_prop_ro("fields", [](const rt::GameDescriptor& value) {
         return std::vector< rt::FieldDescriptor >{value.fields.begin(), value.fields.end()};
      });
   nb::class_< rt::SolverDescriptor >(module, "SolverDescriptor")
      .def_prop_ro("id", &rt::SolverDescriptor::id)
      .def_prop_ro("name", [](const rt::SolverDescriptor& value) {
         return std::string(value.name);
      });
   nb::class_< rt::ProfileDescriptor >(module, "ProfileDescriptor")
      .def_prop_ro("id", &rt::ProfileDescriptor::id)
      .def_prop_ro("solver", &rt::ProfileDescriptor::solver)
      .def_prop_ro("name", [](const rt::ProfileDescriptor& value) {
         return std::string(value.name);
      });
   nb::class_< rt::CapabilityDescriptor >(module, "CapabilityDescriptor")
      .def_prop_ro("game", &rt::CapabilityDescriptor::game)
      .def_prop_ro("solver", &rt::CapabilityDescriptor::solver)
      .def_prop_ro("profile", &rt::CapabilityDescriptor::profile);
   nb::class_< rt::DynamicCapabilityDescriptor >(module, "DynamicCapabilityDescriptor")
      .def_prop_ro(
         "game", [](const rt::DynamicCapabilityDescriptor&) { return rt::GameId::dynamic; }
      )
      .def_prop_ro("solver", &rt::DynamicCapabilityDescriptor::solver)
      .def_prop_ro("profile", &rt::DynamicCapabilityDescriptor::profile)
      .def_prop_ro("name", [](const rt::DynamicCapabilityDescriptor& value) {
         return std::string(value.name);
      });

   nb::class_< rt::RootValue >(module, "RootValue")
      .def(nb::init<>())
      .def_rw("player", &rt::RootValue::player)
      .def_rw("value", &rt::RootValue::value);
   nb::class_< rt::IterationResult >(module, "IterationResult")
      .def_prop_ro("iteration", [](const rt::IterationResult& value) { return value.iteration; })
      .def_prop_ro(
         "root_values", [](const rt::IterationResult& value) { return value.root_values; }
      )
      .def("__len__", [](const rt::IterationResult& value) { return value.root_values.size(); });
   nb::class_< rt::TraceResult >(module, "TraceResult")
      .def_prop_ro("first_iteration", &rt::TraceResult::first_iteration)
      .def_prop_ro("last_iteration", &rt::TraceResult::last_iteration)
      .def_prop_ro("iterations", [](const rt::TraceResult& value) { return value.iterations; })
      .def_prop_ro("empty", &rt::TraceResult::empty)
      .def("__len__", &rt::TraceResult::size)
      .def("__getitem__", [](const rt::TraceResult& value, size_t index) { return value[index]; });
   nb::class_< rt::SessionStats >(module, "SessionStats")
      .def_prop_ro("game", &rt::SessionStats::game)
      .def_prop_ro("solver", &rt::SessionStats::solver)
      .def_prop_ro("profile", &rt::SessionStats::profile)
      .def_prop_ro("iteration", &rt::SessionStats::iteration)
      .def_prop_ro("cycle", &rt::SessionStats::cycle)
      .def_prop_ro("player_count", &rt::SessionStats::player_count)
      .def_prop_ro("current_policy_entries", &rt::SessionStats::current_policy_entries)
      .def_prop_ro("average_policy_entries", &rt::SessionStats::average_policy_entries);

   nb::exception< BindingCapabilityError > capability_error(module, "CapabilityError");
   capability_exception_type = capability_error.ptr();
   nb::register_exception_translator(translate_capability, capability_exception_type);
   nb::exception< StaleViewError > stale_error(module, "StaleViewError");
   nb::exception< ReentrantSessionError > reentrant_error(module, "ReentrantSessionError");

   nb::class_< PythonPolicyRow >(module, "PolicyRow")
      .def_prop_ro("valid", &PythonPolicyRow::valid)
      .def_prop_ro("kind", &PythonPolicyRow::kind)
      .def_prop_ro("generation", &PythonPolicyRow::generation)
      .def_prop_ro("player", &PythonPolicyRow::player)
      .def_prop_ro("info_state", &PythonPolicyRow::info_state)
      .def_prop_ro("size", &PythonPolicyRow::size)
      .def("action_at", &PythonPolicyRow::action_at, "index"_a)
      .def("value_at", &PythonPolicyRow::value_at, "index"_a)
      .def("find", &PythonPolicyRow::find, "action"_a)
      .def("contains", &PythonPolicyRow::contains, "action"_a)
      .def("at", &PythonPolicyRow::at, "action"_a)
      .def(
         "to_entries",
         [](const PythonPolicyRow& value) {
            nb::list result;
            for(const auto& entry : value.entries())
               result.append(
                  nb::make_tuple(PythonAction::from_static(entry.action), entry.probability)
               );
            return result;
         }
      )
      .def("to_tensor", &PythonPolicyRow::tensor);
   nb::class_< PythonPolicy >(module, "Policy")
      .def_prop_ro("valid", &PythonPolicy::valid)
      .def_prop_ro("kind", &PythonPolicy::kind)
      .def_prop_ro("generation", &PythonPolicy::generation)
      .def("find", &PythonPolicy::find, "info_state"_a)
      .def("at", &PythonPolicy::at, "info_state"_a)
      .def("to_entries", [](const PythonPolicy& value) {
         nb::list result;
         for(const auto& entry : value.entries()) {
            nb::list actions;
            for(const auto& action : entry.actions)
               actions.append(
                  nb::make_tuple(PythonAction::from_static(action.action), action.probability)
               );
            result.append(
               nb::make_tuple(entry.player, PythonInfoState::from_static(entry.info_state), actions)
            );
         }
         return result;
      });

   nb::class_< PythonSession, std::shared_ptr< PythonSession > >(module, "Session")
      .def("iterate", &PythonSession::iterate)
      .def("advance", &PythonSession::advance, "n"_a)
      .def("advance_last", &PythonSession::advance_last, "n"_a)
      .def("trace", &PythonSession::trace, "n"_a, "every"_a = 1)
      .def("stats", &PythonSession::stats)
      .def("policy", &PythonSession::policy, "kind"_a = rt::PolicyViewKind::current)
      .def("policy_lookup", &PythonSession::policy, "kind"_a = rt::PolicyViewKind::current)
      .def_prop_ro("closed", &PythonSession::closed);

   nb::class_<
      PyDynamicEnvironmentProvider,
      rt::DynamicEnvironmentProvider,
      std::shared_ptr< PyDynamicEnvironmentProvider > >
      provider(module, "DynamicEnvironmentProvider");
   provider.def(nb::init<>())
      .def("max_player_count", &PyDynamicEnvironmentProvider::max_player_count)
      .def("player_count", &PyDynamicEnvironmentProvider::player_count)
      .def("stochasticity", &PyDynamicEnvironmentProvider::stochasticity)
      .def("serialized", &PyDynamicEnvironmentProvider::serialized)
      .def("unrolled", &PyDynamicEnvironmentProvider::unrolled)
      .def("initial_world_state", &PyDynamicEnvironmentProvider::initial_world_state)
      .def("actions", &PyDynamicEnvironmentProvider::actions)
      .def("players", &PyDynamicEnvironmentProvider::players)
      .def("active_player", &PyDynamicEnvironmentProvider::active_player)
      .def("is_terminal", &PyDynamicEnvironmentProvider::is_terminal)
      .def("is_partaking", &PyDynamicEnvironmentProvider::is_partaking)
      .def("reward", &PyDynamicEnvironmentProvider::reward)
      .def(
         "transition_action",
         [](const PyDynamicEnvironmentProvider& self, PythonWorldState state, PythonAction action) {
            self.transition(state.dynamic(), action.dynamic());
            return state;
         }
      )
      .def(
         "transition_chance",
         [](const PyDynamicEnvironmentProvider& self,
            PythonWorldState state,
            PythonChanceOutcome outcome) {
            self.transition(state.dynamic(), outcome.dynamic());
            return state;
         }
      )
      .def("private_observation_action", &PyDynamicEnvironmentProvider::private_observation)
      .def("public_observation_action", &PyDynamicEnvironmentProvider::public_observation)
      .def("chance_actions", &PyDynamicEnvironmentProvider::chance_actions)
      .def("chance_probability", &PyDynamicEnvironmentProvider::chance_probability)
      .def(
         "private_observation_chance",
         [](const PyDynamicEnvironmentProvider& self,
            nor::Player player,
            PythonWorldState state,
            PythonChanceOutcome outcome,
            PythonWorldState next_state) {
            return PythonObservation::from_dynamic(self.private_observation(
               player, state.dynamic(), outcome.dynamic(), next_state.dynamic()
            ));
         }
      )
      .def(
         "public_observation_chance",
         [](const PyDynamicEnvironmentProvider& self,
            PythonWorldState state,
            PythonChanceOutcome outcome,
            PythonWorldState next_state) {
            return PythonObservation::from_dynamic(
               self.public_observation(state.dynamic(), outcome.dynamic(), next_state.dynamic())
            );
         }
      );

   nb::class_< PythonGame >(module, "Game")
      .def("__init__", &construct_static_game, "game"_a)
      .def("__init__", &construct_named_game, "game"_a)
      .def("__init__", &construct_dynamic_game, "provider"_a)
      .def_static(
         "from_spec",
         [](const PythonGameSpec& spec) { return PythonGame{unwrap(rt::make_game(spec.copy()))}; }
      )
      .def_prop_ro("id", &PythonGame::id)
      .def_prop_ro("game_id", &PythonGame::id)
      .def_prop_ro("name", &PythonGame::name)
      .def_prop_ro("is_dynamic", &PythonGame::is_dynamic)
      .def_prop_ro("spec", &PythonGame::spec)
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

   nb::class_< PythonCatalog >(module, "Catalog")
      .def_prop_ro("games", &PythonCatalog::games)
      .def_prop_ro("solvers", &PythonCatalog::solvers)
      .def_prop_ro("profiles", &PythonCatalog::profiles)
      .def_prop_ro("combinations", &PythonCatalog::combinations);

   module.def("catalog", [] { return PythonCatalog{}; });
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
      "capabilities_for", [](rt::GameId game) { return rt::capabilities_for(game); }, "game"_a
   );
   module.def(
      "profiles_for", [](rt::SolverId solver) { return rt::profiles_for(solver); }, "solver"_a
   );
}
