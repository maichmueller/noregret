#ifndef NOR_BINDING_RUNTIME_DYNAMIC_HPP
#define NOR_BINDING_RUNTIME_DYNAMIC_HPP

#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "types.hpp"

namespace nor::binding::runtime {

/**
 * @brief Value-owned payload used by the dynamic environment boundary.
 *
 * A dynamic provider supplies stable type and identity strings rather than a Python object. The
 * strings are copied into the C++ value, so the provider boundary is crossed once per value and
 * the solver can retain ordinary C++ values in its static data structures.
 */
class DynamicValue {
  public:
   DynamicValue() = default;

   DynamicValue(
      std::string type_name,
      std::string identity,
      std::optional< std::string > text = std::nullopt,
      std::optional< TensorData > tensor = std::nullopt
   )
       : m_type_name(std::move(type_name)),
         m_identity(std::move(identity)),
         m_text(std::move(text)),
         m_tensor(std::move(tensor))
   {
   }

   [[nodiscard]] static DynamicValue named(
      std::string type_name,
      std::string identity,
      std::optional< std::string > text = std::nullopt,
      std::optional< TensorData > tensor = std::nullopt
   )
   {
      return DynamicValue{
         std::move(type_name), std::move(identity), std::move(text), std::move(tensor)};
   }

   [[nodiscard]] bool valid() const noexcept
   {
      return not m_type_name.empty() and not m_identity.empty();
   }
   [[nodiscard]] std::string_view type_name() const noexcept { return m_type_name; }
   [[nodiscard]] std::string_view name() const noexcept { return m_type_name; }
   [[nodiscard]] std::string_view identity() const noexcept { return m_identity; }
   [[nodiscard]] std::optional< std::string > to_string() const { return m_text; }
   [[nodiscard]] std::optional< TensorData > to_tensor() const { return m_tensor; }

   [[nodiscard]] size_t hash() const noexcept
   {
      // FNV-1a over provider-supplied stable names. std::hash is deliberately not used here:
      // dynamic values may be serialized or compared across provider instances.
      size_t result = size_t{1469598103934665603ull};
      const auto add = [&result](std::string_view text) {
         for(const char character : text) {
            result ^= static_cast< size_t >(static_cast< unsigned char >(character));
            result *= size_t{1099511628211ull};
         }
      };
      add(m_type_name);
      result ^= size_t{0xff};
      result *= size_t{1099511628211ull};
      add(m_identity);
      return result;
   }

   // Presentation is intentionally excluded from identity. Two values with the same provider
   // type and identity are the same solver key even when one has an optional display payload.
   friend bool operator==(const DynamicValue& left, const DynamicValue& right) noexcept
   {
      return left.m_type_name == right.m_type_name and left.m_identity == right.m_identity;
   }

  private:
   std::string m_type_name;
   std::string m_identity;
   std::optional< std::string > m_text;
   std::optional< TensorData > m_tensor;
};

template < ValueKind Kind >
class DynamicDomainValue {
  public:
   DynamicDomainValue() = default;
   explicit DynamicDomainValue(DynamicValue value) : m_value(std::move(value)) {}

   [[nodiscard]] static DynamicDomainValue named(
      std::string type_name,
      std::string identity,
      std::optional< std::string > text = std::nullopt,
      std::optional< TensorData > tensor = std::nullopt
   )
   {
      return DynamicDomainValue{DynamicValue::named(
         std::move(type_name), std::move(identity), std::move(text), std::move(tensor)
      )};
   }

   [[nodiscard]] static constexpr ValueKind kind() noexcept { return Kind; }
   [[nodiscard]] bool valid() const noexcept { return m_value.valid(); }
   [[nodiscard]] explicit operator bool() const noexcept { return valid(); }
   [[nodiscard]] std::string_view type_name() const noexcept { return m_value.type_name(); }
   [[nodiscard]] std::string_view name() const noexcept { return m_value.name(); }
   [[nodiscard]] std::string_view identity() const noexcept { return m_value.identity(); }
   [[nodiscard]] std::optional< std::string > to_string() const { return m_value.to_string(); }
   [[nodiscard]] std::optional< TensorData > to_tensor() const { return m_value.to_tensor(); }
   [[nodiscard]] size_t hash() const noexcept
   {
      return m_value.hash() ^ (static_cast< size_t >(Kind) + size_t{1}) * size_t{0x9e3779b9};
   }

   [[nodiscard]] const DynamicValue& value() const noexcept { return m_value; }
   friend bool operator==(const DynamicDomainValue&, const DynamicDomainValue&) = default;

  private:
   DynamicValue m_value;
};

using DynamicAction = DynamicDomainValue< ValueKind::action >;
using DynamicChanceOutcome = DynamicDomainValue< ValueKind::chance_outcome >;
using DynamicObservation = DynamicDomainValue< ValueKind::observation >;
using DynamicWorldValue = DynamicDomainValue< ValueKind::world_state >;

/** Value type used for the world-state object handed to the C++ solver. */
class DynamicWorldState {
  public:
   DynamicWorldState() = default;
   explicit DynamicWorldState(DynamicWorldValue value) : m_value(std::move(value)) {}

   [[nodiscard]] const DynamicWorldValue& value() const noexcept { return m_value; }
   [[nodiscard]] DynamicWorldValue& value() noexcept { return m_value; }
   void set(DynamicWorldValue value) { m_value = std::move(value); }

   friend bool operator==(const DynamicWorldState&, const DynamicWorldState&) = default;
   [[nodiscard]] size_t hash() const noexcept { return m_value.hash(); }

  private:
   DynamicWorldValue m_value;
};

/** Minimal FOSG-compatible dynamic information state. */
class DynamicInfoState {
  public:
   using observation_type = DynamicObservation;
   using history_entry = std::pair< observation_type, observation_type >;

   explicit DynamicInfoState(Player player = Player::unknown) : m_player(player) {}

   [[nodiscard]] Player player() const noexcept { return m_player; }
   [[nodiscard]] size_t size() const noexcept { return m_history.size(); }
   [[nodiscard]] const history_entry& operator[](size_t index) const { return m_history.at(index); }
   [[nodiscard]] const history_entry& latest() const
   {
      if(m_history.empty())
         throw std::out_of_range("DynamicInfoState::latest called on an empty history");
      return m_history.back();
   }
   void
   update(const observation_type& public_observation, const observation_type& private_observation)
   {
      m_history.emplace_back(public_observation, private_observation);
   }

   friend bool operator==(const DynamicInfoState&, const DynamicInfoState&) = default;

   [[nodiscard]] size_t hash() const noexcept
   {
      size_t result = static_cast< size_t >(static_cast< int >(m_player) + 3);
      for(const auto& [public_observation, private_observation] : m_history) {
         result ^= public_observation.hash() + size_t{0x9e3779b9} + (result << 6) + (result >> 2);
         result ^= private_observation.hash() + size_t{0x9e3779b9} + (result << 6) + (result >> 2);
      }
      return result;
   }

  private:
   Player m_player = Player::unknown;
   std::vector< history_entry > m_history;
};

/** Minimal FOSG-compatible dynamic public state. */
class DynamicPublicState {
  public:
   using observation_type = DynamicObservation;

   [[nodiscard]] size_t size() const noexcept { return m_history.size(); }
   [[nodiscard]] const observation_type& operator[](size_t index) const
   {
      return m_history.at(index);
   }
   [[nodiscard]] const observation_type& latest() const
   {
      if(m_history.empty())
         throw std::out_of_range("DynamicPublicState::latest called on an empty history");
      return m_history.back();
   }
   void update(const observation_type& observation) { m_history.push_back(observation); }

   friend bool operator==(const DynamicPublicState&, const DynamicPublicState&) = default;

   [[nodiscard]] size_t hash() const noexcept
   {
      size_t result = size_t{1469598103934665603ull};
      for(const auto& observation : m_history) {
         result ^= observation.hash() + size_t{0x9e3779b9} + (result << 6) + (result >> 2);
      }
      return result;
   }

  private:
   std::vector< observation_type > m_history;
};

}  // namespace nor::binding::runtime

namespace std {

template < nor::binding::runtime::ValueKind Kind >
struct hash< nor::binding::runtime::DynamicDomainValue< Kind > > {
   size_t operator()(const nor::binding::runtime::DynamicDomainValue< Kind >& value) const noexcept
   {
      return value.hash();
   }
};

template <>
struct hash< nor::binding::runtime::DynamicWorldState > {
   size_t operator()(const nor::binding::runtime::DynamicWorldState& value) const noexcept
   {
      return value.hash();
   }
};

template <>
struct hash< nor::binding::runtime::DynamicInfoState > {
   size_t operator()(const nor::binding::runtime::DynamicInfoState& value) const noexcept
   {
      return value.hash();
   }
};

template <>
struct hash< nor::binding::runtime::DynamicPublicState > {
   size_t operator()(const nor::binding::runtime::DynamicPublicState& value) const noexcept
   {
      return value.hash();
   }
};

}  // namespace std

namespace nor::binding::runtime {

/**
 * @brief The one dynamic game-provider boundary.
 *
 * A Python trampoline may implement this interface. Every provider operation is therefore a
 * virtual boundary for a genuinely dynamic game, but the resulting DynamicEnvironment is one
 * ordinary concrete C++ environment type. The compiled registry then statically dispatches the
 * selected solver/profile through a normal SolverSession; it does not add a virtual session or
 * game layer around that traversal.
 *
 * Providers are externally synchronized. C++ sessions do not promise cross-thread safety; the
 * Python wrapper is responsible for serializing calls (normally with a per-session mutex).
 */
class DynamicEnvironmentProvider {
  public:
   virtual ~DynamicEnvironmentProvider() = default;

   [[nodiscard]] virtual size_t max_player_count() const = 0;
   [[nodiscard]] virtual size_t player_count() const = 0;

   /** Runtime declaration checked when a dynamic handle/session is created. */
   [[nodiscard]] virtual Stochasticity stochasticity() const = 0;
   [[nodiscard]] virtual bool serialized() const { return true; }
   [[nodiscard]] virtual bool unrolled() const { return true; }

   [[nodiscard]] virtual DynamicWorldState initial_world_state() const = 0;
   [[nodiscard]] virtual std::vector< DynamicAction >
   actions(Player, const DynamicWorldState& state) const = 0;
   [[nodiscard]] virtual std::vector< Player > players(const DynamicWorldState&) const = 0;
   [[nodiscard]] virtual Player active_player(const DynamicWorldState&) const = 0;
   [[nodiscard]] virtual bool is_terminal(const DynamicWorldState&) const = 0;
   [[nodiscard]] virtual bool is_partaking(const DynamicWorldState&, Player) const = 0;
   [[nodiscard]] virtual double reward(Player, const DynamicWorldState&) const = 0;

   virtual void transition(DynamicWorldState&, const DynamicAction&) const = 0;
   virtual DynamicObservation private_observation(
      Player,
      const DynamicWorldState&,
      const DynamicAction&,
      const DynamicWorldState& next_state
   ) const = 0;
   virtual DynamicObservation public_observation(
      const DynamicWorldState&,
      const DynamicAction&,
      const DynamicWorldState& next_state
   ) const = 0;

   [[nodiscard]] virtual std::vector< DynamicChanceOutcome >
   chance_actions(const DynamicWorldState&) const
   {
      return {};
   }
   [[nodiscard]] virtual double
   chance_probability(const DynamicWorldState&, const DynamicChanceOutcome&) const
   {
      throw std::logic_error("dynamic provider has no chance probability");
   }
   virtual void transition(DynamicWorldState&, const DynamicChanceOutcome&) const
   {
      throw std::logic_error("dynamic provider has no chance transition");
   }
   [[nodiscard]] virtual DynamicObservation
   private_observation(Player, const DynamicWorldState&, const DynamicChanceOutcome&, const DynamicWorldState&)
      const
   {
      return {};
   }
   [[nodiscard]] virtual DynamicObservation
   public_observation(const DynamicWorldState&, const DynamicChanceOutcome&, const DynamicWorldState&)
      const
   {
      return {};
   }
};

/**
 * @brief Concrete FOSG adapter for a provider.
 *
 * The static choice classification is deliberate: it is the compile-time superset needed by the
 * solver traversal. The provider's runtime stochasticity() remains an admission-time declaration
 * and may be deterministic or choice, but sample is rejected because it cannot provide an outcome
 * distribution to this concrete adapter.
 */
class DynamicEnvironment {
  public:
   using action_type = DynamicAction;
   using chance_outcome_type = DynamicChanceOutcome;
   using observation_type = DynamicObservation;
   using info_state_type = DynamicInfoState;
   using public_state_type = DynamicPublicState;
   using world_state_type = DynamicWorldState;
   using action_variant_type = std::variant< action_type, chance_outcome_type >;

   static constexpr Stochasticity stochasticity() noexcept { return Stochasticity::choice; }
   static constexpr bool serialized() noexcept { return true; }
   static constexpr bool unrolled() noexcept { return true; }

   explicit DynamicEnvironment(std::shared_ptr< const DynamicEnvironmentProvider > provider)
       : m_provider(std::move(provider))
   {
      if(not m_provider)
         throw std::invalid_argument("DynamicEnvironment requires a provider");
   }

   [[nodiscard]] const std::shared_ptr< const DynamicEnvironmentProvider >& provider(
   ) const noexcept
   {
      return m_provider;
   }
   [[nodiscard]] size_t max_player_count() const { return m_provider->max_player_count(); }
   [[nodiscard]] size_t player_count() const { return m_provider->player_count(); }
   [[nodiscard]] Stochasticity provider_stochasticity() const
   {
      return m_provider->stochasticity();
   }
   [[nodiscard]] bool provider_serialized() const { return m_provider->serialized(); }
   [[nodiscard]] bool provider_unrolled() const { return m_provider->unrolled(); }
   [[nodiscard]] DynamicWorldState initial_world_state() const
   {
      return m_provider->initial_world_state();
   }
   [[nodiscard]] std::vector< action_type > actions(Player player, const world_state_type& state)
      const
   {
      return m_provider->actions(player, state);
   }
   [[nodiscard]] std::vector< chance_outcome_type > chance_actions(const world_state_type& state
   ) const
   {
      return m_provider->chance_actions(state);
   }
   [[nodiscard]] double
   chance_probability(const world_state_type& state, const chance_outcome_type& outcome) const
   {
      return m_provider->chance_probability(state, outcome);
   }
   [[nodiscard]] std::vector< Player > players(const world_state_type& state) const
   {
      return m_provider->players(state);
   }
   [[nodiscard]] Player active_player(const world_state_type& state) const
   {
      return m_provider->active_player(state);
   }
   [[nodiscard]] bool is_terminal(const world_state_type& state) const
   {
      return m_provider->is_terminal(state);
   }
   [[nodiscard]] bool is_partaking(const world_state_type& state, Player player) const
   {
      return m_provider->is_partaking(state, player);
   }
   [[nodiscard]] double reward(Player player, const world_state_type& state) const
   {
      return m_provider->reward(player, state);
   }
   void transition(world_state_type& state, const action_type& action) const
   {
      m_provider->transition(state, action);
   }
   void transition(world_state_type& state, const chance_outcome_type& outcome) const
   {
      m_provider->transition(state, outcome);
   }
   [[nodiscard]] observation_type private_observation(
      Player player,
      const world_state_type& state,
      const action_type& action,
      const world_state_type& next_state
   ) const
   {
      return m_provider->private_observation(player, state, action, next_state);
   }
   [[nodiscard]] observation_type public_observation(
      const world_state_type& state,
      const action_type& action,
      const world_state_type& next_state
   ) const
   {
      return m_provider->public_observation(state, action, next_state);
   }
   [[nodiscard]] observation_type private_observation(
      Player player,
      const world_state_type& state,
      const chance_outcome_type& outcome,
      const world_state_type& next_state
   ) const
   {
      return m_provider->private_observation(player, state, outcome, next_state);
   }
   [[nodiscard]] observation_type public_observation(
      const world_state_type& state,
      const chance_outcome_type& outcome,
      const world_state_type& next_state
   ) const
   {
      return m_provider->public_observation(state, outcome, next_state);
   }

  private:
   std::shared_ptr< const DynamicEnvironmentProvider > m_provider;
};

class DynamicGameHandle;

struct DynamicCapabilityDescriptor {
   SolverId solver{};
   ProfileId profile{};
   std::string_view name{};
   using Factory = Result< SolverSession > (*)(const DynamicGameHandle&, SessionOptions);
   Factory create = nullptr;
};

class DynamicGameHandle {
  public:
   DynamicGameHandle() = default;

   [[nodiscard]] explicit operator bool() const noexcept
   {
      return m_provider != nullptr and m_spec.game_id() == GameId::dynamic;
   }
   [[nodiscard]] GameId game_id() const noexcept { return m_spec.game_id(); }
   [[nodiscard]] const GameSpec& spec() const noexcept { return m_spec; }
   [[nodiscard]] const std::shared_ptr< const DynamicEnvironmentProvider >& provider(
   ) const noexcept
   {
      return m_provider;
   }

   [[nodiscard]] Result< SolverSession >
   make_session(SolverId solver, ProfileId profile, SessionOptions options = {}) const;

   [[nodiscard]] Result< GameSpec > game_spec() const;

  private:
   DynamicGameHandle(GameSpec spec, std::shared_ptr< const DynamicEnvironmentProvider > provider)
       : m_spec(std::move(spec)), m_provider(std::move(provider))
   {
   }

   friend Result< DynamicGameHandle >
      make_dynamic_game(GameSpec, std::shared_ptr< const DynamicEnvironmentProvider >);

   GameSpec m_spec{GameId::dynamic};
   std::shared_ptr< const DynamicEnvironmentProvider > m_provider;
};

[[nodiscard]] std::span< const DynamicCapabilityDescriptor > dynamic_capabilities() noexcept;
[[nodiscard]] const DynamicCapabilityDescriptor*
find_dynamic_capability(SolverId solver, ProfileId profile) noexcept;

[[nodiscard]] Result< DynamicGameHandle >
   make_dynamic_game(GameSpec, std::shared_ptr< const DynamicEnvironmentProvider >);

}  // namespace nor::binding::runtime

#endif  // NOR_BINDING_RUNTIME_DYNAMIC_HPP
