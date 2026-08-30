#ifndef NOR_BINDING_RUNTIME_DYNAMIC_HPP
#define NOR_BINDING_RUNTIME_DYNAMIC_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
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
   /**
    * @brief The participants of the game.
    *
    * The roster must be the same set at every state, unlike some compiled environments which drop
    * a player once they can no longer act. A dynamic provider signals that a participant is out of
    * the hand through is_partaking() instead, so that a single admitted roster stays checkable at
    * every state -- an out-of-range player array is one of the failures the checked adapter has to
    * rule out, and it cannot do that against a roster that is allowed to change.
    */
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
      throw DynamicProviderError("dynamic provider has no chance probability");
   }
   virtual void transition(DynamicWorldState&, const DynamicChanceOutcome&) const
   {
      throw DynamicProviderError("dynamic provider has no chance transition");
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

namespace detail {

/// The player identifiers a dynamic provider may use for an actual (non-chance) participant.
[[nodiscard]] constexpr bool is_actual_player(Player player) noexcept
{
   const auto value = static_cast< int >(player);
   return value >= static_cast< int >(Player::alex) and value <= static_cast< int >(Player::zoey);
}

/// The number of distinct actual players the Player enumeration can express.
inline constexpr size_t max_actual_player_count = static_cast< size_t >(Player::zoey) + 1;

/**
 * @brief Whether @p values contains the same value twice.
 *
 * Small sets are compared pairwise because that beats building a hash set for a handful of
 * elements, and a legal-action list is usually tiny. The hashed path exists so a provider with a
 * wide action space does not pay a quadratic cost at every node it is checked at.
 */
template < typename Value >
[[nodiscard]] bool contains_duplicate(const std::vector< Value >& values)
{
   constexpr size_t pairwise_limit = 16;
   if(values.size() <= pairwise_limit) {
      for(size_t left = 0; left < values.size(); ++left) {
         for(size_t right = left + 1; right < values.size(); ++right) {
            if(values[left] == values[right])
               return true;
         }
      }
      return false;
   }
   std::unordered_set< Value > seen;
   seen.reserve(values.size());
   for(const auto& value : values) {
      if(not seen.insert(value).second)
         return true;
   }
   return false;
}

}  // namespace detail

/**
 * @brief The immutable facts a provider declares once, at admission time.
 *
 * These are the only provider answers the checked adapter is allowed to reuse instead of asking
 * again: they are declarations about the game, not about a particular state. Caching them keeps
 * the per-node virtual call count identical to an unchecked adapter while giving every later
 * per-state answer a fixed reference to be validated against.
 */
struct DynamicAdmission {
   size_t max_player_count = 0;
   size_t player_count = 0;
   Stochasticity stochasticity = Stochasticity::deterministic;
   /// The admitted actual players, in the order the provider reported them at the initial state.
   std::vector< Player > roster;
   /// The exact initial state that was validated. Sessions are rooted at this value, so validation
   /// and solver construction can never disagree about which snapshot was admitted.
   DynamicWorldState initial_state;
};

/**
 * @brief Validate a provider's declarations and its initial state.
 *
 * On success the returned record is the single admitted snapshot used both by the caller that
 * requested admission and by the environment adapter constructed from it.
 */
[[nodiscard]] Result< std::shared_ptr< const DynamicAdmission > >
admit_dynamic_provider(const std::shared_ptr< const DynamicEnvironmentProvider >&);

/**
 * @brief Checked concrete FOSG adapter for a provider.
 *
 * The static choice classification is deliberate: it is the compile-time superset needed by the
 * solver traversal. The provider's runtime stochasticity() remains an admission-time declaration
 * and may be deterministic or choice, but sample is rejected because it cannot provide an outcome
 * distribution to this concrete adapter.
 *
 * Every value that crosses from the provider into solver code is validated here, not only at the
 * initial state: a provider that is well formed at its root but malformed at a later reachable
 * state is rejected deterministically instead of driving the solver into empty-action sampling,
 * an out-of-range player array, or a non-distribution over chance outcomes. Violations are
 * reported by throwing DynamicProviderError, which the erased session boundary turns into a
 * CapabilityError; there is no valid way to return an expected value from the middle of a
 * concrete traversal.
 *
 * Checking costs no additional provider calls on the normal path. Structural questions that the
 * returned value alone cannot answer -- "is an empty action list legal here?" -- are asked only
 * when the suspicious value actually occurs. A chance node's outcome distribution is summed and
 * checked once per distinct chance state and then remembered, so revisiting that node during
 * later iterations costs exactly the probability queries the solver would have made anyway.
 *
 * Providers are externally synchronized, so the memo needs no locking. Copies of an adapter share
 * one memo, because the solver stores its environment by value.
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

   /// Adopt an already validated admission record. This is the constructor sessions use.
   DynamicEnvironment(
      std::shared_ptr< const DynamicEnvironmentProvider > provider,
      std::shared_ptr< const DynamicAdmission > admission
   )
       : m_provider(std::move(provider)),
         m_admission(std::move(admission)),
         m_checked_chance_states(std::make_shared< std::unordered_set< world_state_type > >())
   {
      if(not m_provider)
         throw std::invalid_argument("DynamicEnvironment requires a provider");
      if(not m_admission)
         throw std::invalid_argument("DynamicEnvironment requires an admission record");
   }

   /// Admit the provider eagerly. Throws DynamicProviderError when the provider is inadmissible.
   explicit DynamicEnvironment(std::shared_ptr< const DynamicEnvironmentProvider > provider)
       : DynamicEnvironment(provider, _admit_or_throw(provider))
   {
   }

   [[nodiscard]] const std::shared_ptr< const DynamicEnvironmentProvider >& provider(
   ) const noexcept
   {
      return m_provider;
   }
   [[nodiscard]] const DynamicAdmission& admission() const noexcept { return *m_admission; }

   [[nodiscard]] size_t max_player_count() const { return m_admission->max_player_count; }
   [[nodiscard]] size_t player_count() const { return m_admission->player_count; }
   [[nodiscard]] Stochasticity provider_stochasticity() const { return m_admission->stochasticity; }
   [[nodiscard]] bool provider_serialized() const { return true; }
   [[nodiscard]] bool provider_unrolled() const { return true; }

   /// The admitted initial snapshot. No provider call is made; the value was validated once.
   [[nodiscard]] const world_state_type& initial_world_state() const noexcept
   {
      return m_admission->initial_state;
   }

   [[nodiscard]] std::vector< action_type > actions(Player player, const world_state_type& state)
      const
   {
      auto result = m_provider->actions(player, state);
      _check_choice_set(result, "legal actions");
      if(result.empty())
         _require_empty_action_set_is_legal(player, state);
      return result;
   }

   [[nodiscard]] std::vector< chance_outcome_type > chance_actions(const world_state_type& state
   ) const
   {
      auto result = m_provider->chance_actions(state);
      _check_choice_set(result, "chance outcomes");
      if(result.empty()) {
         _require_empty_chance_set_is_legal(state);
         return result;
      }
      _check_chance_distribution(state, result);
      return result;
   }

   [[nodiscard]] double
   chance_probability(const world_state_type& state, const chance_outcome_type& outcome) const
   {
      return _checked_probability(m_provider->chance_probability(state, outcome));
   }

   [[nodiscard]] std::vector< Player > players(const world_state_type& state) const
   {
      auto result = m_provider->players(state);
      if(result.size() != m_admission->roster.size()) {
         throw DynamicProviderError(
            "provider players() returned a roster of a different size than the admitted roster"
         );
      }
      for(const auto player : result) {
         if(not detail::is_actual_player(player)) {
            throw DynamicProviderError("provider players() returned a non-actual player");
         }
         if(std::ranges::find(m_admission->roster, player) == m_admission->roster.end()) {
            throw DynamicProviderError(
               "provider players() returned a player outside the admitted roster"
            );
         }
      }
      if(detail::contains_duplicate(result)) {
         throw DynamicProviderError("provider players() returned duplicate players");
      }
      return result;
   }

   [[nodiscard]] Player active_player(const world_state_type& state) const
   {
      const auto result = m_provider->active_player(state);
      if(result == Player::chance) {
         if(m_admission->stochasticity != Stochasticity::choice) {
            throw DynamicProviderError(
               "provider active_player() returned chance although it declared no chance moves"
            );
         }
         return result;
      }
      if(not detail::is_actual_player(result)
         or std::ranges::find(m_admission->roster, result) == m_admission->roster.end()) {
         throw DynamicProviderError(
            "provider active_player() is neither chance nor an admitted actual player"
         );
      }
      return result;
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
      const auto result = m_provider->reward(player, state);
      if(not std::isfinite(result)) {
         throw DynamicProviderError("provider reward() returned a non-finite value");
      }
      return result;
   }

   void transition(world_state_type& state, const action_type& action) const
   {
      m_provider->transition(state, action);
      _check_world(state, "transition(action)");
   }

   void transition(world_state_type& state, const chance_outcome_type& outcome) const
   {
      m_provider->transition(state, outcome);
      _check_world(state, "transition(chance outcome)");
   }

   [[nodiscard]] observation_type private_observation(
      Player player,
      const world_state_type& state,
      const action_type& action,
      const world_state_type& next_state
   ) const
   {
      return _checked_observation(
         m_provider->private_observation(player, state, action, next_state), "private_observation"
      );
   }

   [[nodiscard]] observation_type public_observation(
      const world_state_type& state,
      const action_type& action,
      const world_state_type& next_state
   ) const
   {
      return _checked_observation(
         m_provider->public_observation(state, action, next_state), "public_observation"
      );
   }

   [[nodiscard]] observation_type private_observation(
      Player player,
      const world_state_type& state,
      const chance_outcome_type& outcome,
      const world_state_type& next_state
   ) const
   {
      return _checked_observation(
         m_provider->private_observation(player, state, outcome, next_state), "private_observation"
      );
   }

   [[nodiscard]] observation_type public_observation(
      const world_state_type& state,
      const chance_outcome_type& outcome,
      const world_state_type& next_state
   ) const
   {
      return _checked_observation(
         m_provider->public_observation(state, outcome, next_state), "public_observation"
      );
   }

  private:
   [[nodiscard]] static std::shared_ptr< const DynamicAdmission > _admit_or_throw(
      const std::shared_ptr< const DynamicEnvironmentProvider >& provider
   )
   {
      auto admission = admit_dynamic_provider(provider);
      if(not admission)
         throw DynamicProviderError(admission.error().message);
      return std::move(*admission);
   }

   static void _check_world(const world_state_type& state, const char* context)
   {
      if(not state.value().valid()) {
         throw DynamicProviderError(
            std::string("provider ") + context
            + " produced a world state without a type name and identity"
         );
      }
   }

   [[nodiscard]] static observation_type
   _checked_observation(observation_type observation, const char* context)
   {
      if(not observation.valid()) {
         throw DynamicProviderError(
            std::string("provider ") + context
            + "() returned an observation without a type name and identity"
         );
      }
      return observation;
   }

   [[nodiscard]] static double _checked_probability(double probability)
   {
      if(not std::isfinite(probability) or probability < 0.) {
         throw DynamicProviderError("provider chance_probability() must be finite and nonnegative");
      }
      return probability;
   }

   template < typename Value >
   static void _check_choice_set(const std::vector< Value >& values, const char* what)
   {
      for(const auto& value : values) {
         if(not value.valid()) {
            throw DynamicProviderError(
               std::string("provider ") + what + " must have a nonempty type name and identity"
            );
         }
      }
      if(detail::contains_duplicate(values)) {
         throw DynamicProviderError(std::string("provider ") + what + " must be unique");
      }
   }

   /**
    * @brief Decide whether an empty action list was legal here.
    *
    * Reached only when the provider actually returned one, so the extra queries cost nothing on
    * the normal path. They go through this adapter's own checked accessors rather than straight to
    * the provider, so a provider that answers inconsistently is rejected instead of talking its
    * way out of the emptiness check.
    */
   void _require_empty_action_set_is_legal(Player player, const world_state_type& state) const
   {
      if(is_terminal(state))
         return;
      // A serialized environment has exactly one player to move, so an inactive player having no
      // actions at a nonterminal state is expected rather than a violation.
      if(active_player(state) != player)
         return;
      throw DynamicProviderError(
         "provider returned no legal actions for the active player of a nonterminal state"
      );
   }

   /// @copydoc _require_empty_action_set_is_legal
   void _require_empty_chance_set_is_legal(const world_state_type& state) const
   {
      if(is_terminal(state))
         return;
      if(active_player(state) != Player::chance)
         return;
      throw DynamicProviderError(
         "provider returned no chance outcomes for a nonterminal chance state"
      );
   }

   void _check_chance_distribution(
      const world_state_type& state,
      const std::vector< chance_outcome_type >& outcomes
   ) const
   {
      if(m_checked_chance_states->contains(state))
         return;
      double sum = 0.;
      for(const auto& outcome : outcomes) {
         sum += _checked_probability(m_provider->chance_probability(state, outcome));
      }
      if(not std::isfinite(sum) or std::abs(sum - 1.) > chance_probability_tolerance) {
         throw DynamicProviderError(
            "provider chance probabilities must sum approximately to one at every chance state"
         );
      }
      // Bound the memo so a game with a very large chance-state space degrades to revalidating
      // instead of growing without limit.
      if(m_checked_chance_states->size() < max_checked_chance_states)
         m_checked_chance_states->insert(state);
   }

   static constexpr double chance_probability_tolerance = 1.e-8;
   static constexpr size_t max_checked_chance_states = 1u << 16u;

   std::shared_ptr< const DynamicEnvironmentProvider > m_provider;
   std::shared_ptr< const DynamicAdmission > m_admission;
   std::shared_ptr< std::unordered_set< world_state_type > > m_checked_chance_states;
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
      return m_provider != nullptr and m_admission != nullptr
             and m_spec.game_id() == GameId::dynamic;
   }
   [[nodiscard]] GameId game_id() const noexcept { return m_spec.game_id(); }
   [[nodiscard]] const GameSpec& spec() const noexcept { return m_spec; }
   [[nodiscard]] const std::shared_ptr< const DynamicEnvironmentProvider >& provider(
   ) const noexcept
   {
      return m_provider;
   }
   /// The immutable admission certificate produced when this handle was created. Every session
   /// reuses this exact snapshot, so the Game metadata and the concrete solver adapter cannot
   /// disagree if a pure-Python provider mutates after Game construction.
   [[nodiscard]] const std::shared_ptr< const DynamicAdmission >& admission() const noexcept
   {
      return m_admission;
   }

   [[nodiscard]] Result< SolverSession >
   make_session(SolverId solver, ProfileId profile, SessionOptions options = {}) const;

   [[nodiscard]] Result< GameSpec > game_spec() const;

  private:
   DynamicGameHandle(
      GameSpec spec,
      std::shared_ptr< const DynamicEnvironmentProvider > provider,
      std::shared_ptr< const DynamicAdmission > admission
   )
       : m_spec(std::move(spec)), m_provider(std::move(provider)), m_admission(std::move(admission))
   {
   }

   friend Result< DynamicGameHandle >
      make_dynamic_game(GameSpec, std::shared_ptr< const DynamicEnvironmentProvider >);

   GameSpec m_spec{GameId::dynamic};
   std::shared_ptr< const DynamicEnvironmentProvider > m_provider;
   std::shared_ptr< const DynamicAdmission > m_admission;
};

[[nodiscard]] std::span< const DynamicCapabilityDescriptor > dynamic_capabilities() noexcept;
[[nodiscard]] const DynamicCapabilityDescriptor*
find_dynamic_capability(SolverId solver, ProfileId profile) noexcept;

[[nodiscard]] Result< DynamicGameHandle >
   make_dynamic_game(GameSpec, std::shared_ptr< const DynamicEnvironmentProvider >);

}  // namespace nor::binding::runtime

#endif  // NOR_BINDING_RUNTIME_DYNAMIC_HPP
