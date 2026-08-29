#ifndef NOR_BINDING_RUNTIME_TYPES_HPP
#define NOR_BINDING_RUNTIME_TYPES_HPP

// This header is deliberately independent of Nanobind and of the concrete game/solver
// implementations.  It is the small value-oriented ABI used by the future binding layer.

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "common/common.hpp"
#include "nor/game_defs.hpp"
#include "nor/meta/fosg.hpp"

namespace nor::binding::runtime {

/// Stable wire identifiers.  Values are assigned explicitly and are never derived from C++
/// types, RTTI, enum spelling, or a process-local hash.
enum class GameId : uint16_t {
   kuhn = 0x0101,
   leduc = 0x0102,
   rps = 0x0103,
   stratego = 0x0104,
   texas_holdem = 0x0105,
   goofspiel = 0x0201,
   three_player_goofspiel = 0x0202,
   battleship = 0x0203,
   battleship_gs = 0x0204,
   dark_hex = 0x0205,
   pursuit_evasion = 0x0206,
   oshi_zumo = 0x0207,
   shapley = 0x0208,
   centipede = 0x0209,
   colonel_blotto = 0x020a,
   sheriff = 0x020b,
   liars_dice = 0x020c,

   // Reserved for the type-distinct dynamic fallback. It is intentionally absent from the
   // static game catalog and therefore cannot be passed to make_game().
   dynamic = 0x7f00,

   // Spelling aliases keep the public catalog pleasant to consume without introducing a second
   // identity for an environment.
   texas_holdem_poker = texas_holdem,
   rock_paper_scissors = rps,
   kuhn_poker = kuhn,
   leduc_poker = leduc,
   blotto = colonel_blotto
};

enum class SolverId : uint16_t {
   vanilla_cfr = 0x1001,
   cfr_plus = 0x1002,
   lazy_cfr = 0x1003,
   lazy_cfr_plus = 0x1004,
   extragradient_cfr = 0x1005,
   discounted_cfr = 0x1006,
   linear_cfr = 0x1007,
   exponential_cfr = 0x1008,
   greedy_cfr = 0x1009,
   mccfr = 0x1010,
   mccfr_plus = 0x1011
};

/// A profile is a named compile-time configuration, rather than an unbounded runtime parameter
/// bag.  More profiles can be added to the explicit profile type-list without writing a new
/// game/profile cross-product or a new erased thunk by hand.
enum class ProfileId : uint16_t {
   vanilla_alternating = 0x2001,
   vanilla_simultaneous = 0x2002,
   cfr_plus_alternating = 0x2003,
   lazy_alternating = 0x2004,
   lazy_plus_alternating = 0x2005,
   extragradient_alternating = 0x2006,
   discounted_alternating = 0x2007,
   linear_alternating = 0x2008,
   exponential_alternating = 0x2009,
   greedy_simultaneous = 0x200a,
   mccfr_outcome_lazy = 0x2010,
   mccfr_chance_sampling = 0x2011,
   mccfr_external_sampling = 0x2012,
   mccfr_pure_cfr = 0x2013,
   mccfr_plus_alternating = 0x2014
};

/// Stable field identifiers used in GameSpec.  A field is interpreted only by the descriptor for
/// its game; the explicit field list prevents silently accepting a field meant for another game.
enum class GameFieldId : uint16_t {
   rows = 0x0101,
   cols = 0x0102,
   ships_per_fleet = 0x0103,
   max_shots = 0x0104,
   ship_value = 0x0105,
   loss_multiplier = 0x0106,
   board_size = 0x0107,
   rules_mode = 0x0108,
   move_limit = 0x0109,
   deck_size = 0x0201,
   imp_info = 0x0202,
   split_half_deal = 0x0203,
   n_players = 0x0301,
   dice_per_player = 0x0302,
   n_faces = 0x0303,
   size = 0x0401,
   coins = 0x0402,
   min_bid = 0x0403,
   horizon = 0x0404,
   rounds = 0x0501,
   budget = 0x0601,
   pile_big = 0x0602,
   pile_small = 0x0603,
   v = 0x0701,
   p = 0x0702,
   s = 0x0703,
   n_max = 0x0704,
   b_max = 0x0705,
   starting_stack = 0x0801,
   small_blind = 0x0802,
   big_blind = 0x0803
};

enum class SpecKind : uint8_t { unsigned_integer = 0, floating_point = 1, boolean = 2 };

using SpecValue = std::variant< uint64_t, double, bool >;

struct SpecField {
   GameFieldId id{};
   SpecValue value{};
};

/// Copyable, reusable, deterministic game construction input.  The field vector is maintained in
/// field-ID order so equality and later serialization do not depend on insertion order.
class GameSpec {
  public:
   explicit GameSpec(GameId game) : m_game(game) {}

   [[nodiscard]] GameId game_id() const noexcept { return m_game; }
   [[nodiscard]] const std::vector< SpecField >& fields() const noexcept { return m_fields; }

   GameSpec& set(GameFieldId field, SpecValue value)
   {
      const auto it = std::lower_bound(
         m_fields.begin(),
         m_fields.end(),
         field,
         [](const SpecField& current, GameFieldId requested) {
            return static_cast< uint16_t >(current.id) < static_cast< uint16_t >(requested);
         }
      );
      if(it != m_fields.end() and it->id == field) {
         it->value = std::move(value);
      } else {
         m_fields.insert(it, SpecField{field, std::move(value)});
      }
      return *this;
   }

   template < std::integral Integer >
   GameSpec& set(GameFieldId field, Integer value)
   {
      if constexpr(std::signed_integral< Integer >) {
         // Negative values are kept representable as a failed conversion instead of wrapping to
         // a large unsigned configuration value.
         if(value < 0) {
            return set(field, SpecValue{static_cast< double >(value)});
         }
      }
      return set(field, SpecValue{static_cast< uint64_t >(value)});
   }

   GameSpec& set(GameFieldId field, double value) { return set(field, SpecValue{value}); }
   GameSpec& set(GameFieldId field, bool value) { return set(field, SpecValue{value}); }

   [[nodiscard]] const SpecValue* find(GameFieldId field) const noexcept
   {
      const auto it = std::lower_bound(
         m_fields.begin(),
         m_fields.end(),
         field,
         [](const SpecField& current, GameFieldId requested) {
            return static_cast< uint16_t >(current.id) < static_cast< uint16_t >(requested);
         }
      );
      return it != m_fields.end() and it->id == field ? &it->value : nullptr;
   }

   [[nodiscard]] bool contains(GameFieldId field) const noexcept { return find(field) != nullptr; }

   friend bool operator==(const GameSpec&, const GameSpec&) = default;

   /// Defined by the catalog layer to keep this value type free of concrete game headers.
   [[nodiscard]] static GameSpec defaults(GameId game);

  private:
   GameId m_game;
   std::vector< SpecField > m_fields;
};

enum class CapabilityErrorCode : uint8_t {
   unknown_game = 0,
   invalid_spec,
   unknown_solver,
   unknown_profile,
   profile_solver_mismatch,
   unsupported_combination,
   invalid_handle,
   construction_failure,
   session_failure,
   operation_unavailable
};

struct CapabilityError {
   CapabilityErrorCode code = CapabilityErrorCode::invalid_spec;
   std::string message;
   GameId game = static_cast< GameId >(0);
   SolverId solver = static_cast< SolverId >(0);
   ProfileId profile = static_cast< ProfileId >(0);
};

template < typename T >
using Result = std::expected< T, CapabilityError >;

class GameHandle {
  public:
   explicit GameHandle(GameSpec spec) : m_spec(std::move(spec)) {}

   [[nodiscard]] GameId game_id() const noexcept { return m_spec.game_id(); }
   [[nodiscard]] const GameSpec& spec() const noexcept { return m_spec; }

  private:
   GameSpec m_spec;
};

struct SessionOptions {
   /// Used by sampling profiles.  The static registry validates the finite profile; this value is
   /// the profile's runtime sampling knob and is not used to choose a different concrete type.
   double epsilon = 0.6;
   uint64_t seed = 0;
};

/**
 * @brief Explicit, optional tensor data exposed by an erased domain value.
 *
 * Runtime erasure never guesses a numeric representation. A tensor is present only when the
 * concrete value explicitly provides one (or when a dynamic value was created with one).
 */
struct TensorData {
   std::vector< double > values;
   std::vector< size_t > shape;

   friend bool operator==(const TensorData&, const TensorData&) = default;
};

enum class ValueKind : uint8_t {
   action = 0,
   chance_outcome,
   observation,
   info_state,
   public_state,
   world_state
};

struct ValueCapabilities {
   bool to_string = false;
   bool to_tensor = false;
};

/** Metadata generated at the single concrete-erasure point for a domain value. */
struct ValueMetadata {
   ValueKind kind = ValueKind::action;
   std::string_view type_name{};
   std::string_view name{};
   ValueCapabilities capabilities{};
};

namespace detail {

template < typename T >
concept member_to_string = requires(const T& value) {
   {
      value.to_string()
   } -> std::convertible_to< std::string >;
};

template < typename T >
concept member_to_tensor = requires(const T& value) {
   {
      value.to_tensor()
   } -> std::same_as< TensorData >;
};

template < typename T >
consteval std::string_view reflected_type_name()
{
   // Named types use their identifier. Some valid erased payloads (notably anonymous variants)
   // have no identifier; the compiler's display spelling keeps metadata available for those
   // types without making erasure instantiation fail during constant evaluation.
   if(std::meta::has_identifier(^^T)) {
      return std::meta::identifier_of(^^T);
   }
   return std::meta::display_string_of(^^T);
}

template < typename T >
[[nodiscard]] std::optional< std::string > optional_to_string(const T& value)
{
   if constexpr(member_to_string< T >) {
      return std::string(value.to_string());
   } else if constexpr(common::printable_v< T >) {
      return common::to_string(value);
   } else {
      return std::nullopt;
   }
}

template < typename T >
[[nodiscard]] std::optional< TensorData > optional_to_tensor(const T& value)
{
   if constexpr(member_to_tensor< T >) {
      return value.to_tensor();
   } else {
      return std::nullopt;
   }
}

template < typename T >
inline const char value_type_token{};

}  // namespace detail

/**
 * @brief Copyable value erasure for domain values crossing the binding boundary.
 *
 * The model owns one concrete value copy. Policy views use this class for keys and actions, while
 * the policy probabilities themselves remain borrowed from the solver. Equality is exact within
 * a value kind and concrete type; hash() uses the concrete type's stable std::hash and is salted
 * with deterministic kind/type-name data so values from distinct domains cannot compare equal by
 * accident.
 */
class ErasedValue {
  private:
   struct Concept {
      virtual ~Concept() = default;
      [[nodiscard]] virtual const ValueMetadata& metadata() const noexcept = 0;
      [[nodiscard]] virtual const void* type_token() const noexcept = 0;
      [[nodiscard]] virtual const void* value_ptr() const noexcept = 0;
      [[nodiscard]] virtual size_t value_hash() const noexcept = 0;
      [[nodiscard]] virtual bool equals(const Concept&) const noexcept = 0;
      [[nodiscard]] virtual std::optional< std::string > to_string() const = 0;
      [[nodiscard]] virtual std::optional< TensorData > to_tensor() const = 0;
   };

   template < ValueKind Kind, typename T >
   struct Model final: Concept {
      explicit Model(const T& concrete_value) : value(concrete_value) {}

      [[nodiscard]] const ValueMetadata& metadata() const noexcept final
      {
         static constexpr ValueMetadata result{
            .kind = Kind,
            .type_name = detail::reflected_type_name< T >(),
            .name = detail::reflected_type_name< T >(),
            .capabilities = ValueCapabilities{
               .to_string = detail::member_to_string< T > or common::printable_v< T >,
               .to_tensor = detail::member_to_tensor< T >}};
         return result;
      }

      [[nodiscard]] const void* type_token() const noexcept final
      {
         return std::addressof(detail::value_type_token< T >);
      }

      [[nodiscard]] const void* value_ptr() const noexcept final { return std::addressof(value); }

      [[nodiscard]] size_t value_hash() const noexcept final { return std::hash< T >{}(value); }

      [[nodiscard]] bool equals(const Concept& other) const noexcept final
      {
         if(other.type_token() != type_token()) {
            return false;
         }
         return value == static_cast< const Model& >(other).value;
      }

      [[nodiscard]] std::optional< std::string > to_string() const final
      {
         return detail::optional_to_string(value);
      }

      [[nodiscard]] std::optional< TensorData > to_tensor() const final
      {
         return detail::optional_to_tensor(value);
      }

      T value;
   };

   template < ValueKind Kind, typename T >
   explicit ErasedValue(std::in_place_type_t< T >, const T& value)
       : m_model(std::make_shared< Model< Kind, T > >(value))
   {
   }

  public:
   ErasedValue() = default;

   template < ValueKind Kind, typename T >
      requires std::copy_constructible< std::remove_cvref_t< T > >
               and std::equality_comparable< std::remove_cvref_t< T > >
               and requires(const std::remove_cvref_t< T >& value) {
                      {
                         std::hash< std::remove_cvref_t< T > >{}(value)
                      } -> std::convertible_to< size_t >;
                   }
   [[nodiscard]] static ErasedValue make(const T& value)
   {
      using value_type = std::remove_cvref_t< T >;
      ErasedValue result;
      result.m_model = std::make_shared< Model< Kind, value_type > >(value);
      return result;
   }

   template < typename T >
   [[nodiscard]] static ErasedValue action(const T& value)
   {
      return make< ValueKind::action >(value);
   }
   template < typename T >
   [[nodiscard]] static ErasedValue chance_outcome(const T& value)
   {
      return make< ValueKind::chance_outcome >(value);
   }
   template < typename T >
   [[nodiscard]] static ErasedValue observation(const T& value)
   {
      return make< ValueKind::observation >(value);
   }
   template < typename T >
   [[nodiscard]] static ErasedValue info_state(const T& value)
   {
      return make< ValueKind::info_state >(value);
   }
   template < typename T >
   [[nodiscard]] static ErasedValue public_state(const T& value)
   {
      return make< ValueKind::public_state >(value);
   }
   template < typename T >
   [[nodiscard]] static ErasedValue world_state(const T& value)
   {
      return make< ValueKind::world_state >(value);
   }

   [[nodiscard]] bool valid() const noexcept { return m_model != nullptr; }
   [[nodiscard]] explicit operator bool() const noexcept { return valid(); }

   [[nodiscard]] ValueKind kind() const noexcept
   {
      return m_model == nullptr ? ValueKind::action : m_model->metadata().kind;
   }

   [[nodiscard]] const ValueMetadata* metadata() const noexcept
   {
      return m_model == nullptr ? nullptr : std::addressof(m_model->metadata());
   }

   [[nodiscard]] std::string_view type_name() const noexcept
   {
      return m_model == nullptr ? std::string_view{} : m_model->metadata().type_name;
   }

   [[nodiscard]] std::string_view name() const noexcept
   {
      return m_model == nullptr ? std::string_view{} : m_model->metadata().name;
   }

   [[nodiscard]] size_t hash() const noexcept
   {
      if(m_model == nullptr) {
         return 0;
      }
      size_t result = static_cast< size_t >(m_model->metadata().kind) + 1;
      for(const auto character : m_model->metadata().type_name) {
         result = (result ^ static_cast< unsigned char >(character)) * size_t{1099511628211};
      }
      result ^= m_model->value_hash() + size_t{0x9e3779b97f4a7c15};
      return result;
   }

   [[nodiscard]] std::optional< std::string > to_string() const
   {
      return m_model == nullptr ? std::nullopt : m_model->to_string();
   }

   [[nodiscard]] std::optional< TensorData > to_tensor() const
   {
      return m_model == nullptr ? std::nullopt : m_model->to_tensor();
   }

   template < typename T >
   [[nodiscard]] bool holds() const noexcept
   {
      using value_type = std::remove_cvref_t< T >;
      return m_model != nullptr
             and m_model->type_token() == std::addressof(detail::value_type_token< value_type >);
   }

   template < typename T >
   [[nodiscard]] const std::remove_cvref_t< T >* get_if() const noexcept
   {
      if(not holds< T >()) {
         return nullptr;
      }
      // The model's token has already established the exact concrete type. The value pointer is
      // supplied by the model so this cast is independent of the erased domain kind.
      return static_cast< const std::remove_cvref_t< T >* >(m_model->value_ptr());
   }

   friend bool operator==(const ErasedValue& left, const ErasedValue& right) noexcept
   {
      if(left.m_model == nullptr or right.m_model == nullptr) {
         return left.m_model == nullptr and right.m_model == nullptr;
      }
      if(left.kind() != right.kind() or left.m_model->type_token() != right.m_model->type_token()) {
         return false;
      }
      return left.m_model->equals(*right.m_model);
   }

  private:
   std::shared_ptr< const Concept > m_model;
};

using ErasedAction = ErasedValue;
using ErasedChanceOutcome = ErasedValue;
using ErasedObservation = ErasedValue;
using ErasedInfoState = ErasedValue;
using ErasedPublicState = ErasedValue;
using ErasedWorldState = ErasedValue;

struct RootValue {
   Player player = Player::unknown;
   double value = 0.;
};

/// The result of exactly one solver iteration. Root values are sorted by player before crossing
/// the erased boundary, so the public structure is deterministic even when the concrete map is
/// hash-backed.
struct IterationResult {
   size_t iteration = 0;
   std::vector< RootValue > root_values;
};

/// The only explicit collection result. advance() and advance_last() never construct this type.
struct TraceResult {
   size_t first_iteration = 0;
   size_t last_iteration = 0;
   std::vector< IterationResult > iterations;

   [[nodiscard]] bool empty() const noexcept { return iterations.empty(); }
   [[nodiscard]] size_t size() const noexcept { return iterations.size(); }
   [[nodiscard]] const IterationResult& operator[](size_t index) const
   {
      return iterations.at(index);
   }
   [[nodiscard]] auto begin() const noexcept { return iterations.begin(); }
   [[nodiscard]] auto end() const noexcept { return iterations.end(); }
};

enum class PolicyViewKind : uint8_t { current = 0, average = 1 };

struct PolicyActionEntry {
   ErasedAction action;
   double probability = 0.;
};

/** One copied policy row, created only by an explicit PolicyLookup::to_entries() call. */
struct PolicyEntry {
   Player player = Player::unknown;
   ErasedInfoState info_state;
   std::vector< PolicyActionEntry > actions;
};

class PolicyNodeView;
class PolicyLookup;

/**
 * @brief A generation-checked, erased view of one concrete solver policy row.
 *
 * The backend borrows the concrete node and its action registry. Accessors check the generation
 * before touching that storage; after any solver mutation (or destruction) they fail
 * deterministically with std::logic_error instead of dereferencing a dangling pointer.
 */
class PolicyNodeView {
  public:
   struct Backend {
      virtual ~Backend() = default;
      [[nodiscard]] virtual bool valid() const noexcept = 0;
      [[nodiscard]] virtual size_t generation() const noexcept = 0;
      [[nodiscard]] virtual PolicyViewKind kind() const noexcept = 0;
      [[nodiscard]] virtual Player player() const = 0;
      [[nodiscard]] virtual ErasedInfoState info_state() const = 0;
      [[nodiscard]] virtual size_t size() const = 0;
      [[nodiscard]] virtual ErasedAction action_at(size_t) const = 0;
      [[nodiscard]] virtual double value_at(size_t) const = 0;
      [[nodiscard]] virtual std::optional< double > find(const ErasedAction&) const = 0;
      [[nodiscard]] virtual bool contains(const ErasedAction&) const = 0;
   };

   PolicyNodeView() = default;

   [[nodiscard]] static PolicyNodeView from_backend(std::shared_ptr< const Backend > backend)
   {
      return PolicyNodeView{std::move(backend)};
   }

   [[nodiscard]] bool valid() const noexcept { return m_backend != nullptr and m_backend->valid(); }
   [[nodiscard]] size_t generation() const noexcept
   {
      return m_backend == nullptr ? 0 : m_backend->generation();
   }
   [[nodiscard]] PolicyViewKind kind() const noexcept
   {
      return m_backend == nullptr ? PolicyViewKind::current : m_backend->kind();
   }
   [[nodiscard]] Player player() const
   {
      _check_current();
      return m_backend->player();
   }
   [[nodiscard]] ErasedInfoState info_state() const
   {
      _check_current();
      return m_backend->info_state();
   }
   [[nodiscard]] size_t size() const
   {
      _check_current();
      return m_backend->size();
   }
   [[nodiscard]] ErasedAction action_at(size_t index) const
   {
      _check_current();
      return m_backend->action_at(index);
   }
   [[nodiscard]] double value_at(size_t index) const
   {
      _check_current();
      return m_backend->value_at(index);
   }
   [[nodiscard]] std::optional< double > find(const ErasedAction& action) const
   {
      _check_current();
      return m_backend->find(action);
   }
   [[nodiscard]] bool contains(const ErasedAction& action) const
   {
      _check_current();
      return m_backend->contains(action);
   }
   [[nodiscard]] double at(const ErasedAction& action) const
   {
      auto result = find(action);
      if(not result) {
         throw std::out_of_range("PolicyNodeView: action is not present in this policy row");
      }
      return *result;
   }

   /// Explicitly copy this one row. Normal lookup/access never allocates a probability table.
   [[nodiscard]] std::vector< PolicyActionEntry > to_entries() const
   {
      _check_current();
      std::vector< PolicyActionEntry > result;
      result.reserve(m_backend->size());
      for(size_t index = 0; index < m_backend->size(); ++index) {
         result.push_back(PolicyActionEntry{m_backend->action_at(index), m_backend->value_at(index)}
         );
      }
      return result;
   }

   /// Explicitly copy probabilities as a one-dimensional tensor. No tensor is inferred for an
   /// action or information-state value itself.
   [[nodiscard]] TensorData to_tensor() const
   {
      _check_current();
      TensorData result;
      result.shape = {m_backend->size()};
      result.values.reserve(m_backend->size());
      for(size_t index = 0; index < m_backend->size(); ++index)
         result.values.push_back(m_backend->value_at(index));
      return result;
   }

  private:
   explicit PolicyNodeView(std::shared_ptr< const Backend > backend) : m_backend(std::move(backend))
   {
   }

   void _check_current() const
   {
      if(not valid()) {
         throw std::logic_error(
            "PolicyNodeView is stale; obtain a new view after the solver changes"
         );
      }
   }

   friend class PolicyLookup;
   std::shared_ptr< const Backend > m_backend;
};

/**
 * @brief Generation-checked erased policy lookup over concrete solver-owned rows.
 *
 * Constructing a lookup is O(1) and does not materialize either policy table. to_entries() is the
 * deliberately explicit bulk-copy escape hatch for binding layers that need a Python dict/list.
 */
class PolicyLookup {
  public:
   struct Backend {
      virtual ~Backend() = default;
      [[nodiscard]] virtual bool valid() const noexcept = 0;
      [[nodiscard]] virtual size_t generation() const noexcept = 0;
      [[nodiscard]] virtual std::optional< PolicyNodeView >
      find(PolicyViewKind, const ErasedInfoState&) const = 0;
      [[nodiscard]] virtual size_t
      visit(PolicyViewKind, const std::function< void(const PolicyNodeView&) >&) const = 0;
   };

   PolicyLookup() = default;

   [[nodiscard]] static PolicyLookup from_backend(
      std::shared_ptr< const Backend > backend,
      PolicyViewKind kind = PolicyViewKind::current
   )
   {
      return PolicyLookup{std::move(backend), kind};
   }

   [[nodiscard]] bool valid() const noexcept { return m_backend != nullptr and m_backend->valid(); }
   [[nodiscard]] size_t generation() const noexcept
   {
      return m_backend == nullptr ? 0 : m_backend->generation();
   }
   [[nodiscard]] PolicyViewKind kind() const noexcept { return m_kind; }

   [[nodiscard]] std::optional< PolicyNodeView > find(const ErasedInfoState& info_state) const
   {
      return find(m_kind, info_state);
   }

   [[nodiscard]] std::optional< PolicyNodeView >
   find(PolicyViewKind requested_kind, const ErasedInfoState& info_state) const
   {
      _check_current();
      return m_backend->find(requested_kind, info_state);
   }

   [[nodiscard]] std::optional< PolicyNodeView >
   find(const ErasedInfoState& info_state, PolicyViewKind requested_kind) const
   {
      return find(requested_kind, info_state);
   }

   [[nodiscard]] PolicyNodeView at(const ErasedInfoState& info_state) const
   {
      return at(m_kind, info_state);
   }

   [[nodiscard]] PolicyNodeView at(PolicyViewKind requested_kind, const ErasedInfoState& info_state)
      const
   {
      auto result = find(requested_kind, info_state);
      if(not result) {
         throw std::out_of_range(
            "PolicyLookup: information state is not present in the requested policy"
         );
      }
      return std::move(*result);
   }

   [[nodiscard]] PolicyNodeView at(const ErasedInfoState& info_state, PolicyViewKind requested_kind)
      const
   {
      return at(requested_kind, info_state);
   }

   size_t visit(const std::function< void(const PolicyNodeView&) >& visitor) const
   {
      return visit(m_kind, visitor);
   }

   size_t visit(
      PolicyViewKind requested_kind,
      const std::function< void(const PolicyNodeView&) >& visitor
   ) const
   {
      _check_current();
      return m_backend->visit(requested_kind, visitor);
   }

   /// Explicitly copy all rows in this policy. This is the only operation that walks the complete
   /// policy through the erased API.
   [[nodiscard]] std::vector< PolicyEntry > to_entries() const
   {
      std::vector< PolicyEntry > result;
      visit([&](const PolicyNodeView& node) {
         PolicyEntry entry{.player = node.player(), .info_state = node.info_state(), .actions = {}};
         entry.actions = node.to_entries();
         result.push_back(std::move(entry));
      });
      return result;
   }

  private:
   explicit PolicyLookup(std::shared_ptr< const Backend > backend, PolicyViewKind kind)
       : m_backend(std::move(backend)), m_kind(kind)
   {
   }

   void _check_current() const
   {
      if(not valid()) {
         throw std::logic_error(
            "PolicyLookup is stale; obtain a new lookup after the solver changes"
         );
      }
   }

   std::shared_ptr< const Backend > m_backend;
   PolicyViewKind m_kind = PolicyViewKind::current;
};

// PolicyView remains a source-compatible name for the final lookup object. It no longer denotes a
// materialized ordinal snapshot.
using PolicyView = PolicyLookup;

struct SessionStats {
   GameId game = static_cast< GameId >(0);
   SolverId solver = static_cast< SolverId >(0);
   ProfileId profile = static_cast< ProfileId >(0);
   size_t iteration = 0;
   size_t cycle = 0;
   size_t player_count = 0;
   size_t current_policy_entries = 0;
   size_t average_policy_entries = 0;
};

struct FieldDescriptor {
   GameFieldId id{};
   std::string_view name{};
   SpecKind kind = SpecKind::unsigned_integer;
   SpecValue default_value{};
};

class GameHandle;
class SolverSession;

using GameFactoryFn = Result< GameHandle > (*)(const GameSpec&);
using SessionFactoryFn = Result< SolverSession > (*)(const GameHandle&, SessionOptions);

struct GameDescriptor {
   GameId id{};
   std::string_view name{};
   size_t min_players = 0;
   size_t max_players = 0;
   Stochasticity stochasticity = Stochasticity::deterministic;
   std::span< const FieldDescriptor > fields{};
   GameFactoryFn create = nullptr;
};

struct ProfileDescriptor {
   ProfileId id{};
   SolverId solver{};
   std::string_view name{};
};

struct SolverDescriptor {
   SolverId id{};
   std::string_view name{};
};

struct CapabilityDescriptor {
   GameId game{};
   SolverId solver{};
   ProfileId profile{};
   SessionFactoryFn create = nullptr;
};

struct StaticCatalog {
   std::span< const GameDescriptor > games{};
   std::span< const SolverDescriptor > solvers{};
   std::span< const ProfileDescriptor > profiles{};
   std::span< const CapabilityDescriptor > combinations{};
};

/// Coarse operation vtable.  It is one vtable per concrete solver object, never one dispatch per
/// game-tree node.
struct SolverSessionOps {
   void (*destroy)(void*) noexcept = nullptr;
   Result< IterationResult > (*iterate)(void*) = nullptr;
   Result< void > (*advance)(void*, size_t) = nullptr;
   Result< std::optional< IterationResult > > (*advance_last)(void*, size_t) = nullptr;
   Result< TraceResult > (*trace)(void*, size_t, size_t) = nullptr;
   Result< SessionStats > (*stats)(const void*) = nullptr;
   Result< PolicyView > (*policy_lookup)(const void*, PolicyViewKind) = nullptr;
};

class SolverSession {
  public:
   SolverSession() = default;
   SolverSession(const SolverSession&) = delete;
   SolverSession& operator=(const SolverSession&) = delete;

   SolverSession(SolverSession&& other) noexcept
       : m_object(std::exchange(other.m_object, nullptr)),
         m_ops(std::exchange(other.m_ops, nullptr))
   {
   }

   SolverSession& operator=(SolverSession&& other) noexcept
   {
      if(this != &other) {
         reset();
         m_object = std::exchange(other.m_object, nullptr);
         m_ops = std::exchange(other.m_ops, nullptr);
      }
      return *this;
   }

   ~SolverSession() { reset(); }

   [[nodiscard]] explicit operator bool() const noexcept { return m_object != nullptr; }
   [[nodiscard]] bool empty() const noexcept { return m_object == nullptr; }

   Result< IterationResult > iterate()
   {
      if(not m_ops)
         return missing_operation< IterationResult >("iterate");
      return m_ops->iterate(m_object);
   }

   Result< void > advance(size_t iterations)
   {
      if(not m_ops)
         return missing_operation< void >("advance");
      return m_ops->advance(m_object, iterations);
   }

   Result< std::optional< IterationResult > > advance_last(size_t iterations)
   {
      if(not m_ops)
         return missing_operation< std::optional< IterationResult > >("advance_last");
      return m_ops->advance_last(m_object, iterations);
   }

   Result< TraceResult > trace(size_t iterations, size_t every = 1)
   {
      if(not m_ops)
         return missing_operation< TraceResult >("trace");
      return m_ops->trace(m_object, iterations, every);
   }

   Result< SessionStats > stats() const
   {
      if(not m_ops)
         return missing_operation< SessionStats >("stats");
      return m_ops->stats(m_object);
   }

   Result< PolicyView > policy_lookup(PolicyViewKind kind = PolicyViewKind::current) const
   {
      if(not m_ops)
         return missing_operation< PolicyView >("policy_lookup");
      return m_ops->policy_lookup(m_object, kind);
   }

   /// Compatibility spelling; the returned object is the final borrowed PolicyLookup, not a
   /// materialized ordinal snapshot.
   Result< PolicyView > policy_view(PolicyViewKind kind = PolicyViewKind::current) const
   {
      return policy_lookup(kind);
   }

   /// Internal construction is exposed as a constrained value factory rather than as a public
   /// constructor.  The catalog is the only production caller; keeping this operation templated
   /// preserves the concrete solver type until the single coarse-erasure point.
   template < typename Model, typename... Args >
   static SolverSession make(Args&&... args)
   {
      return SolverSession{new Model(std::forward< Args >(args)...), &Model::ops};
   }

  private:
   SolverSession(void* object, const SolverSessionOps* ops) : m_object(object), m_ops(ops) {}

   void reset() noexcept
   {
      if(m_object and m_ops)
         m_ops->destroy(m_object);
      m_object = nullptr;
      m_ops = nullptr;
   }

   template < typename T >
   [[nodiscard]] static Result< T > missing_operation(std::string_view operation)
   {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::operation_unavailable,
         .message = std::string("solver session is empty; operation unavailable: ")
                    + std::string(operation)});
   }

   void* m_object = nullptr;
   const SolverSessionOps* m_ops = nullptr;
};

// The catalog layer provides the actual lookup/construction functions.
[[nodiscard]] const StaticCatalog& catalog() noexcept;
[[nodiscard]] const GameDescriptor* find_game(GameId) noexcept;
[[nodiscard]] const SolverDescriptor* find_solver(SolverId) noexcept;
[[nodiscard]] const ProfileDescriptor* find_profile(ProfileId) noexcept;
[[nodiscard]] const CapabilityDescriptor* find_capability(GameId, SolverId, ProfileId) noexcept;

[[nodiscard]] Result< GameHandle > make_game(const GameSpec&);
[[nodiscard]] Result< SolverSession >
make_session(const GameHandle&, SolverId, ProfileId, SessionOptions = {});

}  // namespace nor::binding::runtime

namespace std {

template <>
struct hash< nor::binding::runtime::ErasedValue > {
   size_t operator()(const nor::binding::runtime::ErasedValue& value) const noexcept
   {
      return value.hash();
   }
};

}  // namespace std

#endif  // NOR_BINDING_RUNTIME_TYPES_HPP
