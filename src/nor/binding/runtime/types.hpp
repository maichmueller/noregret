#ifndef NOR_BINDING_RUNTIME_TYPES_HPP
#define NOR_BINDING_RUNTIME_TYPES_HPP

// This header is deliberately independent of Nanobind and of the concrete game/solver
// implementations.  It is the small value-oriented ABI used by the future binding layer.

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "nor/game_defs.hpp"

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

struct RootValue {
   Player player = Player::unknown;
   double value = 0.;
};

struct IterationResult {
   size_t iteration = 0;
   std::vector< RootValue > root_values;
};

struct IterateResult {
   size_t first_iteration = 0;
   size_t last_iteration = 0;
   std::vector< IterationResult > iterations;
};

struct TraceRequest {
   size_t max_events = 0;
};

struct TraceEvent {
   size_t depth = 0;
   Player player = Player::unknown;
   size_t action_ordinal = 0;
};

struct TraceResult {
   bool available = false;
   bool complete = false;
   std::vector< TraceEvent > events;
};

enum class PolicyViewKind : uint8_t { current = 0, average = 1 };

/// A temporary value-oriented policy snapshot.  The concrete action/infoset values are retained
/// by the solver and are intentionally not copied through this first runtime ABI; ordinals and
/// probabilities are sufficient for capability/session tests.  The sibling solver's policy-access
/// primitive is the replacement point for stable encoded values before binding exposure.
struct PolicyEntry {
   Player player = Player::unknown;
   size_t info_state_ordinal = 0;
   std::vector< double > action_probabilities;
};

struct PolicyView {
   PolicyViewKind kind = PolicyViewKind::current;
   bool complete = true;
   bool temporary_adapter = true;
   std::vector< PolicyEntry > entries;
};

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
   Result< IterateResult > (*iterate)(void*, size_t) = nullptr;
   Result< IterateResult > (*advance)(void*, size_t) = nullptr;
   Result< TraceResult > (*trace)(const void*, const TraceRequest&) = nullptr;
   Result< SessionStats > (*stats)(const void*) = nullptr;
   Result< PolicyView > (*policy_view)(const void*, PolicyViewKind) = nullptr;
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

   Result< IterateResult > iterate(size_t iterations)
   {
      if(not m_ops)
         return missing_operation< IterateResult >("iterate");
      return m_ops->iterate(m_object, iterations);
   }

   Result< IterateResult > advance(size_t iterations)
   {
      if(not m_ops)
         return missing_operation< IterateResult >("advance");
      return m_ops->advance(m_object, iterations);
   }

   Result< TraceResult > trace(const TraceRequest& request = {}) const
   {
      if(not m_ops)
         return missing_operation< TraceResult >("trace");
      return m_ops->trace(m_object, request);
   }

   Result< SessionStats > stats() const
   {
      if(not m_ops)
         return missing_operation< SessionStats >("stats");
      return m_ops->stats(m_object);
   }

   Result< PolicyView > policy_view(PolicyViewKind kind = PolicyViewKind::current) const
   {
      if(not m_ops)
         return missing_operation< PolicyView >("policy_view");
      return m_ops->policy_view(m_object, kind);
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

#endif  // NOR_BINDING_RUNTIME_TYPES_HPP
