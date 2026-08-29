#ifndef NOR_BINDING_RUNTIME_DYNAMIC_HPP
#define NOR_BINDING_RUNTIME_DYNAMIC_HPP

#include <array>
#include <cmath>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "catalog.hpp"

namespace nor::binding::runtime {

/**
 * @brief Value-owned payload used by the dynamic environment boundary.
 *
 * A dynamic provider supplies stable type and identity strings rather than a Python object. The
 * strings are copied into the C++ value, so a later trampoline can implement the virtual calls
 * without making Python types part of libnor's solver interface.
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
      return not m_type_name.empty() or not m_identity.empty();
   }
   [[nodiscard]] std::string_view type_name() const noexcept { return m_type_name; }
   [[nodiscard]] std::string_view name() const noexcept { return m_type_name; }
   [[nodiscard]] std::string_view identity() const noexcept { return m_identity; }
   [[nodiscard]] std::optional< std::string > to_string() const { return m_text; }
   [[nodiscard]] std::optional< TensorData > to_tensor() const { return m_tensor; }

   [[nodiscard]] size_t hash() const noexcept
   {
      // FNV-1a over provider-supplied stable names. std::hash is deliberately not used here:
      // dynamic values are allowed to be serialized or compared across provider instances.
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
   void set(DynamicWorldValue value) noexcept { m_value = std::move(value); }

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
 * @brief Virtual provider contract for a Python-authored game.
 *
 * This is the only dynamic boundary. The solver, policy tables, generation checks, and coarse
 * session loop remain C++; a later Nanobind trampoline can implement these value operations per
 * game-tree edge. No Python or Nanobind type appears in this header.
 */
class DynamicEnvironmentProvider {
  public:
   virtual ~DynamicEnvironmentProvider() = default;

   [[nodiscard]] virtual size_t max_player_count() const = 0;
   [[nodiscard]] virtual size_t player_count() const = 0;
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

   // These defaults keep deterministic dynamic providers small while still satisfying the
   // stochastic FOSG branch required by a runtime-valued stochasticity() classification.
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

/** Concrete FOSG wrapper owned by a dynamic game boundary. */
class DynamicEnvironment {
  public:
   using action_type = DynamicAction;
   using chance_outcome_type = DynamicChanceOutcome;
   using observation_type = DynamicObservation;
   using info_state_type = DynamicInfoState;
   using public_state_type = DynamicPublicState;
   using world_state_type = DynamicWorldState;
   using action_variant_type = std::variant< action_type, chance_outcome_type >;

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
   [[nodiscard]] Stochasticity stochasticity() const { return m_provider->stochasticity(); }
   [[nodiscard]] bool serialized() const { return m_provider->serialized(); }
   [[nodiscard]] bool unrolled() const { return m_provider->unrolled(); }
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

static_assert(concepts::fosg< DynamicEnvironment >);

class DynamicSolverSession {
  public:
   virtual ~DynamicSolverSession() = default;

   virtual Result< IterationResult > iterate() = 0;
   virtual Result< void > advance(size_t) = 0;
   virtual Result< std::optional< IterationResult > > advance_last(size_t) = 0;
   virtual Result< TraceResult > trace(size_t, size_t every = 1) = 0;
   virtual Result< SessionStats > stats() const = 0;
   virtual Result< PolicyView > policy_lookup(PolicyViewKind) const = 0;

   virtual Result< PolicyView > policy_view(PolicyViewKind kind = PolicyViewKind::current) const
   {
      return policy_lookup(kind);
   }
};

class DynamicGameBoundary {
  public:
   virtual ~DynamicGameBoundary() = default;

   [[nodiscard]] virtual GameSpec game_spec() const = 0;
   [[nodiscard]] virtual Result< std::unique_ptr< DynamicSolverSession > >
      make_session(SolverId, ProfileId, SessionOptions) const = 0;
};

/** A dynamic handle is explicit and cannot enter the static GameHandle API. */
class DynamicGameHandle {
  public:
   explicit DynamicGameHandle(std::shared_ptr< const DynamicGameBoundary > boundary)
       : m_boundary(std::move(boundary))
   {
   }

   [[nodiscard]] explicit operator bool() const noexcept { return bool(m_boundary); }
   [[nodiscard]] const DynamicGameBoundary* boundary() const noexcept { return m_boundary.get(); }

   [[nodiscard]] Result< std::unique_ptr< DynamicSolverSession > >
   make_session(SolverId solver, ProfileId profile, SessionOptions options = {}) const
   {
      if(not m_boundary) {
         return std::unexpected(CapabilityError{
            .code = CapabilityErrorCode::invalid_handle,
            .message = "dynamic game handle is empty",
            .game = GameId::dynamic,
            .solver = solver,
            .profile = profile});
      }
      return m_boundary->make_session(solver, profile, options);
   }

   [[nodiscard]] Result< GameSpec > game_spec() const
   {
      if(not m_boundary) {
         return std::unexpected(CapabilityError{
            .code = CapabilityErrorCode::invalid_handle,
            .message = "dynamic game handle is empty",
            .game = GameId::dynamic});
      }
      return m_boundary->game_spec();
   }

  private:
   std::shared_ptr< const DynamicGameBoundary > m_boundary;
};

struct DynamicCapabilityDescriptor {
   SolverId solver{};
   ProfileId profile{};
   std::string_view name{};
   using Factory = Result< std::unique_ptr< DynamicSolverSession > > (*)(
      std::shared_ptr< const DynamicEnvironmentProvider >,
      SessionOptions
   );
   Factory create = nullptr;
};

namespace detail {

template < typename Solver, SolverId SolverFamily, ProfileId Profile >
struct DynamicSessionModel final: DynamicSolverSession {
   Solver solver;

   explicit DynamicSessionModel(Solver concrete_solver) : solver(std::move(concrete_solver)) {}

   template < typename T >
   [[nodiscard]] static Result< T > operation_error(CapabilityErrorCode code, std::string message)
   {
      return std::unexpected(CapabilityError{
         .code = code,
         .message = std::move(message),
         .game = GameId::dynamic,
         .solver = SolverFamily,
         .profile = Profile});
   }

   Result< IterationResult > iterate() final
   {
      try {
         const size_t iteration = solver.iteration();
         return make_iteration_result(iteration, solver.iterate());
      } catch(const std::exception& exception) {
         return operation_error< IterationResult >(
            CapabilityErrorCode::session_failure,
            std::string("dynamic solver iteration failed: ") + exception.what()
         );
      }
   }

   Result< void > advance(size_t iterations) final
   {
      try {
         solver.advance(iterations);
         return {};
      } catch(const std::exception& exception) {
         return operation_error< void >(
            CapabilityErrorCode::session_failure,
            std::string("dynamic solver advance failed: ") + exception.what()
         );
      }
   }

   Result< std::optional< IterationResult > > advance_last(size_t iterations) final
   {
      try {
         const size_t first_iteration = solver.iteration();
         auto root = solver.advance_last(iterations);
         if(not root)
            return std::optional< IterationResult >{};
         return std::optional< IterationResult >{
            make_iteration_result(first_iteration + iterations - 1, *root)};
      } catch(const std::exception& exception) {
         return operation_error< std::optional< IterationResult > >(
            CapabilityErrorCode::session_failure,
            std::string("dynamic solver advance_last failed: ") + exception.what()
         );
      }
   }

   Result< TraceResult > trace(size_t iterations, size_t every = 1) final
   {
      try {
         if(every == 0) {
            return operation_error< TraceResult >(
               CapabilityErrorCode::invalid_spec, "trace cadence must be greater than zero"
            );
         }
         const size_t first_iteration = solver.iteration();
         auto roots = solver.trace(iterations, every);
         return make_trace_result(first_iteration, solver.iteration(), every, roots);
      } catch(const std::exception& exception) {
         return operation_error< TraceResult >(
            CapabilityErrorCode::session_failure,
            std::string("dynamic solver trace failed: ") + exception.what()
         );
      }
   }

   Result< SessionStats > stats() const final
   {
      try {
         size_t current_entries = 0;
         size_t average_entries = 0;
         solver.visit_current_policy([&](const auto&, const auto& node) {
            current_entries += node.size();
         });
         solver.visit_average_policy([&](const auto&, const auto& node) {
            average_entries += node.size();
         });
         return SessionStats{
            .game = GameId::dynamic,
            .solver = SolverFamily,
            .profile = Profile,
            .iteration = solver.iteration(),
            .cycle = solver.cycle(),
            .player_count = solver.env().players(solver.root_state()).size(),
            .current_policy_entries = current_entries,
            .average_policy_entries = average_entries};
      } catch(const std::exception& exception) {
         return operation_error< SessionStats >(
            CapabilityErrorCode::session_failure,
            std::string("dynamic solver statistics failed: ") + exception.what()
         );
      }
   }

   Result< PolicyView > policy_lookup(PolicyViewKind kind) const final
   {
      try {
         return erase_policy_lookup<
            decltype(solver.policy_lookup()),
            typename Solver::info_state_type,
            typename Solver::action_type >(solver.policy_lookup(), kind);
      } catch(const std::exception& exception) {
         return operation_error< PolicyView >(
            CapabilityErrorCode::session_failure,
            std::string("dynamic policy lookup failed: ") + exception.what()
         );
      }
   }
};

template < typename Profile >
Result< std::unique_ptr< DynamicSolverSession > > make_dynamic_session_impl(
   std::shared_ptr< const DynamicEnvironmentProvider > provider,
   SessionOptions options
)
{
   if(not provider) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::invalid_handle,
         .message = "dynamic session requires a provider",
         .game = GameId::dynamic,
         .solver = Profile::solver,
         .profile = Profile::id});
   }
   if constexpr(not profile_supported< DynamicEnvironment, Profile >()) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::unsupported_combination,
         .message = "solver/profile is not constructible for the dynamic FOSG contract",
         .game = GameId::dynamic,
         .solver = Profile::solver,
         .profile = Profile::id});
   } else {
      try {
         DynamicEnvironment environment{provider};
         auto root = std::make_unique< DynamicWorldState >(environment.initial_world_state());
         auto solver = make_concrete_solver< Profile >(
            std::move(environment), std::move(root), options
         );
         using solver_type = decltype(solver);
         using model_type = DynamicSessionModel< solver_type, Profile::solver, Profile::id >;
         return std::unique_ptr< DynamicSolverSession >{new model_type(std::move(solver))};
      } catch(const std::exception& exception) {
         return std::unexpected(CapabilityError{
            .code = CapabilityErrorCode::construction_failure,
            .message = std::string("failed to create dynamic ") + std::string(Profile::name) + ": "
                       + exception.what(),
            .game = GameId::dynamic,
            .solver = Profile::solver,
            .profile = Profile::id});
      }
   }
}

template < typename... Profiles >
[[nodiscard]] consteval size_t dynamic_capability_count(type_list< Profiles... >)
{
   return (static_cast< size_t >(profile_supported< DynamicEnvironment, Profiles >()) + ...);
}

template < typename Profile, size_t Count >
constexpr void
append_dynamic_capability(std::array< DynamicCapabilityDescriptor, Count >& output, size_t& index)
{
   if constexpr(profile_supported< DynamicEnvironment, Profile >()) {
      output[index++] = DynamicCapabilityDescriptor{
         .solver = Profile::solver,
         .profile = Profile::id,
         .name = Profile::name,
         .create = &make_dynamic_session_impl< Profile >};
   }
}

template < size_t Count, typename... Profiles >
[[nodiscard]] consteval auto make_dynamic_capabilities(type_list< Profiles... >)
{
   std::array< DynamicCapabilityDescriptor, Count > output{};
   size_t index = 0;
   (append_dynamic_capability< Profiles >(output, index), ...);
   return output;
}

inline constexpr size_t dynamic_capability_count_v = dynamic_capability_count(profile_types{});
inline constexpr auto
   dynamic_capability_descriptors = make_dynamic_capabilities< dynamic_capability_count_v >(
      profile_types{}
   );

}  // namespace detail

[[nodiscard]] inline std::span< const DynamicCapabilityDescriptor > dynamic_capabilities() noexcept
{
   return detail::dynamic_capability_descriptors;
}

[[nodiscard]] inline const DynamicCapabilityDescriptor*
find_dynamic_capability(SolverId solver, ProfileId profile) noexcept
{
   for(const auto& capability : dynamic_capabilities()) {
      if(capability.solver == solver and capability.profile == profile)
         return &capability;
   }
   return nullptr;
}

/** Concrete dynamic boundary that registers profiles against one DynamicEnvironment type. */
class DynamicEnvironmentBoundary final: public DynamicGameBoundary {
  public:
   DynamicEnvironmentBoundary(
      GameSpec spec,
      std::shared_ptr< const DynamicEnvironmentProvider > provider
   )
       : m_spec(std::move(spec)), m_provider(std::move(provider))
   {
      if(m_spec.game_id() != GameId::dynamic)
         throw std::invalid_argument("dynamic boundary requires GameId::dynamic");
      if(not m_provider)
         throw std::invalid_argument("dynamic boundary requires a provider");
   }

   [[nodiscard]] GameSpec game_spec() const final { return m_spec; }
   [[nodiscard]] const std::shared_ptr< const DynamicEnvironmentProvider >& provider(
   ) const noexcept
   {
      return m_provider;
   }

   [[nodiscard]] Result< std::unique_ptr< DynamicSolverSession > >
   make_session(SolverId solver, ProfileId profile, SessionOptions options) const final
   {
      if(find_solver(solver) == nullptr) {
         return std::unexpected(CapabilityError{
            .code = CapabilityErrorCode::unknown_solver,
            .message = "solver ID is not present in the immutable solver catalog",
            .game = GameId::dynamic,
            .solver = solver,
            .profile = profile});
      }
      const auto* profile_descriptor = find_profile(profile);
      if(profile_descriptor == nullptr) {
         return std::unexpected(CapabilityError{
            .code = CapabilityErrorCode::unknown_profile,
            .message = "profile ID is not present in the immutable profile catalog",
            .game = GameId::dynamic,
            .solver = solver,
            .profile = profile});
      }
      if(profile_descriptor->solver != solver) {
         return std::unexpected(CapabilityError{
            .code = CapabilityErrorCode::profile_solver_mismatch,
            .message = "profile belongs to a different solver family",
            .game = GameId::dynamic,
            .solver = solver,
            .profile = profile});
      }
      if(not std::isfinite(options.epsilon) or options.epsilon < 0. or options.epsilon > 1.) {
         return std::unexpected(CapabilityError{
            .code = CapabilityErrorCode::invalid_spec,
            .message = "sampling epsilon must be finite and in [0, 1]",
            .game = GameId::dynamic,
            .solver = solver,
            .profile = profile});
      }
      const auto* capability = find_dynamic_capability(solver, profile);
      if(capability == nullptr) {
         return std::unexpected(CapabilityError{
            .code = CapabilityErrorCode::unsupported_combination,
            .message = "solver/profile is not accepted by the dynamic capability matrix",
            .game = GameId::dynamic,
            .solver = solver,
            .profile = profile});
      }
      return capability->create(m_provider, options);
   }

  private:
   GameSpec m_spec;
   std::shared_ptr< const DynamicEnvironmentProvider > m_provider;
};

[[nodiscard]] inline Result< DynamicGameHandle >
make_dynamic_game(GameSpec spec, std::shared_ptr< const DynamicEnvironmentProvider > provider)
{
   if(spec.game_id() != GameId::dynamic) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::invalid_spec,
         .message = "dynamic GameSpec must use the reserved dynamic game ID",
         .game = spec.game_id()});
   }
   if(not provider) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::invalid_handle,
         .message = "dynamic game requires a provider",
         .game = GameId::dynamic});
   }
   try {
      return DynamicGameHandle{
         std::make_shared< DynamicEnvironmentBoundary >(std::move(spec), std::move(provider))};
   } catch(const std::exception& exception) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::invalid_spec,
         .message = std::string("failed to create dynamic game boundary: ") + exception.what(),
         .game = GameId::dynamic});
   }
}

}  // namespace nor::binding::runtime

#endif  // NOR_BINDING_RUNTIME_DYNAMIC_HPP
