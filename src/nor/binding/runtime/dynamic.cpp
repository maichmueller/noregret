#include "dynamic.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <ranges>
#include <utility>

#include "catalog.hpp"

namespace nor::binding::runtime {
namespace {

[[nodiscard]] CapabilityError dynamic_provider_error(std::string message)
{
   return CapabilityError{
      .code = CapabilityErrorCode::invalid_dynamic_provider,
      .message = std::move(message),
      .game = GameId::dynamic};
}

[[nodiscard]] CapabilityError
with_session_context(CapabilityError error, SolverId solver, ProfileId profile)
{
   error.solver = solver;
   error.profile = profile;
   return error;
}

using detail::contains_duplicate;
using detail::is_actual_player;
using detail::max_actual_player_count;

/**
 * @brief Validate the provider's declarations and the state it starts every session from.
 *
 * Only facts that are constant for the whole game are established here. Everything that can vary
 * per state is validated by DynamicEnvironment on each crossing, because a provider that is well
 * formed at its root can still be malformed at a later reachable state.
 */
[[nodiscard]] Result< DynamicAdmission > admit_declarations(
   const DynamicEnvironmentProvider& provider
)
{
   DynamicAdmission admission;
   admission.max_player_count = provider.max_player_count();
   admission.player_count = provider.player_count();
   if(admission.max_player_count == 0 or admission.max_player_count > max_actual_player_count) {
      return std::unexpected(dynamic_provider_error(
         "provider max_player_count must be in [1, " + std::to_string(max_actual_player_count) + "]"
      ));
   }
   if(admission.player_count == 0 or admission.player_count > admission.max_player_count) {
      return std::unexpected(dynamic_provider_error(
         "provider player_count must be nonzero and no greater than max_player_count"
      ));
   }

   admission.stochasticity = provider.stochasticity();
   if(admission.stochasticity == Stochasticity::sample) {
      return std::unexpected(dynamic_provider_error(
         "provider stochasticity::sample is unsupported; declare deterministic or choice"
      ));
   }
   if(admission.stochasticity != Stochasticity::deterministic
      and admission.stochasticity != Stochasticity::choice) {
      return std::unexpected(dynamic_provider_error("provider declared an unknown stochasticity"));
   }
   if(not provider.serialized()) {
      return std::unexpected(
         dynamic_provider_error("dynamic providers must declare serialized() == true")
      );
   }
   if(not provider.unrolled()) {
      return std::unexpected(
         dynamic_provider_error("dynamic providers must declare unrolled() == true")
      );
   }

   admission.initial_state = provider.initial_world_state();
   if(not admission.initial_state.value().valid()) {
      return std::unexpected(dynamic_provider_error(
         "provider initial_world_state() must have a nonempty type name and identity"
      ));
   }

   admission.roster = provider.players(admission.initial_state);
   if(admission.roster.size() != admission.player_count) {
      return std::unexpected(dynamic_provider_error(
         "provider players(initial_world_state) count does not match player_count"
      ));
   }
   for(const auto player : admission.roster) {
      if(not is_actual_player(player)) {
         return std::unexpected(
            dynamic_provider_error("provider player roster contains a non-actual player")
         );
      }
   }
   if(contains_duplicate(admission.roster)) {
      return std::unexpected(
         dynamic_provider_error("provider player roster contains duplicate players")
      );
   }
   return admission;
}

/**
 * @brief Walk the admitted initial state through the checked adapter.
 *
 * This deliberately reuses DynamicEnvironment rather than repeating its rules, so the initial
 * state is admitted by exactly the checks every later state has to pass.
 */
[[nodiscard]] Result< void > admit_initial_state(const DynamicEnvironment& environment)
{
   const auto& initial = environment.initial_world_state();
   // Terminality is settled first: a terminal root has no active player to admit.
   if(environment.is_terminal(initial))
      return {};
   const auto active = environment.active_player(initial);
   if(active == Player::chance) {
      (void) environment.chance_actions(initial);
   } else {
      (void) environment.actions(active, initial);
   }
   return {};
}

}  // namespace

Result< std::shared_ptr< const DynamicAdmission > > admit_dynamic_provider(
   const std::shared_ptr< const DynamicEnvironmentProvider >& provider
)
{
   if(not provider) {
      return std::unexpected(dynamic_provider_error("dynamic game requires a provider"));
   }
   try {
      auto declarations = admit_declarations(*provider);
      if(not declarations)
         return std::unexpected(std::move(declarations.error()));
      auto admission = std::make_shared< const DynamicAdmission >(std::move(*declarations));
      // Constructing the adapter with the record under test is what makes the validated snapshot
      // and the snapshot a session is rooted at literally the same value.
      const DynamicEnvironment environment{provider, admission};
      if(auto initial = admit_initial_state(environment); not initial)
         return std::unexpected(std::move(initial.error()));
      return admission;
   } catch(const DynamicProviderError& violation) {
      return std::unexpected(dynamic_provider_error(violation.what()));
   } catch(const std::exception& exception) {
      return std::unexpected(
         dynamic_provider_error(std::string("provider validation threw: ") + exception.what())
      );
   } catch(...) {
      return std::unexpected(
         dynamic_provider_error("provider validation threw a non-standard exception")
      );
   }
}

namespace {

template < typename Profile >
Result< SolverSession >
make_dynamic_session_impl(const DynamicGameHandle& handle, SessionOptions options)
{
   if(not handle) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::invalid_handle,
         .message = "dynamic session requires a nonempty dynamic game handle",
         .game = GameId::dynamic,
         .solver = Profile::solver,
         .profile = Profile::id});
   }
   // Repeated inside the thunk as well as in the outer entry point: a capability descriptor is a
   // plain function pointer, so a caller that already holds one bypasses the outer checks.
   if(not std::isfinite(options.epsilon) or options.epsilon < 0. or options.epsilon > 1.) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::invalid_spec,
         .message = "sampling epsilon must be finite and in [0, 1]",
         .game = GameId::dynamic,
         .solver = Profile::solver,
         .profile = Profile::id});
   }

   if constexpr(not detail::profile_supported< DynamicEnvironment, Profile >()) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::unsupported_combination,
         .message = "solver/profile is not constructible for the dynamic FOSG contract",
         .game = GameId::dynamic,
         .solver = Profile::solver,
         .profile = Profile::id});
   } else {
      try {
         auto admission = admit_dynamic_provider(handle.provider());
         if(not admission) {
            return std::unexpected(
               with_session_context(std::move(admission.error()), Profile::solver, Profile::id)
            );
         }
         DynamicEnvironment environment{handle.provider(), *admission};
         // The root is the snapshot that was just validated, not a second initial_world_state()
         // query that the provider could answer differently.
         auto root = std::make_unique< DynamicWorldState >(environment.initial_world_state());
         auto solver = detail::make_concrete_solver< Profile >(
            std::move(environment), std::move(root), options
         );
         using solver_type = decltype(solver);
         using model_type = detail::
            SessionModel< solver_type, GameId::dynamic, Profile::solver, Profile::id >;
         return detail::SessionFactory::make< model_type >(std::move(solver));
      } catch(const DynamicProviderError& violation) {
         return std::unexpected(with_session_context(
            dynamic_provider_error(violation.what()), Profile::solver, Profile::id
         ));
      } catch(const std::exception& exception) {
         return std::unexpected(CapabilityError{
            .code = CapabilityErrorCode::construction_failure,
            .message = std::string("failed to create ")
                       + std::string(detail::profile_name< Profile >())
                       + " for the dynamic game: " + exception.what(),
            .game = GameId::dynamic,
            .solver = Profile::solver,
            .profile = Profile::id});
      } catch(...) {
         return std::unexpected(CapabilityError{
            .code = CapabilityErrorCode::construction_failure,
            .message = "failed to create a dynamic solver session: the environment threw a "
                       "non-standard exception",
            .game = GameId::dynamic,
            .solver = Profile::solver,
            .profile = Profile::id});
      }
   }
}

template < typename Profile, size_t Count >
constexpr void
append_dynamic_capability(std::array< DynamicCapabilityDescriptor, Count >& output, size_t& index)
{
   if constexpr(detail::profile_supported< DynamicEnvironment, Profile >()) {
      output[index++] = DynamicCapabilityDescriptor{
         .solver = Profile::solver,
         .profile = Profile::id,
         .name = detail::profile_name< Profile >(),
         .create = &make_dynamic_session_impl< Profile >};
   }
}

template < typename... Profiles >
[[nodiscard]] consteval size_t dynamic_capability_count(detail::type_list< Profiles... >)
{
   return (
      static_cast< size_t >(detail::profile_supported< DynamicEnvironment, Profiles >()) + ...
   );
}

template < size_t Count, typename... Profiles >
[[nodiscard]] consteval auto make_dynamic_capabilities(detail::type_list< Profiles... > profiles)
{
   (void) profiles;
   std::array< DynamicCapabilityDescriptor, Count > output{};
   size_t index = 0;
   (append_dynamic_capability< Profiles >(output, index), ...);
   return output;
}

inline constexpr size_t dynamic_capability_count_v = dynamic_capability_count(
   detail::profile_types{}
);
inline constexpr auto
   dynamic_capability_descriptors = make_dynamic_capabilities< dynamic_capability_count_v >(
      detail::profile_types{}
   );

static_assert(
   dynamic_capability_count_v > 0,
   "the dynamic FOSG contract must admit at least one solver profile"
);

}  // namespace

std::span< const DynamicCapabilityDescriptor > dynamic_capabilities() noexcept
{
   return dynamic_capability_descriptors;
}

const DynamicCapabilityDescriptor*
find_dynamic_capability(SolverId solver, ProfileId profile) noexcept
{
   for(const auto& capability : dynamic_capabilities()) {
      if(capability.solver == solver and capability.profile == profile)
         return &capability;
   }
   return nullptr;
}

Result< SolverSession >
DynamicGameHandle::make_session(SolverId solver, ProfileId profile, SessionOptions options) const
{
   if(not *this) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::invalid_handle,
         .message = "dynamic game handle is empty",
         .game = GameId::dynamic,
         .solver = solver,
         .profile = profile});
   }
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
   // The thunk re-admits the provider itself, so the session is rooted at a snapshot validated
   // now rather than at handle-creation time.
   return capability->create(*this, options);
}

Result< GameSpec > DynamicGameHandle::game_spec() const
{
   if(not *this) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::invalid_handle,
         .message = "dynamic game handle is empty",
         .game = GameId::dynamic});
   }
   return m_spec;
}

Result< DynamicGameHandle >
make_dynamic_game(GameSpec spec, std::shared_ptr< const DynamicEnvironmentProvider > provider)
{
   if(spec.game_id() != GameId::dynamic) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::invalid_spec,
         .message = "dynamic GameSpec must use the reserved dynamic game ID",
         .game = spec.game_id()});
   }
   auto admission = admit_dynamic_provider(provider);
   if(not admission) {
      return std::unexpected(std::move(admission.error()));
   }
   return DynamicGameHandle{std::move(spec), std::move(provider), std::move(*admission)};
}

}  // namespace nor::binding::runtime
