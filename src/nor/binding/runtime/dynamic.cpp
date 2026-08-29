#include "dynamic.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <ranges>

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

[[nodiscard]] constexpr bool is_actual_player(Player player) noexcept
{
   const auto value = static_cast< int >(player);
   return value >= static_cast< int >(Player::alex) and value <= static_cast< int >(Player::zoey);
}

[[nodiscard]] bool contains_player(std::span< const Player > players, Player requested) noexcept
{
   return std::ranges::find(players, requested) != players.end();
}

template < typename Value >
[[nodiscard]] bool contains_duplicate(const std::vector< Value >& values)
{
   for(size_t left = 0; left < values.size(); ++left) {
      for(size_t right = left + 1; right < values.size(); ++right) {
         if(values[left] == values[right])
            return true;
      }
   }
   return false;
}

[[nodiscard]] Result< void > validate_dynamic_provider(
   const std::shared_ptr< const DynamicEnvironmentProvider >& provider
)
{
   if(not provider) {
      return std::unexpected(dynamic_provider_error("dynamic game requires a provider"));
   }

   try {
      constexpr size_t max_actual_players = static_cast< size_t >(Player::zoey) + 1;
      const size_t max_players = provider->max_player_count();
      const size_t player_count = provider->player_count();
      if(max_players == 0 or max_players > max_actual_players) {
         return std::unexpected(dynamic_provider_error(
            "provider max_player_count must be in [1, " + std::to_string(max_actual_players) + "]"
         ));
      }
      if(player_count == 0 or player_count > max_players) {
         return std::unexpected(dynamic_provider_error(
            "provider player_count must be nonzero and no greater than max_player_count"
         ));
      }

      const auto stochasticity = provider->stochasticity();
      if(stochasticity == Stochasticity::sample) {
         return std::unexpected(dynamic_provider_error(
            "provider stochasticity::sample is unsupported; declare deterministic or choice"
         ));
      }
      if(stochasticity != Stochasticity::deterministic and stochasticity != Stochasticity::choice) {
         return std::unexpected(dynamic_provider_error("provider declared an unknown stochasticity")
         );
      }
      if(not provider->serialized()) {
         return std::unexpected(
            dynamic_provider_error("dynamic providers must declare serialized() == true")
         );
      }
      if(not provider->unrolled()) {
         return std::unexpected(
            dynamic_provider_error("dynamic providers must declare unrolled() == true")
         );
      }

      const auto initial = provider->initial_world_state();
      const auto roster = provider->players(initial);
      if(roster.size() != player_count) {
         return std::unexpected(dynamic_provider_error(
            "provider players(initial_world_state) count does not match player_count"
         ));
      }
      for(const auto player : roster) {
         if(not is_actual_player(player)) {
            return std::unexpected(
               dynamic_provider_error("provider player roster contains a non-actual player")
            );
         }
      }
      if(contains_duplicate(roster)) {
         return std::unexpected(
            dynamic_provider_error("provider player roster contains duplicate players")
         );
      }

      const auto active = provider->active_player(initial);
      if(active != Player::chance
         and (not is_actual_player(active) or not contains_player(roster, active))) {
         return std::unexpected(dynamic_provider_error(
            "provider active_player(initial_world_state) is not in the player roster or chance"
         ));
      }
      if(active == Player::chance and stochasticity != Stochasticity::choice) {
         return std::unexpected(
            dynamic_provider_error("a chance initial state requires provider stochasticity::choice")
         );
      }

      if(not provider->is_terminal(initial)) {
         if(active == Player::chance) {
            const auto outcomes = provider->chance_actions(initial);
            if(outcomes.empty()) {
               return std::unexpected(
                  dynamic_provider_error("nonterminal chance initial state has no chance outcomes")
               );
            }
            if(contains_duplicate(outcomes)) {
               return std::unexpected(
                  dynamic_provider_error("initial chance outcomes must be unique")
               );
            }
            double probability_sum = 0.;
            for(const auto& outcome : outcomes) {
               if(not outcome.valid()) {
                  return std::unexpected(dynamic_provider_error(
                     "initial chance outcomes must have nonempty type and identity"
                  ));
               }
               const double probability = provider->chance_probability(initial, outcome);
               if(not std::isfinite(probability) or probability < 0.) {
                  return std::unexpected(dynamic_provider_error(
                     "initial chance probabilities must be finite and nonnegative"
                  ));
               }
               probability_sum += probability;
            }
            if(not std::isfinite(probability_sum) or std::abs(probability_sum - 1.) > 1.e-8) {
               return std::unexpected(dynamic_provider_error(
                  "initial chance probabilities must sum approximately to one"
               ));
            }
         } else {
            const auto actions = provider->actions(active, initial);
            if(actions.empty()) {
               return std::unexpected(
                  dynamic_provider_error("nonterminal initial state has no legal actions")
               );
            }
            if(contains_duplicate(actions)) {
               return std::unexpected(dynamic_provider_error("initial legal actions must be unique")
               );
            }
            for(const auto& action : actions) {
               if(not action.valid()) {
                  return std::unexpected(dynamic_provider_error(
                     "initial legal actions must have nonempty type and identity"
                  ));
               }
            }
         }
      }
   } catch(const std::exception& exception) {
      return std::unexpected(
         dynamic_provider_error(std::string("provider validation threw: ") + exception.what())
      );
   } catch(...) {
      return std::unexpected(
         dynamic_provider_error("provider validation threw a non-standard exception")
      );
   }
   return {};
}

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
   if(not std::isfinite(options.epsilon) or options.epsilon < 0. or options.epsilon > 1.) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::invalid_spec,
         .message = "sampling epsilon must be finite and in [0, 1]",
         .game = GameId::dynamic,
         .solver = Profile::solver,
         .profile = Profile::id});
   }
   if(auto validation = validate_dynamic_provider(handle.provider()); not validation) {
      return std::unexpected(
         with_session_context(std::move(validation.error()), Profile::solver, Profile::id)
      );
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
         DynamicEnvironment environment{handle.provider()};
         auto root = std::make_unique< DynamicWorldState >(environment.initial_world_state());
         auto solver = detail::make_concrete_solver< Profile >(
            std::move(environment), std::move(root), options
         );
         using solver_type = decltype(solver);
         using model_type = detail::
            SessionModel< solver_type, GameId::dynamic, Profile::solver, Profile::id >;
         return detail::SessionFactory::make< model_type >(std::move(solver));
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
            .message = "failed to create a dynamic solver session",
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
   if(auto validation = validate_dynamic_provider(m_provider); not validation) {
      return std::unexpected(with_session_context(std::move(validation.error()), solver, profile));
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
   if(auto validation = validate_dynamic_provider(provider); not validation) {
      return std::unexpected(std::move(validation.error()));
   }
   return DynamicGameHandle{std::move(spec), std::move(provider)};
}

}  // namespace nor::binding::runtime
