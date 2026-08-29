#include <array>
#include <tuple>
#include <vector>

#include "catalog_partitions.hpp"

namespace nor::binding::runtime {
namespace {

using detail::CatalogPartition;

[[nodiscard]] const auto& partitions()
{
   static const auto value = std::array{
      &detail::kuhn_partition(),
      &detail::leduc_partition(),
      &detail::rps_partition(),
      &detail::stratego_partition(),
      &detail::texas_holdem_partition(),
      &detail::goofspiel_partition(),
      &detail::three_player_goofspiel_partition(),
      &detail::battleship_partition(),
      &detail::battleship_gs_partition(),
      &detail::dark_hex_partition(),
      &detail::pursuit_evasion_partition(),
      &detail::oshi_zumo_partition(),
      &detail::shapley_partition(),
      &detail::centipede_partition(),
      &detail::colonel_blotto_partition(),
      &detail::sheriff_partition(),
      &detail::liars_dice_partition()};
   return value;
}

[[nodiscard]] const auto& game_descriptors()
{
   static const auto value = [] {
      using partition_array = std::remove_cvref_t< decltype(partitions()) >;
      std::array< GameDescriptor, std::tuple_size_v< partition_array > > result{};
      size_t index = 0;
      for(const auto* partition : partitions())
         result[index++] = partition->game;
      return result;
   }();
   return value;
}

[[nodiscard]] const auto& profile_descriptors()
{
   static constexpr auto value = detail::make_profile_descriptors(detail::profile_types{});
   return value;
}

[[nodiscard]] const auto& capability_descriptors()
{
   static const auto value = [] {
      std::vector< CapabilityDescriptor > result;
      for(const auto* partition : partitions()) {
         result.insert(
            result.end(), partition->capabilities.begin(), partition->capabilities.end()
         );
      }
      return result;
   }();
   return value;
}

}  // namespace

const StaticCatalog& catalog() noexcept
{
   static const auto& games_storage = game_descriptors();
   static constexpr auto solvers_storage = detail::solver_descriptors;
   static const auto& profiles_storage = profile_descriptors();
   static const auto& capabilities_storage = capability_descriptors();
   static const StaticCatalog value{
      .games = games_storage,
      .solvers = solvers_storage,
      .profiles = profiles_storage,
      .combinations = capabilities_storage};
   return value;
}

std::span< const GameDescriptor > games() noexcept
{
   return catalog().games;
}

std::span< const SolverDescriptor > solvers() noexcept
{
   return catalog().solvers;
}

std::span< const ProfileDescriptor > profiles() noexcept
{
   return catalog().profiles;
}

std::span< const CapabilityDescriptor > capabilities() noexcept
{
   return catalog().combinations;
}

std::vector< CapabilityDescriptor > capabilities_for(GameId game)
{
   std::vector< CapabilityDescriptor > result;
   for(const auto& capability : capabilities()) {
      if(capability.game == game)
         result.emplace_back(capability);
   }
   return result;
}

std::vector< ProfileDescriptor > profiles_for(SolverId solver)
{
   std::vector< ProfileDescriptor > result;
   for(const auto& profile : profiles()) {
      if(profile.solver == solver)
         result.emplace_back(profile);
   }
   return result;
}

const GameDescriptor* find_game(GameId id) noexcept
{
   for(const auto& descriptor : games()) {
      if(descriptor.id == id)
         return &descriptor;
   }
   return nullptr;
}

const SolverDescriptor* find_solver(SolverId id) noexcept
{
   for(const auto& descriptor : solvers()) {
      if(descriptor.id == id)
         return &descriptor;
   }
   return nullptr;
}

const ProfileDescriptor* find_profile(ProfileId id) noexcept
{
   for(const auto& descriptor : profiles()) {
      if(descriptor.id == id)
         return &descriptor;
   }
   return nullptr;
}

const CapabilityDescriptor*
find_capability(GameId game, SolverId solver, ProfileId profile) noexcept
{
   for(const auto& descriptor : capabilities()) {
      if(descriptor.game == game and descriptor.solver == solver and descriptor.profile == profile)
         return &descriptor;
   }
   return nullptr;
}

GameSpec GameSpec::defaults(GameId id)
{
   GameSpec spec{id};
   if(const auto* descriptor = find_game(id); descriptor != nullptr) {
      for(const auto& field : descriptor->fields)
         spec.set(field.id, field.default_value);
   }
   return spec;
}

Result< GameHandle > make_game(const GameSpec& spec)
{
   const auto* descriptor = find_game(spec.game_id());
   if(descriptor == nullptr) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::unknown_game,
         .message = "GameSpec refers to an unknown static game ID",
         .game = spec.game_id()});
   }
   return descriptor->create(spec);
}

Result< SolverSession >
make_session(const GameHandle& handle, SolverId solver, ProfileId profile, SessionOptions options)
{
   if(find_game(handle.game_id()) == nullptr) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::invalid_handle,
         .message = "GameHandle refers to an unknown static game ID",
         .game = handle.game_id(),
         .solver = solver,
         .profile = profile});
   }
   if(find_solver(solver) == nullptr) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::unknown_solver,
         .message = "solver ID is not present in the immutable solver catalog",
         .game = handle.game_id(),
         .solver = solver,
         .profile = profile});
   }
   const auto* profile_descriptor = find_profile(profile);
   if(profile_descriptor == nullptr) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::unknown_profile,
         .message = "profile ID is not present in the immutable profile catalog",
         .game = handle.game_id(),
         .solver = solver,
         .profile = profile});
   }
   if(profile_descriptor->solver != solver) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::profile_solver_mismatch,
         .message = "profile belongs to a different solver family",
         .game = handle.game_id(),
         .solver = solver,
         .profile = profile});
   }
   if(not std::isfinite(options.epsilon) or options.epsilon < 0. or options.epsilon > 1.) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::invalid_spec,
         .message = "sampling epsilon must be finite and in [0, 1]",
         .game = handle.game_id(),
         .solver = solver,
         .profile = profile});
   }
   const auto* capability = find_capability(handle.game_id(), solver, profile);
   if(capability == nullptr) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::unsupported_combination,
         .message = "game/solver/profile combination is not accepted by the static capability "
                    "matrix",
         .game = handle.game_id(),
         .solver = solver,
         .profile = profile});
   }
   return capability->create(handle, options);
}

}  // namespace nor::binding::runtime
