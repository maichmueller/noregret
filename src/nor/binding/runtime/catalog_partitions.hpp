#ifndef NOR_BINDING_RUNTIME_CATALOG_PARTITIONS_HPP
#define NOR_BINDING_RUNTIME_CATALOG_PARTITIONS_HPP

#include "catalog.hpp"

namespace nor::binding::runtime::detail {

/**
 * @brief The single roster of compiled binding-runtime partitions.
 *
 * Every partition is one translation unit that instantiates the concrete solvers of exactly one
 * game. The roster used to be written out three times -- once as declarations, once as the array
 * the assembler walks, and once as the source list in CMake -- which is three places to forget.
 * It is now written once here; the declarations, the assembled array and the partition definition
 * bodies are all generated from it, and the static assertions below fail the build if it ever
 * disagrees with the static game list.
 *
 * Each entry is X(game tag type, partition base name).
 */
#define NOR_BINDING_RUNTIME_PARTITIONS(X)                 \
   X(kuhn_game, kuhn)                                     \
   X(leduc_game, leduc)                                   \
   X(rps_game, rps)                                       \
   X(stratego_game, stratego)                             \
   X(texas_holdem_game, texas_holdem)                     \
   X(goofspiel_game, goofspiel)                           \
   X(three_player_goofspiel_game, three_player_goofspiel) \
   X(battleship_game, battleship)                         \
   X(battleship_gs_game, battleship_gs)                   \
   X(dark_hex_game, dark_hex)                             \
   X(pursuit_evasion_game, pursuit_evasion)               \
   X(oshi_zumo_game, oshi_zumo)                           \
   X(shapley_game, shapley)                               \
   X(centipede_game, centipede)                           \
   X(colonel_blotto_game, colonel_blotto)                 \
   X(sheriff_game, sheriff)                               \
   X(liars_dice_game, liars_dice)

// Keeping these declarations free of concrete solver types is what lets the catalog assembler stay
// small and lets the linker select the same stable registry regardless of partition order.
#define NOR_BINDING_RUNTIME_DECLARE_PARTITION(Tag, Name) \
   [[nodiscard]] const CatalogPartition& Name##_partition() noexcept;

NOR_BINDING_RUNTIME_PARTITIONS(NOR_BINDING_RUNTIME_DECLARE_PARTITION)

#undef NOR_BINDING_RUNTIME_DECLARE_PARTITION

#define NOR_BINDING_RUNTIME_PARTITION_TAG(Tag, Name) , Tag

/// The game tags the roster above actually compiles a partition for. The leading void is a
/// sentinel that lets the list be built by a macro without a trailing comma; it is dropped below.
using partitioned_game_types = type_list<
   void NOR_BINDING_RUNTIME_PARTITIONS(NOR_BINDING_RUNTIME_PARTITION_TAG) >;

#undef NOR_BINDING_RUNTIME_PARTITION_TAG

namespace partition_checks {

template < typename Sentinel, typename... Games >
[[nodiscard]] consteval auto drop_sentinel(type_list< Sentinel, Games... >)
{
   return type_list< Games... >{};
}

using roster = decltype(drop_sentinel(partitioned_game_types{}));

static_assert(
   same_type_set(game_types{}, roster{}),
   "every static game must have exactly one compiled partition, and every compiled partition must "
   "belong to the static game list"
);
static_assert(unique_game_ids(game_types{}), "static game IDs must be unique");
static_assert(
   unique_game_field_ids(game_types{}),
   "fields within a static game specification must be unique"
);
static_assert(unique_profile_ids(profile_types{}), "static profile IDs must be unique");

}  // namespace partition_checks

/// The number of compiled partitions, i.e. the exact number of games the catalog reports.
inline constexpr size_t partition_count = type_list_size(partition_checks::roster{});

/**
 * @brief Emit the definition of one partition.
 *
 * A partition translation unit is exactly this one line, so a new game cannot accidentally get a
 * body that disagrees with its declaration.
 */
#define NOR_BINDING_RUNTIME_DEFINE_PARTITION(Tag, Name) \
   const CatalogPartition& Name##_partition() noexcept  \
   {                                                    \
      return partition_for< Tag >();                    \
   }

}  // namespace nor::binding::runtime::detail

#endif  // NOR_BINDING_RUNTIME_CATALOG_PARTITIONS_HPP
