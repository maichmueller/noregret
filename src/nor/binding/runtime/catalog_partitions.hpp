#ifndef NOR_BINDING_RUNTIME_CATALOG_PARTITIONS_HPP
#define NOR_BINDING_RUNTIME_CATALOG_PARTITIONS_HPP

#include "catalog.hpp"

namespace nor::binding::runtime::detail {

// Each definition lives in a separate translation unit.  Keeping these declarations free of
// concrete solver types is what lets the catalog assembler stay small and lets the linker select
// the same stable registry regardless of partition order.
[[nodiscard]] const CatalogPartition& kuhn_partition() noexcept;
[[nodiscard]] const CatalogPartition& leduc_partition() noexcept;
[[nodiscard]] const CatalogPartition& rps_partition() noexcept;
[[nodiscard]] const CatalogPartition& stratego_partition() noexcept;
[[nodiscard]] const CatalogPartition& texas_holdem_partition() noexcept;
[[nodiscard]] const CatalogPartition& goofspiel_partition() noexcept;
[[nodiscard]] const CatalogPartition& three_player_goofspiel_partition() noexcept;
[[nodiscard]] const CatalogPartition& battleship_partition() noexcept;
[[nodiscard]] const CatalogPartition& battleship_gs_partition() noexcept;
[[nodiscard]] const CatalogPartition& dark_hex_partition() noexcept;
[[nodiscard]] const CatalogPartition& pursuit_evasion_partition() noexcept;
[[nodiscard]] const CatalogPartition& oshi_zumo_partition() noexcept;
[[nodiscard]] const CatalogPartition& shapley_partition() noexcept;
[[nodiscard]] const CatalogPartition& centipede_partition() noexcept;
[[nodiscard]] const CatalogPartition& colonel_blotto_partition() noexcept;
[[nodiscard]] const CatalogPartition& sheriff_partition() noexcept;
[[nodiscard]] const CatalogPartition& liars_dice_partition() noexcept;

}  // namespace nor::binding::runtime::detail

#endif  // NOR_BINDING_RUNTIME_CATALOG_PARTITIONS_HPP
