#include "../catalog_partitions.hpp"

namespace nor::binding::runtime::detail {
const CatalogPartition& liars_dice_partition() noexcept
{
   return partition_for< liars_dice_game >();
}
}  // namespace nor::binding::runtime::detail
