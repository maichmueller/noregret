#include "../catalog_partitions.hpp"

namespace nor::binding::runtime::detail {
const CatalogPartition& battleship_gs_partition() noexcept
{
   return partition_for< battleship_gs_game >();
}
}  // namespace nor::binding::runtime::detail
