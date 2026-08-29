#include "../catalog_partitions.hpp"

namespace nor::binding::runtime::detail {
const CatalogPartition& battleship_partition() noexcept
{
   return partition_for< battleship_game >();
}
}  // namespace nor::binding::runtime::detail
