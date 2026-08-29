#include "../catalog_partitions.hpp"

namespace nor::binding::runtime::detail {
const CatalogPartition& kuhn_partition() noexcept
{
   return partition_for< kuhn_game >();
}
}  // namespace nor::binding::runtime::detail
