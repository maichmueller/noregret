#include "../catalog_partitions.hpp"

namespace nor::binding::runtime::detail {
const CatalogPartition& stratego_partition() noexcept
{
   return partition_for< stratego_game >();
}
}  // namespace nor::binding::runtime::detail
