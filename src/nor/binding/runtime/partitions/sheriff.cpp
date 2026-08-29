#include "../catalog_partitions.hpp"

namespace nor::binding::runtime::detail {
const CatalogPartition& sheriff_partition() noexcept
{
   return partition_for< sheriff_game >();
}
}  // namespace nor::binding::runtime::detail
