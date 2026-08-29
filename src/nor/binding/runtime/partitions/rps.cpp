#include "../catalog_partitions.hpp"

namespace nor::binding::runtime::detail {
const CatalogPartition& rps_partition() noexcept
{
   return partition_for< rps_game >();
}
}  // namespace nor::binding::runtime::detail
