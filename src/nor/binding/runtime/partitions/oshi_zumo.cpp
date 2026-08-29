#include "../catalog_partitions.hpp"

namespace nor::binding::runtime::detail {
const CatalogPartition& oshi_zumo_partition() noexcept
{
   return partition_for< oshi_zumo_game >();
}
}  // namespace nor::binding::runtime::detail
