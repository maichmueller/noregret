#include "../catalog_partitions.hpp"

namespace nor::binding::runtime::detail {
const CatalogPartition& dark_hex_partition() noexcept
{
   return partition_for< dark_hex_game >();
}
}  // namespace nor::binding::runtime::detail
