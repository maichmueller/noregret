#include "../catalog_partitions.hpp"

namespace nor::binding::runtime::detail {
const CatalogPartition& leduc_partition() noexcept
{
   return partition_for< leduc_game >();
}
}  // namespace nor::binding::runtime::detail
