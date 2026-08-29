#include "../catalog_partitions.hpp"

namespace nor::binding::runtime::detail {
const CatalogPartition& colonel_blotto_partition() noexcept
{
   return partition_for< colonel_blotto_game >();
}
}  // namespace nor::binding::runtime::detail
