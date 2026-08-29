#include "../catalog_partitions.hpp"

namespace nor::binding::runtime::detail {
const CatalogPartition& pursuit_evasion_partition() noexcept
{
   return partition_for< pursuit_evasion_game >();
}
}  // namespace nor::binding::runtime::detail
