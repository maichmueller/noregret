#include "../catalog_partitions.hpp"

namespace nor::binding::runtime::detail {
const CatalogPartition& texas_holdem_partition() noexcept
{
   return partition_for< texas_holdem_game >();
}
}  // namespace nor::binding::runtime::detail
