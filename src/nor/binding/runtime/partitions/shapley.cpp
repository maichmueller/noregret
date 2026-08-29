#include "../catalog_partitions.hpp"

namespace nor::binding::runtime::detail {
const CatalogPartition& shapley_partition() noexcept
{
   return partition_for< shapley_game >();
}
}  // namespace nor::binding::runtime::detail
