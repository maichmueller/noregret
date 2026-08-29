#include "../catalog_partitions.hpp"

namespace nor::binding::runtime::detail {
const CatalogPartition& goofspiel_partition() noexcept
{
   return partition_for< goofspiel_game >();
}
}  // namespace nor::binding::runtime::detail
