#include "../catalog_partitions.hpp"

namespace nor::binding::runtime::detail {
const CatalogPartition& centipede_partition() noexcept
{
   return partition_for< centipede_game >();
}
}  // namespace nor::binding::runtime::detail
