#include "../catalog_partitions.hpp"

namespace nor::binding::runtime::detail {
const CatalogPartition& three_player_goofspiel_partition() noexcept
{
   return partition_for< three_player_goofspiel_game >();
}
}  // namespace nor::binding::runtime::detail
