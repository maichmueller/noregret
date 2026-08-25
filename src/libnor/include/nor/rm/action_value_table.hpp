
#ifndef NOR_RM_ACTION_VALUE_TABLE_HPP
#define NOR_RM_ACTION_VALUE_TABLE_HPP

#include <algorithm>
#include <cassert>
#include <deque>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "nor/rm/rm_utils.hpp"

namespace nor::rm {

/// insertion-ordered per-action child-value accumulator used by the tabular
/// CFR traversals (VanillaCFR + MCCFR chance-sampling).
///
/// Replaces the former per-node-visit std::unordered_map<
/// action_variant_type, StateValueMap >: a fresh hash map (one bucket-array +
/// one node allocation per legal action) was constructed and destroyed at
/// EVERY non-terminal node visit of every traversal. The table is instead kept
/// as a flat pair vector whose storage is recycled across visits through the
/// depth-indexed 'ActionValueArena' below, so traversal traffic allocates only
/// on first sight of a new recursion depth.
///
/// Bit-identicality note: the regret/average-policy updates consuming this
/// accumulator write one INDEPENDENT table slot per action ('observe' /
/// 'avg_policy[action] +='), so no summation order over the entries is
/// observable in the numbers; insertion order simply mirrors the child
/// visitation order.
template < typename ActionVariant >
using ActionValueTable = std::vector< std::pair< ActionVariant, StateValueMap > >;

/// per-recursion-depth pool of reusable accumulator slots. Created once per
/// iteration by the solver's iterate routine and threaded down the traversal
/// by reference; slot 'depth' holds the accumulator of the currently visited
/// node at that depth. DFS never interleaves same-depth frames, so slots need
/// only be cleared between successive visitors -- no content save/restore.
/// NOTE: a deque ON PURPOSE -- growing it never moves existing slots, so the
/// slot references held by active recursion frames stay valid (mirrors the
/// solver's m_traversal_state_arena).
template < typename ActionVariant >
using ActionValueArena = std::deque< ActionValueTable< ActionVariant > >;

namespace detail {

/// inserts ('key', 'value') into an ActionValueTable mirroring
/// unordered_map::emplace's keep-the-existing-entry semantics on duplicate
/// keys (legal-action/outcome lists are duplicate-free, so this guard exists
/// purely for transcription fidelity with the replaced hash map).
/// 'key' is deliberately a non-deduced context: callers pass bare actions /
/// chance outcomes which convert into the variant key type exactly like the
/// former unordered_map::emplace conversions did.
template < typename ActionVariant >
inline void emplace_action_value(
   ActionValueTable< ActionVariant >& table,
   std::type_identity_t< ActionVariant > key,
   StateValueMap&& value
)
{
   const auto found = std::ranges::find_if(table, [&](const auto& entry) {
      return entry.first == key;
   });
   if(found == table.end()) {
      table.emplace_back(std::move(key), std::move(value));
   }
}

/// value lookup mirroring unordered_map::at (throws when absent); the key is a
/// non-deduced context accepting any type convertible to the table's key type
template < typename ActionVariant >
[[nodiscard]] inline const StateValueMap& find_action_value(
   const ActionValueTable< ActionVariant >& table,
   std::type_identity_t< ActionVariant > const& key
)
{
   const auto found = std::ranges::find_if(table, [&](const auto& entry) {
      return entry.first == key;
   });
   if(found == table.end()) {
      throw std::out_of_range("action_value: no entry for the given key");
   }
   return found->second;
}

}  // namespace detail

}  // namespace nor::rm

#endif  // NOR_RM_ACTION_VALUE_TABLE_HPP
