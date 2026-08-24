
#ifndef NOR_INFOSTATE_UNPACKER_HPP
#define NOR_INFOSTATE_UNPACKER_HPP

#include <ranges>
#include <unordered_map>
#include <vector>

#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "nor/type_defs.hpp"

namespace nor {

/**
 * @brief B5 (vectorization hook): per-infostate representative records.
 *
 * SKELETON -- the vectorization agent implements the actual logic on top of
 * this trait. The default implementation RECORDS, for every infostate it
 * sees, the world-state histories that represent it ('representatives'),
 * plus the accumulated chance probability of reaching the infostate
 * ('chance_prefix_prob').
 *
 * Recording is only active when enabled via 'record_representatives'
 * (config flag); solvers call record() behind that flag during traversal.
 */
template < typename Env >
class InfostateUnpacker {
  public:
   using env_type = Env;
   using info_state_type = auto_info_state_type< Env >;
   using world_state_type = auto_world_state_type< Env >;

   struct RepresentativeRecord {
      /// one concrete history (sequence of chance outcomes/actions) whose
      /// infostate equals the key; stored as a cloned world state snapshot
      std::shared_ptr< world_state_type > world_state;
   };

   explicit InfostateUnpacker(bool record_representatives = false)
       : m_record(record_representatives)
   {
   }

   [[nodiscard]] bool recording() const { return m_record; }

   /// called by solvers during traversal behind the 'record_representatives'
   /// flag: registers 'world_state' as a representative of 'infostate' and
   /// accumulates its chance prefix probability.
   void record(
      const info_state_type& infostate,
      const world_state_type& world_state,
      double chance_prefix_prob
   )
   {
      if(not m_record) {
         return;
      }
      auto& entry = m_records[infostate];
      entry.representatives.push_back(RepresentativeRecord{
         std::make_shared< world_state_type >(world_state)});
      entry.chance_prefix_prob += chance_prefix_prob;
   }

   /// all recorded representative histories of 'infostate'
   [[nodiscard]] const std::vector< RepresentativeRecord >& representatives(
      const info_state_type& infostate
   ) const
   {
      static const std::vector< RepresentativeRecord > empty{};
      if(auto found = m_records.find(infostate); found != m_records.end()) {
         return found->second.representatives;
      }
      return empty;
   }

   /// accumulated chance probability of all recorded prefixes of 'infostate'
   [[nodiscard]] double chance_prefix_prob(const info_state_type& infostate) const
   {
      if(auto found = m_records.find(infostate); found != m_records.end()) {
         return found->second.chance_prefix_prob;
      }
      return 0.;
   }

   void clear() { m_records.clear(); }

  private:
   struct Entry {
      std::vector< RepresentativeRecord > representatives{};
      double chance_prefix_prob = 0.;
   };
   bool m_record;
   std::unordered_map<
      info_state_type,
      Entry,
      common::value_hasher< info_state_type >,
      common::value_comparator< info_state_type > >
      m_records{};
};

}  // namespace nor

#endif  // NOR_INFOSTATE_UNPACKER_HPP
