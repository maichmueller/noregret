
#ifndef NOR_NODE_HPP
#define NOR_NODE_HPP

#include <ranges>
#include <vector>

#include "common/common.hpp"
#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "nor/rm/minimizers/minimizers.hpp"

namespace nor::rm {

/**
 * @brief the data stored at a single infostate node of the tabular CFR tree.
 *
 * This class is a thin COMPOSITION of an optional extra payload 'NodeDataType'
 * (the regret minimizer's node tables). It is empty-state-optimized via
 * [[no_unique_address]].
 *
 * The payload type has to follow the minimizer node data protocol (see
 * nor/rm/minimizers/minimizers.hpp): it OWNS the infostate's legal actions in
 * registration order ('registry') together with the flat per-action tables that
 * are index-aligned to it, provides 'register_action' to append a fresh action,
 * and exposes its cumulative table as the 'regret' member.
 */
template < typename Action, typename NodeDataType = RegretMatching< Action >::node_data_type >
   requires requires(NodeDataType& data, const Action& action) {
      data.register_action(action);
      data.regret;
   }
class InfostateNodeData {
  public:
   using action_type = Action;
   using node_data_type = NodeDataType;

   InfostateNodeData() = default;

   template < std::ranges::range ActionRange >
   explicit InfostateNodeData(ActionRange actions)
   {
      emplace(std::move(actions));
   }

   /// registers all actions of the given range and their per-action tables
   template < std::ranges::range ActionRange >
   void emplace(ActionRange actions)
   {
      if constexpr(concepts::is::sized< ActionRange >) {
         m_data.registry.actions.reserve(m_data.registry.actions.size() + actions.size());
      }
      for(auto& action : actions) {
         m_data.register_action(action);
      }
   }

   auto& actions() { return m_data.registry.actions; }
   [[nodiscard]] const auto& actions() const { return m_data.registry.actions; }

   /// resolves the per-action-table slot of 'action'
   [[nodiscard]] size_t index_of(const Action& action) const { return m_data.index_of(action); }

   auto& data() { return m_data; }
   [[nodiscard]] const auto& data() const { return m_data; }

   /// the cumulative regret table (lives inside the node data payload)
   auto& regret() { return m_data.regret; }
   [[nodiscard]] const auto& regret() const { return m_data.regret; }
   auto& regret(const Action& action) { return m_data.regret[m_data.index_of(action)]; }
   [[nodiscard]] const auto& regret(const Action& action) const
   {
      return m_data.regret[m_data.index_of(action)];
   }

  private:
   [[no_unique_address]] NodeDataType m_data;
};

}  // namespace nor::rm

#endif  // NOR_NODE_HPP
