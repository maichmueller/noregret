
#ifndef NOR_NODE_HPP
#define NOR_NODE_HPP

#include <vector>

#include "common/common.hpp"
#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "nor/rm/minimizers/minimizers.hpp"

namespace nor::rm {

/**
 * @brief the data stored at a single infostate node of the tabular CFR tree.
 *
 * This class is a thin COMPOSITION of
 *    1. the infostate's legal actions (owned; per-action tables reference into
 *       this storage), and
 *    2. an optional extra payload 'NodeDataType' (the regret minimizer's node
 *       tables). It is empty-state-optimized via [[no_unique_address]].
 *
 * The payload type has to follow the minimizer node data protocol (see
 * nor/rm/minimizers/minimizers.hpp): it provides a 'regret' table and a
 * 'register_action' method that zero-initializes all its per-action tables for
 * a freshly added action.
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
         m_actions.reserve(actions.size());
      }
      for(auto& action : actions) {
         auto& action_in_vec = m_actions.emplace_back(std::move(action));
         m_data.register_action(action_in_vec);
      }
   }

   auto& actions() { return m_actions; }
   [[nodiscard]] const auto& actions() const { return m_actions; }

   auto& data() { return m_data; }
   [[nodiscard]] const auto& data() const { return m_data; }

   /// the cumulative regret table (lives inside the node data payload)
   auto& regret() { return m_data.regret; }
   [[nodiscard]] const auto& regret() const { return m_data.regret; }
   auto& regret(const Action& action) { return m_data.regret[std::cref(action)]; }
   [[nodiscard]] const auto& regret(const Action& action) const
   {
      return m_data.regret.at(std::cref(action));
   }

  private:
   std::vector< Action > m_actions;
   [[no_unique_address]] NodeDataType m_data;
};

}  // namespace nor::rm

#endif  // NOR_NODE_HPP
