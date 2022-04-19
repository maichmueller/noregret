
#ifndef NOR_NODE_HPP
#define NOR_NODE_HPP

#include <map>
#include <vector>

#include "common/common.hpp"
#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "nor/utils/utils.hpp"

namespace nor::rm {

/**
 * @brief the node type to represent game states in the game tree built by CFR.
 *
 * There is no concept checking for this class as any of the template types are supposed to be
 * checked within the CFR class for concept fulfillment and thus allow some flexibility in data
 * storage (e.g. not store the public states, by setting them as empty class - the minimal storage
 * cost)
 * @tparam Action
 * @tparam Infostate
 * @tparam Publicstate
 * @tparam Worldstate
 */

template < typename Action >
class InfostateNodeData {
  public:
   template < ranges::range ActionRange >
   InfostateNodeData(const ActionRange& actions) : m_regret()
   {
      for(const auto& action : actions) {
         m_regret.emplace(action, 0.);
      }
   }
   InfostateNodeData(std::unordered_map< Action, double > regret_per_action)
       : m_regret(std::move(regret_per_action))
   {
   }

   auto& regret(const Action& action) { return m_regret[action]; }
   auto& regret() { return m_regret; }

   [[nodiscard]] auto& regret(const Action& action) const { return m_regret[action]; }
   [[nodiscard]] auto& regret() const { return m_regret; }

  private:
   /// the cumulative regret the active player amassed with each action. Cumulative with regards to
   /// the number of CFR iterations. Defaults to 0 and should be updated later during the traversal.
   std::unordered_map< Action, double > m_regret{};
};

}  // namespace nor::rm

#endif  // NOR_NODE_HPP
