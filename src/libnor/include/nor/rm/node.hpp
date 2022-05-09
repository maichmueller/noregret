
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
   InfostateNodeData() = default;

   template < ranges::range ActionRange >
   InfostateNodeData(ActionRange actions) : m_regret()
   {
      emplace(std::move(actions));
   }

   template < ranges::range ActionRange >
   void emplace(ActionRange actions) {
      if constexpr(concepts::is::sized< ActionRange >) {
         m_legal_actions.reserve(actions.size());
      }
      for(auto& action : actions) {
         auto& action_in_vec = m_legal_actions.emplace_back(std::move(action));
         m_regret.emplace(std::ref(action_in_vec), 0.);
      }
   }

   auto& actions() { return m_legal_actions; }
   auto& regret(const Action& action) { return m_regret[std::ref(action)]; }
   auto& regret() { return m_regret; }

   [[nodiscard]] auto& actions() const { return m_legal_actions; }
   [[nodiscard]] auto& regret(const Action& action) const { return m_regret.at(std::ref(action)); }
   [[nodiscard]] auto& regret() const { return m_regret; }

  private:
   std::vector< Action > m_legal_actions;
   /// the cumulative regret the active player amassed with each action. Cumulative with regards to
   /// the number of CFR iterations. Defaults to 0 and should be updated later during the traversal.
   using action_ref_hasher = decltype([](const std::reference_wrapper< const Action >& action_ref) {
      return std::hash< Action >{}(action_ref.get());
   });
   using action_ref_comparator = decltype([](const std::reference_wrapper< const Action >& ref1,
                                             const std::reference_wrapper< const Action >& ref2) {
      return ref1.get() == ref2.get();
   });
   std::unordered_map<
      std::reference_wrapper< const Action >,
      double,
      action_ref_hasher,
      action_ref_comparator >
      m_regret{};
};

}  // namespace nor::rm

#endif  // NOR_NODE_HPP
