
#ifndef NOR_NODE_HPP
#define NOR_NODE_HPP

#include <map>
#include <vector>

#include "common/common.hpp"
#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "nor/utils/utils.hpp"

namespace nor::rm {

namespace detail {

/// In order to avoid any storage overhead we don't let the empty type have unique storage. This
/// way there won't be individual storage of empty structs in the node, but rather a single address
/// for all. This reduces the memory footprint of each node by a potentially significant amount,
/// depending on compiler struct padding

template < typename Weight >
struct CondWeight {
   constexpr static bool empty_optimization = false;
   Weight weight;
};

/**
 * @brief Empty Weight optimization for when its storage is not required.
 */
template < concepts::is::empty Weight >
struct CondWeight< Weight > {
   constexpr static bool empty_optimization = true;
   [[no_unique_address]] Weight weight;
};

}  // namespace detail

/**
 * @brief the node type to represent information states in the information tree built by CFR.
 *
 * There is no concept checking for this class as any of the template types are supposed to be
 * checked within the CFR class for concept fulfillment
 *
 * @tparam Action
 */

template < typename Action, typename Weight = utils::empty >
class InfostateNodeData: public detail::CondWeight< Weight > {
  public:
   using cond_weight_base = detail::CondWeight< Weight >;

   InfostateNodeData(Weight weight = {}) : cond_weight_base(std::move(weight)), m_regret(){};

   template < ranges::range ActionRange >
   InfostateNodeData(ActionRange actions, Weight weight = {})
       : cond_weight_base(std::move(weight)), m_regret()
   {
      emplace(std::move(actions));
   }

   template < ranges::range ActionRange >
   void emplace(ActionRange actions)
   {
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
   auto& weight() requires(not concepts::is::empty< Weight >) { return cond_weight_base::weight; }

   [[nodiscard]] auto& actions() const { return m_legal_actions; }
   [[nodiscard]] auto& regret(const Action& action) const { return m_regret.at(std::ref(action)); }
   [[nodiscard]] auto& regret() const { return m_regret; }
   [[nodiscard]] auto& weight() const requires(not concepts::is::empty< Weight >) { return cond_weight_base::weight; }

  private:
   std::vector< Action > m_legal_actions;
   /// the cumulative regret the active player amassed with each action. Cumulative with regards to
   /// the number of CFR iterations. Defaults to 0 and should be updated later during the traversal.
   std::unordered_map<
      std::reference_wrapper< const Action >,
      double,
      common::ref_wrapper_hasher< const Action >,
      common::ref_wrapper_comparator< const Action > >
      m_regret{};
};

}  // namespace nor::rm

#endif  // NOR_NODE_HPP
