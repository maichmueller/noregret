
#ifndef NOR_NODE_HPP
#define NOR_NODE_HPP

#include <limits>
#include <ranges>
#include <vector>

#include "common/common.hpp"
#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "nor/rm/minimizers/minimizers.hpp"
#include "nor/rm/pruning.hpp"

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
 *
 * Besides the minimizer payload, the record owns ALL per-infoset solver state
 * (single source of truth -- the solvers hold no parallel TabularPolicy tables
 * anymore):
 *   - 'strategy_sum'  the cumulative average-strategy numerator. It stays zero
 *                     until 'activate_average()' materializes the historical
 *                     fetch-time baseline (a UNIFORM entry, exactly like the
 *                     former fetch_policy<average> default), after which the
 *                     solvers accumulate reach-weighted current-strategy mass.
 *   - 'current_strategy' the last recommendation of the regret minimizer (the
 *                     current policy). An EMPTY table means "never recommended"
 *                     and reads as the uniform distribution, mirroring the
 *                     former fetch_policy<current> default entry creation.
 *   - 'edge_bounds'   per-(infoset,action) probed payoff ranges used by the
 *                     pruning engine (SoA, index-aligned with the registry,
 *                     mirroring the RBPTables pattern). The {+inf,-inf} entry
 *                     marks actions whose subtree was never probed.
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
      _grow_tables();
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

   /////////////////////////////
   /// consolidated record state
   /////////////////////////////

   /// the cumulative average-strategy numerator (see class comment)
   auto& strategy_sum() { return m_strategy_sum; }
   [[nodiscard]] const auto& strategy_sum() const { return m_strategy_sum; }

   /// the last recommendation of the regret minimizer (see class comment)
   auto& current_strategy() { return m_current_strategy; }
   [[nodiscard]] const auto& current_strategy() const { return m_current_strategy; }

   /// the probability the CURRENT policy assigns to registry slot 'idx'
   /// (uniform fallback before the first recommendation)
   [[nodiscard]] double current_prob(size_t idx) const
   {
      const auto n_actions = m_data.registry.actions.size();
      return m_current_strategy.empty()
                ? (n_actions == 0 ? 0. : 1. / static_cast< double >(n_actions))
                : m_current_strategy[idx];
   }

   /// materializes the average-strategy baseline on first touch. With the
   /// default 'with_uniform_baseline' the table is filled with the uniform
   /// distribution exactly like the former fetch_policy<average> entry
   /// creation; callers overlaying user-seeded starting values pass false.
   void activate_average(bool with_uniform_baseline = true)
   {
      if(m_average_active) {
         return;
      }
      m_average_active = true;
      if(with_uniform_baseline) {
         const auto n_actions = m_data.registry.actions.size();
         if(n_actions == 0) {
            return;
         }
         const double uniform_prob = 1. / static_cast< double >(n_actions);
         std::ranges::fill(m_strategy_sum, uniform_prob);
      }
   }
   [[nodiscard]] bool average_active() const { return m_average_active; }

   /// per-action probed payoff ranges of the pruning engine (see class comment)
   auto& edge_bounds() { return m_edge_bounds; }
   [[nodiscard]] const auto& edge_bounds() const { return m_edge_bounds; }

   /// traversal-stamp bookkeeping of the solvers' touched-infoset sweeps:
   /// stamp of the last traversal whose update path touched this node
   size_t sweep_stamp = 0;

  private:
   /// grows the record-owned per-action tables to cover newly registered
   /// actions (fresh slots are inert: zero sums, unprobed bounds). The
   /// current-strategy slots start out UNIFORM -- exactly like the former
   /// fetch_policy<current> default entry creation a fresh traversal read.
   void _grow_tables()
   {
      const auto n_actions = m_data.registry.actions.size();
      if(m_strategy_sum.size() < n_actions) {
         m_strategy_sum.resize(n_actions, 0.);
         const double uniform_prob = n_actions == 0 ? 0. : 1. / static_cast< double >(n_actions);
         m_current_strategy.resize(n_actions, uniform_prob);
      }
      if(m_edge_bounds.size() < n_actions) {
         m_edge_bounds.resize(
            n_actions,
            pruning::PayoffBound{
               .lower = std::numeric_limits< double >::infinity(),
               .upper = -std::numeric_limits< double >::infinity()}
         );
      }
   }

   [[no_unique_address]] NodeDataType m_data;
   std::vector< double > m_strategy_sum{};
   std::vector< double > m_current_strategy{};
   std::vector< pruning::PayoffBound > m_edge_bounds{};
   bool m_average_active = false;
};

}  // namespace nor::rm

#endif  // NOR_NODE_HPP
