#ifndef NOR_CFR_TABULAR_SOLVER_OPERATIONS_HPP
#define NOR_CFR_TABULAR_SOLVER_OPERATIONS_HPP

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "nor/concepts.hpp"
#include "nor/rm/rm_utils.hpp"

namespace nor::rm {

namespace detail {

/// Shared lifetime state for direct policy views. The solver owns the token;
/// views retain only a weak reference so that a view cannot keep a destroyed
/// solver's storage alive (or appear valid after the solver is destroyed).
struct PolicyGeneration {
   size_t value = 0;

   void invalidate() noexcept { ++value; }
};

template < typename T >
inline constexpr bool dependent_false_v = false;

template < PolicyLabel Label, typename Node >
struct NodePolicyValueReader {
   const Node* node = nullptr;

   [[nodiscard]] double value_at(size_t index) const
   {
      if constexpr(Label == PolicyLabel::current) {
         return node->current_prob(index);
      } else {
         const auto& sums = node->strategy_sum();
         // ExponentialCFR keeps the denominator beside the minimizer's node
         // payload. Match the existing average_policy() materialization when
         // that optional table is present; all other families expose raw sums.
         if constexpr(requires(const Node& value_node, size_t i) {
                         value_node.data().avg_policy_denominator[i];
                      }) {
            return sums[index] / node->data().avg_policy_denominator[index];
         } else {
            return sums[index];
         }
      }
   }
};

template < PolicyLabel Label, typename Node, typename ActionPolicy >
struct TablePolicyValueReader {
   const Node* node = nullptr;
   const ActionPolicy* policy = nullptr;

   [[nodiscard]] double value_at(size_t index) const { return policy->at(node->actions()[index]); }
};

}  // namespace detail

template < typename PolicySource, typename InfoState, concepts::action Action >
class TabularPolicyLookup;

namespace detail {

template < typename Derived, typename InfoState, concepts::action Action >
class TabularSolverOperations;

}  // namespace detail

/**
 * @brief allocation-free, read-only view of one tabular information node's policy.
 *
 * A view contains no policy table of its own. It reads the node's action
 * registry and a source-specific current/average value record directly, and
 * returns individual actions and values by value. The view is valid only while
 * the originating solver has not been moved, destroyed, iterated, or otherwise
 * mutated. Every accessor checks the generation token and throws
 * std::logic_error for a stale view; valid() can be used by adapters that prefer
 * a boolean check.
 *
 * For PolicyLabel::current, values are the current strategy probabilities. For
 * PolicyLabel::average, values have the same representation as the solver's
 * existing average_policy() table: cumulative strategy mass for ordinary CFR
 * minimizers, and the exponential minimizer's numerator divided by its stored
 * per-action denominator. No reference or iterator into solver storage escapes
 * this view.
 */
template <
   PolicyLabel Label,
   concepts::action Action,
   typename Node,
   typename ValueReader = detail::NodePolicyValueReader< Label, Node > >
class TabularPolicyNodeView {
  public:
   using action_type = Action;
   using node_type = Node;
   using value_reader_type = ValueReader;

   [[nodiscard]] bool valid() const noexcept
   {
      const auto generation = m_generation.lock();
      return m_node != nullptr and generation != nullptr
             and generation->value == m_expected_generation;
   }

   [[nodiscard]] size_t generation() const noexcept { return m_expected_generation; }

   [[nodiscard]] size_t size() const
   {
      _check_current();
      return m_node->actions().size();
   }

   /// Returns an action by value so callers cannot retain a reference into the node registry.
   [[nodiscard]] action_type action_at(size_t index) const
   {
      _check_current();
      if(index >= m_node->actions().size()) {
         throw std::out_of_range("TabularPolicyNodeView: action index out of range");
      }
      return m_node->actions()[index];
   }

   [[nodiscard]] double value_at(size_t index) const
   {
      _check_current();
      if(index >= m_node->actions().size()) {
         throw std::out_of_range("TabularPolicyNodeView: value index out of range");
      }
      return _value_at_unchecked(index);
   }

   [[nodiscard]] double at(const action_type& action) const
   {
      _check_current();
      return _value_at_unchecked(m_node->index_of(action));
   }

   [[nodiscard]] std::optional< double > find(const action_type& action) const
   {
      _check_current();
      const auto& actions = m_node->actions();
      const auto found = std::ranges::find(actions, action);
      if(found == actions.end()) {
         return std::nullopt;
      }
      return _value_at_unchecked(static_cast< size_t >(found - actions.begin()));
   }

   [[nodiscard]] bool contains(const action_type& action) const
   {
      _check_current();
      return std::ranges::find(m_node->actions(), action) != m_node->actions().end();
   }

   /// Compatibility with action-policy lookup syntax, while retaining value semantics.
   [[nodiscard]] double operator[](const action_type& action) const { return at(action); }

   /**
    * Calls fn(action, value) in the node's deterministic registry order. The
    * callback is the only scope in which the action reference is borrowed.
    */
   template < typename Fn >
   void for_each(Fn&& fn) const
   {
      _check_current();
      for(size_t index = 0; index < m_node->actions().size(); ++index) {
         std::invoke(fn, m_node->actions()[index], _value_at_unchecked(index));
      }
   }

  private:
   using generation_type = detail::PolicyGeneration;

   TabularPolicyNodeView(
      const Node& node,
      ValueReader value_reader,
      std::weak_ptr< const generation_type > generation,
      size_t expected_generation
   )
       : m_node(std::addressof(node)),
         m_value_reader(std::move(value_reader)),
         m_generation(std::move(generation)),
         m_expected_generation(expected_generation)
   {
   }

   void _check_current() const
   {
      if(not valid()) {
         throw std::logic_error(
            "TabularPolicyNodeView is stale; obtain a new view after the solver changes"
         );
      }
   }

   [[nodiscard]] double _value_at_unchecked(size_t index) const
   {
      return m_value_reader.value_at(index);
   }

   template < typename PolicySource, typename InfoState, concepts::action ActionT >
   friend class TabularPolicyLookup;

   const Node* m_node = nullptr;
   ValueReader m_value_reader{};
   std::weak_ptr< const generation_type > m_generation;
   size_t m_expected_generation = 0;
};

namespace detail {

/**
 * @brief policy source for solvers whose node records own both policy values.
 */
template < typename NodeMap, typename InfoState, concepts::action Action >
class NodePolicySource {
  public:
   using node_map_type = NodeMap;
   using info_state_type = InfoState;
   using action_type = Action;
   using node_type = typename node_map_type::mapped_type;

   template < PolicyLabel Label >
   using reader_type = NodePolicyValueReader< Label, node_type >;

   template < PolicyLabel Label >
   using view_type = TabularPolicyNodeView< Label, action_type, node_type, reader_type< Label > >;

   template < PolicyLabel Label >
   struct entry_type {
      const node_type* node;
      reader_type< Label > reader;
   };

   explicit NodePolicySource(const node_map_type& nodes) : m_nodes(std::addressof(nodes)) {}

   [[nodiscard]] bool valid() const noexcept { return m_nodes != nullptr; }

   template < PolicyLabel Label >
   [[nodiscard]] std::optional< entry_type< Label > > find(const info_state_type& infostate) const
   {
      const auto found = m_nodes->find(infostate);
      if(found == m_nodes->end()) {
         return std::nullopt;
      }
      if constexpr(Label == PolicyLabel::average) {
         if(not found->second.average_active()) {
            return std::nullopt;
         }
      }
      return entry_type< Label >{&found->second, reader_type< Label >{&found->second}};
   }

   template < PolicyLabel Label, typename Fn >
   void for_each(Fn&& fn) const
   {
      for(const auto& [infostate, node] : *m_nodes) {
         if constexpr(Label == PolicyLabel::average) {
            if(not node.average_active()) {
               continue;
            }
         }
         std::invoke(fn, *infostate, entry_type< Label >{&node, reader_type< Label >{&node}});
      }
   }

  private:
   const node_map_type* m_nodes;
};

/**
 * @brief policy source for MCCFR's legacy table-backed policy records.
 *
 * MCCFR keeps its current and cumulative average policies in the base policy
 * tables while its information-node map owns the legal-action registry. This
 * source joins those existing records by information state, so direct views
 * still enumerate only visited internal nodes and never materialize a table.
 */
template <
   typename NodeMap,
   typename InfoState,
   concepts::action Action,
   typename CurrentPolicyMap,
   typename AveragePolicyMap >
class TablePolicySource {
  public:
   using node_map_type = NodeMap;
   using info_state_type = InfoState;
   using action_type = Action;
   using node_type = typename node_map_type::mapped_type;
   using current_policy_type = typename CurrentPolicyMap::mapped_type;
   using average_policy_type = typename AveragePolicyMap::mapped_type;
   using current_action_policy_type = typename current_policy_type::action_policy_type;
   using average_action_policy_type = typename average_policy_type::action_policy_type;

   template < PolicyLabel Label >
   using action_policy_type = std::conditional_t<
      Label == PolicyLabel::current,
      current_action_policy_type,
      average_action_policy_type >;

   template < PolicyLabel Label >
   using reader_type = TablePolicyValueReader< Label, node_type, action_policy_type< Label > >;

   template < PolicyLabel Label >
   using view_type = TabularPolicyNodeView< Label, action_type, node_type, reader_type< Label > >;

   template < PolicyLabel Label >
   struct entry_type {
      const node_type* node;
      reader_type< Label > reader;
   };

   TablePolicySource(
      const node_map_type& nodes,
      const CurrentPolicyMap& current_policies,
      const AveragePolicyMap& average_policies
   )
       : m_nodes(std::addressof(nodes)),
         m_current_policies(std::addressof(current_policies)),
         m_average_policies(std::addressof(average_policies))
   {
   }

   [[nodiscard]] bool valid() const noexcept
   {
      return m_nodes != nullptr and m_current_policies != nullptr and m_average_policies != nullptr;
   }

   template < PolicyLabel Label >
   [[nodiscard]] std::optional< entry_type< Label > > find(const info_state_type& infostate) const
   {
      const auto found_node = m_nodes->find(infostate);
      if(found_node == m_nodes->end()) {
         return std::nullopt;
      }
      return _entry< Label >(infostate, found_node->second);
   }

   template < PolicyLabel Label, typename Fn >
   void for_each(Fn&& fn) const
   {
      for(const auto& [infostate, node] : *m_nodes) {
         if(auto entry = _entry< Label >(*infostate, node)) {
            std::invoke(fn, *infostate, std::move(*entry));
         }
      }
   }

  private:
   template < PolicyLabel Label >
   [[nodiscard]] std::optional< entry_type< Label > >
   _entry(const info_state_type& infostate, const node_type& node) const
   {
      const auto& policy_map = [&]() -> const auto& {
         if constexpr(Label == PolicyLabel::current) {
            return *m_current_policies;
         } else {
            return *m_average_policies;
         }
      }();
      const auto found_player = policy_map.find(infostate.player());
      if(found_player == policy_map.end()) {
         return std::nullopt;
      }
      const auto found_policy = found_player->second.find(infostate);
      if(found_policy == found_player->second.end()) {
         return std::nullopt;
      }
      return entry_type< Label >{
         &node, reader_type< Label >{&node, std::addressof(found_policy->second)}};
   }

   const node_map_type* m_nodes;
   const CurrentPolicyMap* m_current_policies;
   const AveragePolicyMap* m_average_policies;
};

}  // namespace detail

/**
 * @brief direct lookup/visitor adapter for a solver-owned information-node map.
 *
 * Constructing this adapter and finding a node do not allocate or materialize a
 * policy table. It is intentionally a short-lived handle: it becomes stale at
 * the same time as views returned from it, namely on the next solver mutation.
 */
template < typename PolicySource, typename InfoState, concepts::action Action >
class TabularPolicyLookup {
  public:
   using source_type = PolicySource;
   using node_map_type = typename source_type::node_map_type;
   using info_state_type = InfoState;
   using action_type = Action;
   using node_type = typename source_type::node_type;

   template < PolicyLabel Label >
   using view_type = typename source_type::template view_type< Label >;

   [[nodiscard]] bool valid() const noexcept
   {
      const auto generation = m_generation.lock();
      return m_source.valid() and generation != nullptr
             and generation->value == m_expected_generation;
   }

   [[nodiscard]] size_t generation() const noexcept { return m_expected_generation; }

   template < PolicyLabel Label >
   [[nodiscard]] std::optional< view_type< Label > > find(const info_state_type& infostate) const
   {
      _check_current();
      auto found = m_source.template find< Label >(infostate);
      if(not found) {
         return std::nullopt;
      }
      // Construct the view here, where the lookup is a friend of the private
      // view constructor, then move the value into the optional. Constructing
      // the optional in-place would make std::optional itself the constructor's
      // access context on some standard library implementations.
      return view_type< Label >{
         *found->node, std::move(found->reader), m_generation, m_expected_generation};
   }

   template < PolicyLabel Label >
   [[nodiscard]] view_type< Label > at(const info_state_type& infostate) const
   {
      auto found = find< Label >(infostate);
      if(not found) {
         throw std::out_of_range(
            "TabularPolicyLookup: information state is not present in the requested policy"
         );
      }
      return std::move(*found);
   }

   /**
    * Calls fn(infostate, view) for every registered node in the requested
    * policy. A one-argument fn(view) is also accepted. The callback runs while
    * the lookup is current; views retained beyond a solver mutation fail their
    * generation check.
    *
    * @return the number of policy nodes visited.
    */
   template < PolicyLabel Label, typename Fn >
   size_t visit(Fn&& fn) const
   {
      _check_current();
      size_t visited = 0;
      m_source.template for_each< Label >([&](const info_state_type& infostate, auto entry) {
         view_type< Label > view{
            *entry.node, std::move(entry.reader), m_generation, m_expected_generation};
         if constexpr(std::invocable< Fn&, const info_state_type&, const view_type< Label >& >) {
            std::invoke(fn, infostate, view);
         } else if constexpr(std::invocable< Fn&, const view_type< Label >& >) {
            std::invoke(fn, view);
         } else {
            static_assert(
               detail::dependent_false_v< Fn >,
               "TabularPolicyLookup visitor must accept (infostate, view) or (view)"
            );
         }
         ++visited;
      });
      return visited;
   }

  private:
   using generation_type = detail::PolicyGeneration;

   TabularPolicyLookup(source_type source, const std::shared_ptr< generation_type >& generation)
       : m_source(std::move(source)),
         m_generation(generation),
         m_expected_generation(generation->value)
   {
   }

   void _check_current() const
   {
      if(not valid()) {
         throw std::logic_error(
            "TabularPolicyLookup is stale; obtain a new lookup after the solver changes"
         );
      }
   }

   template < typename Derived, typename InfoStateT, concepts::action ActionT >
   friend class detail::TabularSolverOperations;

   source_type m_source;
   std::weak_ptr< const generation_type > m_generation;
   size_t m_expected_generation = 0;
};

namespace detail {

/**
 * @brief common operation and direct-policy layer for iterative tabular solvers.
 *
 * Derived solvers provide two private hooks:
 *   - _iterate_one(), which performs exactly one regular iteration and returns
 *     its root StateValueMap, and
 *   - _policy_source(), which returns a small, non-owning source adapter for
 *     the solver's visited node records and policy values.
 *
 * The layer owns the generation token shared by all direct policy handles. Its
 * step wrapper invalidates old handles before entering the derived traversal;
 * derived public mutation entry points can call _invalidate_policy_views() as
 * well when they bypass the step wrapper.
 */
template < typename Derived, typename InfoState, concepts::action Action >
class TabularSolverOperations {
  public:
   using root_value_type = StateValueMap;

   TabularSolverOperations() = default;
   TabularSolverOperations(const TabularSolverOperations&) = delete;
   TabularSolverOperations& operator=(const TabularSolverOperations&) = delete;

   // Transfer the token with the solver's storage. Invalidation makes every pre-move view stale,
   // while leaving the moved-from solver without an owner ensures moved-to views expire when the
   // moved-to solver is destroyed.
   TabularSolverOperations(TabularSolverOperations&& other) noexcept
       : m_generation(std::move(other.m_generation))
   {
      if(m_generation != nullptr) {
         m_generation->invalidate();
      }
   }

   TabularSolverOperations& operator=(TabularSolverOperations&& other) noexcept
   {
      if(this != std::addressof(other)) {
         if(m_generation != nullptr) {
            m_generation->invalidate();
            m_generation.reset();
         }
         m_generation = std::move(other.m_generation);
         if(m_generation != nullptr) {
            m_generation->invalidate();
         }
      }
      return *this;
   }

   /** Execute exactly one regular solver iteration and return its root value map. */
   [[nodiscard]] root_value_type iterate()
   {
      _ensure_generation();
      return _run_step([this] { return _derived()._iterate_one(); });
   }

   /** Execute n iterations and discard each root value immediately. */
   void advance(size_t n_iters)
   {
      _ensure_generation();
      for([[maybe_unused]] auto _ : std::views::iota(size_t{0}, n_iters)) {
         (void) _run_step([this] { return _derived()._iterate_one(); });
      }
   }

   /**
    * Execute n iterations while retaining only the last root value. For n ==
    * 0, no iteration is performed and std::nullopt is returned.
    */
   [[nodiscard]] std::optional< root_value_type > advance_last(size_t n_iters)
   {
      _ensure_generation();
      if(n_iters == 0) {
         return std::nullopt;
      }
      std::optional< root_value_type > last;
      for([[maybe_unused]] auto _ : std::views::iota(size_t{0}, n_iters)) {
         last.emplace(_run_step([this] { return _derived()._iterate_one(); }));
      }
      return last;
   }

   /**
    * Execute n iterations and collect the root value after iterations every,
    * 2*every, ... up to n. A zero cadence is invalid; a final partial interval
    * is intentionally not added, so callers that need the final value should
    * use advance_last().
    */
   [[nodiscard]] std::vector< root_value_type > trace(size_t n_iters, size_t every = 1)
   {
      if(every == 0) {
         throw std::invalid_argument("TabularSolverOperations::trace requires every > 0");
      }
      _ensure_generation();
      std::vector< root_value_type > values;
      values.reserve(n_iters / every);
      for(size_t completed = 0; completed < n_iters; ++completed) {
         auto value = _run_step([this] { return _derived()._iterate_one(); });
         if((completed + 1) % every == 0) {
            values.emplace_back(std::move(value));
         }
      }
      return values;
   }

   [[nodiscard]] size_t policy_generation() const
   {
      _ensure_generation();
      return m_generation->value;
   }

   /** Return a non-owning, generation-checked direct lookup adapter. */
   [[nodiscard]] auto policy_lookup() const
   {
      _ensure_generation();
      using source_type = std::remove_cvref_t< decltype(_derived()._policy_source()) >;
      return TabularPolicyLookup< source_type, InfoState, Action >{
         _derived()._policy_source(), m_generation};
   }

   template < PolicyLabel Label >
   [[nodiscard]] auto policy_at(const InfoState& infostate) const
   {
      return policy_lookup().template find< Label >(infostate);
   }

   [[nodiscard]] auto current_policy_at(const InfoState& infostate) const
   {
      return policy_at< PolicyLabel::current >(infostate);
   }

   [[nodiscard]] auto average_policy_at(const InfoState& infostate) const
   {
      return policy_at< PolicyLabel::average >(infostate);
   }

   template < PolicyLabel Label, typename Fn >
   size_t visit_policy_nodes(Fn&& fn) const
   {
      return policy_lookup().template visit< Label >(std::forward< Fn >(fn));
   }

   template < typename Fn >
   size_t visit_current_policy(Fn&& fn) const
   {
      return visit_policy_nodes< PolicyLabel::current >(std::forward< Fn >(fn));
   }

   template < typename Fn >
   size_t visit_average_policy(Fn&& fn) const
   {
      return visit_policy_nodes< PolicyLabel::average >(std::forward< Fn >(fn));
   }

  protected:
   template < typename Fn >
   decltype(auto) _run_step(Fn&& fn)
   {
      _invalidate_policy_views();
      return std::invoke(std::forward< Fn >(fn));
   }

   void _invalidate_policy_views()
   {
      _ensure_generation();
      m_generation->invalidate();
   }

   void _ensure_generation() const
   {
      if(m_generation == nullptr) {
         m_generation = std::make_shared< PolicyGeneration >();
      }
   }

  private:
   [[nodiscard]] Derived& _derived() noexcept { return static_cast< Derived& >(*this); }
   [[nodiscard]] const Derived& _derived() const noexcept
   {
      return static_cast< const Derived& >(*this);
   }

   mutable std::shared_ptr< PolicyGeneration > m_generation = std::make_shared< PolicyGeneration >(
   );
};

}  // namespace detail

}  // namespace nor::rm

#endif  // NOR_CFR_TABULAR_SOLVER_OPERATIONS_HPP
