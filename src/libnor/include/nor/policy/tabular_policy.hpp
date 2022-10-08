
#ifndef NOR_TABULAR_POLICY_HPP
#define NOR_TABULAR_POLICY_HPP

#include <concepts>
#include <range/v3/all.hpp>

#include "default_policy.hpp"
#include "nor/concepts.hpp"

namespace nor {

template <
   typename Infostate,
   typename ActionPolicy,
   typename DefaultPolicy = UniformPolicy< Infostate, ActionPolicy >,
   typename Table = std::unordered_map< Infostate, ActionPolicy > >
// clang-format off
requires
   concepts::map< Table >
   && concepts::default_state_policy<
         DefaultPolicy,
         Infostate,
         ActionPolicy
      >
   && concepts::action_policy< ActionPolicy >
// clang-format on
class TabularPolicy {
  public:
   using info_state_type = Infostate;
   using action_policy_type = ActionPolicy;
   using action_type = typename fosg_auto_traits< action_policy_type >::action_type;
   using default_policy_type = DefaultPolicy;
   using table_type = Table;

   TabularPolicy()
      requires common::
                  all_predicate_v< std::is_default_constructible, table_type, default_policy_type >
   = default;
   TabularPolicy(table_type table) : m_table(std::move(table)) {}
   TabularPolicy(table_type table, default_policy_type default_policy)
       : m_table(std::move(table)), m_default_policy(std::move(default_policy))
   {
   }

   auto begin() { return m_table.begin(); }
   auto end() { return m_table.end(); }
   [[nodiscard]] auto begin() const { return m_table.begin(); }
   [[nodiscard]] auto end() const { return m_table.end(); }

   template < typename... Args >
   auto emplace(Args&&... args)
   {
      return m_table.emplace(std::forward< Args >(args)...);
   }

   inline auto find(const info_state_type& infostate) { return m_table.find(infostate); }
   [[nodiscard]] inline auto find(const info_state_type& infostate) const
   {
      return m_table.find(infostate);
   }

   auto find_or_default(
      const std::tuple< const info_state_type&, const std::vector< action_type >& >&
         state_action_pair)
   {
      const auto& infostate = std::get< 0 >(state_action_pair);
      auto found_action_policy_iter = find(infostate);
      if(found_action_policy_iter == m_table.end()) {
         return m_table.emplace(infostate, m_default_policy[state_action_pair]).first;
      }
      return found_action_policy_iter;
   }

   action_policy_type& operator[](
      std::tuple< const info_state_type&, const std::vector< action_type >& > state_action_pair)
   {
      return find_or_default(state_action_pair)->second;
   }

   action_policy_type operator[](
      std::tuple< const info_state_type&, const std::vector< action_type >&, tag::normalize >
         state_action_pair)
   {
      const auto& [istate, actions, tag] = state_action_pair;
      auto& found_action_policy = find_or_default(std::tuple{istate, actions})->second;
      return normalize_action_policy(found_action_policy);
   }

   const action_policy_type& at(const info_state_type& infostate) const
   {
      return m_table.at(infostate);
   }

   const action_policy_type& at(
      std::tuple< const info_state_type&, const std::vector< action_type >& > state_any_pair) const
   {
      return at(std::get< 0 >(state_any_pair));
   }

   action_policy_type at(std::tuple< const info_state_type&, tag::normalize > infostate_tag) const
   {
      return normalize_action_policy(m_table.at(std::get< 0 >(infostate_tag)));
   }

   action_policy_type at(
      std::tuple< const info_state_type&, const std::vector< action_type >&, tag::normalize >
         state_any_pair) const
   {
      return (*this)[std::get< 0 >(state_any_pair)];
   }

   [[nodiscard]] auto size() const
      requires(concepts::is::sized< table_type >)
   {
      return m_table.size();
   }

   [[nodiscard]] auto& table() const { return m_table; }

  private:
   table_type m_table;
   /// the fallback policy to use when the encountered infostate has not been observed before
   default_policy_type m_default_policy{};
};

}  // namespace nor

#endif  // NOR_TABULAR_POLICY_HPP
