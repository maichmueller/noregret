
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
   typename Table = std::unordered_map< Infostate, ActionPolicy > >
// clang-format off
requires
   concepts::map< Table >
   && concepts::action_policy< ActionPolicy >
// clang-format on
class TabularPolicy {
  public:
   using info_state_type = Infostate;
   using action_policy_type = ActionPolicy;
   using action_type = typename fosg_auto_traits< action_policy_type >::action_type;
   using table_type = Table;

   TabularPolicy()
      requires std::is_default_constructible_v< table_type >
   = default;
   TabularPolicy(table_type table) : m_table(std::move(table)) {}

   auto begin() { return m_table.begin(); }
   auto end() { return m_table.end(); }
   [[nodiscard]] auto begin() const { return m_table.begin(); }
   [[nodiscard]] auto end() const { return m_table.end(); }

   template < typename... Args >
   auto emplace(Args&&... args)
   {
      return m_table.emplace(std::forward< Args >(args)...);
   }
   template < typename IstateType, typename... Args >
   auto emplace(IstateType&& infostate, Args&&... args)
   {
      return m_table.emplace(
         std::piecewise_construct,
         /*key*/ std::forward_as_tuple(std::forward< IstateType >(infostate)),
         /*value*/ std::forward_as_tuple(std::initializer_list{std::forward< Args >(args)...})
      );
   }

   inline auto find(const info_state_type& infostate) { return m_table.find(infostate); }
   [[nodiscard]] inline auto find(const info_state_type& infostate) const
   {
      return m_table.find(infostate);
   }

   template < typename DefaultPolicy = UniformPolicy< Infostate, ActionPolicy > >
      requires concepts::
         default_state_policy< DefaultPolicy, info_state_type, action_type, action_policy_type >
      auto find_or_default(
         const info_state_type& infostate,
         const std::vector< action_type >& actions,
         const DefaultPolicy& default_policy
      )
   {
      auto found_action_policy_iter = find(infostate);
      if(found_action_policy_iter == m_table.end()) {
         return m_table.emplace(infostate, default_policy(infostate, actions)).first;
      }
      return found_action_policy_iter;
   }

   action_policy_type& operator()(const info_state_type& infostate)
   {
      auto found = find(infostate);
      if(found == end()) {
         throw std::invalid_argument(
            "Given Infostate not found in table and no default method provided."
         );
      }
      return found->second;
   }

   template < typename DefaultPolicy = UniformPolicy< Infostate, ActionPolicy > >
      requires concepts::
         default_state_policy< DefaultPolicy, info_state_type, action_type, action_policy_type >
      action_policy_type& operator()(
         const info_state_type& infostate,
         const std::vector< action_type >& actions,
         DefaultPolicy default_policy = {}
      )
   {
      return find_or_default(infostate, actions, default_policy)->second;
   }

   template < typename DefaultPolicy = UniformPolicy< Infostate, ActionPolicy > >
      requires concepts::
         default_state_policy< DefaultPolicy, info_state_type, action_type, action_policy_type >
      action_policy_type operator()(
         const info_state_type& infostate,
         const std::vector< action_type >& actions,
         tag::normalize,
         DefaultPolicy default_policy = {}
      )
   {
      auto& found_action_policy = find_or_default(infostate, actions, default_policy)->second;
      return normalize_action_policy(found_action_policy);
   }

   template < typename DefaultPolicy = UniformPolicy< Infostate, ActionPolicy > >
      requires concepts::
         default_state_policy< DefaultPolicy, info_state_type, action_type, action_policy_type >
      action_policy_type operator()(
         const info_state_type& infostate,
         const std::vector< action_type >& actions,
         DefaultPolicy default_policy,
         tag::normalize
      )
   {
      auto& found_action_policy = find_or_default(infostate, actions, default_policy)->second;
      return normalize_action_policy(found_action_policy);
   }

   const action_policy_type& at(const info_state_type& infostate) const
   {
      return m_table.at(infostate);
   }

   action_policy_type at(const info_state_type& infostate, tag::normalize) const
   {
      return normalize_action_policy(m_table.at(infostate));
   }

   action_policy_type at(
      std::tuple< const info_state_type&, const std::vector< action_type >&, tag::normalize >
         state_any_pair
   ) const
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
};

}  // namespace nor

#endif  // NOR_TABULAR_POLICY_HPP
