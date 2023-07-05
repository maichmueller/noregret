
#ifndef NOR_TABULAR_POLICY_HPP
#define NOR_TABULAR_POLICY_HPP

#include <concepts>
#include <range/v3/all.hpp>

#include "default_policy.hpp"
#include "nor/concepts.hpp"
#include "nor/holder.hpp"

namespace nor {

template <
   typename Infostate,
   typename ActionPolicy,
   typename Table = std::unordered_map<
      InfostateHolder< Infostate >,
      ActionPolicy,
      common::value_hasher< Infostate >,
      common::value_comparator< Infostate > > >
// clang-format off
requires
   concepts::map_specced< Table, InfostateHolder< Infostate >, ActionPolicy >
   && concepts::action_policy< ActionPolicy >
// clang-format on
class TabularPolicy {
  public:
   using info_state_type = Infostate;
   using action_policy_type = ActionPolicy;
   using action_type = auto_action_type< action_policy_type >;
   using table_type = std::remove_cvref_t< Table >;
   using value_type = typename Table::value_type;
   using key_type = typename Table::key_type;
   using mapped_type = typename Table::mapped_type;

  private:
   static constexpr bool _allows_heterogenous_lookup = requires(table_type table) {
      table.find(std::declval< info_state_type >());
   };
   template < typename InfostateOrHolder >
   static constexpr bool _accepts_holder_and_value =
      concepts::is::holder_specialization< InfostateOrHolder, tag::infostate, info_state_type >
      or (std::same_as< InfostateOrHolder, info_state_type > and _allows_heterogenous_lookup);

  public:
   TabularPolicy()
      requires std::is_default_constructible_v< table_type >
   = default;

   template < typename OtherTableType >
      requires(not common::
                  is_specialization_v< std::remove_cvref_t< OtherTableType >, TabularPolicy >)
              and concepts::map< std::remove_cvref_t< OtherTableType > >
   TabularPolicy(OtherTableType&& table) : m_table(std::forward< OtherTableType >(table))
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
   template < typename IstateType, typename... Args >
   auto emplace(IstateType&& infostate, Args&&... args)
   {
      return m_table.emplace(
         std::piecewise_construct,
         /*key*/ std::forward_as_tuple(std::forward< IstateType >(infostate)),
         /*value*/ std::forward_as_tuple(std::forward< Args >(args)...)
      );
   }

   inline auto find(const info_state_type& infostate) { return m_table.find(infostate); }
   [[nodiscard]] inline auto find(const info_state_type& infostate) const
   {
      return m_table.find(infostate);
   }

   template <
      typename ActionContainer,
      concepts::is::holder_specialization< tag::infostate, info_state_type > InfostateHolderType,
      typename DefaultPolicy = UniformPolicy< Infostate, ActionPolicy > >
      requires concepts::
         default_state_policy< DefaultPolicy, info_state_type, action_type, action_policy_type >
      auto find_or_default(
         const InfostateHolderType& infostate,
         ActionContainer&& actions,
         const DefaultPolicy& default_policy = {}
      )
   {
      auto found_action_policy_iter = find(infostate);
      if(found_action_policy_iter == m_table.end()) {
         return m_table
            .emplace(
               infostate.template copy< InfostateHolder< info_state_type > >(),
               default_policy(infostate, std::span{actions})
            )
            .first;
      }
      return found_action_policy_iter;
   }

   template <
      typename ActionContainer,
      typename DefaultPolicy = UniformPolicy< Infostate, ActionPolicy > >
      requires concepts::default_state_policy<
                  DefaultPolicy,
                  info_state_type,
                  action_type,
                  action_policy_type >
               and _allows_heterogenous_lookup
   auto find_or_default(
      const info_state_type& infostate,
      ActionContainer&& actions,
      const DefaultPolicy& default_policy = {}
   )
   {
      auto found_action_policy_iter = find(infostate);
      if(found_action_policy_iter == m_table.end()) {
         return m_table
            .emplace(
               InfostateHolder< Infostate >{infostate},
               default_policy(infostate, std::span{actions})
            )
            .first;
      }
      return found_action_policy_iter;
   }

   template < typename InfostateOrHolder >
      requires _accepts_holder_and_value< InfostateOrHolder >
   auto& operator()(const InfostateOrHolder& infostate)
   {
      auto found = find(infostate);
      if(found == end()) {
         throw std::invalid_argument(
            "Given Infostate not found in table and no default method provided."
         );
      }
      return found->second;
   }

   template <
      typename InfostateOrHolder,
      typename ActionContainer,
      typename DefaultPolicy = UniformPolicy< Infostate, ActionPolicy > >
      requires concepts::default_state_policy<
                  DefaultPolicy,
                  info_state_type,
                  action_type,
                  action_policy_type >
               and _accepts_holder_and_value< InfostateOrHolder >
   action_policy_type& operator()(
      const InfostateOrHolder& infostate,
      ActionContainer&& actions,
      DefaultPolicy default_policy = {}
   )
   {
      return find_or_default(infostate, std::forward< ActionContainer >(actions), default_policy)
         ->second;
   }

   template < typename InfostateOrHolder, typename ActionContainer, typename DefaultPolicy >
      requires concepts::default_state_policy<
                  DefaultPolicy,
                  info_state_type,
                  action_type,
                  action_policy_type >
               and _accepts_holder_and_value< InfostateOrHolder >
   action_policy_type operator()(
      const InfostateOrHolder& infostate,
      ActionContainer&& actions,
      tag::normalize,
      DefaultPolicy default_policy = UniformPolicy< Infostate, ActionPolicy >{}
   )
   {
      auto& found_action_policy = find_or_default(
                                     infostate,
                                     std::forward< ActionContainer >(actions),
                                     default_policy
      )
                                     ->second;
      return normalize_action_policy(found_action_policy);
   }

   template <
      typename InfostateOrHolder,
      typename ActionContainer,
      typename DefaultPolicy = UniformPolicy< Infostate, ActionPolicy > >
      requires concepts::default_state_policy<
                  DefaultPolicy,
                  info_state_type,
                  action_type,
                  action_policy_type >
               and _accepts_holder_and_value< InfostateOrHolder >
               and std::
                  constructible_from< std::span< ActionHolder< action_type > >, ActionContainer >
   action_policy_type operator()(
      const InfostateOrHolder& infostate,
      ActionContainer&& actions,
      DefaultPolicy default_policy,
      tag::normalize
   )
   {
      auto& found_action_policy = find_or_default(
                                     infostate,
                                     std::forward< ActionContainer >(actions),
                                     default_policy
      )
                                     ->second;
      return normalize_action_policy(found_action_policy);
   }

   template < typename InfostateOrHolder >
      requires _accepts_holder_and_value< InfostateOrHolder >
   const action_policy_type& at(const InfostateOrHolder& infostate) const
   {
      return m_table.at(infostate);
   }

   template < typename InfostateOrHolder >
      requires _accepts_holder_and_value< InfostateOrHolder >
   action_policy_type at(const InfostateOrHolder& infostate, tag::normalize) const
   {
      return normalize_action_policy(at(infostate));
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

// deduction-guide
template < typename OtherTableType >
   requires(not common::is_specialization_v< std::remove_cvref_t< OtherTableType >, TabularPolicy >)
           and concepts::map< std::remove_cvref_t< OtherTableType > >
           and common::is_specialization_v<
              typename std::remove_cvref_t< OtherTableType >::key_type,
              InfostateHolder >
TabularPolicy(OtherTableType&& table) -> TabularPolicy<
   typename std::remove_cvref_t< OtherTableType >::key_type::type,
   typename std::remove_cvref_t< OtherTableType >::mapped_type,
   std::remove_cvref_t< OtherTableType > >;

}  // namespace nor

#endif  // NOR_TABULAR_POLICY_HPP
