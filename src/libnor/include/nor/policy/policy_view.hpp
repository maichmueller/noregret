
#ifndef NOR_POLICY_VIEW_HPP
#define NOR_POLICY_VIEW_HPP

#include "nor/concepts.hpp"
#include "nor/utils/utils.hpp"

/// This file provides type erasure classes which allow accessing the underlying contained values
/// without infering the precise types of the policies used. This enables e.g. type agnostic
/// containers of policies.

namespace nor {

template < concepts::action Action >
class ActionPolicyView {
  public:
   using action_type = Action;

   template < typename ActionPolicy >
      requires concepts::action_policy< ActionPolicy, action_type >
   ActionPolicyView(const ActionPolicy& policy)
       : m_getitem_impl(std::bind_front(
          static_cast< double (ActionPolicy::*)(const action_type&) const >(
             &ActionPolicy::operator[]),
          policy)),
         m_at_impl(std::bind_front(
            static_cast< double (ActionPolicy::*)(const action_type&) const >(&ActionPolicy::at),
            policy)),
         m_size_impl(std::bind_front(&ActionPolicy::size, policy))
   {
   }

   auto operator[](const action_type& action) const { return m_getitem_impl(action); }
   auto at(const action_type& action) const { return m_at_impl(action); }

  private:
   std::function< double(const action_type&) > m_getitem_impl;
   std::function< double(const action_type&) > m_at_impl;
   std::function< size_t() > m_size_impl;
};

template < typename Infostate, concepts::action Action >
class StatePolicyView {
  public:
   using info_state_type = Infostate;
   using action_type = Action;
   using action_policy_view_type = ActionPolicyView< Action >;

   using getitem_sig = common::tuple_of_const_ref< info_state_type, std::vector< action_type > >;

   template < typename StatePolicy >
      requires concepts::reference_state_policy<
                  StatePolicy,
                  info_state_type,
                  action_type,
                  typename StatePolicy::action_policy_type >
   StatePolicyView(StatePolicy& policy)
       : m_getitem_tuple_impl([&]< typename... Args >(Args&&... args) {
            return ActionPolicyView< Action >{policy.operator[](std::forward< Args >(args)...)};
         }),
         m_at_tuple_impl([&]< typename... Args >(Args&&... args) {
            return ActionPolicyView< Action >{policy.at(std::forward< Args >(args)...)};
         }),
         m_at_infostate_impl(
            [const_policy = std::cref(policy)]< typename... Args >(Args&&... args) {
               return ActionPolicyView< Action >{
                  ActionPolicyView< Action >{const_policy.get().at(std::forward< Args >(args)...)}};
            })
   {
   }

   auto operator[](getitem_sig params) const { return m_getitem_tuple_impl(std::move(params)); }
   auto at(getitem_sig params) const { return m_tuple_impl(std::move(params)); }
   auto at(const info_state_type& infostate) const { return m_at_infostate_impl(infostate); }

  private:
   std::function< action_policy_view_type(getitem_sig) > m_getitem_tuple_impl;
   std::function< action_policy_view_type(getitem_sig) > m_at_tuple_impl;
   std::function< action_policy_view_type(const info_state_type&) > m_at_infostate_impl;
};
}  // namespace nor
#endif  // NOR_POLICY_VIEW_HPP
