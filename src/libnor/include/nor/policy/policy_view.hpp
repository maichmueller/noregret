
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
       : m_getitem_impl([&]< typename... Args >(Args&&... args) {
            return policy.operator[](std::forward< Args >(args)...);
         })
   {
   }

   auto operator[](const action_type& action) const { return m_getitem_impl(action); }

  private:
   std::function< const action_type&(const action_type&) > m_getitem_impl;
};

template < typename Infostate, concepts::action Action >
class StatePolicyView {
  public:
   using info_state_type = Infostate;
   using action_type = Action;
   using action_policy_view_type = ActionPolicyView< Action >;

   using getitem_sig = std::tuple< const info_state_type&, const std::vector< action_type >& >;

   template < typename StatePolicy >
      requires concepts::reference_state_policy<
         StatePolicy,
         info_state_type,
         action_type,
         HashmapActionPolicy< action_type > >
   StatePolicyView(const StatePolicy& policy)
       : m_getitem_impl([&]< typename... Args >(Args&&... args) {
            return action_policy_view_type{policy.operator[](std::forward< Args >(args)...)};
         })
   {
   }

   auto operator[](getitem_sig params) const { return m_getitem_impl(std::move(params)); }

  private:
   std::function< action_policy_view_type(getitem_sig) > m_getitem_impl;
};
}  // namespace nor
#endif  // NOR_POLICY_VIEW_HPP
