
#ifndef NOR_STATE_POLICY_VIEW_HPP
#define NOR_STATE_POLICY_VIEW_HPP

#include "nor/concepts.hpp"
#include "nor/utils/utils.hpp"

namespace nor {

template < typename Infostate, concepts::action Action, typename ActionPolicy >
class StatePolicyView {
  public:
   using info_state_type = Infostate;
   using action_type = Action;
   using action_policy_type = ActionPolicy;

   using getitem_sig = std::tuple< const info_state_type&, const std::vector< action_type >& >;

   template < typename StatePolicy >
      requires concepts::state_policy<
         StatePolicy,
         info_state_type,
         action_type,
         HashmapActionPolicy< action_type > >
   StatePolicyView(const StatePolicy& policy)
       : m_getitem_impl([&]< typename... Args >(Args&&... args) {
            return policy.operator[](std::forward< Args >(args)...);
         })
   {
   }

   auto operator[](getitem_sig params) const { return m_getitem_impl(std::move(params)); }

  private:
   const action_policy_type& (*m_getitem_impl)(getitem_sig);
};
}  // namespace nor
#endif  // NOR_STATE_POLICY_VIEW_HPP
