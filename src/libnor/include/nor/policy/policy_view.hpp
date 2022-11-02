
#ifndef NOR_POLICY_VIEW_HPP
#define NOR_POLICY_VIEW_HPP

#include "default_policy.hpp"
#include "nor/concepts.hpp"
#include "nor/utils/utils.hpp"

/// This file provides type erasure classes which allow accessing the underlying contained values
/// without infering the precise types of the policies used. This enables e.g. type agnostic
/// containers of policies.

namespace nor {

template < concepts::action Action >
struct ActionPolicyInterface {
   using action_type = Action;

   template < typename T >
   ActionPolicyInterface(const T& t) : iterator_source(t)
   {
   }

   virtual ~ActionPolicyInterface() = default;

   using mapped_type = std::pair< const action_type, double >;
   using view_iterator = decltype(std::declval< ranges::any_view< const mapped_type& > >().begin());

   view_iterator begin() { return iterator_source.begin(); };
   view_iterator end() { return iterator_source.end(); }

   view_iterator begin() const { return iterator_source.begin(); };
   view_iterator end() const { return iterator_source.end(); }

   virtual size_t size() const = 0;

   virtual double at(const action_type& action) const = 0;

  protected:
   /// type erased view to provide an iterator basis for the underlying policies.
   /// while costly, this view allows us to provide a standard begin(), end() range functionality to
   /// the policies under the view
   ranges::any_view< const mapped_type& > iterator_source;
};

template < concepts::action Action >
class ActionPolicyView {
  public:
   using action_type = Action;
   using interface_type = ActionPolicyInterface< action_type >;

   template < typename ActionPolicy >
      requires concepts::action_policy< std::remove_cvref_t< ActionPolicy >, action_type >
               and (not std::is_same_v< std::remove_cvref_t< ActionPolicy >, ActionPolicyView >)
   ActionPolicyView(ActionPolicy&& obj) : view(_init(std::forward< ActionPolicy >(obj)))
   {
   }

   ~ActionPolicyView() = default;
   ActionPolicyView& operator=(const ActionPolicyView& other) = default;
   ActionPolicyView& operator=(ActionPolicyView&& other) = default;
   ActionPolicyView(const ActionPolicyView& obj) = default;
   ActionPolicyView(ActionPolicyView&& obj) = default;

   auto begin() const { return view->begin(); };
   auto end() const { return view->end(); }

   size_t size() const { return view->size(); }

   double at(const action_type& action) const { return view->at(action); }

   template < typename T >
   struct View: interface_type {
      View(T& t) : interface_type(t), policy(&t) {}

      virtual size_t size() const { return policy->size(); }

      virtual double at(const action_type& action) const override { return policy->at(action); }

     private:
      T* policy;
   };

   template < typename T >
   struct OwningView: interface_type {
      OwningView(T t) : policy(std::move(t)), interface_type(policy) {}

      virtual size_t size() const { return policy->size(); }

      virtual double at(const action_type& action) const override { return policy.at(action); }

     private:
      T policy;
   };

   sptr< const interface_type > view;

  private:
   template < typename T >
   auto _init(T&& obj)
   {
      if constexpr(std::is_lvalue_reference_v< T >) {
         return std::make_shared< View< std::remove_reference_t< T > > >(obj);
      } else {
         return std::make_shared< OwningView< T > >(std::move(obj));
      }
   }
};

// template < concepts::action Action >
// class ActionPolicyView {
//   public:
//    using action_type = Action;
//
//    template < typename ActionPolicy >
//       requires concepts::action_policy< ActionPolicy, action_type >
//    ActionPolicyView(const ActionPolicy& policy)
//        : m_getitem_impl(std::bind_front(
//           static_cast< double (ActionPolicy::*)(const action_type&) const >(
//              &ActionPolicy::operator[]
//           ),
//           policy
//        )),
//          m_at_impl(std::bind_front(
//             static_cast< double (ActionPolicy::*)(const action_type&) const >(&ActionPolicy::at),
//             policy
//          )),
//          m_size_impl(std::bind_front(&ActionPolicy::size, policy))
//    {
//    }
//
//    auto operator[](const action_type& action) const { return m_getitem_impl(action); }
//    auto at(const action_type& action) const { return m_at_impl(action); }
//
//   private:
//    std::function< double(const action_type&) > m_getitem_impl;
//    std::function< double(const action_type&) > m_at_impl;
//    std::function< size_t() > m_size_impl;
// };

template < typename Infostate, concepts::action Action >
struct StatePolicyInterface {
   using action_type = Action;
   using info_state_type = Infostate;
   using action_policy_view_type = ActionPolicyView< Action >;

   virtual ~StatePolicyInterface() = default;

   virtual action_policy_view_type at(const info_state_type&) const = 0;
};

template < typename Infostate, concepts::action Action >
class StatePolicyView {
  public:
   using action_type = Action;
   using info_state_type = Infostate;
   using state_policy_view_type = StatePolicyInterface< info_state_type, action_type >;
   using action_policy_view_type = typename StatePolicyInterface< info_state_type, action_type >::
      action_policy_view_type;

   template < typename StatePolicy >
      requires concepts::
         state_policy_no_default< std::remove_cvref_t< StatePolicy >, info_state_type, action_type >
      StatePolicyView(StatePolicy&& obj) : view(_init(std::forward< StatePolicy >(obj)))
   {
   }

   ~StatePolicyView() = default;
   StatePolicyView& operator=(const StatePolicyView& other) = default;
   StatePolicyView& operator=(StatePolicyView&& other) = default;
   StatePolicyView(const StatePolicyView& obj) = default;
   StatePolicyView(StatePolicyView&& obj) = default;

   auto at(const info_state_type& infostate) const { return view->at(infostate); }

   template < typename T >
   struct View: state_policy_view_type {
      View(T& t) : policy(&t) {}

      action_policy_view_type at(const info_state_type& infostate) const override
      {
         return policy->at(infostate);
      }

     private:
      T* policy;
   };

   template < typename T >
   struct OwningView: state_policy_view_type {
      OwningView(T t) : policy(std::move(t)) {}

      action_policy_view_type at(const info_state_type& infostate) const override
      {
         return policy.at(infostate);
      }

     private:
      T policy;
   };

  private:
   sptr< state_policy_view_type > view;

   template < typename T >
   auto _init(T&& obj)
   {
      if constexpr(std::is_lvalue_reference_v< T >) {
         return std::make_shared< View< std::remove_reference_t< T > > >(obj);
      } else {
         return std::make_shared< OwningView< T > >(std::move(obj));
      }
   }
};

// template < typename Infostate, concepts::action Action >
// class StatePolicyView {
//   public:
//    using info_state_type = Infostate;
//    using action_type = Action;
//    using action_policy_view_type = ActionPolicyView< Action >;
//
//    using getitem_sig = common::tuple_of_const_ref< info_state_type, std::vector< action_type > >;
//
//    template < typename StatePolicy >
//       requires concepts::reference_state_policy<
//                   StatePolicy,
//                   info_state_type,
//                   action_type,
//                   typename StatePolicy::action_policy_type >
//    StatePolicyView(StatePolicy& policy)
//        : m_getitem_tuple_impl([&]< typename... Args >(Args&&... args) {
//             return ActionPolicyView< Action >{policy.operator[](std::forward< Args >(args)...)};
//          }),
//          m_at_tuple_impl([&]< typename... Args >(Args&&... args) {
//             return ActionPolicyView< Action >{policy.at(std::forward< Args >(args)...)};
//          }),
//          m_at_infostate_impl([const_policy = std::cref(policy)]< typename... Args >(Args&&...
//          args
//                              ) {
//             return ActionPolicyView< Action >{
//                ActionPolicyView< Action >{const_policy.get().at(std::forward< Args >(args)...)}};
//          })
//    {
//    }
//
//    auto operator[](getitem_sig params) const { return m_getitem_tuple_impl(std::move(params)); }
//    auto at(getitem_sig params) const { return m_tuple_impl(std::move(params)); }
//    auto at(const info_state_type& infostate) const { return m_at_infostate_impl(infostate); }
//
//   private:
//    std::function< action_policy_view_type(getitem_sig) > m_getitem_tuple_impl;
//    std::function< action_policy_view_type(getitem_sig) > m_at_tuple_impl;
//    std::function< action_policy_view_type(const info_state_type&) > m_at_infostate_impl;
// };

}  // namespace nor
#endif  // NOR_POLICY_VIEW_HPP
