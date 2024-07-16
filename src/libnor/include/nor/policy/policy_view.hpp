
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

   auto begin() { return iterator_source.begin(); };
   auto end() { return iterator_source.end(); }

   auto begin() const { return iterator_source.begin(); };
   auto end() const { return iterator_source.end(); }

   [[nodiscard]] virtual size_t size() const = 0;

   [[nodiscard]] virtual double at(const action_type& action) const = 0;

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

   //   auto begin() const { return view->begin(); };
   //   auto end() const { return view->end(); }

   size_t size() const { return view->size(); }

   double at(const action_type& action) const { return view->at(action); }

   template < typename T >
   struct View: interface_type {
      View(T& t) : interface_type(t), policy(&t) {}

      size_t size() const override { return policy->size(); }

      double at(const action_type& action) const override { return policy->at(action); }

     private:
      T* policy;
   };

   template < typename T >
   struct OwningView: interface_type {
      OwningView(T t) : interface_type(policy), policy(std::move(t)) {}

      size_t size() const override { return policy.size(); }

      double at(const action_type& action) const override { return policy.at(action); }

     private:
      T policy;
   };

   sptr< const interface_type > view;

  private:
   template < typename T >
   auto _init(T&& obj)
   {
      using T_raw = std::remove_reference_t< T >;
      using view_type = std::
         conditional_t< std::is_lvalue_reference_v< T >, View< T_raw >, OwningView< T_raw > >;
      return std::make_shared< view_type >(std::forward< T >(obj));
   }
};

// deduction guide for how to deduce the action type from an action policy type if not explicitly
// provided upon instantiation.
template < typename ActionPolicy >
ActionPolicyView(ActionPolicy&& policy)
   -> ActionPolicyView< auto_action_type< std::decay_t< ActionPolicy > > >;

template < typename Infostate, concepts::action Action >
struct StatePolicyInterface {
   using action_type = Action;
   using info_state_type = Infostate;
   using action_policy_view_type = ActionPolicyView< Action >;

   virtual ~StatePolicyInterface() = default;

   virtual action_policy_view_type at(const info_state_type&) const = 0;
   [[nodiscard]] virtual size_t size() const = 0;
};

template < typename Infostate, concepts::action Action >
class StatePolicyView {
  public:
   using action_type = Action;
   using info_state_type = Infostate;
   using state_policy_type = StatePolicyInterface< info_state_type, action_type >;
   using action_policy_type = typename StatePolicyInterface< info_state_type, action_type >::
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
   struct View: state_policy_type {
      View(T& t) : policy(&t) {}

      action_policy_type at(const info_state_type& infostate) const override
      {
         return policy->at(infostate);
      }
      size_t size() const override { return policy->size(); }

     private:
      T* policy;
   };

   template < typename T >
   struct OwningView: state_policy_type {
      OwningView(T t) : policy(std::move(t)) {}

      action_policy_type at(const info_state_type& infostate) const override
      {
         return policy.at(infostate);
      }
      [[nodiscard]] size_t size() const override { return policy.size(); }

     private:
      T policy;
   };

  private:
   sptr< state_policy_type > view;

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

// deduction guide for how to deduce infostate and action from a state policy type if not explicitly
// provided upon instantiation.
template < typename StatePolicy >
StatePolicyView(StatePolicy&& policy) -> StatePolicyView<
   auto_info_state_type< std::decay_t< StatePolicy > >,
   auto_action_type< std::decay_t< StatePolicy > > >;

}  // namespace nor

#endif  // NOR_POLICY_VIEW_HPP
