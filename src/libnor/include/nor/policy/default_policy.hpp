
#ifndef NOR_UNIFORM_POLICY_HPP
#define NOR_UNIFORM_POLICY_HPP

#include <concepts>
#include <range/v3/all.hpp>

#include "nor/concepts.hpp"

namespace nor {

/**
 * @brief Creates a uniform state_policy taking in the given state and action type.
 *
 * This will return a uniform probability vector over the legal actions as provided by the
 * LegalActionsFilter. The filter takes in an object of state type and returns the legal actions. In
 * the case of fixed action size it will simply return dummy 0 values to do nothing.
 * Each probability vector can be further accessed by the action index to receive the action
 * probability.
 *
 * @tparam Infostate, the state type. This is the key type for 'lookup' (not actually a lookup here)
 * in the state_policy.
 * @tparam extent, the number of legal actions at any given time. If std::dynamic_extent, then the
 * legal actions have to be supplied.
 * @tparam LegalActionFunctor, type of callable that provides the legal actions in the case of
 * dynamic_extent.
 */
template < typename Infostate, typename ActionPolicy, std::size_t extent = std::dynamic_extent >
   requires concepts::info_state< Infostate > && concepts::action_policy< ActionPolicy >
class UniformPolicy {
  public:
   using info_state_type = Infostate;
   using action_policy_type = ActionPolicy;
   using action_type = auto_action_type< action_policy_type >;

   UniformPolicy() = default;

   action_policy_type operator()(
      const info_state_type&,
      std::span< const action_type > legal_actions
   ) const
   {
      if constexpr(extent == std::dynamic_extent) {
         double uniform_p = 1. / static_cast< double >(legal_actions.size());
         return action_policy_type(legal_actions, uniform_p);
      } else {
         // Otherwise we can compute directly the uniform probability vector
         constexpr double uniform_p = 1. / static_cast< double >(extent);
         return action_policy_type(extent, uniform_p);
      }
   }
};

/**
 * @brief Creates a state_policy with 0 as the probability value for any new states.

 * @tparam Infostate, the state type. This is the key type for 'lookup' (not actually a lookup here)
 * in the state_policy.
 * @tparam extent, the number of legal actions at any given time. If std::dynamic_extent, then the
 * legal actions have to be supplied.
 * @tparam LegalActionFunctor, type of callable that provides the legal actions in the case of
 * dynamic_extent.
 */
template < typename Infostate, typename ActionPolicy, std::size_t extent = std::dynamic_extent >
   requires concepts::info_state< Infostate > && concepts::action_policy< ActionPolicy >
class ZeroDefaultPolicy {
  public:
   using info_state_type = Infostate;
   using action_policy_type = ActionPolicy;
   using action_type = auto_action_type< action_policy_type >;

   ZeroDefaultPolicy() = default;

   action_policy_type operator()(
      const info_state_type&,
      std::span< const action_type > legal_actions
   ) const
   {
      if constexpr(extent == std::dynamic_extent) {
         return action_policy_type(legal_actions, 0.);
      } else {
         return action_policy_type(extent, 0.);
      }
   }

   action_policy_type operator()(const info_state_type&) const
      requires(extent != std::dynamic_extent)
   {
      return action_policy_type(extent, 0.);
   }
};

}  // namespace nor

#endif  // NOR_UNIFORM_POLICY_HPP
