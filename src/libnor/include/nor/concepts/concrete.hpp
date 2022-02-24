
#ifndef NOR_CONCRETE_HPP
#define NOR_CONCRETE_HPP

#include <concepts>
#include <range/v3/all.hpp>
#include <utility>
#include <vector>

#include "has.hpp"
#include "is.hpp"
#include "nor/game_defs.hpp"

namespace nor::concepts {

template < typename T >
// clang-format off
concept iterable =
/**/  has::method::begin< T >
   && has::method::begin< const T >
   && has::method::end< T >
   && has::method::end< const T >;
// clang-format on

template < typename T >
concept action = is::hashable< T >;

template < typename T >
concept observation = is::hashable< T >;

template <
   typename T,
   typename Action = typename T::action_type,
   typename Observation = typename T::observation_type >
// clang-format off
concept public_state =
/**/  action< Action >
   && observation< Observation >
   && is::sized< T >
   && has::method::append< T, Action >
   && has::method::append< T, Observation >;
// clang-format on

template <
   typename T,
   typename Action = typename T::action_type,
   typename Observation = typename T::observation_type >
// clang-format off
concept info_state =
/**/  action< Action >
   && observation< Observation >
   && is::sized< T >
   && is::hashable< T >
   && std::equality_comparable< T >;
// clang-format on

template < typename T >
concept world_state = std::equality_comparable< T >;

template < typename T, typename Action = typename T::action_type >
// clang-format off
concept action_policy =
/**/  std::is_move_constructible_v< T >
   && std::is_move_assignable_v< T >
   && is::sized< T >
   && iterable< T >
   && has::method::getitem< T, Action, double >
   && has::method::getitem< const T, Action, double >;
// clang-format on

template <
   typename T,
   typename Infostate = typename T::info_state_type,
   typename Action = typename T::action_type,
   typename Observation = typename T::observation_type >
// clang-format off
concept state_policy =
/**/  info_state< Infostate, Action, Observation>
   && std::is_move_constructible_v< T >
   && std::is_move_assignable_v< T >
   && has::trait::action_policy_type< T>
   && has::method::getitem<T, std::pair<Infostate, Action>, double>
   && has::method::getitem<const T, std::pair<Infostate, Action>, double>
   && requires(T obj, Infostate istate)
   {
      /// these getitem methods need to specified explicitly since concepts cannot be passed as
      /// typenames to other concepts (no nested concepts allowed). This would be necessary for
      /// has::method::getitem at the return type.
      { obj[istate] } -> action_policy< Action >;
   }
   && requires(T const obj, Infostate istate)
   {
      { obj[istate] } -> action_policy< Action >;
   };
// clang-format on

template <
   typename Env,
   typename Action = typename Env::action_type,
   typename Infostate = typename Env::info_state_type,
   typename Publicstate = typename Env::public_state_type,
   typename Worldstate = typename Env::world_state_type,
   typename Observation = typename Env::observation_type >
// clang-format off
concept fosg =
/**/  std::is_copy_constructible_v< Env >
   && std::is_copy_assignable_v< Env >
   && std::is_move_constructible_v< Env >
   && std::is_move_assignable_v< Env >
   && action< Action >
   && info_state< Infostate >
   && public_state< Publicstate >
   && world_state< Worldstate >
   && observation< Observation >
   && has::method::actions< Env >
   && has::method::transition< Env >
   && has::method::private_observation< Env >
   && has::method::public_observation< Env >
   && has::method::observation< Env >
   && has::method::reset< Env, Worldstate& >
   && has::method::world_state< Env, Worldstate& >
   && has::method::world_state< const Env, uptr< Worldstate > >
   && has::method::reward< const Env >
   && has::method::is_terminal< Env, Worldstate& >
   && has::method::players< Env >
   && has::trait::max_player_count< Env >
   && has::trait::turn_dynamic< Env >;
// clang-format on

template <
   typename T,
   typename Game,
   typename Policy,
   typename Action = typename Game::action_type,
   typename Infostate = typename Game::info_state_type,
   typename Publicstate = typename Game::public_state_type,
   typename Worldstate = typename Game::world_state_type,
   typename Observation = typename Game::observation_type,
   typename... PolicyUpdateArgs,
   typename... ReachProbabilityArgs >
// clang-format off
concept cfr_variant =
/**/  fosg< Game >
   && concepts::state_policy< Policy, Infostate, Action >
   && has::method::current_policy< T, Policy >
   && has::method::current_policy< const T, Policy >
   && has::method::avg_policy< T, Policy >
   && has::method::avg_policy< const T, Policy >
   && has::method::iteration< T >
   && has::method::iteration< const T >
   && has::method::game_tree< T >
   && has::method::game_tree< const T >
   && requires(T t, int n_iterations, Player player_to_update) {
   /// Method to iterate over the current policy and improve it by rolling out the game.
   { t.iterate(n_iterations) } ->std::same_as<const Policy*>;
    /// Alternating updates version of iterating.
   { t.iterate(player_to_update, n_iterations) } ->std::same_as<const Policy*>;
} && requires(
      T t,
      typename Policy::action_policy_type& policy,
      PolicyUpdateArgs&&...policy_update_args
) {
   /// Method to update the currently stored policy in-place with incoming (probably) regret values.
   /// This would be, in the vanilla rm case, the regret-matching procedure.
   { t.policy_update(
         policy,
         std::forward<PolicyUpdateArgs>(policy_update_args)...
     )
   } -> std::same_as< typename Policy::action_policy_type& >;
} && requires(
      T t,
      typename Policy::action_policy_type& policy,
      const typename T::cfr_node_type& node,
      ReachProbabilityArgs&&...reach_prob_args
) {
    /// the factual reach probability of the given node
   { t.m_reach_prob(
         node,
         std::forward<ReachProbabilityArgs>(reach_prob_args)...
     )
   } -> std::same_as< double >;
    /// the counterfactual reach probability of the given node.
   { t.cf_reach_probability(
         node,
         std::forward<ReachProbabilityArgs>(reach_prob_args)...
      )
   } -> std::same_as< double >;
};
// clang-format on

}  // namespace nor::concepts

#endif  // NOR_CONCRETE_HPP
