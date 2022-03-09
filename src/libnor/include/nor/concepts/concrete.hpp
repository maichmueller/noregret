
#ifndef NOR_CONCRETE_HPP
#define NOR_CONCRETE_HPP

#include <concepts>
#include <range/v3/all.hpp>
#include <utility>
#include <vector>

#include "has.hpp"
#include "is.hpp"
#include "nor/fosg_traits.hpp"
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

template <
   typename Map,
   typename KeyType = typename Map::key_type,
   typename MappedType = typename Map::mapped_type >
concept map = iterable< Map > && requires(Map m, KeyType key, MappedType mapped)
{
   typename Map::key_type;
   typename Map::mapped_type;
   {
      m.find(key)
      } -> std::same_as< typename Map::iterator >;
   {
      m.at(key)
      } -> std::same_as< MappedType& >;
   {
      std::as_const(m).at(key)
      } -> std::same_as< const MappedType& >;
};

template < typename T >
concept action = is::hashable< T >;

template < typename T >
concept observation = is::hashable< T >;

template < typename T, typename Observation = typename fosg_auto_traits< T >::observation_type >
// clang-format off
concept public_state =
/**/  observation< Observation >
   && is::sized< T >
   && is::hashable< T >
   && std::equality_comparable< T >
   && has::method::append<
         T,
         std::pair< /*action_=*/Observation, /*state_=*/Observation>&,
         std::pair< /*action_=*/Observation, /*state_=*/Observation>
      >
   && has::method::getitem<
         T,
         std::pair< /*action_=*/Observation, /*state_=*/Observation>&,
         size_t
      >;
// clang-format on

template < typename T, typename Observation = typename fosg_auto_traits< T >::observation_type >
// clang-format off
concept info_state =
/**/  public_state< T, Observation >
   && has::method::player< const T >;
// clang-format on

template < typename T >
concept world_state = true;

template < typename T, typename Action = typename fosg_auto_traits< T >::action_type >
// clang-format off
concept action_policy = true;
//      is::sized< T >
//   && iterable< T >
//   && has::method::getitem< T, double&, Action >
//   && has::method::getitem< const T, double, Action >;
// clang-format on

template <
   typename T,
   typename Infostate,
   typename Observation,
   typename ActionPolicy = typename T::action_policy_type >
// clang-format off
concept state_policy =
/**/  info_state< Infostate, Observation >
   && action_policy< ActionPolicy >
   && has::trait::action_policy_type< T >
   && has::method::getitem<
         T,
         ActionPolicy&,
         Infostate
      >
   && has::method::getitem<
         T const,
         const ActionPolicy&,
         Infostate
      >;
// clang-format on

template <
   typename T,
   typename Infostate,
   typename Observation,
   typename ActionPolicy = typename T::action_policy_type >
// clang-format off
concept default_state_policy =
/**/  info_state< Infostate, Observation >
   && action_policy< ActionPolicy >
   && has::trait::action_policy_type< T >
   && has::method::getitem<
         T,
         ActionPolicy,
         const std::pair<
            Infostate,
            std::vector<typename fosg_auto_traits<ActionPolicy>::action_type>
         >&
      >
   && has::method::getitem<
         T const,
         ActionPolicy,
         const std::pair<
            Infostate,
            std::vector<typename fosg_auto_traits<ActionPolicy>::action_type>
         >&
      >;
// clang-format on

template <
   typename Env,
   typename Action = typename nor::fosg_auto_traits< Env >::action_type,
   typename Observation = typename nor::fosg_auto_traits< Env >::observation_type,
   typename Infostate = typename nor::fosg_auto_traits< Env >::info_state_type,
   typename Publicstate = typename nor::fosg_auto_traits< Env >::public_state_type,
   typename Worldstate = typename nor::fosg_auto_traits< Env >::world_state_type >
// clang-format off
concept fosg =
/**/  action< Action >
   && observation< Observation >
   && info_state< Infostate >
   && public_state< Publicstate >
   && world_state< Worldstate >
   && is::copyable_someway< Worldstate >
   && has::method::actions< Env, const Worldstate& >
   && has::method::transition< Env, Worldstate& >
   && has::method::private_observation< Env, Worldstate, Observation >
   && has::method::public_observation< Env, Worldstate, Observation >
   && has::method::private_observation< Env, Action, Observation >
   && has::method::public_observation< Env, Action, Observation >
   && has::method::reset< Env, Worldstate& >
   && has::method::reward< const Env >
   && has::method::is_terminal< Env, Worldstate& >
   && has::method::active_player< Env >
   && has::method::players< Env >
   && has::method::max_player_count< Env >
   && has::method::player_count< Env >
   && has::method::turn_dynamic< Env >;
// clang-format on
//
// template <
//   typename T,
//   typename Game,
//   typename Policy,
//   typename Action = typename Game::action_type,
//   typename Infostate = typename Game::info_state_type,
//   typename Publicstate = typename Game::public_state_type,
//   typename Worldstate = typename Game::world_state_type,
//   typename Observation = typename Game::observation_type,
//   typename... PolicyUpdateArgs,
//   typename... ReachProbabilityArgs >
//// clang-format off
// concept cfr_variant =
///**/  fosg< Game >
//   && concepts::state_policy< Policy, Infostate, Action >
//   && has::method::current_policy< T, Policy >
//   && has::method::current_policy< const T, Policy >
//   && has::method::avg_policy< T, Policy >
//   && has::method::avg_policy< const T, Policy >
//   && has::method::iteration< T >
//   && has::method::iteration< const T >
//   && has::method::game_tree< T >
//   && has::method::game_tree< const T >
//   && requires(T t, int n_iterations, Player player_to_update) {
//   /// Method to iterate over the current policy and improve it by rolling out the game.
//   { t.iterate(n_iterations) } ->std::same_as<const Policy*>;
//    /// Alternating updates version of iterating.
//   { t.iterate(player_to_update, n_iterations) } ->std::same_as<const Policy*>;
//} && requires(
//      T t,
//      typename Policy::action_policy_type& policy,
//      PolicyUpdateArgs&&...policy_update_args
//) {
//   /// Method to update the currently stored policy in-place with incoming (probably) regret
//   values.
//   /// This would be, in the vanilla rm case, the regret-matching procedure.
//   { t.policy_update(
//         policy,
//         std::forward<PolicyUpdateArgs>(policy_update_args)...
//     )
//   } -> std::same_as< typename Policy::action_policy_type& >;
//} && requires(
//      T t,
//      typename Policy::action_policy_type& policy,
//      const typename T::cfr_node_type& node,
//      ReachProbabilityArgs&&...reach_prob_args
//) {
//    /// the factual reach probability of the given node
//   { t.m_reach_prob(
//         node,
//         std::forward<ReachProbabilityArgs>(reach_prob_args)...
//     )
//   } -> std::same_as< double >;
//    /// the counterfactual reach probability of the given node.
//   { t.cf_reach_probability(
//         node,
//         std::forward<ReachProbabilityArgs>(reach_prob_args)...
//      )
//   } -> std::same_as< double >;
//};
//// clang-format on

}  // namespace nor::concepts

#endif  // NOR_CONCRETE_HPP
