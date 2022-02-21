
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
   && has::method::const_begin< T >
   && has::method::end< T >
   && has::method::const_end< T >;
// clang-format on

template < typename T >
concept action = requires(T t)
{
   is::hashable< T >;
};

template < typename T >
concept observation = requires(T t)
{
   is::hashable< T >;
};

template <
   typename T,
   typename Action = typename T::action_type,
   typename Observation = typename T::observation_type >
concept public_state =
   action< Action > && observation< Observation > && is::sized< T > && requires(T t)
{
   true;
};

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
concept maplike_policy =
/**/  std::is_move_constructible_v< T >
   && std::is_move_assignable_v< T >
   && is::sized<T>
   && iterable< T>
   && has::method::getitem< T, Action, double >
   && has::method::const_getitem< T, Action, double >;
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
   && has::method::getitem<T, std::pair<Infostate, Action>, double>
   && has::method::const_getitem<T, std::pair<Infostate, Action>, double>
   && requires(T obj, Infostate istate)
   {
      /// these getitem methods need to specified explicitly since concepts cannot be passed as
      /// typenames to other concepts (no nested concepts allowed). This would be necessary for
      /// has::method::getitem at the return type.
      { obj[istate] } -> maplike_policy< Action >;
   }
   && requires(T const obj, Infostate istate)
   {
      { obj[istate] } -> maplike_policy< Action >;
   };
// clang-format on

/**
 * A merely semantic specification of state policy. This concept expects to have all entries
 * initialized to 0. This cannot be enforce by mere syntactic concepts though.
 */
template <
   typename T,
   typename Infostate = typename T::info_state_type,
   typename Action = typename T::action_type,
   typename Observation = typename T::observation_type >
concept zeroed_state_policy = state_policy< T, Infostate, Action, Observation >;

template <
   typename Game,
   typename Action = typename Game::action_type,
   typename Infostate = typename Game::info_state_type,
   typename Publicstate = typename Game::public_state_type,
   typename Worldstate = typename Game::world_state_type,
   typename Observation = typename Game::observation_type >
// clang-format off
concept fosg =
/**/  std::is_copy_constructible_v< Game >
   && std::is_copy_assignable_v< Game >
   && std::is_move_constructible_v< Game >
   && std::is_move_assignable_v< Game >
   && action<Action>
   && info_state< Infostate>
   && public_state< Publicstate >
   && world_state< Worldstate >
   && observation< Observation >
   && has::method::actions< Game >
   && has::method::transition< Game >
   && has::method::private_observation< Game>
   && has::method::public_observation<Game >
   && has::method::observation< Game >
   && has::method::reset< Game, Worldstate& >
   && has::method::info_state< Game, std::shared_ptr<Infostate> >
   && has::method::public_state< Game, std::shared_ptr<Publicstate> >
   && has::method::world_state< Game, std::shared_ptr<Worldstate> >
   && has::method::reward< Game >
   && has::method::is_terminal< Game, Worldstate& >
   && has::method::players< Game >
   && has::trait::player_count< Game >
   && has::trait::max_player_count< Game >
   && has::trait::turn_dynamic< Game >;
// clang-format on

}  // namespace nor::concepts

#endif  // NOR_CONCRETE_HPP
