
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
concept action = requires(T t)
{
   // placeholder for possible refinement later
   true;
};

template < typename T >
concept observation = requires(T t)
{
   // placeholder for possible refinement later
   true;
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
   typename Observation = typename T::observation_type>
concept info_state =
   action< Action > && observation< Observation > && is::sized< T > && requires(T t)
{
   true;
};

template < typename T >
concept world_state = requires(T t)
{
   // placeholder for possible refinement later
   true;
};

template <
   typename T,
   typename Infostate = typename T::info_state_type,
   typename Action = typename T::action_type,
   typename Observation = typename T::observation_type >
concept policy = info_state< Infostate > && requires(T obj, Infostate istate, Action action)
{
   std::is_move_constructible_v< T >;
   std::is_move_assignable_v< T >;

   /// getitem methods by (istate, action) tuple and istate only
   {
      obj[std::pair{istate, action}]
      } -> std::convertible_to< double >;
   {
      obj[istate]
      } -> std::convertible_to< std::vector< Action > >;

   /// const getitem methods
} && requires(T const obj, Infostate istate, Action action)
{
   {
      obj[std::pair{istate, action}]
      } -> std::convertible_to< double >;
   {
      obj[istate]
      } -> std::convertible_to< std::vector< Action > >;
};

template <
   typename T,
   typename Action = typename T::action_type,
   typename Infostate = typename T::info_state_type,
   typename Publicstate = typename T::public_state_type,
   typename Worldstate = typename T::world_state_type,
   typename Observation = typename T::observation_type >
// clang-format off
concept fosg =
   action<Action>
   && info_state< Infostate>
   && public_state< Publicstate >
   && world_state< Worldstate >
   && observation< Observation >
   && has::method::actions< T >
   && has::method::transition< T >
   && has::method::private_observation< T>
   && has::method::public_observation<T >
   && has::method::observation< T >;
// clang-format on

}  // namespace nor::concepts

#endif  // NOR_CONCRETE_HPP
