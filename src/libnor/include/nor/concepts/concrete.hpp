
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

template < typename T >
concept public_state = action< typename T::action_t > && observation<
   typename T::observation_t > && is::sized< T > && requires(T t)
{
   true;
};

template < typename T >
concept info_state = action< typename T::action_t > && observation<
   typename T::observation_t > && is::sized< T > && requires(T t)
{
   true;
};

template < typename T >
concept world_state = requires(T t)
{
   // placeholder for possible refinement later
   true;
};

template < typename T, typename Infostate >
concept policy =
   info_state< Infostate > && requires(T obj,
   Infostate istate, typename Infostate::action_t action)
{
   std::is_move_constructible_v< T >;
   std::is_move_assignable_v< T >;

   /// getitem methods by (istate, action) tuple and istate only
   {
      obj[std::pair{istate, action}]
      } -> std::convertible_to< double >;
   {
      obj[istate]
      } -> std::convertible_to< std::vector< typename Infostate::action_t > >;

   /// const getitem methods
} && requires(T const obj, Infostate istate, typename Infostate::action_t action)
{
   {
      obj[std::pair{istate, action}]
      } -> std::convertible_to< double >;
   {
      obj[istate]
      } -> std::convertible_to< std::vector< typename Infostate::action_t > >;
};

// clang-format off
template < typename T >
concept game =
   info_state< typename T::info_state_t >
   && public_state< typename T::public_state_t >
   && world_state< typename T::world_state_t >
   && observation< typename T::observation_t >
   && has::method::actions< T >
   && has::method::transition< T,  >
   && has::method::private_observation< T >
   && has::method::public_observation<T >
   && has::method::observation< T >;
// clang-format on

template < typename T, typename WorldState >
concept action_function = world_state< WorldState > && requires(T const t, WorldState w)
{
   {
      t(w)
      } -> std::convertible_to< std::vector< typename WorldState::action_t > >;
};

template < typename T, typename WorldState, typename Action >
concept transition_function =
   action< Action > && world_state< WorldState > && requires(T const t, WorldState w, Action a)
{
   {
      t(w, a)
      } -> std::same_as< WorldState >;
};

template < typename T, typename WorldState, typename Action >
concept reward_function =
   action< Action > && world_state< WorldState > && requires(T const t, WorldState w, Action a)
{
   {
      t(w, a)
      } -> std::convertible_to< double >;
};

template < typename T, typename WorldState, typename Action >
concept observation_function =
   action< Action > && world_state< WorldState > && requires(T const t, WorldState w, Action a)
{
   {
      t(w, a)
      } -> std::same_as< WorldState >;
};;

template <
   typename Action,
   typename Infostate,
   typename Policy,
   typename W,
   typename A,
   typename R,
   typename O,
   typename T,
   typename Valuator >
// clang-format off
concept fosg =
      action< Action >
      && info_state< Infostate >
      && world_state< W >
      && policy< Policy, Infostate, Action >
      && action_function< A, W >
      && transition_function< T, W, Action >
      && reward_function< R, W, Action >
      && observation_function< O, W, Action >
      && valuator< Valuator, Infostate, Action >;
// clang-format on

}  // namespace nor::concepts

#endif  // NOR_CONCRETE_HPP
