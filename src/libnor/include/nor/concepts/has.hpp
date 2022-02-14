
#ifndef NOR_HAS_HPP
#define NOR_HAS_HPP

#include <optional>
#include <utility>
#include <vector>

#include "nor/game_defs.hpp"

namespace nor::concepts::has::method {

template < typename T >
concept actions = requires(T const t)
{
   // legal actions getter a the given player
   {
      t.actions(std::declval< Player >())
      } -> std::convertible_to< std::span< typename T::action_t > >;
   // legal actions getter for each given player
   {
      t.actions(/*player_list=*/std::declval< std::vector< Player > >())
      } -> std::convertible_to< std::map< Player, std::span< typename T::action_t > > >;
};

template < typename T, typename State, typename Action >
concept transition = requires(T t, State state, Action action)
{
   // sample a new state from the given state and action
   // Can return a nullopt if the given state is terminal and thus doesnt have a state to transition
   // into
   {
      t.transition(state, action)
      } -> std::same_as< std::optional< State > >;
};

template < typename T >
concept run_game = requires(T const t)
{
   {
      t.run()
      } -> std::same_as< void >;
};

template < typename T >
concept private_observation = requires(T const t)
{
   // get only private obervations
   {
      t.private_observation(/*player_list=*/std::declval< std::vector< Player > >())
      } -> std::convertible_to< std::map< Player, typename T::observation_t > >;
   {
      t.private_observation(/*player=*/std::declval< Player >())
      } -> std::convertible_to< typename T::observation_t >;
};

template < typename T >
concept public_observation = requires(T const t)
{
   // get only public obervations
   {
      t.public_observation(/*player_list=*/std::declval< std::vector< Player > >())
      } -> std::convertible_to< std::map< Player, typename T::observation_t > >;
   {
      t.public_observation(/*player=*/std::declval< Player >())
      } -> std::convertible_to< typename T::observation_t >;
};

template < typename T >
concept observation = requires(T const t)
{
   // get the full observation: private and public
   {
      t.observation(/*player_list=*/std::declval< std::vector< Player > >())
      } -> std::convertible_to<
         std::map< Player, std::pair< typename T::observation_t, typename T::observation_t > > >;
   {
      t.observation(/*player=*/std::declval< Player >())
      } -> std::convertible_to< std::pair< typename T::observation_t, typename T::observation_t > >;
};

}  // namespace nor::concepts::has::method

#endif  // NOR_HAS_HPP
