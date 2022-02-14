
#ifndef NOR_HAS_HPP
#define NOR_HAS_HPP

#include <optional>
#include <utility>
#include <vector>

#include "nor/game_defs.hpp"

namespace nor::concepts::has {
namespace method {

template < typename T, typename Action = typename T::action_type >
concept actions = requires(T const t)
{
   // legal actions getter a the given player
   {
      t.actions(std::declval< Player >())
      } -> std::convertible_to< std::span< typename T::action_type > >;
   // legal actions getter for each given player
   {
      t.actions(/*player_list=*/std::declval< std::vector< Player > >())
      } -> std::convertible_to< std::map< Player, std::span< Action > > >;
};

template <
   typename T,
   typename State = typename T::state_type,
   typename Action = typename T::action_type >
concept transition = requires(T t, State state, Action action)
{
   // sample a new state from the given state and action
   // Can return a nullopt if the given state is terminal and thus doesnt have a state to transition
   // into
   {
      t.transition(state, action)
      } -> std::convertible_to< int >;
};

template <
   typename T,
   typename Worldstate = typename T::world_state_type,
   typename Action = typename T::action_type >
concept reward = requires(T const t, Worldstate wstate, Action action)
{
   {
      t.reward(wstate, action)
      } -> std::convertible_to< double >;
};

template < typename T >
concept run = requires(T const t)
{
   {
      t.run()
      } -> std::same_as< void >;
};

template < typename T, typename Observation = typename T::observation_type >
concept private_observation = requires(T const t)
{
   // get only private obervations
   {
      t.private_observation(/*player_list=*/std::declval< std::vector< Player > >())
      } -> std::convertible_to< std::map< Player, Observation > >;
   {
      t.private_observation(/*player=*/std::declval< Player >())
      } -> std::convertible_to< Observation >;
};

template < typename T, typename Observation = typename T::observation_type >
concept public_observation = requires(T const t)
{
   // get only public obervations
   {
      t.public_observation(/*player_list=*/std::declval< std::vector< Player > >())
      } -> std::convertible_to< std::map< Player, Observation > >;
   {
      t.public_observation(/*player=*/std::declval< Player >())
      } -> std::convertible_to< Observation >;
};

template < typename T, typename Observation = typename T::observation_type >
concept observation = requires(T const t)
{
   // get the full observation: private and public
   {
      t.observation(/*player_list=*/std::declval< std::vector< Player > >())
      } -> std::convertible_to< std::map< Player, std::pair< Observation, Observation > > >;
   {
      t.observation(/*player=*/std::declval< Player >())
      } -> std::convertible_to< std::pair< Observation, Observation > >;
};


}  // namespace method

namespace trait {


}
}  // namespace nor::concepts::has

#endif  // NOR_HAS_HPP
