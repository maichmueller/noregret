
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
      } -> std::convertible_to< std::vector< typename T::action_type > >;
   // legal actions getter for each given player
   {
      t.actions(/*player_list=*/std::declval< std::vector< Player > >())
      } -> std::convertible_to< std::map< Player, std::vector< Action > > >;
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

template < typename T, typename Worldstate = typename T::world_state_type >
concept reward = requires(T const t, Worldstate wstate)
{
   {
      t.reward(wstate)
      } -> std::convertible_to< double >;
};

template < typename T >
concept run = requires(T const t)
{
   {
      t.run()
      } -> std::same_as< void >;
};

template < typename T, typename ReturnType = void >
concept reset = requires(T const t)
{
   {
      t.reset()
      } -> std::same_as< ReturnType >;
};

template < typename T, typename Worldstate = typename T::world_state_type >
concept is_terminal = requires(T const t, Worldstate wstate)
{
   {
      t.is_terminal(wstate)
      } -> std::same_as< bool >;
};

template < typename T, typename Worldstate = typename T::world_state_type >
concept world_state = requires(T const t)
{
   {
      t.world_state()
      } -> std::same_as< Worldstate >;
};

template < typename T, typename Publicstate = typename T::public_state_type >
concept public_state = requires(T const t)
{
   {
      t.public_state()
      } -> std::same_as< Publicstate >;
};

template < typename T, typename Privatestate = typename T::private_state_type >
concept info_state = requires(T const t, Player p)
{
   {
      t.info_state(p)
      } -> std::same_as< Privatestate >;
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
