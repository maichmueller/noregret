
#ifndef NOR_HAS_HPP
#define NOR_HAS_HPP

#include <optional>
#include <utility>
#include <vector>

#include "nor/game_defs.hpp"

namespace nor::concepts::has {
namespace method {

template <
   typename T,
   typename Worldstate = typename T::world_state_type,
   typename Action = typename T::action_type >
concept actions = requires(T const t, Worldstate worldstate)
{
   // legal actions getter a the given player
   {
      t.actions(worldstate, std::declval< Player >())
      } -> std::convertible_to< std::vector< typename T::action_type > >;
   // legal actions getter for each given player
   {
      t.actions(worldstate, /*player_list=*/std::declval< std::vector< Player > >())
      } -> std::convertible_to< std::map< Player, std::vector< Action > > >;
};

template <
   typename T,
   typename Worldstate = typename T::world_state_type,
   typename Action = typename T::action_type >
concept transition = requires(T t, Worldstate worldstate, Action action)
{
   // apply the action on the currently stored world state inplace
   {
      t.transition(action)
      } -> std::convertible_to< std::optional< Worldstate > >;
}
&&requires(T t, Worldstate& worldstate, Action action)
{
   // apply the action on the given world state inplace.
   {
      t.transition(worldstate, action)
      } -> std::same_as< void >;
};

template < typename T, typename Worldstate = typename T::world_state_type >
concept reward = requires(T t, Worldstate wstate)
{
   {
      t.reward()
      } -> std::convertible_to< double >;
}
&&requires(T const t, Worldstate& wstate)
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
concept is_terminal = requires(T t)
{
   {
      t.is_terminal()
      } -> std::same_as< bool >;
}
&&requires(T const t, Worldstate& wstate)
{
   {
      t.is_terminal(wstate)
      } -> std::same_as< bool >;
};

template < typename T, typename Worldstate = typename T::world_state_type >
concept active_player = requires(T const t)
{
   {
      t.active_player()
      } -> std::same_as< Player >;
}
&&requires(T const t, Worldstate& wstate)
{
   {
      t.active_player(wstate)
      } -> std::same_as< Player >;
};

template < typename T, typename Worldstate = typename T::world_state_type >
concept world_state = requires(T const t)
{
   {
      t.world_state()
      } -> std::same_as< Worldstate >;
};

template <
   typename T,
   typename Worldstate = typename T::world_state_type,
   typename Publicstate = typename T::public_state_type >
concept public_state = requires(T t)
{
   {
      t.public_state()
      } -> std::same_as< Publicstate >;
} && requires(T const t, Worldstate& wstate)
{
   {
      t.public_state(wstate)
      } -> std::same_as< Publicstate >;
};;

template <
   typename T,
   typename Worldstate = typename T::world_state_type,
   typename Privatestate = typename T::private_state_type >
concept info_state = requires(T const t, Worldstate wstate, Player player)
{
   {
      t.info_state(wstate, player)
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

template < typename T, typename InputT, typename OutputT >
concept getitem = requires(T t, InputT inp)
{
   /// getitem method for input type returning an output type
   {
      t[inp]
      } -> std::same_as< OutputT >;
};

template < typename T, typename InputT, typename OutputT >
concept const_getitem = requires(T const t, InputT inp)
{
   /// getitem method for input type returning an output type
   {
      t[inp]
      } -> std::same_as< OutputT >;
};

}  // namespace method

namespace trait {

template < typename T >
concept max_player_count = requires(T t)
{
   T::max_player_count;
};

template < typename T >
concept player_count = requires(T t)
{
   T::player_count;
};

template < typename T >
concept turn_dynamic = requires(T t)
{
   std::same_as< typename T::turn_dynamic, TurnDynamic >;
};

}  // namespace trait

}  // namespace nor::concepts::has

#endif  // NOR_HAS_HPP
