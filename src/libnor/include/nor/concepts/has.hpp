
#ifndef NOR_HAS_HPP
#define NOR_HAS_HPP

#include <optional>
#include <utility>
#include <vector>

#include "nor/game_defs.hpp"
#include "nor/type_defs.hpp"
#include "nor/utils/type_traits.hpp"

namespace nor::concepts::has {
namespace method {

template < typename T, typename GameType = typename T::game_type >
concept game = requires(T&& t)
{
   {
      t.game()
      } -> std::same_as< nor::const_as_t< T, GameType >& >;
};

template < typename T, typename Policy = typename T::policy_type >
concept current_policy = requires(T&& t)
{
   {
      t.current_policy()
      } -> std::same_as< nor::const_as_t< T, Policy >& >;
};

template < typename T, typename Policy = typename T::policy_type >
concept avg_policy = requires(T&& t)
{
   {
      t.avg_policy()
      } -> std::same_as< nor::const_as_t< T, Policy >& >;
};

template < typename T >
concept iteration = requires(T&& t)
{
   {
      t.iteration()
      } -> std::convertible_to< size_t >;
};

template < typename T, typename TreeContainer = typename T::tree_type >
concept game_tree = requires(T&& t)
{
   {
      t.game_tree()
      } -> std::same_as< nor::const_as_t< T, TreeContainer >& >;
};

template <
   typename T,
   typename Worldstate = typename T::world_state_type,
   typename Action = typename T::action_type >
concept actions = requires(T const t, Worldstate&& worldstate, Player player)
{
   // legal actions getter for the given player
   {
      t.actions(player, std::forward< Worldstate >(worldstate))
      } -> std::convertible_to< std::vector< Action > >;
};

template < typename T, typename ReturnType, typename... Args >
concept append = requires(T&& t, Args&&... args)
{
   // append object u to t
   {
      t.append(std::forward< Args >(args)...)
      } -> std::same_as< ReturnType& >;
};

template <
   typename T,
   typename Worldstate = typename T::world_state_type,
   typename Action = typename T::action_type >
concept transition = requires(T&& t, Worldstate&& worldstate, Action action)
{
   // apply the action on the given world state inplace.
   {
      t.transition(action, std::forward< Worldstate >(worldstate))
      } -> std::same_as< void >;
};

template <
   typename T,
   typename Worldstate = typename T::world_state_type,
   typename Infostate = typename T::info_state_type,
   typename Publicstate = typename T::public_state_type,
   typename Action = typename T::action_type >
concept transition_jointly = requires(
   T&& t,
   Action action,
   Worldstate& worldstate,
   std::map< Player, Infostate >& infostates,
   Publicstate& pubstate)
{
   // apply the action on the given world state inplace.
   {
      t.transition(action, worldstate, infostates, pubstate)
      } -> std::same_as< void >;
};

template < typename T, typename Worldstate = typename T::world_state_type >
concept reward = requires(T&& t, Worldstate wstate, Player player)
{
   {
      t.reward(player, wstate)
      } -> std::convertible_to< double >;
};

template < typename T, typename Worldstate = typename T::world_state_type >
concept reward_multi = requires(T&& t, Worldstate& wstate, const std::vector< Player >& players)
{
   {
      t.reward(players, wstate)
      } -> std::convertible_to< std::vector< double > >;
};

template < typename T >
concept run = requires(T&& t)
{
   {
      t.run()
      } -> std::same_as< void >;
};

template < typename T, typename ReturnType = void >
concept reset = requires(T&& t)
{
   {
      t.reset()
      } -> std::same_as< ReturnType >;
};

template < typename T >
concept players = requires(T&& t)
{
   {
      t.players()
      } -> std::same_as< std::vector< Player > >;
};

template < typename T >
concept player = requires(T&& t)
{
   {
      t.player()
      } -> std::same_as< Player >;
};

template < typename T, typename Worldstate = typename T::world_state_type >
concept is_terminal = requires(T&& t, Worldstate& wstate)
{
   {
      t.is_terminal(wstate)
      } -> std::same_as< bool >;
};

template < typename T, typename Worldstate = typename T::world_state_type >
concept active_player = requires(T&& t, Worldstate wstate)
{
   {
      t.active_player(wstate)
      } -> std::same_as< Player >;
};

template < typename T, typename Worldstate = typename T::world_state_type >
concept initial_world_state = requires(T&& t)
{
   {
      t.initial_world_state()
      } -> std::same_as< Worldstate >;
};

template <
   typename T,
   typename Publicstate = typename T::public_state_type,
   typename Worldstate = typename T::world_state_type >
concept public_state = requires(T&& t, const Worldstate& wstate)
{
   {
      t.public_state(wstate)
      } -> std::same_as< Publicstate >;
};

template <
   typename T,
   typename Worldstate = typename T::world_state_type,
   typename Infostate = typename T::info_state_type >
concept info_state = requires(T&& t, Worldstate wstate, Player player)
{
   {
      t.info_states(wstate, player)
      } -> std::same_as< Infostate >;
};

template <
   typename T,
   typename Worldstate = typename T::world_state_type,
   typename Observation = typename T::observation_type >
concept private_observation = requires(T&& t, Player player, Worldstate wstate)
{
   {
      t.private_observation(player, wstate)
      } -> std::convertible_to< Observation >;
};
template <
   typename T,
   typename Worldstate = typename T::world_state_type,
   typename Observation = typename T::observation_type >
concept private_observation_multi =
   requires(T&& t, const std::vector< Player >& player_list, Worldstate wstate)
{
   // get only private obervations.
   // Output is a paired vector of observations, i.e. output[i] is paired to player_list[i].
   {
      t.private_observation(player_list, wstate)
      } -> std::same_as< std::vector< Observation > >;
};

template <
   typename T,
   typename Worldstate = typename T::world_state_type,
   typename Observation = typename T::observation_type >
concept public_observation = requires(T&& t, Player player, Worldstate wstate)
{
   {
      t.public_observation(player, wstate)
      } -> std::convertible_to< Observation >;
};

template <
   typename T,
   typename Worldstate = typename T::world_state_type,
   typename Observation = typename T::observation_type >
concept public_observation_multi =
   requires(T&& t, const std::vector< Player >& player_list, Worldstate wstate)
{
   // get only private obervations for the vector of players.
   // Output is a paired vector of observations, i.e. output[i] is paired to player_list[i].
   {
      t.public_observation(player_list, wstate)
      } -> std::same_as< std::vector< Observation > >;
};

template <
   typename T,
   typename Worldstate = typename T::world_state_type,
   typename Observation = typename T::observation_type >
concept observation = requires(T&& t, Player player, Worldstate wstate)
{
   {
      t.observation(player, wstate)
      } -> std::convertible_to< Observation >;
};

template <
   typename T,
   typename Worldstate = typename T::world_state_type,
   typename Observation = typename T::observation_type >
concept observation_multi =
   requires(T&& t, const std::vector< Player >& player_list, Worldstate wstate)
{
   // get full observation: private and public.
   // Output is a paired vector of observations, i.e. output[i] is paired to player_list[i].
   {
      t.observation(player_list, wstate)
      } -> std::same_as< std::vector< Observation > >;
};

template < typename T, template < typename > class Ptr, typename ElemT >
concept clone_other = std::is_pointer_v< Ptr< ElemT > > && requires(T&& t, Ptr< ElemT > ptr)
{
   {
      t->clone(ptr)
      } -> std::convertible_to< Ptr< ElemT > >;
};

template < typename T, typename Output = T >
concept clone_self = requires(T const t)
{
   {
      t.clone()
      } -> std::convertible_to< Output >;
};

template < class Ptr >
concept clone_ptr = std::is_pointer_v< Ptr > && requires(Ptr const ptr)
{
   {
      ptr->clone()
      } -> std::convertible_to< uptr< typename Ptr::element_type > >;
};

template < typename T, typename U = T >
concept copy = requires(T const t)
{
   {
      t.copy()
      } -> std::same_as< U >;
};

template < typename T, typename OutputT, typename InputT >
concept getitem = requires(T&& t, InputT inp)
{
   /// getitem method for input type returning an output type
   {
      t[inp]
      } -> std::same_as< OutputT >;
};

template < typename T, typename OutputT, typename InputT >
concept at = requires(T&& t, InputT inp)
{
   /// getitem method for input type returning an output type
   {
      t.at(inp)
      } -> std::same_as< OutputT >;
};

//template < typename T, typename OutputT, typename...InputTs >
//concept getitem_overloaded = requires(T&& t, InputTs&&... inp)
//{
//   /// getitem method for input type returning an output type
//   {
//      t[inp]
//      } -> std::same_as< OutputT >;
//};

//template < typename T, typename OutputT, typename... InputTs >
//concept getitem = requires(T&& t, InputTs&&... inp)
//{
//   /// getitem method for input type returning an output type
//   {
//      t[inp]
//      } -> std::same_as< OutputT >;
//};
//
//template < typename T, typename OutputT, typename... InputT1, typename... InputTs >
//concept getitem_for_each_arg = requires(T&& t, InputT1 inp, InputTs&&... rest_inp)
//{
//   /// getitem method for input type returning an output type
//   invocable_with_each_v
//};

template < typename T, typename U = typename T::iterator >
concept begin = requires(T t)
{
   {
      t.begin()
      }
      -> std::same_as< std::conditional_t< std::is_const_v< T >, typename T::const_iterator, U > >;
};

template < typename T, typename U = typename T::iterator >
concept end = requires(T&& t)
{
   {
      t.end()
      }
      -> std::same_as< std::conditional_t< std::is_const_v< T >, typename T::const_iterator, U > >;
};

template < typename T, typename MapLikePolicy, typename Action >
concept policy_update = requires(T t, MapLikePolicy policy, std::map< Action, double > regrets)
{
   {
      t.policy_update(policy, regrets)
      } -> std::same_as< void >;
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

template < typename T >
concept action_policy_type = requires(T t)
{
   typename T::action_policy_type;
};

template < typename T >
concept action_type = requires(T t)
{
   typename T::action_type;
};

template < typename T >
concept observation_type = requires(T t)
{
   typename T::observation_type;
};

template < typename T >
concept info_state_type = requires(T t)
{
   typename T::info_state_type;
};

template < typename T >
concept public_state_type = requires(T t)
{
   typename T::public_state_type;
};

template < typename T >
concept world_state_type = requires(T t)
{
   typename T::world_state_type;
};

}  // namespace trait

}  // namespace nor::concepts::has

#endif  // NOR_HAS_HPP
