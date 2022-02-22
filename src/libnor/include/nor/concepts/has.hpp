
#ifndef NOR_HAS_HPP
#define NOR_HAS_HPP

#include <optional>
#include <utility>
#include <vector>

#include "nor/game_defs.hpp"
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
concept transition = requires(T&& t, const Action& action)
{
   // apply the action on the currently stored world state inplace
   {
      t.transition(action)
      } -> std::convertible_to< std::optional< Worldstate > >;
}
&&requires(T&& t, Worldstate& worldstate, const Action& action)
{
   // apply the action on the given world state inplace.
   {
      t.transition(worldstate, action)
      } -> std::same_as< void >;
};

template < typename T, typename Worldstate = typename T::world_state_type >
concept reward = requires(T&& t, Player player)
{
   {
      t.reward(player)
      } -> std::convertible_to< double >;
}
&&requires(T&& t, Worldstate&& wstate, Player player)
{
   {
      t.reward(wstate, player)
      } -> std::convertible_to< double >;
};

template < typename T, typename Worldstate = typename T::world_state_type >
concept reward_all = requires(T&& t)
{
   {
      t.reward()
      } -> std::convertible_to< std::map< Player, double > >;
}
&&requires(T&& t, Worldstate& wstate)
{
   {
      t.reward(wstate)
      } -> std::convertible_to< std::map< Player, double > >;
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

template < typename T, typename Worldstate = typename T::world_state_type >
concept is_terminal = requires(T&& t)
{
   {
      t.is_terminal()
      } -> std::same_as< bool >;
}
&&requires(T&& t, Worldstate& wstate)
{
   {
      t.is_terminal(wstate)
      } -> std::same_as< bool >;
};

template < typename T, typename Worldstate = typename T::world_state_type >
concept active_player = requires(T&& t)
{
   {
      t.active_player()
      } -> std::same_as< Player >;
};

template < typename T, typename Worldstate = typename T::world_state_type >
concept world_state = requires(T&& t)
{
   {
      t.world_state()
      } -> std::same_as< nor::const_as_t< T, Worldstate >& >;
};

template < typename T, typename Publicstate = typename T::public_state_type >
concept public_state = requires(T&& t)
{
   {
      t.public_state()
      } -> std::same_as< nor::const_as_t< T, Publicstate >& >;
};

template <
   typename T,
   typename Worldstate = typename T::world_state_type,
   typename Privatestate = typename T::private_state_type >
concept info_state = requires(T&& t, Worldstate&& wstate, Player player)
{
   {
      t.info_state(wstate, player)
      } -> std::same_as< Privatestate >;
};

template < typename T, typename Observation = typename T::observation_type >
concept private_observation = requires(T&& t, const std::vector< Player >& player_list)
{
   // get only private obervations
   {
      t.private_observation(player_list)
      } -> std::same_as< std::map< Player, Observation > >;
}
&&requires(T&& t, Player player)
{
   {
      t.private_observation(player)
      } -> std::convertible_to< Observation >;
};

template < typename T, typename Observation = typename T::observation_type >
concept public_observation = requires(T&& t, const std::vector< Player >& player_list)
{
   // get only private obervations
   {
      t.public_observation(player_list)
      } -> std::same_as< std::map< Player, Observation > >;
}
&&requires(T&& t, Player player)
{
   {
      t.public_observation(player)
      } -> std::convertible_to< Observation >;
};

template < typename T, typename Observation = typename T::observation_type >
concept observation = requires(T&& t, const std::vector< Player >& player_list)
{
   // get full observation: private and public
   {
      t.observation(player_list)
      } -> std::same_as< std::map< Player, Observation > >;
}
&&requires(T&& t, Player player)
{
   {
      t.observation(player)
      } -> std::convertible_to< Observation >;
};

template < typename T, typename InputT, typename OutputT >
concept getitem = requires(T&& t, InputT inp)
{
   /// getitem method for input type returning an output type
   {
      t[inp]
      } -> std::same_as< OutputT >;
};

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
concept policy_update = requires(T t, MapLikePolicy policy)
{
   {
      t.policy_update(policy, std::declval< std::map< Action, double > >())
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
   T::action_policy_type;
};

}  // namespace trait

}  // namespace nor::concepts::has

#endif  // NOR_HAS_HPP
