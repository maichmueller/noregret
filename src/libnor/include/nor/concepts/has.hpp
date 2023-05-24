
#ifndef NOR_HAS_HPP
#define NOR_HAS_HPP

#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "nor/game_defs.hpp"
#include "nor/type_defs.hpp"
#include "nor/utils/player_informed_type.hpp"

namespace nor::concepts::has {
namespace method {

template < typename T, typename GameType = typename T::game_type >
concept game = requires(T t) {
   {
      t.game()
   } -> std::same_as< common::const_as_t< T, GameType >& >;
};

template < typename T, typename Policy = typename T::policy_type >
concept current_policy = requires(T t) {
   {
      t.current_policy()
   } -> std::same_as< common::const_as_t< T, Policy >& >;
};

template < typename T, typename Policy = typename T::policy_type >
concept avg_policy = requires(T t) {
   {
      t.avg_policy()
   } -> std::same_as< common::const_as_t< T, Policy >& >;
};

template < typename T >
concept iteration = requires(T t) {
   {
      t.iteration()
   } -> std::convertible_to< size_t >;
};

template < typename T, typename TreeContainer = typename T::tree_type >
concept game_tree = requires(T t) {
   {
      t.game_tree()
   } -> std::same_as< common::const_as_t< T, TreeContainer >& >;
};

template <
   typename T,
   typename Worldstate = typename T::world_state_type,
   typename Action = typename T::action_type,
   typename ChanceOutcome = typename T::chance_outcome_type >
concept private_history = requires(T const t, Worldstate worldstate, Player player) {
   // get the history of privately observable (by the player) actions up
   // to this point
   {
      t.private_history(player, worldstate)
   } -> std::convertible_to<
      std::vector< PlayerInformedType< std::optional< std::variant< ChanceOutcome, Action > > > > >;
};

template <
   typename T,
   typename Worldstate = typename T::world_state_type,
   typename Action = typename T::action_type,
   typename ChanceOutcome = typename T::chance_outcome_type >
concept public_history = requires(T const t, Worldstate worldstate) {
   // get the history of public actions played up to this state
   {
      t.public_history(worldstate)
   } -> std::convertible_to<
      std::vector< PlayerInformedType< std::optional< std::variant< ChanceOutcome, Action > > > > >;
};

template <
   typename T,
   typename Worldstate = typename T::world_state_type,
   typename Action = typename T::action_type,
   typename ChanceOutcome = typename T::chance_outcome_type >
concept open_history = requires(T const t, Worldstate worldstate) {
   // get the history of all applied actions, regardless whether some actions
   // were hidden from other players
   {
      t.open_history(worldstate)
   } -> std::convertible_to<
      std::vector< PlayerInformedType< std::variant< ChanceOutcome, Action > > > >;
};

template <
   typename T,
   typename Worldstate = typename T::world_state_type,
   typename Action = typename T::action_type >
concept actions = requires(T const t, Worldstate worldstate, Player player) {
   // legal actions getter for the given player
   {
      t.actions(player, worldstate)
   } -> std::convertible_to< std::vector< Action > >;
};

template <
   typename T,
   typename Worldstate = typename T::world_state_type,
   typename Outcome = typename T::chance_outcome_type >
concept chance_actions = requires(T const t, Worldstate worldstate) {
   // legal actions getter for the given player
   {
      t.chance_actions(worldstate)
   } -> std::convertible_to< std::vector< Outcome > >;
};

template <
   typename T,
   typename Worldstate = typename T::world_state_type,
   typename Outcome = typename T::chance_outcome_type >
concept chance_probability = requires(T const t, Worldstate worldstate, Outcome outcome) {
   // legal actions getter for the given player
   {
      t.chance_probability(worldstate, outcome)
   } -> std::convertible_to< double >;
};

template < typename T, typename ReturnType, typename Observation >
concept update_infostate = requires(T t, Observation public_obs, Observation private_obs) {
   // append objects Us... to t
   {
      t.update(public_obs, private_obs)
   } -> std::same_as< ReturnType >;
};

template < typename T, typename ReturnType, typename Observation >
concept update_publicstate = requires(T t, Observation public_obs) {
   // append objects Us... to t
   {
      t.update(public_obs)
   } -> std::same_as< ReturnType >;
};

template <
   typename T,
   typename Worldstate = typename T::world_state_type,
   typename Action = typename T::action_type >
concept transition = requires(T t, Worldstate&& worldstate, Action action) {
   // apply the action on the given world state inplace.
   {
      t.transition(std::forward< Worldstate >(worldstate), action)
   } -> std::same_as< void >;
};

template <
   typename T,
   typename Worldstate = typename T::world_state_type,
   typename Infostate = typename T::info_state_type,
   typename Publicstate = typename T::public_state_type,
   typename Action = typename T::action_type >
concept transition_jointly = requires(
   T t,
   Action action,
   Worldstate& worldstate,
   std::map< Player, Infostate >& infostates,
   Publicstate& pubstate
) {
   // apply the action on the given world state inplace.
   {
      t.transition(worldstate, action)
   } -> std::same_as< void >;
};

template < typename T, typename Worldstate = typename T::world_state_type >
concept reward = requires(T t, Worldstate wstate, Player player) {
   {
      t.reward(player, wstate)
   } -> std::convertible_to< double >;
};

template < typename T, typename Worldstate = typename T::world_state_type >
concept reward_multi = requires(T t, Worldstate wstate, const std::vector< Player >& players) {
   {
      t.reward(players, wstate)
   } -> std::convertible_to< std::vector< double > >;
};

template < typename T >
concept run = requires(T t) {
   {
      t.run()
   } -> std::same_as< void >;
};

template <
   typename T,
   typename Worldstate = typename T::world_state_type,
   typename ReturnType = void >
concept reset = requires(T t, Worldstate& wstate) {
   {
      t.reset(wstate)
   } -> std::same_as< ReturnType >;
};

template < typename T, typename Worldstate = typename T::world_state_type >
concept players = requires(T t, const Worldstate& wstate) {
   {
      t.players(wstate)
   } -> std::same_as< std::vector< Player > >;
};

template < typename T, typename Worldstate = typename T::world_state_type >
concept is_partaking = requires(T t, const Worldstate& wstate, Player player) {
   // a function for answering whether a given player is still partaking in
   // the game or e.g. already lost
   {
      t.is_partaking(wstate, player)
   } -> std::convertible_to< bool >;
};

template < typename T >
concept player = requires(T t) {
   {
      t.player()
   } -> std::same_as< Player >;
};

template < typename T, typename Worldstate = typename T::world_state_type >
concept is_terminal = requires(T t, Worldstate& wstate) {
   {
      t.is_terminal(wstate)
   } -> std::same_as< bool >;
};

template < typename T, typename Worldstate = typename T::world_state_type >
concept active_player = requires(T t, Worldstate wstate) {
   // returns the current active player at this world state. This may also
   // return Player::chance if the current state requires a chance outcome
   // to proceed next!
   {
      t.active_player(wstate)
   } -> std::same_as< Player >;
};

template < typename T, typename Worldstate = typename T::world_state_type >
concept initial_world_state = requires(T t) {
   {
      t.initial_world_state()
   } -> std::same_as< Worldstate >;
};

template <
   typename T,
   typename Worldstate = typename T::world_state_type,
   typename Infostate = typename T::info_state_type >
concept adhoc_info_state = requires(T t, Worldstate wstate, Player player) {
   {
      t.adhoc_info_state(wstate, player)
   } -> std::same_as< Infostate >;
};

template <
   typename T,
   typename Worldstate = typename T::world_state_type,
   typename Infostate = typename T::info_state_type >
concept adhoc_public_state = requires(T t, Worldstate wstate, Player player) {
   {
      t.adhoc_public_state(wstate, player)
   } -> std::same_as< Infostate >;
};

template <
   typename T,
   typename Worldstate = typename T::world_state_type,
   typename Action = typename T::action_type,
   typename Observation = typename T::observation_type >

concept private_observation = requires(
   T t,
   Player player,
   Worldstate wstate,
   Action action,
   Worldstate next_wstate
) {
   {
      t.private_observation(player, wstate, action, next_wstate)
   } -> std::convertible_to< Observation >;
};

template <
   typename T,
   typename Worldstate = typename T::world_state_type,
   typename Action = typename T::action_type,
   typename Observation = typename T::observation_type >
concept public_observation = requires(
   T t,
   Worldstate wstate,
   Action action,
   Worldstate next_wstate
) {
   {
      t.public_observation(wstate, action, next_wstate)
   } -> std::convertible_to< Observation >;
};

template <
   typename T,
   typename Worldstate = typename T::world_state_type,
   typename Observation = typename T::observation_type >
concept observation = requires(T t, Player player, Worldstate wstate) {
   {
      t.observation(player, wstate)
   } -> std::convertible_to< Observation >;
};

template <
   typename T,
   typename Worldstate = typename T::world_state_type,
   typename Observation = typename T::observation_type >
concept observation_multi = requires(
   T t,
   const std::vector< Player >& player_list,
   Worldstate wstate
) {
   // get full observation: private and public.
   // Output is a paired vector of observations, i.e. output[i] is
   // paired to player_list[i].
   {
      t.observation(player_list, wstate)
   } -> std::same_as< std::vector< Observation > >;
};

template < typename T, template < typename > class Ptr, typename ElemT >
concept clone_other = std::is_pointer_v< Ptr< ElemT > > && requires(T&& t, Ptr< ElemT > ptr) {
   {
      t->clone(ptr)
   } -> std::convertible_to< Ptr< ElemT > >;
};

template < typename T >
concept clone = requires(T const t) { t.clone(); };

template < typename T, typename U = T >
concept copy = requires(T const t) {
   {
      t.copy()
   } -> std::same_as< U >;
};

template < typename T, typename... InputTs >
concept call = requires(T t, InputTs&&... inp) {
   /// call method for input type. No return type check
   t(std::forward< InputTs >(inp)...);
};

template < typename T, typename OutputT, typename... InputTs >
concept call_r = requires(T t, InputTs&&... inp) {
   /// call method for input type returning an output type
   {
      t(std::forward< InputTs >(inp)...)
   } -> std::convertible_to< OutputT >;
};

template < typename T, typename InputT >
concept getitem = requires(T t, InputT&& inp) {
   /// getitem method for input type. No return type check
   t[std::forward< InputT >(inp)];
};

template < typename T, typename OutputT, typename InputT >
concept getitem_r = requires(T t, InputT&& inp) {
   /// getitem method for input type returning an output type
   {
      t[std::forward< InputT >(inp)]
   } -> std::convertible_to< OutputT >;
};

template < typename T, typename... InputTs >
concept at = requires(T t, InputTs&&... inp) {
   /// getitem method for input type returning an output type
   t.at(std::forward< InputTs >(inp)...);
};

template < typename T, typename OutputT, typename... InputTs >
concept at_r = requires(T t, InputTs&&... inp) {
   /// getitem method for input type returning an output type
   {
      t.at(std::forward< InputTs >(inp)...)
   } -> std::convertible_to< OutputT >;
};

template < typename T >
concept begin = requires(T t) { t.begin(); };

template < typename T >
concept end = requires(T&& t) { t.end(); };

template < typename T, typename MapLikePolicy, typename Action >
concept policy_update = requires(T t, MapLikePolicy policy, std::map< Action, double > regrets) {
   {
      t.policy_update(policy, regrets)
   } -> std::same_as< void >;
};

template < typename T >
concept max_player_count = requires(T t) {
   {
      t.max_player_count()
   } -> std::same_as< size_t >;
};

template < typename T >
concept player_count = requires(T t) {
   {
      t.player_count()
   } -> std::same_as< size_t >;
};

template < typename T >
concept serialized = requires(T t) {
   {
      t.serialized()
   } -> std::convertible_to< bool >;
};

template < typename T >
concept unrolled = requires(T t) {
   {
      t.unrolled()
   } -> std::convertible_to< bool >;
};

template < typename T >
concept stochasticity = requires(T t) {
   {
      t.stochasticity()
   } -> std::same_as< Stochasticity >;
};

}  // namespace method

namespace trait {

template < typename T >
concept action_policy_type = requires(T t) { typename T::action_policy_type; };

template < typename T >
concept chance_outcome_type = requires(T t) { typename T::chance_outcome_type; };

template < typename T >
concept chance_distribution_type = requires(T t) { typename T::chance_distribution_type; };

template < typename T >
concept action_type = requires(T t) { typename T::action_type; };

template < typename T >
concept observation_type = requires(T t) { typename T::observation_type; };

template < typename T >
concept info_state_type = requires(T t) { typename T::info_state_type; };

template < typename T >
concept public_state_type = requires(T t) { typename T::public_state_type; };

template < typename T >
concept world_state_type = requires(T t) { typename T::world_state_type; };

}  // namespace trait

}  // namespace nor::concepts::has

#endif  // NOR_HAS_HPP
