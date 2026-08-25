#ifndef NOR_CFR_BASE_TABULAR_HPP
#define NOR_CFR_BASE_TABULAR_HPP

#include <algorithm>
#include <array>
#include <deque>
#include <iostream>
#include <list>
#include <map>
#include <named_type.hpp>
#include <queue>
#include <ranges>
#include <span>
#include <stack>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/common.hpp"
#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "nor/rm/forest.hpp"
#include "nor/rm/node.hpp"
#include "nor/rm/rm_utils.hpp"
#include "nor/type_defs.hpp"
#include "nor/utils/utils.hpp"

namespace nor::rm {

/// hard upper bound of simultaneously participating seats: the Player enum in
/// game_defs.hpp enumerates exactly 'Player::zoey + 1' actual player slots
/// (chance/unknown are negative and excluded). Bounds all fixed-size stack
/// storage of the seat-indexed traversal containers below.
inline constexpr size_t max_player_seats = static_cast< size_t >(Player::zoey) + 1;

/// validates and converts a player into its raw seat index; players outside
/// the enumerable seat range (chance, unknown) are rejected
[[nodiscard]] inline size_t player_seat(Player player)
{
   const auto idx = static_cast< size_t >(player);
   if(player == Player::chance or player == Player::unknown or idx >= max_player_seats) {
      throw std::invalid_argument("player_seat: player is not a valid seat (chance/unknown?)");
   }
   return idx;
}

///////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////// seat-indexed traversal containers ////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief seat-indexed slot table of per-player infostates.
 *
 * Replaces the former player_hashmap<sptr<InfoState>> traversal container:
 * built once per iteration from the root roster, then indexed O(1) by the
 * player's raw seat -- no hashing, no node allocation on the hot path. A null
 * slot mirrors the former map's missing key ('find' yields nullptr,
 * 'at'/'contains' report absence).
 */
template < typename InfoState >
class InfostateSptrTable {
  public:
   using mapped_type = sptr< InfoState >;
   static constexpr size_t seat_count = max_player_seats;

   [[nodiscard]] mapped_type& at(Player player)
   {
      auto& slot = m_slots[player_seat(player)];
      if(not slot) {
         throw std::out_of_range("InfostateSptrTable: no infostate entry for player");
      }
      return slot;
   }
   [[nodiscard]] const mapped_type& at(Player player) const
   {
      return const_cast< InfostateSptrTable* >(this)->at(player);
   }

   /// map-'find' equivalent: nullptr stands in for the former end() iterator
   [[nodiscard]] mapped_type* find(Player player)
   {
      if(player == Player::chance or player == Player::unknown) {
         return nullptr;
      }
      const auto idx = static_cast< size_t >(player);
      if(idx >= seat_count) {
         return nullptr;
      }
      auto& slot = m_slots[idx];
      return slot ? &slot : nullptr;
   }
   [[nodiscard]] const mapped_type* find(Player player) const
   {
      return const_cast< InfostateSptrTable* >(this)->find(player);
   }

   [[nodiscard]] bool contains(Player player) const { return find(player) != nullptr; }

   /// roster-initialization insert (replaces map::emplace during setup only)
   void emplace(Player player, mapped_type infostate)
   {
      m_slots[player_seat(player)] = std::move(infostate);
   }

   /// calls 'fn(player, sptr&)' for every initialized slot (ascending seats)
   template < typename F >
   void for_each_present(F&& fn)
   {
      for(auto idx : std::views::iota(size_t{0}, seat_count)) {
         if(m_slots[idx]) {
            fn(static_cast< Player >(idx), m_slots[idx]);
         }
      }
   }
   template < typename F >
   void for_each_present(F&& fn) const
   {
      for(auto idx : std::views::iota(size_t{0}, seat_count)) {
         if(m_slots[idx]) {
            fn(static_cast< Player >(idx), std::as_const(m_slots[idx]));
         }
      }
   }

  private:
   std::array< mapped_type, seat_count > m_slots{};
};

/**
 * @brief seat-indexed slot table of per-player observation buffers.
 *
 * Replaces the former player_hashmap<vector<pair<Obs, Obs>>> traversal
 * container: fixed stack storage indexed by raw seat; each occupied seat owns
 * one reusable heap buffer whose capacity survives across edges/iterations.
 */
template < typename Observation >
class ObservationbufferTable {
  public:
   using buffer_type = std::vector< std::pair< Observation, Observation > >;
   static constexpr size_t seat_count = max_player_seats;

   [[nodiscard]] buffer_type& at(Player player)
   {
      const auto idx = player_seat(player);
      if(not m_present[idx]) {
         throw std::out_of_range("ObservationbufferTable: no buffer entry for player");
      }
      return m_buffers[idx];
   }
   [[nodiscard]] const buffer_type& at(Player player) const
   {
      return const_cast< ObservationbufferTable* >(this)->at(player);
   }

   /// map-'operator[]' equivalent: lazily marks a seat present (former map
   /// semantics inserted an empty buffer on first touch)
   [[nodiscard]] buffer_type& operator[](Player player)
   {
      const auto idx = player_seat(player);
      m_present[idx] = true;
      return m_buffers[idx];
   }

   /// map-'find'-style presence query used by the flush decision
   [[nodiscard]] bool contains(Player player) const
   {
      if(player == Player::chance or player == Player::unknown) {
         return false;
      }
      const auto idx = static_cast< size_t >(player);
      return idx < seat_count and m_present[idx];
   }

   /// roster-initialization insert
   void emplace(Player player, buffer_type buffer = {})
   {
      auto& slot = (*this)[player];
      slot = std::move(buffer);
   }

   /// calls 'fn(player, buffer&)' for every present slot (ascending seats)
   template < typename F >
   void for_each_present(F&& fn)
   {
      for(auto idx : std::views::iota(size_t{0}, seat_count)) {
         if(m_present[idx]) {
            fn(static_cast< Player >(idx), m_buffers[idx]);
         }
      }
   }
   template < typename F >
   void for_each_present(F&& fn) const
   {
      for(auto idx : std::views::iota(size_t{0}, seat_count)) {
         if(m_present[idx]) {
            fn(static_cast< Player >(idx), std::as_const(m_buffers[idx]));
         }
      }
   }

  private:
   std::array< buffer_type, seat_count > m_buffers{};
   std::array< bool, seat_count > m_present{};
};

/**
 * @brief cold-path adapters: the best-response / payoff-probe walks (declared
 * outside this migration's ownership on classic player_hashmaps) materialize
 * the seat-indexed traversal containers through these views. The walks restore
 * every mutation per edge, so operating on the materialized copies is
 * observationally equivalent.
 */
template < typename InfoState >
[[nodiscard]] inline player_hashmap< sptr< InfoState > > raw_infostate_view(
   const InfostateSptrTable< InfoState >& table
)
{
   player_hashmap< sptr< InfoState > > out;
   out.reserve(table.seat_count);
   table.for_each_present([&](Player p, const sptr< InfoState >& istate) { out.emplace(p, istate); }
   );
   return out;
}

template < typename Observation >
[[nodiscard]] inline player_hashmap< std::vector< std::pair< Observation, Observation > > >
raw_observation_buffer_view(const ObservationbufferTable< Observation >& table)
{
   player_hashmap< std::vector< std::pair< Observation, Observation > > > out;
   out.reserve(table.seat_count);
   table.for_each_present([&](Player p, const auto& buffer) { out.emplace(p, buffer); });
   return out;
}

/**
 * @brief single-trajectory counterpart of
 * utils::next_infostate_and_obs_buffers_inplace over the seat-indexed tables:
 * folds the edge's observations into the live buffers / advances the next
 * active player's infostate in place. Mutation order over players matches the
 * generic helper exactly.
 */
template <
   typename Env,
   typename ObservationbufferMapT,
   typename InfostateSptrMapT,
   typename Worldstate = auto_world_state_type< std::remove_cvref_t< Env > > >
   requires concepts::fosg< std::remove_cvref_t< Env > >
void next_infostate_and_obs_buffers_seated(
   Env&& env,
   ObservationbufferMapT& observation_buffer,
   InfostateSptrMapT& infostate_map,
   const Worldstate& state,
   const auto& action_or_outcome,
   const Worldstate& next_state
)
{
   // the public observation will be given to every player, so we can establish it outside the loop
   auto public_obs = env.public_observation(state, action_or_outcome, next_state);

   auto active_player = env.active_player(next_state);
   for(auto player : env.players(next_state)) {
      if(player == Player::chance) {
         continue;
      }
      if(player != active_player) {
         // for all but the active player we simply append action and state observation to the
         // buffer. They will be written to an actual infostate once that player becomes the
         // active player again
         auto& player_obs_buffer = observation_buffer[player];
         player_obs_buffer.emplace_back(
            public_obs, env.private_observation(player, state, action_or_outcome, next_state)
         );
      } else {
         // for the active player we first append all recent actions and state observations to
         // the info state, and then follow it up by adding the current action and state
         // observations. We consume these observations by moving them into the appendix of the
         // infostate. The cleared observation buffer is reused, but is now empty.
         auto& infostate_holder = infostate_map.at(active_player);
         auto& obs_history = observation_buffer[active_player];
         for(auto& obs : obs_history) {
            ::nor::detail::update_infostate(
               infostate_holder, std::move(obs.first), std::move(obs.second)
            );
         }
         obs_history.clear();
         ::nor::detail::update_infostate(
            infostate_holder,
            public_obs,
            env.private_observation(player, state, action_or_outcome, next_state)
         );
      }
   }
}

/**
 * A Counterfactual Regret Minimization algorithm base class following the
 * terminology of the Factored-Observation Stochastic Games (FOSG) formulation.
 *
 * This class defines the common template and constructor definitions, as well as the tabular
 * members to be used by specific implementations.
 *
 */
template < bool alternating_update, typename Env, typename Policy, typename AveragePolicy >
   requires concepts::tabular_cfr_requirements<
      Env,
      Policy,
      AveragePolicy,
      UniformPolicy< auto_info_state_type< Env >, auto_action_policy_type< Policy > >,
      ZeroDefaultPolicy< auto_info_state_type< Env >, auto_action_policy_type< AveragePolicy > > >
class TabularCFRBase {
   ////////////////////////////
   /// API: public typedefs ///
   ////////////////////////////
  public:
   /// aliases for the template types
   using env_type = Env;
   using policy_type = Policy;
   /// store the flag for alternating updates
   static constexpr bool alternating_updates = alternating_update;
   /// import all fosg aliases to be used in this class from the env type.
   using action_type = auto_action_type< Env >;
   using world_state_type = auto_world_state_type< Env >;
   using info_state_type = auto_info_state_type< Env >;
   using public_state_type = auto_public_state_type< Env >;
   using observation_type = auto_observation_type< Env >;
   using chance_outcome_type = auto_chance_outcome_type< Env >;
   using chance_distribution_type = auto_chance_distribution_type< Env >;

   using uniform_policy_type = UniformPolicy<
      auto_info_state_type< Env >,
      auto_action_policy_type< Policy > >;
   using zero_policy_type = UniformPolicy<
      auto_info_state_type< Env >,
      auto_action_policy_type< Policy > >;

   /// the data to store per infostate entry
   using infostate_data_type = InfostateNodeData< action_type >;
   /// seat-indexed traversal containers (B3): built once per iteration from
   /// the root roster and passed down the recursion by reference -- no player
   /// hashing or node allocation on the hot path (formerly NamedTypes over
   /// player_hashmap).
   using InfostateSptrMap = InfostateSptrTable< info_state_type >;

   using ObservationbufferMap = ObservationbufferTable< observation_type >;

   ////////////////////
   /// Constructors ///
   ////////////////////

  protected:
   /// the constructors are protected since this class is not intended to be instantiated on its own
   /// but only used as base class for cfr algorithms

   TabularCFRBase(
      Env game,
      uptr< world_state_type > root_state,
      const Policy& policy = Policy(),
      const AveragePolicy& avg_policy = AveragePolicy()
   )
      // clang-format off
       requires
          common::all_predicate_v<
             std::is_copy_constructible,
             Policy,
             AveragePolicy >
       // clang-format on
       : m_env(std::move(game)),
         m_root_state(std::move(root_state)),
         m_curr_policy(),
         m_avg_policy()
   {
      // note: query the players from the ALREADY MOVED INTO m_env instance.
      // Reading the by-value parameter 'game' here would be a use-after-move.
      for(auto player : m_env.players(*m_root_state) | utils::is_actual_player_filter) {
         m_curr_policy.emplace(player, policy);
         m_avg_policy.emplace(player, avg_policy);
         ++m_root_player_count;
      }
      _init_player_update_schedule();
   }

   TabularCFRBase(
      Env env,
      const Policy& policy = Policy(),
      const AveragePolicy& avg_policy = AveragePolicy()
   )
      // clang-format off
       requires
          concepts::has::method::initial_world_state< Env >
       // clang-format on
       // note: this delegation is well-defined: all arguments of the delegated
       // constructor call are evaluated (incl. env.initial_world_state())
       // before the target constructor's by-value parameters are initialized
       // from them, so 'env' cannot be moved-from prematurely.
       : TabularCFRBase(
          std::move(env),
          std::make_unique< world_state_type >(env.initial_world_state()),
          policy,
          avg_policy
       )
   {
      _init_player_update_schedule();
   }

   TabularCFRBase(
      Env game,
      uptr< world_state_type > root_state,
      std::unordered_map< Player, Policy > policy,
      std::unordered_map< Player, AveragePolicy > avg_policy
   )
       : m_env(std::move(game)),
         m_root_state(std::move(root_state)),
         m_curr_policy(std::move(policy)),
         m_avg_policy(std::move(avg_policy))
   {
      _init_player_update_schedule();
   }

   ////////////////////////////////////
   /// API: public member functions ///
   ////////////////////////////////////

  public:
   /**
    * @brief gets the current or average state policy of a node
    *
    * Depending on the template parameter either the current policy (true) or the average policy
    * (false) over the last iterations is queried. If the current node has not been emplaced in the
    * policy yet, then the default policy will be asked to provide an initial entry.
    * @param current_policy switch for querying either the current (true) or the average (false)
    * policy.
    * @return action_policy_type the player's state policy (distribution) over all actions at this
    * node
    */
   template < bool current_policy >
   auto& fetch_policy(const info_state_type& infostate, const std::vector< action_type >& actions);
   /**
    * @brief Policy fetching overload for explicit naming of the policy.
    */
   template < PolicyLabel label >
   decltype(auto)
   fetch_policy(const info_state_type& infostate, const std::vector< action_type >& actions)
   {
      static_assert(
         label == PolicyLabel::current or label == PolicyLabel::average,
         "Policy label has to be either 'current' or 'average'."
      );
      return fetch_policy< label == PolicyLabel::current >(infostate, actions);
   }

   /**
    *
    * @param action the action to select at this node
    * @return
    */
   template < bool current_policy >
   inline auto& fetch_policy(
      const info_state_type& infostate,
      const std::vector< action_type >& actions,
      const action_type& action
   )
   {
      return fetch_policy< current_policy >(infostate, actions)[action];
   }

   /// getter methods for stored data

   [[nodiscard]] inline const auto& root_state() const { return *m_root_state; }
   [[nodiscard]] inline auto iteration() const { return m_iteration; }
   /// the CYCLE index: iteration / num_actual_players. Used e.g. by the
   /// 'weight_by_cycle' option of DiscountedCFR so that per-player weighting
   /// stays consistent under alternating updates (B3 hook).
   [[nodiscard]] inline size_t cycle() const { return m_iteration / m_root_player_count; }
   [[nodiscard]] inline const auto& policy() const { return m_curr_policy; }
   [[nodiscard]] inline const auto& average_policy() const { return m_avg_policy; }
   [[nodiscard]] inline const auto& env() const { return m_env; }

   //////////////////////////////////
   /// protected member functions ///
   //////////////////////////////////

  protected:
   /// protected member access for derived classes
   [[nodiscard]] inline auto& _env() { return m_env; }
   [[nodiscard]] inline const auto& _root_state_uptr() const { return m_root_state; }
   [[nodiscard]] inline auto& _iteration() { return m_iteration; }
   [[nodiscard]] inline auto& _policy() { return m_curr_policy; }
   [[nodiscard]] inline auto& _average_policy() { return m_avg_policy; }
   [[nodiscard]] inline auto& _player_update_schedule() { return m_player_update_schedule; }

   /**
    * @brief Cycles the update schedule by popping the next player to update and requeueing them as
    * last.
    *
    * The update schedule for alternative updates is a cycle P1-P2-...-PN. After each update the
    * schedule moves by returning the first player and reattaching it to the back of the queue. This
    * way every other player is pushed up by one position in the update queue.
    * @param player_to_update the optional player to update next. If not given, the next in line
    * will be chosen.
    * @return the player to update next.
    */
   Player _cycle_player_to_update(std::optional< Player > player_to_update = std::nullopt);

   Player _preview_next_player_to_update() const { return m_player_update_schedule.front(); }

   /**
    * @brief initializes the player cycle buffer with all available players at the current state
    */
   inline void _init_player_update_schedule()
   {
      for(auto player : m_env.players(*m_root_state)) {
         if(player != Player::chance) {
            if constexpr(alternating_updates) {
               m_player_update_schedule.push_back(player);
            }
            ++m_root_player_count;
         }
      }
   }

   bool _partial_pruning_condition(
      [[maybe_unused]] std::optional< Player > player_to_update,
      const ReachProbabilityMap& reach_probability
   )
   {
      if constexpr(alternating_update) {
         // if all players have 0 reach probability for reaching an infostate then the
         // entire subtree visited from that infostate can be pruned, because both, the regret
         // updates (which depends on the counterfactual values, i.e. pi_{-i}), and
         // the average strategy updates (which depends on pi_i) will be 0. If only the
         // opponent reach prob is 0, then we can merely skip regret updates which would not
         // improve the speed much in this implementation.
         Player traverser = *player_to_update;
         auto traversing_player_rp_is_zero = reach_probability.get().at(traverser)
                                             <= std::numeric_limits< double >::epsilon();
         return traversing_player_rp_is_zero
                and std::ranges::any_of(reach_probability.get(), [&](const auto& player_rp_pair) {
                       const auto& [player, rp] = player_rp_pair;
                       return player != traverser
                              and rp <= std::numeric_limits< double >::epsilon();
                    });
      } else {
         // A mere check on ONE of the opponents having reach prob 0 and the active player
         // having reach prob 0 would not suffice in the multiplayer case as some average
         // strategy updates of other opponent with reach prob > 0 would be missed in the case
         // of simultaneous updates.
         return std::ranges::all_of(reach_probability.get(), [&](const auto& player_rp_pair) {
            const auto& [player, rp] = player_rp_pair;
            return player != Player::chance and rp <= std::numeric_limits< double >::epsilon();
         });
      }
   }

   ///////////////////////////////////////////
   /// private member variable definitions ///
   ///////////////////////////////////////////

  protected:
   /// B3: per-depth reusable scratch buffer for the flush-target observation
   /// buffer SWAP at recursion boundaries (replaces the former per-edge heap
   /// copy of the flush target's buffer). A deque on purpose -- growing it
   /// never moves existing slots, keeping references held by active frames
   /// valid (same discipline as the derived classes' world-state arena).
   /// Access through _obs_scratch_slot(depth).
   std::deque< std::vector< std::pair< observation_type, observation_type > > >
      m_obs_scratch_arena{};

   /// grows (if needed) and returns the flush-swap scratch slot of 'depth'
   auto& _obs_scratch_slot(size_t depth)
   {
      if(m_obs_scratch_arena.size() <= depth) {
         m_obs_scratch_arena.resize(depth + 1);
      }
      return m_obs_scratch_arena[depth];
   }

  private:
   /// the environment object to maneuver the states with.
   env_type m_env;
   /// the root game state.
   uptr< world_state_type > m_root_state;
   /// a map of the current policy $\pi^t$ that each player is following in this iteration (t).
   player_hashmap< Policy > m_curr_policy;
   /// the average policy table. The values stored in this table are the UNNORMALIZED average state
   /// policy cumulative values. This means that the state policy p(s, . ) for a given info state s
   /// needs to normalize its probabilities p(s, . ) by \sum_a p(s,a) when used for evaluation.
   player_hashmap< AveragePolicy > m_avg_policy;
   /// the next player to update when doing alternative updates. Otherwise this member will be
   /// unused.
   std::deque< Player > m_player_update_schedule{};
   /// the number of iterations we have run so far.
   size_t m_iteration = 0;
   /// number of ACTUAL players at the root (chance excluded); basis of cycle()
   size_t m_root_player_count = 0;
};

template < bool alternating_updates, typename Env, typename Policy, typename AveragePolicy >
   requires concepts::tabular_cfr_requirements<
      Env,
      Policy,
      AveragePolicy,
      UniformPolicy< auto_info_state_type< Env >, auto_action_policy_type< Policy > >,
      ZeroDefaultPolicy< auto_info_state_type< Env >, auto_action_policy_type< AveragePolicy > > >
Player TabularCFRBase< alternating_updates, Env, Policy, AveragePolicy >::_cycle_player_to_update(
   std::optional< Player > player_to_update
)
{
   // we assert here that the chosen player to update is not the chance player.
   if(player_to_update.value_or(Player::alex) == Player::chance) {
      std::stringstream ssout;
      ssout << "Given combination of '";
      ssout << Player::chance;
      ssout << "' and '";
      ssout << "alternating updates'";
      ssout << "is incompatible. Did you forget to pass the correct player parameter?";
      throw std::invalid_argument(ssout.str());
   }

   auto player_q_iter = std::find(
      m_player_update_schedule.begin(),
      m_player_update_schedule.end(),
      player_to_update.value_or(m_player_update_schedule.front())
   );

   if(player_q_iter == m_player_update_schedule.end()) {
      std::stringstream ssout;
      ssout << "Given player to update ";
      ssout << player_to_update.value();
      ssout << " is not a member of the update schedule.";
      common::print_bracketed(ssout, m_player_update_schedule);
      ssout << ".";
      throw std::invalid_argument(ssout.str());
   }
   Player next_to_update = *player_q_iter;
   m_player_update_schedule.erase(player_q_iter);
   m_player_update_schedule.push_back(next_to_update);
   return next_to_update;
}

template < bool alternating_updates, typename Env, typename Policy, typename AveragePolicy >
   requires concepts::tabular_cfr_requirements<
      Env,
      Policy,
      AveragePolicy,
      UniformPolicy< auto_info_state_type< Env >, auto_action_policy_type< Policy > >,
      ZeroDefaultPolicy< auto_info_state_type< Env >, auto_action_policy_type< AveragePolicy > > >
template < bool current_policy >
auto& TabularCFRBase< alternating_updates, Env, Policy, AveragePolicy >::fetch_policy(
   const info_state_type& infostate,
   const std::vector< action_type >& actions
)
{
   if constexpr(current_policy) {
      auto& player_policy = m_curr_policy[infostate.player()];
      return player_policy(infostate, actions, uniform_policy_type{});
   } else {
      auto& player_policy = m_avg_policy[infostate.player()];
      return player_policy(infostate, actions, zero_policy_type{});
   }
}

}  // namespace nor::rm

#endif  // NOR_CFR_BASE_TABULAR_HPP
