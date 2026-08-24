
#ifndef NOR_RM_UTILS_HPP
#define NOR_RM_UTILS_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <named_type.hpp>
#include <numeric>
#include <ranges>
#include <unordered_map>
#include <vector>

#include "common/common.hpp"
#include "node.hpp"
#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "nor/type_defs.hpp"
#include "nor/utils/utils.hpp"

namespace nor::rm {

// NOTE on parallelism: all kernels in this header as well as the end-of-
// iteration regret-minimization sweeps of the tabular solvers run strictly
// SERIAL. Per-infostate workloads are far too small to amortize thread-pool
// dispatch, while parallel scheduling would additionally make floating-point
// summation orders non-deterministic. Serial execution guarantees bit-wise
// reproducible regret/policy tables across runs given identical seeds.

enum class PolicyLabel { current = 0, average = 1 };
/// strong-types for passing arguments around with intent
using Probability = fluent::NamedType< double, struct prob_tag >;
using Weight = fluent::NamedType< double, struct weight_tag >;
using StateValue = fluent::NamedType< double, struct state_value_tag >;

/////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// player value tables ////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief small fixed-size player-indexed double table.
 *
 * Replaces the previous std::unordered_map<Player, double> churn inside the
 * solvers' traversals: the root participant roster is known at the root call,
 * so per-recursion-level value maps never need hashing or node allocations.
 * The class mimics the subset of the map API its former users rely on
 * (operator[], at, emplace/try_emplace, iteration over (Player, double)
 * pairs) and converts to/from player_hashmap<double> at API boundaries.
 */
class PlayerValueTable {
  public:
   using key_type = Player;
   using mapped_type = double;
   using UnderlyingType = PlayerValueTable;

   using value_type = std::pair< Player, double >;

   template < bool Const >
   struct iterator_t {
      using difference_type = std::ptrdiff_t;
      using value_type = std::pair< Player, double >;
      using reference = value_type;
      using pointer = void;
      using iterator_concept = std::random_access_iterator_tag;

      std::conditional_t< Const, const PlayerValueTable*, PlayerValueTable* > table{};
      size_t index = 0;

      constexpr value_type operator*() const
      {
         return {table->m_players[index], table->m_values[index]};
      }
      constexpr iterator_t& operator++()
      {
         ++index;
         return *this;
      }
      constexpr iterator_t operator++(int)
      {
         auto tmp = *this;
         ++index;
         return tmp;
      }
      constexpr iterator_t& operator--()
      {
         --index;
         return *this;
      }
      constexpr iterator_t operator--(int)
      {
         auto tmp = *this;
         --index;
         return tmp;
      }
      constexpr iterator_t& operator+=(difference_type n)
      {
         index += static_cast< size_t >(n);
         return *this;
      }
      constexpr iterator_t& operator-=(difference_type n)
      {
         index -= static_cast< size_t >(n);
         return *this;
      }
      constexpr iterator_t operator+(difference_type n) const
      {
         return iterator_t{table, index + static_cast< size_t >(n)};
      }
      constexpr iterator_t operator-(difference_type n) const
      {
         return iterator_t{table, index - static_cast< size_t >(n)};
      }
      constexpr difference_type operator-(const iterator_t& other) const
      {
         return static_cast< difference_type >(index - other.index);
      }
      constexpr bool operator==(const iterator_t&) const = default;
      constexpr auto operator<=>(const iterator_t&) const = default;
   };

  public:
   using iterator = iterator_t< false >;
   using const_iterator = iterator_t< true >;

   PlayerValueTable() = default;

   /// conversion from a classic player hashmap (API boundary helper)
   PlayerValueTable(const std::unordered_map< Player, double >& map)
   {
      reserve(map.size());
      for(const auto& [player, value] : map) {
         emplace(player, value);
      }
   }

   /// conversion from a classic player hashmap (moving variant)
   explicit PlayerValueTable(std::unordered_map< Player, double >&& map)
   {
      reserve(map.size());
      for(auto&& [player, value] : map) {
         emplace(player, std::move(value));
      }
   }

   /// conversion back into a classic player hashmap (API boundary helper)
   [[nodiscard]] std::unordered_map< Player, double > to_hashmap() const
   {
      std::unordered_map< Player, double > map;
      map.reserve(m_players.size());
      for(size_t i : std::views::iota(size_t{0}, m_size)) {
         map.emplace(m_players[i], m_values[i]);
      }
      return map;
   }

   // ---- map-like interface -------------------------------------------------------------

   [[nodiscard]] double& operator[](Player player)
   {
      const auto idx = find_slot(player);
      if(idx < m_size and m_players[idx] == player) {
         return m_values[idx];
      }
      return insert_at(idx, player, 0.);
   }

   [[nodiscard]] const double& at(Player player) const
   {
      for(size_t i : std::views::iota(size_t{0}, m_size)) {
         if(m_players[i] == player) {
            return m_values[i];
         }
      }
      throw std::out_of_range("PlayerValueTable: player not present");
   }

   [[nodiscard]] double& at(Player player)
   {
      return const_cast< double& >(std::as_const(*this).at(player));
   }

   std::pair< iterator, bool > emplace(Player player, double value)
   {
      const auto idx = find_slot(player);
      if(idx < m_size and m_players[idx] == player) {
         return std::pair{iterator{this, idx}, false};
      }
      insert_at(idx, player, value);
      return std::pair{iterator{this, idx}, true};
   }

   std::pair< iterator, bool > try_emplace(Player player, double value = 0.)
   {
      const auto idx = find_slot(player);
      if(idx < m_size and m_players[idx] == player) {
         return std::pair{iterator{this, idx}, false};
      }
      insert_at(idx, player, value);
      return std::pair{iterator{this, idx}, true};
   }

   [[nodiscard]] size_t count(Player player) const
   {
      const auto idx = find_slot(player);
      return (idx < m_size and m_players[idx] == player) ? size_t{1} : size_t{0};
   }

   [[nodiscard]] size_t size() const { return m_size; }
   [[nodiscard]] bool empty() const { return m_size == 0; }
   void clear() { m_size = 0; }
   void reserve(size_t n)
   {
      m_players.reserve(n);
      m_values.reserve(n);
   }

   iterator begin() { return iterator{this, 0}; }
   iterator end() { return iterator{this, m_size}; }
   const_iterator begin() const { return const_iterator{this, 0}; }
   const_iterator end() const { return const_iterator{this, m_size}; }
   const_iterator cbegin() const { return begin(); }
   const_iterator cend() const { return end(); }

  private:
   /// insertion position that keeps players sorted ascending (binary search)
   [[nodiscard]] size_t find_slot(Player player) const
   {
      return static_cast< size_t >(
         std::lower_bound(m_players.begin(), m_players.begin() + m_size, player) - m_players.begin()
      );
   }

   double& insert_at(size_t idx, Player player, double value)
   {
      assert(idx <= m_size and m_size <= max_player_slots);
      if(idx == m_size) {
         m_players.emplace_back(player);
         m_values.emplace_back(value);
      } else {
         m_players.insert(m_players.begin() + static_cast< long >(idx), player);
         m_values.insert(m_values.begin() + static_cast< long >(idx), value);
      }
      ++m_size;
      return m_values[idx];
   }

   static constexpr size_t max_player_slots = 64;  ///< far beyond any real roster
   /// compacted storage: [0, m_size) holds the entries in ascending player order
   std::vector< Player > m_players;
   std::vector< double > m_values;
   size_t m_size = 0;
};

/// strong-type wrapper kept API-compatible with the former NamedType over
/// unordered_map (incl. the 'get()' accessor and 'UnderlyingType' alias)
class StateValueMap {
  public:
   using UnderlyingType = PlayerValueTable;

   /// NOTE: deliberately no initializer-list constructor -- 'StateValueMap
   /// m{{}}' must construct an EMPTY table exactly like the former
   /// NamedType-over-unordered_map did.
   StateValueMap(UnderlyingType underlying = {}) : m_table(std::move(underlying)) {}
   explicit StateValueMap(const std::unordered_map< Player, double >& map) : m_table(map) {}

   [[nodiscard]] UnderlyingType& get() { return m_table; }
   [[nodiscard]] const UnderlyingType& get() const { return m_table; }

   friend bool operator==(const StateValueMap&, const StateValueMap&) = default;

  private:
   UnderlyingType m_table;
};

using ReachProbabilityMap = fluent::
   NamedType< std::unordered_map< Player, double >, struct reach_prob_map_tag >;

/////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// kernels ///////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief computes the reach probability of the node.
 *
 * Since each player's compounding likelihood contribution is stored in the nodes themselves, the
 * actual computation is nothing more than merely multiplying all player's individual
 * contributions.
 * @param reach_probability_contributions the compounded reach probability of each player for
 * this node
 * @return the reach probability of the nde
 */
template < concepts::mapping_of< double > KVdouble >
[[nodiscard]] inline double reach_probability(const KVdouble& reach_probability_contributions)
{
   auto values_view = reach_probability_contributions | std::views::values;
   return std::reduce(values_view.begin(), values_view.end(), double(1.), std::multiplies{});
}
/**
 * @brief computes the counterfactual reach probability of the player for this node.
 *
 * The method delegates the actual computation to the overload with an already provided reach
 * probability.
 * @param node the node at which the counterfactual reach probability is to be computed
 * @param player the player for which the value is computed
 * @return the counterfactual reach probability
 */
template < concepts::mapping_of< double > KVdouble >
   requires requires(KVdouble m) {
      // the keys have to of type 'Player' as well
      std::is_convertible_v< decltype(*(std::views::keys(m).begin())), Player >;
   }
inline double
cf_reach_probability(const Player& player, const KVdouble& reach_probability_contributions)
{
   auto values_view = reach_probability_contributions
                      | std::views::filter([&](const auto& player_rp_pair) {
                           return std::get< 0 >(player_rp_pair) != player;
                        })
                      | std::views::values;
   return std::reduce(values_view.begin(), values_view.end(), double(1.), std::multiplies{});
}

/**
 * @brief Performs regret-matching on the given policy with respect to the provided regret
 *
 * @tparam Action
 * @tparam Policy
 */
template < concepts::action Action, concepts::action_policy< Action > Policy >
void regret_matching(Policy& policy_map, const std::unordered_map< Action, double >& cumul_regret)
{
   // sum up the positivized regrets and store them in a new vector
   std::unordered_map< Action, double > pos_regrets;
   double pos_regret_sum{0.};
   for(const auto& [action, regret] : cumul_regret) {
      double pos_regret = std::max(0., regret);
      pos_regrets.emplace(action, pos_regret);
      pos_regret_sum += pos_regret;
   }
   // apply the new policy to the vector policy
   if(pos_regret_sum > 0) {
      if(cumul_regret.size() != policy_map.size()) {
         throw std::invalid_argument(
            "Passed regrets and policy maps do not have the same number of elements"
         );
      }
      std::for_each(policy_map.begin(), policy_map.end(), [&](auto& entry) {
         return std::get< 1 >(entry) = std::max(0., cumul_regret.at(std::get< 0 >(entry)))
                                       / pos_regret_sum;
      });
   } else {
      std::for_each(policy_map.begin(), policy_map.end(), [&](auto& entry) {
         return std::get< 1 >(entry) = 1. / static_cast< double >(policy_map.size());
      });
   }
}

/**
 * @brief Performs regret-matching on the given policy with respect to the provided regret
 *
 * @tparam Action
 * @tparam Policy
 */
template <
   typename Policy,
   typename RegretMap,
   typename ActionWrapper,
   typename Action = auto_action_type< Policy > >
// clang-format off
requires
   concepts::map< RegretMap >
   and std::is_convertible_v< typename RegretMap::mapped_type, double>
   and std::invocable< ActionWrapper, Action >
   and concepts::action_policy<
      Policy
   >
// clang-format on
void regret_matching(
   Policy& policy_map,
   const RegretMap& cumul_regret,
   ActionWrapper action_wrapper = [](const Action& action) { return action; }
)
{
   // sum up the positivized regrets and store them in a new vector
   RegretMap pos_regrets;
   double pos_regret_sum{0.};
   for(const auto& [action, regret] : cumul_regret) {
      double pos_regret = std::max(0., regret);
      pos_regrets.emplace(action, pos_regret);
      pos_regret_sum += pos_regret;
   }
   // apply the new policy to the vector policy
   if(pos_regret_sum > 0) {
      if(cumul_regret.size() != policy_map.size()) {
         throw std::invalid_argument(
            "Passed regrets and policy maps do not have the same number of elements"
         );
      }
      std::for_each(policy_map.begin(), policy_map.end(), [&](auto& entry) {
         return std::get< 1 >(entry
                ) = std::max(0., cumul_regret.at(action_wrapper(std::get< 0 >(entry))))
                    / pos_regret_sum;
      });
   } else {
      std::for_each(policy_map.begin(), policy_map.end(), [&](auto& entry) {
         return std::get< 1 >(entry) = 1. / static_cast< double >(policy_map.size());
      });
   }
}

/**
 * @brief Performs regret-matching (plus resetting to 0 if regret < 0) on the given policy with
 * respect to the provided regret
 *
 * @tparam Action
 * @tparam Policy
 */
template < typename Policy, typename RegretMap, typename Action = auto_action_type< Policy > >
// clang-format off
requires
   concepts::map< RegretMap >
   and std::is_convertible_v< typename RegretMap::mapped_type, double>
   and concepts::action_policy< Policy >
// clang-format on
void regret_matching_plus(Policy& policy_map, RegretMap& cumul_regret)
{
   double pos_regret_sum{0.};
   for(auto& [action, regret] : cumul_regret) {
      regret = std::max(0., regret);
      pos_regret_sum += regret;
   }
   // apply the new policy to the vector policy
   if(pos_regret_sum > 0) {
      if(cumul_regret.size() != policy_map.size()) {
         throw std::invalid_argument(
            "Passed regrets and policy maps do not have the same number of elements"
         );
      }
      std::for_each(policy_map.begin(), policy_map.end(), [&](auto& entry) {
         return std::get< 1 >(entry) = std::max(0., cumul_regret.at(std::get< 0 >(entry)))
                                       / pos_regret_sum;
      });
   } else {
      std::for_each(policy_map.begin(), policy_map.end(), [&](auto& entry) {
         return std::get< 1 >(entry) = 1. / static_cast< double >(policy_map.size());
      });
   }
}

/**
 * @brief Performs regret-matching+ on the given policy with respect to the provided regret when
 * regret-based-pruning is also performed.
 *
 * @tparam Action
 * @tparam Policy
 */
template <
   typename Policy,
   typename RegretMap,
   typename InstantRegretMap,
   typename Action = auto_action_type< Policy > >
// clang-format off
requires
   concepts::action_policy< Policy >
   and concepts::map< RegretMap >
   and std::is_convertible_v< typename RegretMap::mapped_type, double >
   and concepts::map< InstantRegretMap >
   and std::is_convertible_v< typename InstantRegretMap::mapped_type, double >
// clang-format on
void regret_matching_plus_rbp(
   Policy& policy_map,
   RegretMap& cumul_regret_map,
   InstantRegretMap& instant_regret_map
)
{
   double pos_regret_sum{0.};
   for(auto& [action, cumul_reg] : cumul_regret_map) {
      auto& instant_regret = instant_regret_map[action];
      cumul_reg = instant_regret > 0. and cumul_reg < 0. ? instant_regret
                                                         : cumul_reg + instant_regret;
      instant_regret = 0.;
      pos_regret_sum += std::max(0., cumul_reg);
   }
   // apply the new policy to the vector policy
   if(pos_regret_sum > 0) {
      if(cumul_regret_map.size() != policy_map.size()) {
         throw std::invalid_argument(
            "Passed regrets and policy maps do not have the same number of elements"
         );
      }
      std::for_each(policy_map.begin(), policy_map.end(), [&](auto& entry) {
         return std::get< 1 >(entry
                ) = std::max(0., cumul_regret_map.at(action_wrapper(std::get< 0 >(entry))))
                    / pos_regret_sum;
      });
   } else {
      std::for_each(policy_map.begin(), policy_map.end(), [&](auto& entry) {
         return std::get< 1 >(entry) = 1. / static_cast< double >(policy_map.size());
      });
   }
}

/**
 * @brief Performs regret-matching on the given policy with respect to the provided regret
 *
 * @tparam Action
 * @tparam Policy
 */
template <
   typename Policy,
   typename RegretMap,
   typename ActionWrapper,
   typename Action = auto_action_type< Policy > >
// clang-format off
requires
   concepts::map< RegretMap >
   and std::is_convertible_v< typename RegretMap::mapped_type, double>
   and std::invocable< ActionWrapper, Action >
   and concepts::action_policy<
      Policy
   >
// clang-format on
void regret_matching_plus(
   Policy& policy_map,
   RegretMap& cumul_regret_map,
   ActionWrapper action_wrapper = [](const Action& action) { return action; }
)
{
   double pos_regret_sum{0.};
   for(auto& [action, cumul_regret] : cumul_regret_map) {
      cumul_regret = std::max(0., cumul_regret);
      pos_regret_sum += cumul_regret;
   }
   // apply the new policy to the vector policy
   if(pos_regret_sum > 0) {
      if(cumul_regret_map.size() != policy_map.size()) {
         throw std::invalid_argument(
            "Passed regrets and policy maps do not have the same number of elements"
         );
      }
      std::for_each(policy_map.begin(), policy_map.end(), [&](auto& entry) {
         return std::get< 1 >(entry
                ) = std::max(0., cumul_regret_map.at(action_wrapper(std::get< 0 >(entry))))
                    / pos_regret_sum;
      });
   } else {
      std::for_each(policy_map.begin(), policy_map.end(), [&](auto& entry) {
         return std::get< 1 >(entry) = 1. / static_cast< double >(policy_map.size());
      });
   }
}

/**
 * @brief emplaces the environment rewards for a terminal state and stores them in the node.
 *
 * No terminality checking is done within this method! Hence only call this method if you are
 * already certain that the node is a terminal one. Whether the environment rewards for
 * non-terminal states would be problematic is dependant on the environment.
 * @param[in] terminal_wstate the terminal state to collect rewards for.
 */
template < typename Env, typename Worldstate = auto_world_state_type< std::remove_cvref_t< Env > > >
   requires concepts::fosg< std::remove_cvref_t< Env > >
// clang-format off
auto collect_rewards(
   Env&& env,
   common::const_ref_if_t<   // the fosg concept asserts a reward function taking world_state_type.
                     // But if it can be passed a const world state then do so instead
      nor::concepts::has::method::reward_multi< std::remove_cvref_t< Env >, const Worldstate& >
         or concepts::has::method::reward< std::remove_cvref_t< Env >, const Worldstate& >,
      Worldstate > terminal_wstate,
   std::vector< Player > players = {})
// clang-format on
{
   using env_type = std::remove_cvref_t< Env >;
   if(players.empty()) {
      players = env.players(terminal_wstate);
   }
   // erase non-actual player elements (e.g. chance or unknown)
   std::erase_if(players, common::not_pred(utils::is_actual_player_pred));

   std::unordered_map< Player, double > rewards;
   rewards.reserve(players.size());

   if constexpr(nor::concepts::has::method::reward_multi< env_type >) {
      // if the environment has a method for returning all rewards for given players at
      // once, then we will assume this is a more performant alternative and use it
      // instead (e.g. when it is costly to compute the reward of each player
      // individually).
      auto all_rewards = env.reward(players, terminal_wstate);
      for(Player player : players) {
         rewards.emplace(player, all_rewards[player]);
      };
   } else {
      // otherwise we just loop over the per player reward method
      for(auto player : players) {
         rewards.emplace(player, env.reward(player, terminal_wstate));
      }
   }
   return rewards;
}

}  // namespace nor::rm

#endif  // NOR_RM_UTILS_HPP
