
#ifndef NOR_POLICY_HPP
#define NOR_POLICY_HPP

#include <concepts>
#include <range/v3/all.hpp>

#include "nor/concepts.hpp"
#include "nor/utils/utils.hpp"

namespace nor {

template < typename T >
inline T _zero()
{
   return T(0);
}

/**
 * @brief Adaptor class for using std::unordered_map as a valid type of an action policy.
 *
 *
 * @tparam Action
 */
template < concepts::action Action, typename default_value_generator = decltype(&_zero< double >) >
   requires std::is_invocable_r_v< double, default_value_generator >
class HashmapActionPolicy {
  public:
   using action_type = Action;
   using map_type = std::unordered_map< Action, double >;

   using iterator = typename map_type::iterator;
   using const_iterator = typename map_type::const_iterator;

   HashmapActionPolicy(default_value_generator dvg = &_zero< double >) : m_def_value_gen(dvg) {}
   HashmapActionPolicy(
      ranges::range auto const& actions,
      double value,
      default_value_generator dvg = &_zero< double >)
       : m_map(), m_def_value_gen(dvg)
   {
      for(const auto& action : actions) {
         emplace(action, value);
      }
   }
   HashmapActionPolicy(size_t n_actions, default_value_generator dvg = &_zero< double >)
      requires std::is_integral_v< action_type >
   : m_map(), m_def_value_gen(dvg)
   {
      for(auto a : ranges::views::iota(size_t(0), n_actions)) {
         emplace(a, m_def_value_gen());
      }
   }
   HashmapActionPolicy(map_type map, default_value_generator dvg = &_zero< double >)
       : m_map(std::move(map)), m_def_value_gen(dvg)
   {
   }

   template < typename... Args >
   inline auto emplace(Args&&... args)
   {
      return m_map.emplace(std::forward< Args >(args)...);
   }

   inline auto begin() { return m_map.begin(); }
   [[nodiscard]] inline auto begin() const { return m_map.begin(); }
   inline auto end() { return m_map.end(); }
   [[nodiscard]] inline auto end() const { return m_map.end(); }

   inline auto find(const action_type& action) { return m_map.find(action); }
   [[nodiscard]] inline auto find(const action_type& action) const { return m_map.find(action); }

   [[nodiscard]] inline auto size() const noexcept { return m_map.size(); }

   [[nodiscard]] bool operator==(const HashmapActionPolicy& other) const
   {
      if(size() != other.size()) {
         return false;
      }
      return std::all_of(begin(), end(), [&](const auto& action_and_prob) {
         return other.at(std::get< 0 >(action_and_prob)) == std::get< 1 >(action_and_prob);
      });
   }

   inline auto& operator[](const action_type& action)
   {
      if(auto found = find(action); found != end()) {
         return found->second;
      } else {
         return emplace(action, m_def_value_gen()).first->second;
      }
   }
   [[nodiscard]] inline auto at(const action_type& action) const
   {
      if(auto found = find(action); found != end()) {
         return found->second;
      } else {
         return m_def_value_gen();
      }
   }

  private:
   map_type m_map;
   default_value_generator m_def_value_gen;
};

template < typename State, std::size_t extent >
constexpr size_t placeholder_filter([[maybe_unused]] Player player, const State&)
{
   return extent;
}

/**
 * @brief Creates a uniform state_policy taking in the given state and action type.
 *
 * This will return a uniform probability vector over the legal actions as provided by the
 * LegalActionsFilter. The filter takes in an object of state type and returns the legal actions. In
 * the case of fixed action size it will simply return dummy 0 values to do nothing.
 * Each probability vector can be further accessed by the action index to receive the action
 * probability.
 *
 * @tparam Infostate, the state type. This is the key type for 'lookup' (not actually a lookup here)
 * in the state_policy.
 * @tparam extent, the number of legal actions at any given time. If std::dynamic_extent, then the
 * legal actions have to be supplied.
 * @tparam LegalActionFunctor, type of callable that provides the legal actions in the case of
 * dynamic_extent.
 */
template < typename Infostate, typename ActionPolicy, std::size_t extent = std::dynamic_extent >
   requires concepts::info_state< Infostate > && concepts::action_policy< ActionPolicy >
class UniformPolicy {
  public:
   using info_state_type = Infostate;
   using action_policy_type = ActionPolicy;

   UniformPolicy() = default;

   auto operator[](const std::pair<
                   info_state_type,
                   std::vector< typename fosg_auto_traits< action_policy_type >::action_type > >&
                      infostate_legalactions) const
   {
      if constexpr(extent == std::dynamic_extent) {
         const auto& [info_state, legal_actions] = infostate_legalactions;
         double uniform_p = 1. / static_cast< double >(legal_actions.size());
         return action_policy_type(legal_actions, uniform_p);
      } else {
         // Otherwise we can compute directly the uniform probability vector
         constexpr double uniform_p = 1. / static_cast< double >(extent);
         return action_policy_type(extent, uniform_p);
      }
   }
};

/**
 * @brief Creates a state_policy with 0 as the probability value for any new states.

 * @tparam Infostate, the state type. This is the key type for 'lookup' (not actually a lookup here)
 * in the state_policy.
 * @tparam extent, the number of legal actions at any given time. If std::dynamic_extent, then the
 * legal actions have to be supplied.
 * @tparam LegalActionFunctor, type of callable that provides the legal actions in the case of
 * dynamic_extent.
 */
template < typename Infostate, typename ActionPolicy, std::size_t extent = std::dynamic_extent >
   requires concepts::info_state< Infostate > && concepts::action_policy< ActionPolicy >
class ZeroDefaultPolicy {
  public:
   using info_state_type = Infostate;
   using action_policy_type = ActionPolicy;

   ZeroDefaultPolicy() = default;

   auto operator[](const std::pair<
                   info_state_type,
                   std::vector< typename fosg_auto_traits< action_policy_type >::action_type > >&
                      infostate_legalactions) const
   {
      if constexpr(extent == std::dynamic_extent) {
         const auto& [info_state, legal_actions] = infostate_legalactions;
         return action_policy_type(legal_actions, 0.);
      } else {
         return action_policy_type(extent, 0.);
      }
   }
};

template <
   typename Infostate,
   typename ActionPolicy,
   typename DefaultPolicy = UniformPolicy< Infostate, ActionPolicy >,
   typename Table = std::unordered_map< Infostate, ActionPolicy > >
// clang-format off
requires
   concepts::map< Table >
   && concepts::default_state_policy<
         DefaultPolicy,
         Infostate,
         ActionPolicy
      >
   && concepts::action_policy< ActionPolicy >
// clang-format on
class TabularPolicy {
  public:
   using info_state_type = Infostate;
   using action_policy_type = ActionPolicy;
   using action_type = typename fosg_auto_traits< action_policy_type >::action_type;
   using default_policy_type = DefaultPolicy;
   using table_type = Table;

   TabularPolicy()
      requires all_predicate_v< std::is_default_constructible, table_type, default_policy_type >
   = default;
   TabularPolicy(table_type table) : m_table(std::move(table)) {}
   TabularPolicy(table_type table, default_policy_type default_policy)
       : m_table(std::move(table)), m_default_policy(std::move(default_policy))
   {
   }

   auto begin() { return m_table.begin(); }
   auto end() { return m_table.end(); }
   [[nodiscard]] auto begin() const { return m_table.begin(); }
   [[nodiscard]] auto end() const { return m_table.end(); }

   template < typename... Args >
   auto emplace(Args&&... args)
   {
      return m_table.emplace(std::forward< Args >(args)...);
   }

   inline auto find(const info_state_type& infostate) { return m_table.find(infostate); }
   [[nodiscard]] inline auto find(const info_state_type& infostate) const
   {
      return m_table.find(infostate);
   }

   template < typename T >
   const action_policy_type& operator[](const std::pair< info_state_type, T >& state_any_pair) const
   {
      const auto& infostate = std::get< 0 >(state_any_pair);
      return m_table.at(infostate);
   }

   action_policy_type& operator[](
      const std::pair< info_state_type, std::vector< action_type > >& state_action_pair)
   {
      const auto& infostate = std::get< 0 >(state_action_pair);
      auto found_action_policy = find(infostate);
      if(found_action_policy == m_table.end()) {
         return m_table.emplace(infostate, m_default_policy[state_action_pair]).first->second;
      }
      return found_action_policy->second;
   }

   [[nodiscard]] auto size() const
      requires(concepts::is::sized< table_type >)
   {
      return m_table.size();
   }

   [[nodiscard]] auto& table() const { return m_table; }

  private:
   table_type m_table;
   /// the fallback policy to use when the encountered infostate has not been obseved before
   default_policy_type m_default_policy{};
};

template < concepts::world_state Worldstate, concepts::action Action >
class FixedActionsChanceDistribution {
   using world_state_type = Worldstate;
   using action_type = Action;
   FixedActionsChanceDistribution(std::unordered_map< action_type, double > action_dist)
       : m_action_dist(std::move(action_dist))
   {
   }

   double operator[](const std::pair< Worldstate, action_type >& state_action_pair)
   {
      return m_action_dist.at(state_action_pair.second);
   }

  private:
   std::unordered_map< action_type, double > m_action_dist;
};

template <
   concepts::info_state Infostate,
   concepts::action Action,
   typename TabularOpponentPolicy,
   typename Table = std::unordered_map< Infostate, Action > >
// clang-format off
requires
   concepts::state_policy<
      TabularOpponentPolicy,
      Infostate,
      typename fosg_auto_traits< Infostate >::observation_type,
      Action>
   && concepts::map< Table, Infostate, Action >
   // clang-format on
   class TabularBestResponse {
   TabularBestResponse(
      Player best_responding_player,
      const TabularOpponentPolicy& opponent_policy,
      Table best_response_table)
       : m_responder(best_responding_player),
         m_opponent_policy(&opponent_policy),
         m_best_response(std::move(best_response_table))
   {
   }

  private:
   /// the player who is acting according to the best response
   Player m_responder;
   /// the tabular opponent policy the given player is searching the best response to
   const TabularOpponentPolicy* m_opponent_policy;
   /// the table of a best response to a given infostate
   Table m_best_response;
};

}  // namespace nor

#endif  // NOR_POLICY_HPP
