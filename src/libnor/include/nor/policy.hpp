
#ifndef NOR_POLICY_HPP
#define NOR_POLICY_HPP

#include "nor/concepts.hpp"
#include "nor/utils/utils.hpp"

namespace nor {

template < typename Worldstate, std::size_t extent >
constexpr auto fixed_size_filter([[maybe_unused]] Player player, const Worldstate&)
{
   return std::vector< int >(extent, 0);
}

/**
 * @brief Creates a uniform state_policy taking in the given state and action type.
 *
 * This will return a uniform probability vector over the legal actions as provided by the
 * LegalActionsFilter. The filter takes in an object of state type and returns the legal actions. In
 * the case of fixed action size it will simply return dummy 0 values to do nothing.
 * Each probabability vector can be further accessed by the action index to receive the action
 * probability.
 *
 * @tparam Infostate, the state type. This is the key type for 'lookup' (not actually a lookup here)
 * in the state_policy.
 * @tparam extent, the number of legal actions at any given time. If std::dynamic_extent, then the
 * legal action filter has to be supplied.
 * @tparam LegalActionFunctor, type of callable that provides the legal actions in the case of
 * dynamic_extent.
 */
template <
   typename Infostate,
   std::size_t extent = std::dynamic_extent,
   std::invocable< Player, const Infostate& >
      LegalActionGetterType = decltype(&fixed_size_filter< Infostate, extent >) >
requires concepts::info_state< Infostate >
class UniformPolicy {
   using info_state_type = Infostate;

   UniformPolicy() requires(extent != std::dynamic_extent)
       : m_la_getter(&fixed_size_filter< State, extent >)
   {
   }
   explicit UniformPolicy(LegalActionGetterType la_getter) requires(extent == std::dynamic_extent)
       : m_la_getter(la_getter)
   {
   }

   auto& operator[](const info_state_type& info_state) const
   {
      if constexpr(extent == std::dynamic_extent) {
         // if we have non-fixed-size action vectors coming in, then we need to ask the filter to
         // provide us with the legal actions.
         auto legal_actions = m_la_getter(info_state);
         double uniform_p = 1. / static_cast< double >(legal_actions.size());
         return std::vector< double >(legal_actions.size(), uniform_p);
      } else {
         // Otherwise we can compute directly the uniform probability vector
         constexpr double uniform_p = 1. / static_cast< double >(extent);
         return std::vector< double >(extent, uniform_p);
      }
   }
   template < typename Action = NEW_EMPTY_TYPE >
   auto& operator[](const std::pair< info_state_type, Action >& state_action) const
   {
      return (*this)[std::get< 0 >(state_action)][0];
   }

  private:
   LegalActionGetterType m_la_getter;
};

template <
   typename Map,
   typename DefaultPolicy,
   typename KeyType = typename Map::key_type,
   typename MappedType = typename Map::mapped_type >
requires concepts::map< Map, KeyType, MappedType > && concepts::state_policy< DefaultPolicy >
class TabularPolicy {
   using state_type = typename Map::key_type;
   using mapped_type = typename Map::mapped_type;
   using default_policy_type = DefaultPolicy;

   TabularPolicy() requires std::is_default_constructible_v< Map >
   = default;
   explicit TabularPolicy(Map&& map, DefaultPolicy&& def_policy = DefaultPolicy())
       : m_table(std::forward< Map >(map)),
         m_default_policy(std::forward< DefaultPolicy >(def_policy))
   {
   }

   auto& operator[](const state_type& state)
   {
      auto found_policy = m_table.find(state);
      if(found_policy == m_table.end()) {
         auto default_policy_vec = m_default_policy[state];
         m_table.insert({state, default_policy_vec});
         return default_policy_vec;
      }
      return found_policy->second;
   }

   template < concepts::action Action >
   auto& operator[](const std::pair< state_type, Action >& state_action)
   {
      auto&& [state, action] = state_action;
      return m_table[state][action];
   }

   auto& operator[](const state_type& state) const { return m_table.at(state); }

   template < concepts::action Action >
   auto& operator[](const std::pair< state_type, Action >& state_action) const
   {
      auto&& [state, action] = state_action;
      return m_table.at(state)[action];
   }

  private:
   Map m_table;
   default_policy_type m_default_policy;
};

}  // namespace nor

#endif  // NOR_POLICY_HPP
