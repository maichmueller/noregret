
#ifndef NOR_POLICY_HPP
#define NOR_POLICY_HPP

#include "nor/concepts.hpp"
#include "nor/utils/utils.hpp"

namespace nor {

/**
 * @brief Adaptor class for using std::unordered_map as a valid type of an action policy.
 *
 *
 * @tparam Action
 * @tparam Pred
 * @tparam Alloc
 */
template <
   concepts::action Action,
   class Hash = std::hash< Action >,
   class KeyEqual = std::equal_to< Action >,
   class Alloc = std::allocator< std::pair< const Action, double > >,
   std::invocable default_value_generator = decltype([]() { return 0.; }) >
class HashMapActionPolicy {
  public:
   using action_type = Action;
   using map_type = std::unordered_map< Action, double, Hash, KeyEqual, Alloc >;

   using iterator = typename map_type::iterator;
   using const_iterator = typename map_type::const_iterator;

   HashMapActionPolicy() = default;
   HashMapActionPolicy(std::span< action_type > actions, double value) : m_map()
   {
      for(const auto& action : actions) {
         emplace(action, value);
      }
   }
   HashMapActionPolicy(size_t actions, default_value_generator) : m_map() {}

   template < typename... Args >
   inline auto emplace(Args&&... args)
   {
      return m_map.emplace(std::forward< Args >(args)...);
   }

   inline auto begin() { return m_map.begin(); }
   inline auto begin() const { return m_map.begin(); }
   inline auto end() { return m_map.end(); }
   inline auto end() const { return m_map.end(); }

   inline auto find(const action_type& action) { return m_map.find(action); }
   inline auto find(const action_type& action) const { return m_map.find(action); }

   inline auto& operator[](const action_type& action)
   {
      if(auto found = find(action); found != end()) {
         return found->second;
      } else {
         return emplace(action, default_value_generator{}()).first->second;
      }
   }
   inline auto operator[](const action_type& action) const
   {
      if(auto found = find(action); found != end()) {
         return found->second;
      } else {
         return default_value_generator{}();
      }
   }

   [[nodiscard]] size_t size() const { return m_map.size(); }

  private:
   map_type m_map;
};

template < typename Worldstate, std::size_t extent >
constexpr size_t placeholder_filter([[maybe_unused]] Player player, const Worldstate&)
{
   return extent;
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
   typename Action,
   typename ActionPolicy = HashMapActionPolicy< Action >,
   std::size_t extent = std::dynamic_extent,
   std::invocable< const Infostate& >
      LegalActionGetterType = decltype(&placeholder_filter< Infostate, extent >) >
requires concepts::info_state< Infostate > && concepts::action_policy< ActionPolicy, Action >
class UniformPolicy {
  public:
   using info_state_type = Infostate;
   using action_type = Action;
   using action_policy_type = ActionPolicy;

   UniformPolicy() requires(extent != std::dynamic_extent)
       : m_la_getter(&placeholder_filter< Infostate, extent >)
   {
   }
   explicit UniformPolicy(LegalActionGetterType la_getter, Hint< Infostate, Action >) requires(
      extent == std::dynamic_extent)
       : m_la_getter(std::move(la_getter))
   {
   }
   explicit UniformPolicy(LegalActionGetterType la_getter, Hint< Infostate, Action, ActionPolicy >) requires(
      extent == std::dynamic_extent && std::is_default_constructible_v< ActionPolicy >)
       : m_la_getter(std::move(la_getter))
   {
   }

   auto operator[](const info_state_type& info_state) const
   {
      if constexpr(extent == std::dynamic_extent) {
         // if we have non-fixed-size action vectors coming in, then we need to ask the filter to
         // provide us with the legal actions.
         auto legal_actions = m_la_getter(info_state);
         double uniform_p = 1. / static_cast< double >(legal_actions.size());
         return action_policy_type(std::span(legal_actions), uniform_p);
      } else {
         // Otherwise we can compute directly the uniform probability vector
         constexpr double uniform_p = 1. / static_cast< double >(extent);
         return action_policy_type(extent, uniform_p);
      }
   }
   auto operator[](const std::pair< info_state_type, Action >& state_action) const
   {
      if constexpr(extent == std::dynamic_extent) {
         // if we have non-fixed-size action vectors coming in, then we need to ask the filter to
         // provide us with the legal actions.
         return 1. / static_cast< double >(m_la_getter(std::get< 0 >(state_action)).size());
      } else {
         // Otherwise we can compute directly the uniform probability vector
         return 1. / static_cast< double >(extent);
      }
   }

  private:
   LegalActionGetterType m_la_getter;
};

template < typename DefaultPolicy, typename Table >
// clang-format off
requires
   concepts::map< Table >
   && concepts::default_state_policy< DefaultPolicy >
   && concepts::action_policy< typename DefaultPolicy::action_policy_type >
// clang-format on
class TabularPolicy {
  public:
   using default_policy_type = DefaultPolicy;
   using action_policy_type = typename DefaultPolicy::action_policy_type;

   TabularPolicy() requires
      std::is_default_constructible_v< Table > && std::is_default_constructible_v< DefaultPolicy >
   = default;

   explicit TabularPolicy(
      DefaultPolicy def_policy = DefaultPolicy()) requires std::is_default_constructible_v< Table >:
       m_table(),
       m_default_policy(std::forward< DefaultPolicy >(def_policy))
   {
   }
   explicit TabularPolicy(Table&& table, DefaultPolicy&& def_policy = DefaultPolicy())
       : m_table(std::forward< Table >(table)),
         m_default_policy(std::forward< DefaultPolicy >(def_policy))
   {
   }
   explicit TabularPolicy(DefaultPolicy def_policy)
       : m_table(), m_default_policy(std::move(def_policy))
   {
   }

   template <typename Infostate>
   auto& operator[](const Infostate& state)
   {
      auto found_policy = m_table.find(state);
      if(found_policy == m_table.end()) {
         return m_table.emplace(state, m_default_policy[state]).first->second;
      }
      return found_policy->second;
   }

   template <typename Infostate>
   const auto& operator[](const Infostate& state) const { return m_table.at(state); }

  private:
   Table m_table;
   default_policy_type m_default_policy;
};

}  // namespace nor

#endif  // NOR_POLICY_HPP
