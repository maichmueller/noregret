
#ifndef NOR_POLICY_HPP
#define NOR_POLICY_HPP

#include <concepts>

#include "nor/concepts.hpp"
#include "nor/utils/utils.hpp"

namespace nor {

template < typename T >
inline T _zero()
{
   return T(0);
}

///**
// * @brief Adaptor class for using std::unordered_map as a valid type of an action policy.
// *
// *
// * @tparam Action
// */
template <
   concepts::action Action,
   //           std::invocable default_value_generator = decltype([]() { return 0.; })
   typename default_value_generator = decltype(&_zero< double >) >
requires std::is_invocable_r_v< double, default_value_generator >
class HashmapActionPolicy {
  public:
   using action_type = Action;
   using map_type = std::unordered_map< Action, double >;

   using iterator = typename map_type::iterator;
   using const_iterator = typename map_type::const_iterator;

   HashmapActionPolicy(default_value_generator dvg = &_zero< double >) : m_def_value_gen(dvg) {}
   HashmapActionPolicy(
      ranges::span< const action_type > actions,
      double value,
      default_value_generator dvg = &_zero< double >)
       : m_map(), m_def_value_gen(dvg)
   {
      for(const auto& action : actions) {
         emplace(action, value);
      }
   }
   HashmapActionPolicy(size_t n_actions, default_value_generator dvg = &_zero< double >)
       : m_map(), m_def_value_gen(dvg)
   {
      for(auto a : ranges::views::iota(size_t(0), n_actions)) {
         emplace(a, m_def_value_gen());
      }
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

   inline auto& operator[](const action_type& action)
   {
      if(auto found = find(action); found != end()) {
         return found->second;
      } else {
         return emplace(action, m_def_value_gen()).first->second;
      }
   }
   [[nodiscard]] inline auto operator[](const action_type& action) const
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
 * legal action filter has to be supplied.
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
         return action_policy_type(ranges::span(legal_actions), uniform_p);
      } else {
         // Otherwise we can compute directly the uniform probability vector
         constexpr double uniform_p = 1. / static_cast< double >(extent);
         return action_policy_type(extent, uniform_p);
      }
   }
};

template <
   typename Infostate,
   typename ActionPolicy,
   typename Table = std::unordered_map< Infostate, ActionPolicy > >
// clang-format off
requires
   concepts::map< Table >
   && concepts::action_policy< ActionPolicy >
// clang-format on
class TabularPolicy {
  public:
   using info_state_type = Infostate;
   using action_policy_type = ActionPolicy;
   using table_type = Table;

   TabularPolicy() requires std::is_default_constructible_v< table_type >
   = default;
   TabularPolicy(table_type&& table) : m_table(std::forward< table_type >(table)) {}

   auto begin() { return m_table.begin(); }
   auto end() { return m_table.end(); }
   [[nodiscard]] auto begin() const { return m_table.begin(); }
   [[nodiscard]] auto end() const { return m_table.end(); }

   template < typename... Args >
   auto emplace(Args&&... args)
   {
      return m_table.emplace(std::forward< Args >(args)...);
   }

   inline auto find(const info_state_type& action) { return m_table.find(action); }
   [[nodiscard]] inline auto find(const info_state_type& action) const
   {
      return m_table.find(action);
   }

   inline auto& operator[](const info_state_type& state) { return m_table[state]; }
   inline const auto& operator[](const info_state_type& state) const { return m_table.at(state); }

   [[nodiscard]] auto& table() const { return m_table; }

   [[nodiscard]] auto size() const requires(concepts::is::sized< table_type >)
   {
      return m_table.size();
   }

  private: table_type m_table;
};

}  // namespace nor

#endif  // NOR_POLICY_HPP
