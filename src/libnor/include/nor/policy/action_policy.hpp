

#ifndef NOR_ACTION_POLICY_HPP
#define NOR_ACTION_POLICY_HPP

#include <concepts>
#include <range/v3/all.hpp>

#include "nor/concepts.hpp"

namespace nor {

template < typename T, typename... Args >
inline T _zero(Args&&... args)
{
   return T(0);
}

/**
 * @brief Adaptor class for using std::unordered_map as a valid type of an action policy.
 *
 *
 * @tparam Action
 */
template < concepts::action Action>
class HashmapActionPolicy {
  public:
   using action_type = Action;
   using map_type = std::unordered_map< Action, double >;

   using iterator = typename map_type::iterator;
   using const_iterator = typename map_type::const_iterator;

   template <typename default_value_generator>
   HashmapActionPolicy(default_value_generator dvg = &_zero< double >) : m_def_value_gen(dvg) {}
   template <typename default_value_generator>
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
   template <typename default_value_generator>
   HashmapActionPolicy(size_t n_actions, default_value_generator dvg = &_zero< double >)
      requires std::is_integral_v< action_type >
   : m_map(), m_def_value_gen(dvg)
   {
      for(auto a : ranges::views::iota(size_t(0), n_actions)) {
         emplace(a, m_def_value_gen());
      }
   }
   template <typename default_value_generator>
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
   inline auto operator[](const action_type& action) const
   {
      if(auto found = find(action); found != end()) {
         return found->second;
      } else {
         return m_def_value_gen();
      }
   }

  private:
   map_type m_map;
   std::function<double()> m_def_value_gen;
};

}  // namespace nor

#endif  // NOR_ACTION_POLICY_HPP
