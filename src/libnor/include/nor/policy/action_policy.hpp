

#ifndef NOR_ACTION_POLICY_HPP
#define NOR_ACTION_POLICY_HPP

#include <concepts>
#include <range/v3/all.hpp>

#include "nor/concepts.hpp"

namespace nor {

// template < typename Action >
//    requires std::equality_comparable< Action >
// class DeterministicActionPolicy {
//    using action_type = Action;
//
//       DeterministicActionPolicy(Action action) : m_choice({std::move(action)}) {}
//       DeterministicActionPolicy() : m_choice({std::nullopt}) {}
//
//    inline auto emplace(action_type action) { return m_choice[0] = std::move(action); }
//
//    inline auto begin() { return m_choice.begin(); }
//    [[nodiscard]] inline auto begin() const { return m_choice.begin(); }
//    inline auto end() { return m_choice.end(); }
//    [[nodiscard]] inline auto end() const { return m_choice.end(); }
//
//    [[nodiscard]] inline auto find(const action_type& action) const
//    {
//       return std::find_if(
//          begin(),
//          end(),
//          [&](const std::optional< action_type >& maybe_action) {
//             return maybe_action.has_value() ? maybe_action.value() == action : false;
//          }
//       );
//    }
//
//    [[nodiscard]] inline auto size() const noexcept { return m_choice[0].has_value() ? 1 : 0; }
//
//    [[nodiscard]] bool operator==(const DeterministicActionPolicy& other) const
//    {
//       return m_choice[0] == other.m_choice;
//    }
//
//    inline auto& operator[](const action_type& action)
//    {
//       if(auto found = find(action); found != end()) {
//          return 1.;
//       } else {
//          return 0.;
//       }
//    }
//    [[nodiscard]] inline auto at(const action_type& action) const
//    {
//       if(auto found = find(action); found != end()) {
//          return found->second;
//       } else {
//          return m_def_value_gen();
//       }
//    }
//    inline auto operator[](const action_type& action) const
//    {
//       if(auto found = find(action); found != end()) {
//          return found->second;
//       } else {
//          return m_def_value_gen();
//       }
//    }
//
//   private:
//    std::array< std::optional< action_type >, 1 > m_choice;
//    double m_prob = 1.;
// };

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
template < concepts::action Action >
class HashmapActionPolicy {
  public:
   using action_type = Action;
   using map_type = std::unordered_map< Action, double >;

   using iterator = typename map_type::iterator;
   using const_iterator = typename map_type::const_iterator;

   HashmapActionPolicy(std::function< double() > dvg = &_zero< double >)
       : m_def_value_gen(std::move(dvg))
   {
   }

   HashmapActionPolicy(ranges::range auto const& actions, double value, std::function< double() > dvg = &_zero< double >)
       : m_map(), m_def_value_gen(std::move(dvg))
   {
      for(const auto& action : actions) {
         emplace(action, value);
      }
   }

   template < typename T >
      requires concepts::maps< T, action_type > and concepts::mapping_of< T, double >
   HashmapActionPolicy(T&& action_value_pairs, std::function< double() > dvg = &_zero< double >)
       : m_map(), m_def_value_gen(std::move(dvg))
   {
      for(const auto& [action, value] : action_value_pairs) {
         emplace(action, value);
      }
   }

   HashmapActionPolicy(std::initializer_list< std::pair< action_type, double > > init_list)
       : m_map(), m_def_value_gen(&_zero< double >)
   {
      for(const auto& [action, value] : init_list) {
         emplace(action, value);
      }
   }
   HashmapActionPolicy(size_t n_actions, std::function< double() > dvg = &_zero< double >)
      requires std::is_integral_v< action_type >
   : m_map(), m_def_value_gen(std::move(dvg))
   {
      for(auto a : ranges::views::iota(size_t(0), n_actions)) {
         emplace(a, m_def_value_gen());
      }
   }
   //   HashmapActionPolicy(map_type map, std::function< double() > dvg = &_zero< double >)
   //       : m_map(std::move(map)), m_def_value_gen(std::move(dvg))
   //   {
   //   }

   ~HashmapActionPolicy() = default;
   HashmapActionPolicy(const HashmapActionPolicy& other) = default;
   HashmapActionPolicy(HashmapActionPolicy&& other) = default;
   HashmapActionPolicy& operator=(const HashmapActionPolicy& other) = default;
   HashmapActionPolicy& operator=(HashmapActionPolicy&& other) = default;

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
   std::function< double() > m_def_value_gen;
};

}  // namespace nor

#endif  // NOR_ACTION_POLICY_HPP
