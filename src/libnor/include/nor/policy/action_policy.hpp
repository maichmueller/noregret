

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
inline T _zero(Args&&...)
{
   return T(0);
}

/**
 * @brief Adaptor class for using std::unordered_map as a valid type of an action policy.
 *
 *
 * @tparam Action
 */
template <
   concepts::action Action,
   concepts::map Table = std::unordered_map<
      Action,
      double,
      common::value_hasher< Action >,
      common::value_comparator< Action > > >
class HashmapActionPolicy {
  public:
   using action_type = Action;
   using table_type = Table;
   using mapped_type = typename Table::mapped_type;
   using key_type = typename Table::key_type;
   using value_type = typename table_type::value_type;

   using iterator = typename table_type::iterator;
   using const_iterator = typename table_type::const_iterator;

   HashmapActionPolicy(std::function< double() > dvg = &_zero< double >)
       : m_def_value_gen(std::move(dvg))
   {
   }

   HashmapActionPolicy(table_type table, std::function< double() > dvg = &_zero< double >)
       : m_map(std::move(table)), m_def_value_gen(std::move(dvg))
   {
   }

   template < typename FirstArg, typename SecondArg, typename... TailArgs >
      requires(not std::same_as< HashmapActionPolicy, std::remove_cvref_t< FirstArg > >
               and not std::same_as< table_type, std::remove_cvref_t< FirstArg > >)
   HashmapActionPolicy(FirstArg&& first_arg, SecondArg&& second_arg, TailArgs&&... args)
       : m_map{std::forward< FirstArg >(first_arg), std::forward< SecondArg >(second_arg), std::forward< TailArgs >(args)...},
         m_def_value_gen(&_zero< double >)
   {
   }

   HashmapActionPolicy(ranges::range auto&& actions, double value, std::function< double() > dvg = &_zero< double >)
       : m_map(), m_def_value_gen(std::move(dvg))
   {
      for(auto&& action : actions) {
         emplace(std::forward< decltype(action) >(action), value);
      }
   }

   template < typename T >
      requires(concepts::map_specced< std::remove_cvref_t< T >, action_type, double >
               and not std::same_as< HashmapActionPolicy, std::remove_cvref_t< T > >
               and not std::same_as< table_type, std::remove_cvref_t< T > >)
   HashmapActionPolicy(T&& mapping, std::function< double() > dvg = &_zero< double >)
       : m_map(), m_def_value_gen(std::move(dvg))
   {
      for(auto&& [action, value] : std::forward< T >(mapping)) {
         emplace(std::forward< decltype(action) >(action), std::forward< decltype(value) >(value));
      }
   }

   template < typename ActionType, std::floating_point Float >
   HashmapActionPolicy(std::initializer_list< std::pair< ActionType, Float > > init_list, std::function< double() > dvg = &_zero< double >)
       : m_map(), m_def_value_gen(std::move(dvg))
   {
      for(auto& [action, value] : init_list) {
         emplace(std::move(action), static_cast< double >(value));
      }
   }

   HashmapActionPolicy(std::initializer_list< value_type > init_list, std::function< double() > dvg = &_zero< double >)
       : m_map(), m_def_value_gen(std::move(dvg))
   {
      for(auto& value : init_list) {
         emplace(std::move(value));
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

   ~HashmapActionPolicy() = default;
   HashmapActionPolicy(const HashmapActionPolicy& other) = default;
   HashmapActionPolicy(HashmapActionPolicy&& other) noexcept = default;
   HashmapActionPolicy& operator=(const HashmapActionPolicy& other) = default;
   HashmapActionPolicy& operator=(HashmapActionPolicy&& other) noexcept = default;

   template < typename... Args >
   auto emplace(Args&&... args)
   {
      return m_map.emplace(std::forward< Args >(args)...);
   }

   auto begin() { return m_map.begin(); }
   [[nodiscard]] auto begin() const { return m_map.begin(); }
   auto end() { return m_map.end(); }
   [[nodiscard]] auto end() const { return m_map.end(); }

   auto find(const action_type& action) { return m_map.find(action); }
   [[nodiscard]] auto find(const action_type& action) const { return m_map.find(action); }

   [[nodiscard]] auto size() const noexcept { return m_map.size(); }

   [[nodiscard]] bool operator==(const HashmapActionPolicy& other) const
   {
      if(size() != other.size()) {
         return false;
      }
      return std::all_of(begin(), end(), [&](const auto& action_and_prob) {
         return other.at(std::get< 0 >(action_and_prob)) == std::get< 1 >(action_and_prob);
      });
   }

   auto& operator[](const action_type& action)
   {
      if(auto found = find(action); found != end()) {
         return found->second;
      } else {
         return emplace(action, m_def_value_gen()).first->second;
      }
   }

   [[nodiscard]] auto at(const action_type& action) const
   {
      if(auto found = find(action); found != end()) {
         return found->second;
      } else {
         return m_def_value_gen();
      }
   }
   auto operator[](const action_type& action) const
   {
      if(auto found = find(action); found != end()) {
         return found->second;
      }
      return m_def_value_gen();
   }

   void default_value_generator(std::function< double() > def_value_gen)
   {
      m_def_value_gen = std::move(def_value_gen);
   }
   const auto& default_value_generator() const { return m_def_value_gen; }

  private:
   table_type m_map;
   std::function< double() > m_def_value_gen;
};

template < typename T >
HashmapActionPolicy(T&& mapping, std::function< double() > dvg = &_zero< double >)
   -> HashmapActionPolicy< typename std::remove_cvref_t< T >::key_type >;

template < typename ActionType, std::floating_point Float >
HashmapActionPolicy(std::initializer_list< std::pair< ActionType, Float > > init_list, std::function< double() > dvg = &_zero< double >)
   -> HashmapActionPolicy< ActionType >;
}  // namespace nor

#endif  // NOR_ACTION_POLICY_HPP
