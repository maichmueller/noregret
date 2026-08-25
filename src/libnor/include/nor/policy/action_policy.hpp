#ifndef NOR_ACTION_POLICY_HPP
#define NOR_ACTION_POLICY_HPP

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <ranges>
#include <utility>
#include <vector>

#include "nor/concepts.hpp"

namespace nor {

/**
 * @brief Tabular action policy storing (action, probability) entries.
 *
 * The internal table is an insertion-ordered flat std::vector<std::pair<const Action, double>>
 * instead of a hash map: the set of legal actions per infoset is tiny (typically <= 6), so a
 * linear scan beats hashing on every lookup while avoiding per-entry allocation and cache
 * misses. As a consequence iteration is deterministic in INSERTION ORDER (an improvement over
 * unordered_map's bucket order) and lookups are O(n) with a very small n.
 *
 * NOTE on reference stability: like std::vector, any emplace that grows the storage beyond its
 * capacity invalidates references/iterators into this policy. Callers holding a reference across
 * further inserts of the SAME policy object must size it first via reserve().
 *
 * Unseen actions are assigned a plain default value (0. by default) upon first access through
 * operator[]; const accessors at()/operator[] return the default without inserting.
 */
template < concepts::action Action >
class HashmapActionPolicy {
  public:
   using action_type = Action;
   using key_type = Action;
   using mapped_type = double;
   /// NOTE: keys are immutable (as with std::unordered_map's value_type) so that type-erased
   /// views over the entries remain reference-compatible; the store is append-only.
   using table_type = std::vector< std::pair< const Action, double > >;
   using value_type = typename table_type::value_type;

   using iterator = typename table_type::iterator;
   using const_iterator = typename table_type::const_iterator;
   using size_type = typename table_type::size_type;

   /// an empty policy assigning 0. to unseen actions
   HashmapActionPolicy() = default;

   /**
    * @brief constructs an empty policy whose default value is obtained by EAGERLY evaluating the
    * given generator once (replaces the former lazily-stored std::function<double()> generator).
    */
   template < typename DefaultGen >
      requires(std::regular_invocable< DefaultGen& > and std::convertible_to< std::invoke_result_t< DefaultGen& >, double > and not std::same_as< HashmapActionPolicy, std::remove_cvref_t< DefaultGen > >)
   HashmapActionPolicy(DefaultGen&& def_value_gen)
       : m_def_value(static_cast< double >(std::invoke(def_value_gen)))
   {
   }

   HashmapActionPolicy(table_type table, double def_value = 0.)
       : m_map(std::move(table)), m_def_value(def_value)
   {
   }

   template < typename FirstArg, typename SecondArg, typename... TailArgs >
      requires(not std::same_as< HashmapActionPolicy, std::remove_cvref_t< FirstArg > > and not std::same_as< table_type, std::remove_cvref_t< FirstArg > >)
   HashmapActionPolicy(FirstArg&& first_arg, SecondArg&& second_arg, TailArgs&&... args)
       : m_map{
          std::forward< FirstArg >(first_arg),
          std::forward< SecondArg >(second_arg),
          std::forward< TailArgs >(args)...}
   {
   }

   /// fills every given action with the same probability value
   HashmapActionPolicy(std::ranges::range auto&& actions, double value)
   {
      if constexpr(std::ranges::sized_range< decltype(actions) >) {
         m_map.reserve(static_cast< size_type >(std::ranges::size(actions)));
      }
      for(auto&& action : actions) {
         emplace(std::forward< decltype(action) >(action), value);
      }
   }

   template < typename T >
      requires(concepts::map_specced< std::remove_cvref_t< T >, action_type, double > and not std::same_as< HashmapActionPolicy, std::remove_cvref_t< T > > and not std::same_as< table_type, std::remove_cvref_t< T > >)
   HashmapActionPolicy(T&& mapping)
   {
      if constexpr(std::ranges::sized_range< T >) {
         m_map.reserve(static_cast< size_type >(std::ranges::size(mapping)));
      }
      for(auto&& [action, value] : std::forward< T >(mapping)) {
         emplace(std::forward< decltype(action) >(action), std::forward< decltype(value) >(value));
      }
   }

   template < typename ActionType, std::floating_point Float >
   HashmapActionPolicy(std::initializer_list< std::pair< ActionType, Float > > init_list)
   {
      m_map.reserve(init_list.size());
      for(auto& [action, value] : init_list) {
         emplace(std::move(action), static_cast< double >(value));
      }
   }

   HashmapActionPolicy(std::initializer_list< value_type > init_list)
   {
      m_map.reserve(init_list.size());
      for(auto& value : init_list) {
         emplace(std::move(value));
      }
   }

   /// enumerates actions [0, n_actions) with equal initial value (integral action types only)
   HashmapActionPolicy(size_t n_actions, double value = 0.)
      requires std::is_integral_v< action_type >
   {
      m_map.reserve(n_actions);
      for(auto a : std::views::iota(size_t(0), n_actions)) {
         emplace(a, value);
      }
   }

   ~HashmapActionPolicy() = default;
   HashmapActionPolicy(const HashmapActionPolicy& other) = default;
   HashmapActionPolicy(HashmapActionPolicy&& other) noexcept = default;
   // NOTE: custom assignment implementations -- the const-keyed entries are not assignable
   // (mirroring std::unordered_map value_type semantics), so entries are rebuilt instead.
   HashmapActionPolicy& operator=(const HashmapActionPolicy& other)
   {
      if(this != &other) {
         m_def_value = other.m_def_value;
         m_map.clear();
         m_map.reserve(other.size());
         for(const auto& [action, prob] : other.m_map) {
            m_map.emplace_back(action, prob);
         }
      }
      return *this;
   }
   HashmapActionPolicy& operator=(HashmapActionPolicy&& other) noexcept
   {
      if(this != &other) {
         m_def_value = other.m_def_value;
         m_map.clear();
         m_map.reserve(other.size());
         for(auto& [action, prob] : other.m_map) {
            m_map.emplace_back(std::move(action), prob);
         }
         other.m_map.clear();
      }
      return *this;
   }

   /// pre-sizes the storage so that subsequent inserts keep references/iterators valid
   void reserve(size_type n_entries) { m_map.reserve(n_entries); }
   [[nodiscard]] size_type capacity() const noexcept { return m_map.capacity(); }

   auto emplace(const action_type& action, double value) -> std::pair< iterator, bool >
   {
      if(auto found = find(action); found != end()) {
         return {found, false};
      }
      m_map.emplace_back(action, value);
      return {std::prev(end()), true};
   }

   template < typename PairLike >
      requires std::convertible_to< PairLike, value_type >
   auto emplace(PairLike&& action_and_value) -> std::pair< iterator, bool >
   {
      value_type entry(std::forward< PairLike >(action_and_value));
      return emplace(entry.first, entry.second);
   }

   auto begin() { return m_map.begin(); }
   [[nodiscard]] auto begin() const { return m_map.begin(); }
   auto end() { return m_map.end(); }
   [[nodiscard]] auto end() const { return m_map.end(); }

   auto find(const action_type& action) { return _find(action); }
   [[nodiscard]] auto find(const action_type& action) const { return _find(action); }

   [[nodiscard]] bool contains(const action_type& action) const { return _find(action) != end(); }

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

   /// access to the probability of 'action'; inserts the default value if not yet present
   auto& operator[](const action_type& action)
   {
      if(auto found = find(action); found != end()) {
         return found->second;
      } else {
         return emplace(action, m_def_value).first->second;
      }
   }

   /// read access returning the default value for unseen actions (without inserting)
   [[nodiscard]] auto at(const action_type& action) const
   {
      if(auto found = find(action); found != end()) {
         return found->second;
      }
      return m_def_value;
   }
   auto operator[](const action_type& action) const
   {
      if(auto found = find(action); found != end()) {
         return found->second;
      }
      return m_def_value;
   }

   void default_value(double def_value) noexcept { m_def_value = def_value; }
   [[nodiscard]] double default_value() const noexcept { return m_def_value; }

  private:
   table_type m_map{};
   double m_def_value = 0.;

   template < typename Self >
   static decltype(auto) _find(Self& self, const action_type& action)
   {
      return std::ranges::find_if(self.m_map, [&](const auto& action_and_prob) {
         return action_and_prob.first == action;
      });
   }
   auto _find(const action_type& action) { return _find(*this, action); }
   [[nodiscard]] auto _find(const action_type& action) const { return _find(*this, action); }
};

template < typename T >
HashmapActionPolicy(T&& mapping)
   -> HashmapActionPolicy< typename std::remove_cvref_t< T >::key_type >;

template < typename ActionType, std::floating_point Float >
HashmapActionPolicy(std::initializer_list< std::pair< ActionType, Float > > init_list)
   -> HashmapActionPolicy< ActionType >;
}  // namespace nor

#endif  // NOR_ACTION_POLICY_HPP
