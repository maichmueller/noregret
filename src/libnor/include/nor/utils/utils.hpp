#ifndef NOR_UTILS_HPP
#define NOR_UTILS_HPP

#include <iterator>
#include <type_traits>

//#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"


namespace nor {

// using this macro gave cryptic allocator problems
#ifndef NEW_EMPTY_TYPE
   #define NEW_EMPTY_TYPE decltype([]() {})
#endif

/**
 * A type hint to help identify the remaining template types in a constructor without specifying
 * them twice.
 * @tparam T
 */
template < typename... Ts >
struct Hint {
};

}  // namespace nor

namespace nor::utils {


constexpr const char* btos(bool b)
{
   if(b) {
      return "true";

   } else {
      return "false";
   }
}

template < class >
inline constexpr bool always_false_v = false;

template < bool condition, typename T >
inline std::conditional_t< condition, T &&, T & > move_if(T &obj)
{
   if constexpr(condition) {
      return std::move(obj);
   } else {
      return obj;
   }
}

template < template < typename > class UnaryPredicate, typename T >
inline std::conditional_t< UnaryPredicate< T >::value, T &&, T & > move_if(T &obj)
{
   if constexpr(UnaryPredicate< T >::value) {
      return std::move(obj);
   } else {
      return obj;
   }
}

template < typename T >
auto clone_any_way(const T &obj)
{
   if constexpr(std::is_pointer_v< T > && concepts::has::method::clone_ptr< T >) {
      return obj->clone();
   } else if constexpr(concepts::has::method::clone_self< T >) {
      return obj.clone();
   } else if constexpr(concepts::has::method::copy< T >) {
      return obj.copy();
   } else if constexpr(std::is_copy_constructible_v< T >) {
      return T(obj);
   } else {
      static_assert(always_false_v< T >, "No cloning/copying method available in given type.");
   }
}

template < typename Iter >
class ConstView {
  public:
   static_assert(
      concepts::is::const_iter< Iter >,
      "ConstView can be constructed from const_iterators only.");

   ConstView(Iter begin, Iter end) : m_begin(begin), m_end(end) {}

   [[nodiscard]] auto begin() const { return m_begin; }
   [[nodiscard]] auto end() const { return m_end; }

  private:
   Iter m_begin;
   Iter m_end;
};

template < typename Iter >
auto advance(Iter &&iter, typename Iter::difference_type n)
{
   Iter it = std::forward< Iter >(iter);
   std::advance(it, n);
   return it;
}

template < typename Key, typename Value, std::size_t Size >
struct CEMap {
   std::array< std::pair< Key, Value >, Size > data;

   [[nodiscard]] constexpr Value at(const Key &key) const
   {
      const auto itr = std::find_if(
         begin(data), end(data), [&key](const auto &v) { return v.first == key; });
      if(itr != end(data)) {
         return itr->second;
      } else {
         throw std::range_error("Not Found");
      }
   }
};

template < nor::concepts::is::enum_ Enum >
std::string enum_name(Enum e);

template <>
inline std::string enum_name(Player e)
{
   constexpr CEMap< Player, const char *, 27 > name_lookup = {
      std::pair{Player::chance, "chance"},     std::pair{Player::alex, "alex"},
      std::pair{Player::bob, "bob"},           std::pair{Player::cedric, "cedric"},
      std::pair{Player::dexter, "dexter"},     std::pair{Player::emily, "emily"},
      std::pair{Player::florence, "florence"}, std::pair{Player::gustavo, "gustavo"},
      std::pair{Player::henrick, "henrick"},   std::pair{Player::ian, "ian"},
      std::pair{Player::julia, "julia"},       std::pair{Player::kelvin, "kelvin"},
      std::pair{Player::lea, "lea"},           std::pair{Player::michael, "michael"},
      std::pair{Player::norbert, "norbert"},   std::pair{Player::oscar, "oscar"},
      std::pair{Player::pedro, "pedro"},       std::pair{Player::quentin, "quentin"},
      std::pair{Player::rosie, "rosie"},       std::pair{Player::sophia, "sophia"},
      std::pair{Player::tristan, "tristan"},   std::pair{Player::ulysses, "ulysses"},
      std::pair{Player::victoria, "victoria"}, std::pair{Player::william, "william"},
      std::pair{Player::xavier, "xavier"},     std::pair{Player::yeet, "yeet"},
      std::pair{Player::zoey, "zoey"}};
   return name_lookup.at(e);
}
template <>
inline std::string enum_name(TurnDynamic e)
{
   constexpr CEMap< TurnDynamic, const char *, 2 > name_lookup = {
      std::pair{TurnDynamic::sequential, "sequential"},
      std::pair{TurnDynamic::simultaneous, "simultaneous"}};
   return name_lookup.at(e);
}
template <>
inline std::string enum_name(Stochasticity e)
{
   constexpr CEMap< Stochasticity, const char *, 3 > name_lookup = {
      std::pair{Stochasticity::deterministic, "deterministic"},
      std::pair{Stochasticity::sample, "sample"},
      std::pair{Stochasticity::choice, "choice"}};
   return name_lookup.at(e);
}

template < nor::concepts::is::enum_ Enum >
requires nor::concepts::is::any_of< Enum, Player, TurnDynamic, Stochasticity >
inline auto &operator<<(std::ostream &os, Enum e)
{
   os << nor::utils::enum_name(e);
   return os;
}

}  // namespace nor::utils

template < nor::concepts::is::enum_ Enum, typename T >
requires nor::concepts::is::any_of< Enum, nor::Player, nor::TurnDynamic, nor::Stochasticity >
inline std::string operator+(const T &other, Enum e)
{
   return std::string_view(other) + nor::utils::enum_name(e);
}
template < nor::concepts::is::enum_ Enum, typename T >
requires nor::concepts::is::any_of< Enum, nor::Player, nor::TurnDynamic, nor::Stochasticity >
inline std::string operator+(Enum e, const T &other)
{
   return nor::utils::enum_name(e) + std::string_view(other);
}

template < nor::concepts::is::enum_ Enum >
requires nor::concepts::is::any_of< Enum, nor::Player, nor::TurnDynamic, nor::Stochasticity >
inline auto &operator<<(std::stringstream &os, Enum e)
{
   os << nor::utils::enum_name(e);
   return os;
}

#endif  // NOR_UTILS_HPP
