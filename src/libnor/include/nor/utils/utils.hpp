#ifndef NOR_UTILS_HPP
#define NOR_UTILS_HPP

#include <iterator>
#include <type_traits>

#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"

template < typename T >
using uptr = std::unique_ptr< T >;
template < typename T >
using sptr = std::shared_ptr< T >;
template < typename T >
using wptr = std::weak_ptr< T >;

namespace nor::utils {

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
