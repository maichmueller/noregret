#pragma once

#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <range/v3/all.hpp>
#include <sstream>
#include <string>
#include <utility>

template < typename T >
using uptr = std::unique_ptr< T >;
template < typename T >
using sptr = std::shared_ptr< T >;
template < typename T >
using wptr = std::weak_ptr< T >;

namespace common {

namespace random {

using RNG = std::mt19937_64;
/**
 * @brief Creates and returns a new random number generator from a potential seed.
 * @param seed the seed for the Mersenne Twister algorithm.
 * @return The Mersenne Twister RNG object
 */
inline auto create_rng()
{
   return RNG{std::random_device{}()};
}
inline auto create_rng(size_t seed)
{
   return RNG{seed};
}
inline auto create_rng(RNG rng)
{
   return rng;
}

template < typename Container >
auto choose(const Container& cont, RNG& rng)
{
   return cont[std::uniform_int_distribution(0ul, cont.size())(rng)];
}
template < typename Container >
auto choose(const Container& cont)
{
   auto dev = std::random_device{};
   return cont[std::uniform_int_distribution(0ul, cont.size())(dev)];
}

}  // namespace random

template < typename T >
inline std::map< T, unsigned int > counter(const std::vector< T >& vals)
{
   std::map< T, unsigned int > rv;

   for(auto val = vals.begin(); val != vals.end(); ++val) {
      rv[*val]++;
   }

   return rv;
}

template < typename T, size_t... Is >
inline constexpr auto make_enum_vec(std::index_sequence< Is... >)
{
   return std::vector{T(Is)...};
}

template < typename T, typename Allocator, typename Accessor >
inline auto counter(
   const std::vector< T, Allocator >& vals,
   Accessor acc = [](const auto& iter) { return *iter; })
{
   std::map< T, unsigned int > rv;

   for(auto val_iter = vals.begin(); val_iter != vals.end(); ++val_iter) {
      rv[acc(val_iter)]++;
   }

   return rv;
}

template < typename Container, typename Accessor >
inline auto counter(
   const Container& vals,
   Accessor acc = [](const auto& iter) { return *iter; })
{
   std::map< typename Container::mapped_type, unsigned int > rv;

   for(auto val_iter = vals.begin(); val_iter != vals.end(); ++val_iter) {
      rv[acc(val_iter)]++;
   }

   return rv;
}

template < class Lambda, int = (Lambda{}(), 0) >
constexpr bool is_constexpr(Lambda)
{
   return true;
}
constexpr bool is_constexpr(...)
{
   return false;
}

template < class first, class second, class... types >
auto min(first f, second s, types... t)
{
   return std::min(f, s) ? min(f, t...) : min(s, t...);
}

template < class first, class second >
auto min(first f, second s)
{
   return std::min(f, s);
}

};  // namespace common
