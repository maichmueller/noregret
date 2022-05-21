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


template < typename... Args >
struct debug;

using noop = decltype([](auto&&...) { return; });

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

template < typename RAContainer >
   requires ranges::range< RAContainer >
inline auto& choose(const RAContainer& cont, RNG& rng)
{
   auto chooser = [&](const auto& actual_ra_container) -> auto& {
      return actual_ra_container[std::uniform_int_distribution(
         0ul, actual_ra_container.size())(rng)];
   };
   if constexpr(
      std::random_access_iterator<
         decltype(std::declval< RAContainer >().begin()) > and requires { cont.size(); }) {
      return chooser(cont);
   } else {
      auto cont_as_vec = ranges::to_vector(
         cont | ranges::views::transform([](const auto& elem) { return std::ref(elem); }));
      return chooser(cont_as_vec).get();
   }
}

template < typename RAContainer, typename Policy >
   requires ranges::range< RAContainer > and requires(Policy p) {
                                                {
                                                   // policy has to be a callable returning the
                                                   // probability of the input matching the
                                                   // container's contained type
                                                   p(std::declval< decltype(*(
                                                        std::declval< RAContainer >().begin())) >())
                                                   } -> std::convertible_to< double >;
                                             }
inline auto& choose(const RAContainer& cont, const Policy& policy, RNG& rng)
{
   if constexpr(
      std::random_access_iterator<
         decltype(std::declval< RAContainer >().begin()) > and requires { cont.size(); }) {
      auto weights = ranges::to_vector(
         cont | ranges::views::transform([&](const auto& elem) { return policy(elem); }));
      return cont[std::discrete_distribution< size_t >(weights.begin(), weights.end())(rng)];
   } else {
      std::vector< double > weights;
      if constexpr(requires { cont.size(); }) {
         // if we can know how many elements are in the container, then reserve that amount.
         weights.reserve(cont.size());
      }
      auto cont_as_vec = ranges::to_vector(cont | ranges::views::transform([&](const auto& elem) {
                                              weights.emplace_back(policy(elem));
                                              return std::ref(elem);
                                           }));
      return cont_as_vec[std::discrete_distribution< size_t >(weights.begin(), weights.end())(rng)]
         .get();
   }
}
template < typename RAContainer >
inline auto choose(const RAContainer& cont)
{
   auto rng = create_rng(std::random_device{}());
   return choose(cont, rng);
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
