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
   auto chooser = [&](const auto& actual_ra_container) -> auto&
   {
      //      auto choice = std::uniform_int_distribution(
      //         0ul, actual_ra_container.size())(rng);
      //      LOGD2("Choice", choice);
      //      return cont[choice];
      return actual_ra_container[std::uniform_int_distribution(
         0ul, actual_ra_container.size() - 1)(rng)];
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
      std::vector< double > weights;
      weights.reserve(cont.size());
      for(const auto& elem : cont) {
         weights.emplace_back(policy(elem));
      }
      // the ranges::to_vector method here fails with a segmentation fault for no apparent reason
      //      auto weights = ranges::to_vector(cont | ranges::views::transform(policy));
      //      auto choice = std::discrete_distribution< size_t >(weights.begin(),
      //      weights.end())(rng); LOGD2("Choice", choice); return cont[choice];
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

template < ranges::range Container, typename Accessor >
inline auto counter(
   const Container& vals,
   Accessor acc = [](const auto& iter) { return *iter; })
{
   using mapped_type = std::remove_cvref_t< decltype(*(vals.begin())) >;
   constexpr bool hashable_mapped_type =
      requires(Container t)
   {
      std::hash< mapped_type >{}(t);
      std::equality_comparable< mapped_type >;
   };
   std::conditional_t<
      hashable_mapped_type,
      std::unordered_map< mapped_type, unsigned int >,
      std::map< mapped_type, unsigned int > >
      rv;

   for(auto val_iter = vals.begin(); val_iter != vals.end(); ++val_iter) {
      rv[acc(val_iter)]++;
   }

   return rv;
}

template < typename T >
struct Printer;

template < typename T >
struct Printer< std::span< T > > {
   std::span< T > value;
   std::string_view delimiter;

   Printer(std::span< T > span, const std::string& delim = std::string(", "))
       : value(span), delimiter(delim)
   {
   }

   friend auto& operator<<(std::ostream& os, const Printer& printer)
   {
      os << "[";
      for(unsigned int i = 0; i < printer.value.size() - 1; ++i) {
         os << printer.value[i] << printer.delimiter;
      }
      os << printer.value.back() << "]";
      return os;
   }
};

// additional deduction guide needed to disambiguate type T
template < class T >
Printer(std::span< T >) -> Printer< std::span< T > >;

template < class Lambda, int = (Lambda{}(), 0) >
constexpr bool is_constexpr(Lambda)
{
   return true;
}
constexpr bool is_constexpr(...)
{
   return false;
}

template < typename Container, typename T >
constexpr bool contains(Container cont, T value)
{
   return std::find(cont.begin(), cont.end(), value) != cont.end();
}

template < class first, class second, class... types >
constexpr auto min(first f, second s, types... t)
{
   return std::min(f, s) ? min(f, t...) : min(s, t...);
}

template < class first, class second >
constexpr auto min(first f, second s)
{
   return std::min(f, s);
}

};  // namespace common
