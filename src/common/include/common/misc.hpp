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

template < typename Return = void>
struct noop {
   template < typename... Args >
      requires std::same_as< Return, void >
   void operator()(Args&&...) const noexcept
   {
   }
   template < typename... Args >
      requires (not std::same_as< Return, void >)
   auto operator()(Args&&...) const noexcept
   {
      return Return{};
   }
};

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
      return actual_ra_container[std::uniform_int_distribution(0ul, actual_ra_container.size() - 1)(
         rng
      )];
   };
   if constexpr(ranges::random_access_range< RAContainer > and ranges::sized_range< RAContainer >) {
      return chooser(cont);
   } else {
      auto cont_as_vec = ranges::to_vector(cont | ranges::views::transform([](const auto& elem) {
                                              return std::ref(elem);
                                           }));
      return chooser(cont_as_vec).get();
   }
}

template < typename RAContainer, typename Policy >
   requires ranges::range< RAContainer > and requires(Policy p) {
      {
         // policy has to be a callable returning the
         // probability of the input matching the
         // container's contained type
         p(std::declval< decltype(*(std::declval< RAContainer >().begin())) >())
      } -> std::convertible_to< double >;
   }
inline auto& choose(const RAContainer& cont, const Policy& policy, RNG& rng)
{
   if constexpr(ranges::random_access_range< RAContainer > and ranges::sized_range< RAContainer >) {
      std::vector< double > weights;
      weights.reserve(cont.size());
      for(const auto& elem : cont) {
         weights.emplace_back(policy(elem));
      }
      return cont[std::discrete_distribution< size_t >(weights.begin(), weights.end())(rng)];
   } else {
      std::vector< double > weights;
      if constexpr(ranges::sized_range< RAContainer >) {
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
   Accessor acc = [](const auto& iter) { return *iter; }
)
{
   std::map< T, unsigned int > rv;

   for(auto val_iter = vals.begin(); val_iter != vals.end(); ++val_iter) {
      rv[acc(val_iter)]++;
   }

   return rv;
}

template < ranges::range Container, typename Accessor >
inline auto counter(const Container& vals, Accessor acc = [](const auto& iter) { return *iter; })
{
   using mapped_type = std::remove_cvref_t< decltype(*(vals.begin())) >;
   constexpr bool hashable_mapped_type = requires(Container t) {
      requires std::hash< mapped_type > {}
      (t);
      requires std::equality_comparable< mapped_type >;
   };
   std::conditional_t<
      hashable_mapped_type,
      std::unordered_map< mapped_type, unsigned int >,
      std::map< mapped_type, unsigned int > >
      rv;

   for(auto val_iter = vals.begin(); val_iter != vals.end(); ++val_iter) {
      ++rv[acc(val_iter)];
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
      if(printer.value.empty()) {
         return os << "[]";
      }
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

template < ranges::range Rng >
   requires requires(Rng t) { std::cout << *(t.begin()); }
struct RangePrinter {
   ranges::ref_view< Rng > value;
   std::string delimiter;

   RangePrinter(Rng& range, std::string delim = ", ") : value(range), delimiter(std::move(delim)) {}

   friend auto& operator<<(std::ostream& os, RangePrinter printer)
   {
      auto iter = printer.value.begin();
      auto end = printer.value.end();
      if(iter == end) {
         // empty range
         return os << "[]";
      }
      os << "[";
      // we need a lookahead because not all iterators admissible by this printer type are
      // bidirectional. So we cannot simply use end() - 1 to print the last element.
      auto lookahead = std::next(iter);
      do {
         os << *iter << printer.delimiter << std::flush;
         std::advance(iter, 1);
         std::advance(lookahead, 1);
      } while(lookahead != end);
      os << *iter << "]";
      return os;
   }
};

template < ranges::range Rng >
   requires requires(Rng t) {
      // elements must be decomposable into key and value
      ranges::views::keys(t);
      ranges::views::values(t);
      // elements must be printable
      std::cout << *(ranges::views::keys(t).begin());
      std::cout << *(ranges::views::values(t).begin());
   }
struct KeyValueRangePrinter {
   ranges::ref_view< Rng > view;
   std::string delimiter;

   KeyValueRangePrinter(Rng& range, std::string delim = ", ")
       : view(range), delimiter(std::move(delim))
   {
   }

   friend auto& operator<<(std::ostream& os, KeyValueRangePrinter printer)
   {
      auto iter = printer.view.begin();
      auto end = printer.view.end();
      if(iter == end) {
         // empty range
         return os << "[]";
      }
      os << "[";
      // we need a lookahead because not all iterators admissible by this printer type are
      // bidirectional. So we cannot simply use end() - 1 to print the last element.
      auto lookahead = std::next(printer.view.begin());
      do {
         const auto& [key, value] = *iter;
         os << key << ": " << value << printer.delimiter;
         std::advance(iter, 1);
         std::advance(lookahead, 1);
      } while(lookahead != end);
      const auto& [key, value] = *iter;
      os << key << ": " << value << "]";
      return os;
   }
};

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
constexpr bool contains(Container&& cont, T&& value)
{
   if constexpr(requires { cont.contains(value); }) {
      if(std::is_constant_evaluated()) {
         if constexpr(is_constexpr([] { Container{}.contains(T{}); })) {
            return cont.contains(value);
         } else {
            return std::find(cont.begin(), cont.end(), value) != cont.end();
         }
      } else {
         return cont.contains(value);
      }
   } else if constexpr(requires { cont.find(value); }) {
      if(std::is_constant_evaluated()) {
         if constexpr(is_constexpr([] { Container{}.find(T{}); })) {
            return cont.find(value) != cont.end();
         } else {
            return std::find(cont.begin(), cont.end(), value) != cont.end();
         }
      } else {
         return cont.find(value) != cont.end();
      }
   } else {
      return std::find(cont.begin(), cont.end(), value) != cont.end();
   }
}

template < typename T, typename Container >
constexpr bool isin(T&& value, Container&& cont)
{
   return contains(std::forward< Container >(cont), std::forward< T >(value));
}
template < typename T >
constexpr bool isin(T&& value, const std::initializer_list< std::remove_cvref_t< T > >& cont)
{
   return contains(cont, std::forward< T >(value));
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

/// prints a type's name
/// taken from:
/// https://stackoverflow.com/questions/81870/is-it-possible-to-print-a-variables-type-in-standard-c/56766138#56766138
template < typename T >
constexpr auto type_name()
{
   std::string_view name, prefix, suffix;
#ifdef __clang__
   name = __PRETTY_FUNCTION__;
   prefix = "auto type_name() [T = ";
   suffix = "]";
#elif defined(__GNUC__)
   name = __PRETTY_FUNCTION__;
   prefix = "constexpr auto type_name() [with T = ";
   suffix = "]";
#elif defined(_MSC_VER)
   name = __FUNCSIG__;
   prefix = "auto __cdecl type_name<";
   suffix = ">(void)";
#endif
   name.remove_prefix(prefix.size());
   name.remove_suffix(suffix.size());
   return name;
}

};  // namespace common
