
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

#include "common/common.hpp"

template < typename T >
using uptr = std::unique_ptr< T >;
template < typename T >
using sptr = std::shared_ptr< T >;
template < typename T >
using wptr = std::weak_ptr< T >;

namespace aze::utils {

template < std::string_view const&... Strs >
struct join {
   // Join all strings into a single std::array of chars
   static constexpr auto impl() noexcept
   {
      constexpr std::size_t len = (Strs.size() + ... + 0);
      std::array< char, len + 1 > arr{};
      auto append = [i = 0, &arr](auto const& s) mutable {
         for(auto c : s)
            arr[i++] = c;
      };
      (append(Strs), ...);
      arr[len] = 0;
      return arr;
   }
   // Give the joined string static storage
   static constexpr auto arr = impl();
   // View as a std::string_view
   static constexpr std::string_view value{arr.data(), arr.size() - 1};
};
// Helper to get the value out
template < std::string_view const&... Strs >
static constexpr auto join_v = join< Strs... >::value;

/**
 * Literal class type that wraps a constant expression string.
 * Can be used as template parameter to differentiate via 'strings'
 */
template < size_t N >
struct StringLiteral {
   constexpr StringLiteral(const char (&str)[N]) { std::copy_n(str, N, value); }

   char value[N];
};

template < typename T >
requires requires(T t)
{
   std::cout << t;
}
struct VectorPrinter {
   const std::vector< T >& value;
   std::string_view delimiter;

   VectorPrinter(const std::vector< T >& vec, const std::string& delim = std::string(", "))
       : value(vec), delimiter(delim)
   {
   }

   friend auto& operator<<(std::ostream& os, const VectorPrinter& printer)
   {
      os << "[";
      for(unsigned int i = 0; i < printer.value.size() - 1; ++i) {
         os << printer.value[i] << printer.delimiter;
      }
      os << printer.value.back() << "]";
      return os;
   }
};

template < typename T >
requires requires(T t)
{
   std::cout << t;
}
struct SpanPrinter {
   std::span< T > value;
   std::string_view delimiter;

   SpanPrinter(std::span< T > vec, const std::string& delim = std::string(", "))
       : value(vec), delimiter(delim)
   {
   }

   friend auto& operator<<(std::ostream& os, const SpanPrinter& printer)
   {
      os << "[";
      for(unsigned int i = 0; i < printer.value.size() - 1; ++i) {
         os << printer.value[i] << printer.delimiter;
      }
      os << printer.value.back() << "]";
      return os;
   }
};

template < typename... Ts >
struct Overload: Ts... {
   using Ts::operator()...;
};
template < typename... Ts >
Overload(Ts...) -> Overload< Ts... >;

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

inline std::string repeat(std::string str, const std::size_t n)
{
   if(n == 0) {
      str.clear();
      str.shrink_to_fit();
      return str;
   } else if(n == 1 || str.empty()) {
      return str;
   }
   const auto period = str.size();
   if(period == 1) {
      str.append(n - 1, str.front());
      return str;
   }
   str.reserve(period * n);
   std::size_t m{2};
   for(; m < n; m *= 2)
      str += str;
   str.append(str.c_str(), (n - (m / 2)) * period);
   return str;
}

inline std::string center(const std::string& str, int width, const char* fillchar)
{
   size_t len = str.length();
   if(std::cmp_less(width, len)) {
      return str;
   }

   int diff = static_cast< int >(static_cast< unsigned long >(width) - len);
   int pad1 = diff / 2;
   int pad2 = diff - pad1;
   return std::string(static_cast< unsigned long >(pad2), *fillchar) + str
          + std::string(static_cast< unsigned long >(pad1), *fillchar);
}

inline std::string operator*(std::string str, std::size_t n)
{
   return repeat(std::move(str), n);
}

template < typename BoardType, typename PieceType >
inline void print_board(const BoardType& board, bool flip_board = false, bool hide_unknowns = false)
{
   std::string output = board_str_rep< BoardType, PieceType >(board, flip_board, hide_unknowns);
   std::cout << output << std::endl;
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

template < int N >
struct faculty {
   static constexpr int n_faculty() { return N * faculty< N - 1 >::n_faculty(); }
};

template <>
struct faculty< 0 > {
   static constexpr int n_faculty() { return 1; }
};

template < class first, class second, class... types >
auto min(first f, second s, types... t)
{
   return f < s ? min(f, t...) : min(s, t...);
}

template < class first, class second >
auto min(first f, second s)
{
   return f < s ? f : s;
}

template < typename T >
concept is_enum = std::is_enum_v< T >;

template < class T, class... Ts >
struct is_any: ::std::disjunction< ::std::is_same< T, Ts >... > {
};
template < class T, class... Ts >
inline constexpr bool is_any_v = is_any< T, Ts... >::value;

template < class T, class... Ts >
struct is_same: ::std::conjunction< ::std::is_same< T, Ts >... > {
};
template < class T, class... Ts >
inline constexpr bool is_same_v = is_same< T, Ts... >::value;

template < typename Key, typename Value, std::size_t Size >
struct CEMap {
   std::array< std::pair< Key, Value >, Size > data;

   [[nodiscard]] constexpr Value at(const Key& key) const
   {
      const auto itr = std::find_if(
         begin(data), end(data), [&key](const auto& v) { return v.first == key; });
      if(itr != end(data)) {
         return itr->second;
      } else {
         throw std::range_error("Not Found");
      }
   }
};

template < typename Key, typename Value, std::size_t Size >
struct CEBijection {
   std::array< std::pair< Key, Value >, Size > data;

   template < typename T >
   requires is_any_v< T, Key, Value >[[nodiscard]] constexpr auto at(const T& elem) const
   {
      const auto itr = std::find_if(begin(data), end(data), [&elem](const auto& v) {
         if constexpr(std::is_same_v< T, Key >) {
            return v.first == elem;
         } else {
            return v.second == elem;
         }
      });
      if(itr != end(data)) {
         if constexpr(std::is_same_v< T, Key >) {
            return itr->second;
         } else {
            return itr->first;
         }
      } else {
         throw std::range_error("Not Found");
      }
   }
};

};  // namespace aze::utils

#include <tuple>

namespace tuple {

template < typename TT >
struct hash {
   size_t operator()(TT const& tt) const { return std::hash< TT >()(tt); }
};

namespace {
template < class T >
inline void hash_combine(std::size_t& seed, T const& v)
{
   seed ^= tuple::hash< T >()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
}  // namespace

namespace {
// Recursive template code derived from Matthieu M.
template < class Tuple, size_t Index = std::tuple_size< Tuple >::value - 1 >
struct HashValueImpl {
   static void apply(size_t& seed, Tuple const& tuple)
   {
      HashValueImpl< Tuple, Index - 1 >::apply(seed, tuple);
      hash_combine(seed, std::get< Index >(tuple));
   }
};

template < class Tuple >
struct HashValueImpl< Tuple, 0 > {
   static void apply(size_t& seed, Tuple const& tuple) { hash_combine(seed, std::get< 0 >(tuple)); }
};
}  // namespace

template < typename... TT >
struct hash< std::tuple< TT... > > {
   size_t operator()(std::tuple< TT... > const& tt) const
   {
      size_t seed = 0;
      HashValueImpl< std::tuple< TT... > >::apply(seed, tt);
      return seed;
   }
};
}  // namespace tuple

namespace {
template < typename T, typename... Ts >
auto head(std::tuple< T, Ts... > const& t)
{
   return std::get< 0 >(t);
}

template < std::size_t... Ns, typename... Ts >
auto tail_impl(std::index_sequence< Ns... > const&, std::tuple< Ts... > const& t)
{
   return std::make_tuple(std::get< Ns + 1u >(t)...);
}

template < typename... Ts >
auto tail(std::tuple< Ts... > const& t)
{
   return tail_impl(std::make_index_sequence< sizeof...(Ts) - 1u >(), t);
}

template < typename TT >
struct eqcomp {
   bool operator()(TT const& tt1, TT const& tt2) const { return ! ((tt1 < tt2) || (tt2 < tt1)); }
};
}  // namespace

namespace std {
template < typename T1, typename... TT >
struct equal_to< std::tuple< T1, TT... > > {
   bool operator()(std::tuple< T1, TT... > const& tuple1, std::tuple< T1, TT... > const& tuple2)
      const
   {
      return eqcomp< T1 >()(std::get< 0 >(tuple1), std::get< 0 >(tuple2))
             && eqcomp< std::tuple< TT... > >()(tail(tuple1), tail(tuple2));
   }
};

template <>
struct hash< tuple< string, int > > {
   size_t operator()(const std::tuple< std::string, int >& s) const
   {
      return std::hash< std::string >()(
         std::get< 0 >(s) + "!@#$%^&*()_" + std::to_string(std::get< 1 >(s)));
   }
};
}  // namespace std
