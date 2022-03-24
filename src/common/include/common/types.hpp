
#ifndef NOR_COMMON_TYPES_HPP
#define NOR_COMMON_TYPES_HPP

#include <iostream>
#include <vector>
#include <range/v3/all.hpp>

namespace common {

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
   ranges::span< T > value;
   std::string_view delimiter;

   SpanPrinter(ranges::span< T > vec, const std::string& delim = std::string(", "))
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
   requires is_any_v< T, Key, Value >
   [[nodiscard]] constexpr auto at(const T& elem) const
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

}  // namespace common

#endif  // NOR_COMMON_TYPES_HPP
