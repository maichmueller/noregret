
#ifndef NOR_COMMON_STRING_UTILS_HPP
#define NOR_COMMON_STRING_UTILS_HPP

#include <fmt/core.h>
#include <fmt/ostream.h>

#include <array>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace common {

template < typename T >
std::string to_string(const T& value);

template <>
inline std::string to_string(const ::std::nullopt_t&)
{
   return "std::nullopt";
}

template < typename To >
To from_string(std::string_view str);

template < typename T >
struct printable: std::false_type {};

template < typename T >
constexpr bool printable_v = printable< T >::value;

}  // namespace common

template <>
struct common::printable< std::nullopt_t >: std::true_type {};

// template < typename T >
//    requires(common::printable_v< T >)
// auto& operator<<(std::ostream& os, const T& value)
// {
//    return os << common::to_string(value);
// }
//
// template < typename T >
//    requires(common::printable_v< T >)
// auto& operator<<(std::stringstream& os, const T& value)
// {
//    return os << common::to_string(value);
// }

template < class T >
   requires(common::printable_v< T >)
struct fmt::formatter< T >: public fmt::ostream_formatter {};

#ifndef COMMON_ENABLE_PRINT
   #define COMMON_ENABLE_PRINT(nmspace, type)                        \
      namespace nmspace {                                            \
      inline auto& operator<<(std::ostream& ostr, const type& value) \
      {                                                              \
         return ostr << ::common::to_string(value);                  \
      }                                                              \
      }                                                              \
      template <>                                                    \
      struct ::common::printable< nmspace::type >: std::true_type {}
#endif

/// note that if you get an error here, you might need to include <fmt/ranges.h>
#ifndef COMMON_NO_RANGE_FORMATTER
   #define COMMON_NO_RANGE_FORMATTER(type)            \
      template <>                                     \
      struct fmt::detail::range_format_kind_< type >: \
          std::integral_constant< fmt::range_format, fmt::range_format::disabled > {};
#endif

namespace common {

inline std::vector< std::string_view > split(std::string_view str, std::string_view delim)
{
   std::vector< std::string_view > splits;
   std::string_view curr_substr = str;
   size_t pos = 0;
   std::string segment;
   while((pos = curr_substr.find(delim)) != std::string_view::npos) {
      // extract the left half of the delimited segment
      splits.emplace_back(curr_substr.substr(0, pos));
      // reassign the right half as the curr substr
      curr_substr = curr_substr.substr(pos + delim.size(), std::string_view::npos);
   }
   // the remaining string is the last cutoff segment and needs to be added manually
   splits.emplace_back(curr_substr);
   return splits;
}

template < std::string_view const&... Strs >
struct join {
   // Join all strings into a single std::array of chars
   static constexpr auto impl() noexcept
   {
      constexpr std::size_t len = (Strs.size() + ... + 0);
      std::array< char, len + 1 > arr_{};
      auto append = [i = 0, &arr_](auto const& s) mutable {
         for(auto c : s)
            arr_[i++] = c;
      };
      (append(Strs), ...);
      arr_[len] = 0;
      return arr_;
   }
   // Give the joined string static storage
   static constexpr auto arr = impl();
   // View as a std::string_view
   static constexpr std::string_view value{arr.data(), arr.size() - 1};
};
// Helper to get the value out
template < std::string_view const&... Strs >
static constexpr auto join_v = join< Strs... >::value;

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

inline std::string left(const std::string& str, int width, const char* fillchar)
{
   size_t len = str.length();
   if(std::cmp_less(width, len)) {
      return str;
   }

   int diff = static_cast< int >(static_cast< unsigned long >(width) - len);
   int pad1 = diff;
   return str + std::string(static_cast< unsigned long >(pad1), *fillchar);
}

inline std::string operator*(std::string str, std::size_t n)
{
   return repeat(std::move(str), n);
}

inline std::string
replace(std::string str, std::string_view str_to_replace, std::string_view replacement)
{
   return str.replace(str.find(str_to_replace), sizeof(str_to_replace) - 1, replacement);
}

inline std::string
replace_all(std::string str, std::string_view str_to_replace, std::string_view replacement)
{
   size_t curr_pos = str.find(str_to_replace);
   while(curr_pos != std::string::npos) {
      str = str.replace(curr_pos, str_to_replace.size(), replacement);
      curr_pos = str.find(str_to_replace);
   }
   return str;
}

}  // namespace common

#endif  // NOR_COMMON_STRING_UTILS_HPP
