
#ifndef NOR_COMMON_STRING_UTILS_HPP
#define NOR_COMMON_STRING_UTILS_HPP

#include <array>
#include <string>
#include <vector>

namespace common {

template < typename T >
std::string to_string(const T& value);

template < typename To >
To from_string(std::string_view str);

template < typename T >
struct printable: std::false_type {};

template < typename T >
constexpr bool printable_v = printable< T >::value;

}  // namespace common

template < typename T >
   requires(::common::printable_v< T >)
inline auto& operator<<(std::ostream& os, const T& value)
{
   return os << common::to_string(value);
}

template < typename T >
   requires(::common::printable_v< T >)
inline auto& operator<<(std::stringstream& os, const T& value)
{
   return os << common::to_string(value);
}

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

}  // namespace common

#endif  // NOR_COMMON_STRING_UTILS_HPP
