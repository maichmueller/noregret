#ifndef NOR_STRING_UTILS_HPP
#define NOR_STRING_UTILS_HPP

#include <string>
#include <string_view>
#include <vector>

namespace nor::utils {

std::vector<std::string_view> split(std::string_view str, std::string_view delim) {
   std::vector<std::string_view> splits;
   std::string_view curr_substr = str;
   size_t pos = 0;
   std::string segment;
   while ((pos = curr_substr.find(delim)) != std::string_view::npos) {
      // extract the left half of the delimited segment
      splits.emplace_back(curr_substr.substr(0, pos));
      // reassign the right half as the curr substr
      curr_substr = curr_substr.substr(pos + delim.size(), std::string_view::npos);
   }
   // the remaining string is the last cutoff segment and needs to be added manually
   splits.emplace_back(curr_substr);
   return splits;
}

}

#endif  // NOR_STRING_UTILS_HPP
