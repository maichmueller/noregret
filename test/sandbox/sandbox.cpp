

#include "sandbox.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <range/v3/all.hpp>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "common/common.hpp"
#include "nor/nor.hpp"


struct M {


   inline auto begin() { return m_map.begin(); }
   [[nodiscard]] inline auto begin() const { return m_map.begin(); }
   inline auto end() { return m_map.end(); }
   [[nodiscard]] inline auto end() const { return m_map.end(); }
   std::unordered_map< int, double> m_map;
};

int main()
{
   using list_type = std::initializer_list< std::pair< int, double > >;
   auto list = std::initializer_list< std::pair< int, double > >{
      std::pair{10, 1}, std::pair{11, 4}, std::pair{12, 7}};
   std::cout << nor::concepts::maps< list_type, int > << "\n";
   std::cout << nor::concepts::mapping< list_type > << "\n";

   std::cout << std::convertible_to< int, decltype(*(std::ranges::views::keys(list).begin())) >
             << "\n";
   std::cout << nor::concepts::mapping< std::unordered_map< int, double > > << "\n";
   std::cout << nor::concepts::mapping< list_type > << "\n";

   for(auto v : list | ranges::views::keys) {
      std::cout << v << "\n";
   }

   nor::HashmapActionPolicy< int > policy{std::pair{0, 5.}, std::pair{1, 2.}, std::pair{2, 3.}};
   nor::HashmapActionPolicy< int > policy2(M{std::unordered_map< int, double >{
      std::pair{0, 5.}, std::pair{1, 2.}, std::pair{2, 3.}}});
   nor::HashmapActionPolicy< int > policy3(std::unordered_map< int, double >{
      std::pair{0, 5.}, std::pair{1, 2.}, std::pair{2, 3.}});
}