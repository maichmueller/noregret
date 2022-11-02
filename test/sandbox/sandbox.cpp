

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
   M() { std::cout << "CTOR\n"; }
   ~M() { std::cout << "DTOR\n"; }
   M(const M&) { std::cout << "COPY\n"; }
   M(M&&) { std::cout << "MOVE\n"; }
   M& operator=(const M&) { std::cout << "COPY assignment\n"; }
   M& operator=(M&&) { std::cout << "MOVE assignment\n"; }

   auto begin() const { return vec.begin(); }
   auto end() const { return vec.end(); }
   std::vector< int > vec{10, 20, 30, 40};
};

int main()
{
   //   using list_type = std::initializer_list< std::pair< int, double > >;
   //   auto list = std::initializer_list< std::pair< int, double > >{
   //      std::pair{10, 1}, std::pair{11, 4}, std::pair{12, 7}};
   //   std::cout << nor::concepts::maps< list_type, int > << "\n";
   //   std::cout << nor::concepts::mapping< list_type > << "\n";
   //
   //   std::cout << std::convertible_to< int, decltype(*(std::ranges::views::keys(list).begin())) >
   //             << "\n";
   //   std::cout << nor::concepts::mapping< std::unordered_map< int, double > > << "\n";
   //   std::cout << nor::concepts::mapping< list_type > << "\n";
   //
   //   for(auto v : list | ranges::views::keys) {
   //      std::cout << v << "\n";
   //   }
   //
   //   nor::HashmapActionPolicy< int > policy{std::pair{0, 5.}, std::pair{1, 2.},
   //   std::pair{2, 3.}}; nor::HashmapActionPolicy< int > policy2(M{std::unordered_map< int, double
   //   >{
   //      std::pair{0, 5.}, std::pair{1, 2.}, std::pair{2, 3.}}});
   //   nor::HashmapActionPolicy< int > policy3(std::unordered_map< int, double >{
   //      std::pair{0, 5.}, std::pair{1, 2.}, std::pair{2, 3.}});
   struct A {
      //      auto begin() const { return ranges::any_view< const int& >{vec}; }
      //      auto end() const { return ranges::any_view< const int& >{vec}; }
      auto begin() const { return vec.begin(); }
      auto end() const { return vec.end(); }
      std::vector< int > vec{10, 20, 30, 40};
   };
   //   A a;
   std::vector< M > vec{};
   vec.emplace_back();
   ranges::any_view< const M > v{vec};

   for(auto& d : v) {
      std::cout << "hello"
                << "\n";
   }

   ranges::any_view< const M& > v1{vec};

   for(auto& d : v1) {
      std::cout << "hello"
                << "\n";
   }
}