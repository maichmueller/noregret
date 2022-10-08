

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

#include "common/common.hpp"
// #include "nor/nor.hpp"

#include <variant>

using var = std::variant< int, double >;

struct Visitor {
   auto operator()(int i)
   {
      throw std::logic_error("sd");
      return i;
   }
   auto operator()(double k) { return int(k); }
};

template < class T, template < class... > class Template >
struct is_specialization: std::false_type {};

template < template < class... > class Template, class... Args >
struct is_specialization< Template< Args... >, Template >: std::true_type {};

template < class T, template < class... > class Template >
constexpr bool is_specialization_v = is_specialization< T, Template>::value;

int main()
{
   var v = 3;
   //   std::visit(Visitor{}, v);
   std::cout << "Ok";
   int k = 5;
   std::reference_wrapper< int > r = k;
   std::cout << is_specialization_v< std::reference_wrapper< inis_specializationt >, std::reference_wrapper >;
}