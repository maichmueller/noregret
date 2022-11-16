

#include "sandbox.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <named_type.hpp>
#include <range/v3/all.hpp>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "common/common.hpp"
// #include "nor/nor.hpp"

struct M {
   M() { std::cout << "CTOR\n"; }
   ~M() { std::cout << "DTOR\n"; }
   M(const M&) { std::cout << "COPY\n"; }
   M(M&&) { std::cout << "MOVE\n"; }
   M& operator=(const M&)
   {
      std::cout << "COPY assignment\n";
      return *this;
   }
   M& operator=(M&&)
   {
      std::cout << "MOVE assignment\n";
      return *this;
   }

   auto begin() const { return vec.begin(); }
   auto end() const { return vec.end(); }
   std::vector< int > vec{10, 20, 30, 40};
};

template < typename... Ts >
struct Overload: Ts... {
   using Ts::operator()...;
};
template < typename... Ts >
Overload(Ts...) -> Overload< Ts... >;

namespace detail {

template < typename ReturnType >
struct monostate_error_visitor {
   ReturnType operator()(std::monostate) const
   {
      // this should never be visited, but if so --> error
      throw std::logic_error("We entered a std::monostate visit branch.");
      if constexpr(not std::same_as< ReturnType, void >) {
         return {};
      } else {
         return;
      }
   }
};

template < typename... Ts >
struct monostate_filtered_variant {
   static_assert(sizeof...(Ts), "Passed type pack consists of std::monostates types only.");
};

template < typename... Ts >
   requires common::is_none_v< std::monostate, Ts... >
struct monostate_filtered_variant< std::variant< Ts... > > {
   using type = std::variant< Ts... >;
};

template < typename T, typename... Tail >
   requires(std::same_as< std::monostate, T > or common::is_any_v< std::monostate, Tail... >)
struct monostate_filtered_variant< std::variant< T, Tail... > >:
    public std::conditional_t<
       std::same_as< std::monostate, T >,
       std::conditional_t<
          (sizeof...(Tail) > 1),
          monostate_filtered_variant< std::variant< Tail... > >,
          monostate_filtered_variant< Tail... > >,
       monostate_filtered_variant< std::variant< Tail..., T > > > {};

template < typename T >
   requires(not std::same_as< T, std::monostate >)
struct monostate_filtered_variant< T > {
   using type = std::variant< T, T >;
};

template < typename... Ts >
using monostate_filtered_variant_t = typename monostate_filtered_variant< Ts... >::type;

}  // namespace detail

template < typename TargetVariant, typename... Ts >
auto make_overload_with_monostate(Ts...)
   -> Overload< detail::monostate_error_visitor< std::invoke_result_t<
      decltype(&std::visit<
               Overload< Ts... >,
               detail::monostate_filtered_variant_t< TargetVariant > >),
      Overload< Ts... >,
      detail::monostate_filtered_variant_t< TargetVariant > > > >
{
   return {};
}

struct visitor {
   void operator()(int val) { std::cout << val; }
};

int main()
{
   using var_type = std::variant< int, std::monostate >;
   //   common::debug< detail::monostate_filtered_variant_t< std::variant< int, std::monostate > > >
   //   d{};
   std::cout << std::boolalpha
             << common::is_none_v< std::monostate, int, double, std::string, unsigned long >
             << "\n";

   std::cout << std::boolalpha
             << std::is_same_v<
                   detail::monostate_filtered_variant_t< std::variant< int, std::monostate > >,
                   std::variant< int, int > >
             << "\n";
   std::cout
      << std::boolalpha
      << std::is_same_v<
            detail::monostate_filtered_variant_t< std::variant< int, double, std::monostate > >,
            std::variant< int, double > >
      << "\n";
   //      detail::monostate_error_visitor< void > vis;
   var_type v;
   common::debug< std::invoke_result_t< decltype(&std::visit<visitor, var_type>), visitor, var_type> {};
//   make_overload_with_monostate< var_type >(visitor{});
   //   std::visit(common::Overload{[](auto value) { std::cout << value; }, vis}, v);
}