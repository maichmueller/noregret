

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

enum class A { one = 4, two = 4534 };
enum class B { one = 11111111, two = 33 };

using var = std::variant< A, std::monostate >;

struct Visitor {
   auto operator()(A a) { return std::hash< A >{}(a); }
   auto operator()(B b) { return std::hash< B >{}(b); }
};

using av_vector_sptr_hasher = decltype([](const sptr< std::vector< var > >& av_vec_ptr) {
   size_t start = 0;
   ranges::for_each(*av_vec_ptr, [&](const var& av) {
      common::hash_combine(
         start,
         std::visit(
            [&]< typename T >(const T& action_or_outcome) {
               return std::hash< T >{}(action_or_outcome);
            },
            av
         )
      );
   });
   return start;
});

int main()
{
   sptr< std::vector< var > > ptr = std::make_shared< std::vector< var > >(std::vector< var >{
      A::one, A::two});
   using av_sequence_infostate_map = std::unordered_map<
      sptr< std::vector< var > >,
      std::string,
      av_vector_sptr_hasher,
      common::sptr_value_comparator< std::vector<  var > > > ;

   av_sequence_infostate_map map;
   map.emplace(ptr, std::string("Mydear"));
   std::cout << map[ptr];
}