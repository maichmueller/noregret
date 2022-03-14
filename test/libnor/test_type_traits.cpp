#include <gtest/gtest.h>

#include "nor/utils/type_traits.hpp"

TEST(add_const_if, expected_constness)
{
   //   using Type = int;
   //
   //   EXPECT_TRUE((std::is_same_v< nor::add_const_if_t< Type const, Type >, const Type >) );
   //   EXPECT_TRUE((std::is_same_v< nor::add_const_if_t< Type const*, Type >, const Type >) );
   //   EXPECT_TRUE((std::is_same_v< nor::add_const_if_t< Type* const, Type >, Type >) );
   //   EXPECT_TRUE((std::is_same_v< nor::add_const_if_t< Type const * const, Type >, const Type >)
   //   ); EXPECT_TRUE((std::is_same_v< nor::add_const_if_t< Type const &, Type >, const Type >) );
}

TEST(all_predicate, simple)
{
   EXPECT_TRUE((nor::all_predicate_v< std::is_default_constructible, int, double, std::string >) );
}
