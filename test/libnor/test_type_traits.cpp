#include <gtest/gtest.h>

#include "common/types.hpp"

// TEST(add_const_if, expected_constness)
//{
//    //   using Type = int;
//    //
//    //   EXPECT_TRUE((std::is_same_v< nor::add_const_if_t< Type const, Type >, const Type >) );
//    //   EXPECT_TRUE((std::is_same_v< nor::add_const_if_t< Type const*, Type >, const Type >) );
//    //   EXPECT_TRUE((std::is_same_v< nor::add_const_if_t< Type* const, Type >, Type >) );
//    //   EXPECT_TRUE((std::is_same_v< nor::add_const_if_t< Type const * const, Type >, const Type
//    >)
//    //   ); EXPECT_TRUE((std::is_same_v< nor::add_const_if_t< Type const &, Type >, const Type >)
//    );
// }

TEST(all_predicate, simple)
{
   EXPECT_TRUE((common::all_predicate_v< std::is_default_constructible, int, double, std::string >)
   );
}

struct Conditions {
   bool _1 = false;
   bool _2 = false;
   bool _3 = false;
   bool _4 = false;
};

template < typename Expected, Conditions config >
struct SwitchTypedTestParamPack {
   using expected = Expected;
   static constexpr auto conditions = config;
};

template < typename T >
struct switch_fixture: public ::testing::Test {
   switch_fixture() = default;
};

using switch_options = std::tuple< char, int, float, double >;

using switch_types = ::testing::Types<
   SwitchTypedTestParamPack<
      std::tuple_element< 0, switch_options >,
      Conditions{true, false, false, false} >,
   SwitchTypedTestParamPack<
      std::tuple_element< 1, switch_options >,
      Conditions{false, true, false, false} >,
   SwitchTypedTestParamPack<
      std::tuple_element< 2, switch_options >,
      Conditions{false, false, true, false} >,
   SwitchTypedTestParamPack<
      std::tuple_element< 3, switch_options >,
      Conditions{false, false, false, true} >,
   SwitchTypedTestParamPack<
      std::tuple_element< 1, switch_options >,
      Conditions{false, true, true, true} >,
   SwitchTypedTestParamPack<
      std::tuple_element< 1, switch_options >,
      Conditions{false, true, false, true} >,
   SwitchTypedTestParamPack<
      std::tuple_element< 2, switch_options >,
      Conditions{false, false, true, true} >,
   SwitchTypedTestParamPack<
      std::tuple_element< 0, switch_options >,
      Conditions{true, false, false, true} >,
   SwitchTypedTestParamPack<
      std::tuple_element< 0, switch_options >,
      Conditions{true, true, true, true} >,
   SwitchTypedTestParamPack< void, Conditions{false, false, false, false} > >;

TYPED_TEST_SUITE(switch_fixture, switch_types);

TYPED_TEST(switch_fixture, _)
{
   constexpr auto conditions = TypeParam::conditions;
   using expected = typename TypeParam::expected;
   using switch_choice = common::switch_t<
      common::case_< conditions._1, std::tuple_element< 0, switch_options > >,
      common::case_< conditions._2, std::tuple_element< 1, switch_options > >,
      common::case_< conditions._3, std::tuple_element< 2, switch_options > >,
      common::case_< conditions._4, std::tuple_element< 3, switch_options > > >;

   EXPECT_TRUE((std::same_as< expected, switch_choice >) );
}
