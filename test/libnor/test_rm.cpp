
#include <gtest/gtest.h>

#include "nor/cfr/rm.hpp"
#include "nor/policy.hpp"

class RegretMatchingParamsF:
    public ::testing::TestWithParam< std::tuple<
       std::vector< double >,  // regret values
       std::unordered_map< int, double >,  // the expected policy
       nor::HashmapActionPolicy< int >  // the policy
       > > {
  protected:
   std::vector< int > actions{1, 2, 3, 4, 5};
};

using namespace nor;

TEST_P(RegretMatchingParamsF, integer_actions)
{
   auto [regret, expected, policy] = GetParam();

   std::unordered_map< int, double > regret_map;
   for(auto [a, r] : ranges::views::zip(actions, regret)) {
      regret_map[a] = r;
   }
   rm::regret_matching(policy, regret_map);
   ASSERT_EQ(policy, HashmapActionPolicy< int >{expected});
}

template < size_t N >
auto value_pack();

template <>
auto value_pack< 0 >()
{
   std::vector< double > r{1., 2., 3., 4., 5.};
   std::unordered_map< int, double > exp{
      {1, 1. / 15.}, {2, 2. / 15.}, {3, 3. / 15.}, {4, 4. / 15.}, {5, 5. / 15.}};
   nor::HashmapActionPolicy< int > pol{std::unordered_map< int, double >{
      {1, 1. / 15.}, {2, 2. / 15.}, {3, 3. / 15.}, {4, 4. / 15.}, {5, 5. / 15.}}};
   return std::tuple{r, exp, pol};
}

template <>
auto value_pack< 1 >()
{
   std::vector< double > r{1, -1, 1, -1, 1};
   std::unordered_map< int, double > exp{
      {1, 1. / 3.}, {2, 0.}, {3, 1. / 3.}, {4, 0.}, {5, 1. / 3.}};
   nor::HashmapActionPolicy< int > pol{std::unordered_map< int, double >{
      {1, 1. / 15.}, {2, 2. / 15.}, {3, 3. / 15.}, {4, 4. / 15.}, {5, 5. / 15.}}};
   return std::tuple{r, exp, pol};
}

template <>
auto value_pack< 2 >()
{
   std::vector< double > r{-1, -1, 0, -1, -1};
   std::unordered_map< int, double > exp{{1, 0.2}, {2, 0.2}, {3, 0.2}, {4, 0.2}, {5, 0.2}};
   nor::HashmapActionPolicy< int > pol{std::unordered_map< int, double >{
      {1, 1. / 15.}, {2, 2. / 15.}, {3, 3. / 15.}, {4, 4. / 15.}, {5, 5. / 15.}}};
   return std::tuple{r, exp, pol};
}

INSTANTIATE_TEST_SUITE_P(
   integer_actions_simple_test,
   RegretMatchingParamsF,
   ::testing::Values(value_pack< 0 >(), value_pack< 1 >(), value_pack< 2 >()));