
#include <gtest/gtest.h>

#include "common/common.hpp"
#include "nor/concepts.hpp"
#include "nor/env/kuhn.hpp"
#include "nor/factory.hpp"
#include "nor/policy/action_policy.hpp"
#include "nor/utils/utils.hpp"

using namespace nor;

template < size_t Index >
struct counter {
   static size_t copy_constructor_counter;
   static size_t copy_method_counter;
   static size_t clone_method_counter;

  protected:
   counter() = default;
   counter(const counter&) { copy_constructor_counter += 1; }
   counter& operator=(const counter&)
   {
      copy_constructor_counter += 1;
      return *this;
   }
   counter(counter&&) = default;
   counter& operator=(counter&&) = default;
   ~counter() = default;

   static auto copy() { copy_method_counter += 1; }
   static auto clone() { clone_method_counter += 1; }
};
template < size_t I >
size_t counter< I >::copy_constructor_counter = 0;
template < size_t I >
size_t counter< I >::copy_method_counter = 0;
template < size_t I >
size_t counter< I >::clone_method_counter = 0;

struct TestConfig {
   bool copy_constructible;
   bool copy_method;
   bool clone_method;
};

template < size_t CounterIndex, TestConfig cfg >
struct Tester: public counter< CounterIndex > {
   static constexpr auto config = cfg;
   static constexpr size_t index = CounterIndex;
   using counter_access = counter< CounterIndex >;

   Tester() = default;
   ~Tester() = default;

   Tester(const Tester&)
      requires(config.copy_constructible)
   = default;

   Tester& operator=(const Tester&)
      requires(config.copy_constructible)
   = default;

   Tester(Tester&&) = default;

   Tester& operator=(Tester&&) = default;

   auto copy() const
      requires(config.copy_method)
   {
      counter_access::copy();
      return Tester();
   }
   auto clone() const
      requires(config.clone_method)
   {
      counter_access::clone();
      return std::make_unique< Tester >();
   }
};

template < typename T >
struct clone_any_way_fixture: public ::testing::Test {
   clone_any_way_fixture() = default;
   ~clone_any_way_fixture() override { delete rawptr; }

   uptr< T > uniq_ptr = std::make_unique< T >();
   sptr< T > shrd_ptr = std::make_shared< T >();
   T* rawptr = new T();
   T value{};
   T& ref{*rawptr};
   const T& const_ref{*rawptr};
   std::reference_wrapper< T > ref_wrapper = {value};
};

using clone_any_way_types = ::testing::Types<
   Tester< 0, TestConfig{.copy_constructible = true, .copy_method = false, .clone_method = false} >,
   Tester< 1, TestConfig{.copy_constructible = false, .copy_method = true, .clone_method = false} >,
   Tester< 2, TestConfig{.copy_constructible = false, .copy_method = false, .clone_method = true} >,
   Tester< 3, TestConfig{.copy_constructible = true, .copy_method = true, .clone_method = false} >,
   Tester< 4, TestConfig{.copy_constructible = true, .copy_method = false, .clone_method = true} >,
   Tester<
      5,
      TestConfig{.copy_constructible = false, .copy_method = true, .clone_method = true} > >;

TYPED_TEST_SUITE(clone_any_way_fixture, clone_any_way_types);

TYPED_TEST(clone_any_way_fixture, test_all_paths)
{
   auto uptr_clone = nor::utils::clone_any_way(this->uniq_ptr);
   auto sptr_clone = nor::utils::clone_any_way(this->shrd_ptr);
   auto rawptr_clone = nor::utils::clone_any_way(this->rawptr);
   auto value_clone = nor::utils::clone_any_way(this->value);
   auto ref_clone = nor::utils::clone_any_way(this->ref);
   auto const_ref_clone = nor::utils::clone_any_way(this->const_ref);
   auto ref_wrapper_clone = nor::utils::clone_any_way(this->ref_wrapper);

   struct ExpectedCounts {
      size_t copy_cstructor_count = 0;
      size_t copy_meth_count = 0;
      size_t clone_meth_count = 0;
   };

   // these expected values are associated with the Tester types in 'clone_any_way_types' above
   constexpr std::array< ExpectedCounts, 6 > expected_counts = std::array{
      ExpectedCounts{.copy_cstructor_count = 7},
      ExpectedCounts{.copy_meth_count = 7},
      ExpectedCounts{.clone_meth_count = 7},
      ExpectedCounts{.copy_meth_count = 7},
      ExpectedCounts{.clone_meth_count = 7},
      ExpectedCounts{.clone_meth_count = 7}};

   EXPECT_EQ(
      TypeParam::counter_access::copy_constructor_counter,
      expected_counts[TypeParam::index].copy_cstructor_count
   );
   EXPECT_EQ(
      TypeParam::counter_access::copy_method_counter,
      expected_counts[TypeParam::index].copy_meth_count
   );
   EXPECT_EQ(
      TypeParam::counter_access::clone_method_counter,
      expected_counts[TypeParam::index].clone_meth_count
   );
}

TEST(ChildState, create_kuhn_child)
{
   using namespace nor;
   using namespace nor::games::kuhn;

   auto env = Environment{};
   State state{};

   state.apply_action(ChanceOutcome{kuhn::Player::one, Card::king});
   state.apply_action(ChanceOutcome{kuhn::Player::two, Card::queen});

   uptr< State > child = child_state(env, state, Action::check);
   auto state_copy = state;
   state_copy.apply_action(Action::check);
   ASSERT_EQ(child->history(), state_copy.history());
}

TEST(Normalizing, action_policy)
{
   nor::HashmapActionPolicy< int > policy{std::pair{0, 5.}, std::pair{1, 2.}, std::pair{2, 3.}};

   nor::normalize_action_policy_inplace(policy);

   EXPECT_EQ(policy[0], .5);
   EXPECT_EQ(policy[1], .2);
   EXPECT_EQ(policy[2], .3);
}

TEST(Normalizing, state_policy)
{
   using namespace nor;
   using namespace nor::games::kuhn;

   auto env = Environment{};
   State state{}, next_state{};
   Infostate istate1{nor::Player::alex}, istate2{nor::Player::alex};

   auto action = ChanceOutcome{kuhn::Player::one, Card::king};

   next_state.apply_action(action);

   istate2.update(
      env.public_observation(state, action, next_state),
      env.private_observation(nor::Player::bob, state, action, next_state)
   );
   state = next_state;

   action = ChanceOutcome{kuhn::Player::two, Card::queen};
   next_state.apply_action(action);
   istate1 = istate2;
   istate2.update(
      env.public_observation(state, action, next_state),
      env.private_observation(nor::Player::bob, state, action, next_state)
   );
   state = next_state;

   auto policy = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< int > >{}
   );
   policy.emplace(istate1, std::pair{0, 5.}, std::pair{1, 2.}, std::pair{2, 3.});
   policy.emplace(istate2, std::pair{0, 8.}, std::pair{1, 2.}, std::pair{2, 1.}, std::pair{3, 9.});

   auto policy_copy = policy;

   nor::normalize_state_policy_inplace(policy);

   EXPECT_EQ(policy(istate1)[0], .5);
   EXPECT_EQ(policy(istate1)[1], .2);
   EXPECT_EQ(policy(istate1)[2], .3);

   EXPECT_EQ(policy(istate2)[0], .4);
   EXPECT_EQ(policy(istate2)[1], .1);
   EXPECT_EQ(policy(istate2)[2], .05);
   EXPECT_EQ(policy(istate2)[3], .45);

   auto normalized_pol = nor::normalize_state_policy(policy_copy);

   EXPECT_EQ(normalized_pol(istate1)[0], .5);
   EXPECT_EQ(normalized_pol(istate1)[1], .2);
   EXPECT_EQ(normalized_pol(istate1)[2], .3);

   EXPECT_EQ(normalized_pol(istate2)[0], .4);
   EXPECT_EQ(normalized_pol(istate2)[1], .1);
   EXPECT_EQ(normalized_pol(istate2)[2], .05);
   EXPECT_EQ(normalized_pol(istate2)[3], .45);

   EXPECT_EQ(policy_copy(istate1)[0], 5.);
   EXPECT_EQ(policy_copy(istate1)[1], 2.);
   EXPECT_EQ(policy_copy(istate1)[2], 3.);

   EXPECT_EQ(policy_copy(istate2)[0], 8.);
   EXPECT_EQ(policy_copy(istate2)[1], 2.);
   EXPECT_EQ(policy_copy(istate2)[2], 1.);
   EXPECT_EQ(policy_copy(istate2)[3], 9.);
}
