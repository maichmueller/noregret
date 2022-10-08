
#include <gtest/gtest.h>

#include "common/common.hpp"
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
   std::reference_wrapper< T > value_ref = {value};
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
   auto value_ref_clone = nor::utils::clone_any_way(this->value_ref);

   struct ExpectedCounts {
      size_t copy_cstructor_count = 0;
      size_t copy_meth_count = 0;
      size_t clone_meth_count = 0;
   };

   constexpr std::array< ExpectedCounts, 6 > expected_counts = std::array{
      ExpectedCounts{.copy_cstructor_count = 5},
      ExpectedCounts{.copy_meth_count = 5},
      ExpectedCounts{.clone_meth_count = 5},
      ExpectedCounts{.copy_meth_count = 5},
      ExpectedCounts{.clone_meth_count = 5},
      ExpectedCounts{.clone_meth_count = 5}};

   ASSERT_EQ(
      TypeParam::counter_access::copy_constructor_counter,
      expected_counts[TypeParam::index].copy_cstructor_count);
   ASSERT_EQ(
      TypeParam::counter_access::copy_method_counter,
      expected_counts[TypeParam::index].copy_meth_count);
   ASSERT_EQ(
      TypeParam::counter_access::clone_method_counter,
      expected_counts[TypeParam::index].clone_meth_count);
}