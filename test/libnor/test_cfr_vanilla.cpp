#include <gtest/gtest.h>

#include <concepts>
#include <memory>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../games/stratego/fixtures.hpp"
#include "cfr_run_funcs.hpp"
#include "nor/env.hpp"
#include "nor/exploitability.hpp"
#include "nor/nor.hpp"
#include "rm_specific_testing_utils.hpp"

using namespace nor;

namespace {

constexpr rm::CFRConfig operations_config{
   .update_mode = rm::UpdateMode::simultaneous,
   .regret_minimizing_mode = rm::RegretMinimizingMode::regret_matching,
   .weighting_mode = rm::CFRWeightingMode::uniform};

constexpr rm::CFRConfig alternating_operations_config{
   .update_mode = rm::UpdateMode::alternating,
   .regret_minimizing_mode = rm::RegretMinimizingMode::regret_matching,
   .weighting_mode = rm::CFRWeightingMode::uniform};

template < auto config >
auto make_rps_operation_solver()
{
   auto setup = setup_rps_test();
   return factory::make_cfr< config >(
      std::move(std::get< 0 >(setup)),
      std::make_unique< games::rps::State >(),
      std::unordered_map{
         std::pair{Player::alex, std::move(std::get< 3 >(setup))},
         std::pair{Player::bob, std::move(std::get< 4 >(setup))}},
      std::unordered_map{
         std::pair{Player::alex, std::move(std::get< 1 >(setup))},
         std::pair{Player::bob, std::move(std::get< 2 >(setup))}}
   );
}

void expect_same_root_value(const rm::StateValueMap& actual, const rm::StateValueMap& expected)
{
   ASSERT_EQ(actual.get().size(), expected.get().size());
   for(const auto [player, value] : actual.get()) {
      ASSERT_EQ(expected.get().count(player), 1u);
      EXPECT_DOUBLE_EQ(value, expected.get().at(player));
   }
}

}  // namespace

TEST(KuhnPoker, CFR_VANILLA_alternating)
{
   run_cfr_on_kuhn_poker< rm::CFRDiscountedConfig{.update_mode = rm::UpdateMode::alternating} >();
}

TEST(KuhnPoker, CFR_VANILLA_simultaneous)
{
   run_cfr_on_kuhn_poker< rm::CFRDiscountedConfig{.update_mode = rm::UpdateMode::simultaneous} >();
}

TEST(RockPaperScissors, CFR_VANILLA_alternating)
{
   run_cfr_on_rps< rm::CFRDiscountedConfig{.update_mode = rm::UpdateMode::alternating} >();
}

TEST(RockPaperScissors, CFR_VANILLA_simultaneous)
{
   run_cfr_on_rps< rm::CFRDiscountedConfig{.update_mode = rm::UpdateMode::simultaneous} >();
}

TEST(TabularSolverOperations, VanillaStepTraceAndAdvance)
{
   using solver_type = decltype(make_rps_operation_solver< operations_config >());
   static_assert(std::same_as<
                 decltype(std::declval< solver_type& >().iterate()),
                 rm::StateValueMap >);
   static_assert(std::same_as< decltype(std::declval< solver_type& >().advance(size_t{})), void >);
   static_assert(std::same_as<
                 decltype(std::declval< solver_type& >().advance_last(size_t{})),
                 std::optional< rm::StateValueMap > >);
   static_assert(std::same_as<
                 decltype(std::declval< solver_type& >().trace(size_t{}, size_t{})),
                 std::vector< rm::StateValueMap > >);

   auto one_step_solver = make_rps_operation_solver< operations_config >();
   std::vector< rm::StateValueMap > one_step_values;
   one_step_values.reserve(5);
   for(size_t i = 0; i < 5; ++i) {
      one_step_values.emplace_back(one_step_solver.iterate());
   }
   EXPECT_EQ(one_step_solver.iteration(), 5u);

   auto legacy_solver = make_rps_operation_solver< operations_config >();
   const auto legacy_values = legacy_solver.iterate(5);
   ASSERT_EQ(legacy_values.size(), one_step_values.size());
   for(const auto [legacy, one_step] : std::views::zip(legacy_values, one_step_values)) {
      expect_same_root_value(legacy, one_step);
   }

   auto trace_solver = make_rps_operation_solver< operations_config >();
   const auto traced_values = trace_solver.trace(5, 2);
   ASSERT_EQ(traced_values.size(), 2u);
   expect_same_root_value(traced_values[0], one_step_values[1]);
   expect_same_root_value(traced_values[1], one_step_values[3]);
   EXPECT_EQ(trace_solver.iteration(), 5u);

   auto advance_solver = make_rps_operation_solver< operations_config >();
   advance_solver.advance(5);
   EXPECT_EQ(advance_solver.iteration(), 5u);

   auto last_solver = make_rps_operation_solver< operations_config >();
   const auto last_value = last_solver.advance_last(5);
   ASSERT_TRUE(last_value.has_value());
   expect_same_root_value(*last_value, one_step_values.back());
   EXPECT_EQ(last_solver.iteration(), 5u);

   auto empty_solver = make_rps_operation_solver< operations_config >();
   const auto initial_generation = empty_solver.policy_generation();
   EXPECT_FALSE(empty_solver.advance_last(0).has_value());
   EXPECT_TRUE(empty_solver.trace(0, 2).empty());
   EXPECT_EQ(empty_solver.iteration(), 0u);
   EXPECT_EQ(empty_solver.policy_generation(), initial_generation);
   EXPECT_THROW((void) empty_solver.trace(0, 0), std::invalid_argument);
   EXPECT_EQ(empty_solver.iteration(), 0u);

   using alternating_solver_type = decltype(make_rps_operation_solver<
                                            alternating_operations_config >());
   static_assert(std::same_as<
                 decltype(std::declval< alternating_solver_type& >().iterate(std::nullopt)),
                 std::vector< rm::StateValueMap > >);
   auto compatibility_solver = make_rps_operation_solver< alternating_operations_config >();
   auto one_step_alternating_solver = make_rps_operation_solver< alternating_operations_config >();
   const auto compatibility_values = compatibility_solver.iterate(std::nullopt);
   const auto one_step_alternating_value = one_step_alternating_solver.iterate();
   ASSERT_EQ(compatibility_values.size(), 1u);
   expect_same_root_value(compatibility_values.front(), one_step_alternating_value);
}

TEST(TabularSolverOperations, VanillaDirectPolicyLookupAndInvalidation)
{
   using solver_type = decltype(make_rps_operation_solver< operations_config >());
   using view_optional_type = decltype(std::declval< const solver_type& >().current_policy_at(
      std::declval< const games::rps::Infostate& >()
   ));

   auto solver = make_rps_operation_solver< operations_config >();
   const games::rps::Infostate alex_root{Player::alex};
   auto missing = games::rps::Infostate{Player::alex};
   missing.update("missing", "missing");

   EXPECT_FALSE(solver.current_policy_at(alex_root).has_value());
   EXPECT_FALSE(solver.average_policy_at(alex_root).has_value());
   EXPECT_FALSE(solver.current_policy_at(missing).has_value());

   (void) solver.iterate();
   auto lookup = solver.policy_lookup();
   ASSERT_TRUE(lookup.valid());
   auto current = lookup.at< rm::PolicyLabel::current >(alex_root);
   auto average = lookup.at< rm::PolicyLabel::average >(alex_root);
   ASSERT_TRUE(current.valid());
   ASSERT_TRUE(average.valid());
   EXPECT_EQ(current.size(), 3u);
   EXPECT_EQ(average.size(), 3u);
   EXPECT_EQ(current.action_at(0), games::rps::Action::paper);
   EXPECT_EQ(current.action_at(1), games::rps::Action::rock);
   EXPECT_EQ(current.action_at(2), games::rps::Action::scissors);
   EXPECT_THROW((void) current.action_at(current.size()), std::out_of_range);
   EXPECT_THROW((void) current.value_at(current.size()), std::out_of_range);
   EXPECT_TRUE(current.contains(games::rps::Action::rock));

   const auto& materialized_current = solver.policy().at(Player::alex).table().at(alex_root);
   const auto&
      materialized_average = solver.average_policy().at(Player::alex).table().at(alex_root);
   for(const auto action :
       {games::rps::Action::paper, games::rps::Action::rock, games::rps::Action::scissors}) {
      EXPECT_DOUBLE_EQ(current.at(action), materialized_current.at(action));
      EXPECT_DOUBLE_EQ(average.at(action), materialized_average.at(action));
   }

   size_t current_nodes = 0;
   const auto current_visited = solver.visit_current_policy(
      [&](const auto& infostate, const auto& node_view) {
         ++current_nodes;
         EXPECT_TRUE(node_view.valid());
         EXPECT_TRUE(infostate.player() == Player::alex or infostate.player() == Player::bob);
      }
   );
   EXPECT_EQ(current_visited, 2u);
   EXPECT_EQ(current_nodes, current_visited);

   size_t average_nodes = 0;
   const auto average_visited = solver.visit_average_policy([&](const auto& node_view) {
      ++average_nodes;
      node_view.for_each([&](const auto&, double value) { EXPECT_GE(value, 0.); });
   });
   EXPECT_EQ(average_visited, 2u);
   EXPECT_EQ(average_nodes, average_visited);

   const auto direct_current = solver.current_policy_at(alex_root);
   ASSERT_TRUE(direct_current.has_value());
   const auto direct_average = solver.average_policy_at(alex_root);
   ASSERT_TRUE(direct_average.has_value());

   auto stale_lookup = solver.policy_lookup();
   auto stale_view = stale_lookup.at< rm::PolicyLabel::current >(alex_root);
   const auto stale_generation = stale_lookup.generation();
   solver.advance(1);
   EXPECT_FALSE(stale_lookup.valid());
   EXPECT_FALSE(stale_view.valid());
   EXPECT_GT(solver.policy_generation(), stale_generation);
   EXPECT_THROW((void) stale_lookup.find< rm::PolicyLabel::current >(alex_root), std::logic_error);
   EXPECT_THROW((void) stale_view.at(games::rps::Action::rock), std::logic_error);

   view_optional_type orphaned_view;
   {
      auto temporary_solver = make_rps_operation_solver< operations_config >();
      (void) temporary_solver.iterate();
      orphaned_view = temporary_solver.current_policy_at(alex_root);
      ASSERT_TRUE(orphaned_view.has_value());
      EXPECT_TRUE(orphaned_view->valid());
   }
   ASSERT_TRUE(orphaned_view.has_value());
   EXPECT_FALSE(orphaned_view->valid());
   EXPECT_THROW((void) orphaned_view->size(), std::logic_error);
}

TEST(TabularSolverOperations, VanillaMoveTransfersPolicyGeneration)
{
   const games::rps::Infostate alex_root{Player::alex};
   auto solver = make_rps_operation_solver< operations_config >();
   (void) solver.iterate();
   auto pre_move_lookup = solver.policy_lookup();
   auto pre_move_view = pre_move_lookup.at< rm::PolicyLabel::current >(alex_root);
   ASSERT_TRUE(pre_move_lookup.valid());
   ASSERT_TRUE(pre_move_view.valid());

   auto moved_view = [&] {
      auto moved_solver = std::move(solver);
      EXPECT_FALSE(pre_move_lookup.valid());
      EXPECT_FALSE(pre_move_view.valid());

      auto moved_lookup = moved_solver.policy_lookup();
      EXPECT_TRUE(moved_lookup.valid());
      auto view = moved_lookup.at< rm::PolicyLabel::current >(alex_root);
      EXPECT_TRUE(view.valid());

      // A moved-from solver has no shared ownership of the moved-to token, but its next lookup
      // receives a fresh token lazily.
      auto moved_from_lookup = solver.policy_lookup();
      EXPECT_TRUE(moved_from_lookup.valid());
      return view;
   }();

   EXPECT_FALSE(moved_view.valid());
   EXPECT_THROW((void) moved_view.size(), std::logic_error);
}

TEST(TabularSolverOperations, VanillaMoveAssignmentInvalidatesBothSources)
{
   const games::rps::Infostate alex_root{Player::alex};
   auto destination = make_rps_operation_solver< operations_config >();
   (void) destination.iterate();
   auto destination_lookup = destination.policy_lookup();
   auto destination_view = destination_lookup.at< rm::PolicyLabel::current >(alex_root);

   auto source = make_rps_operation_solver< operations_config >();
   (void) source.iterate();
   auto source_lookup = source.policy_lookup();
   auto source_view = source_lookup.at< rm::PolicyLabel::current >(alex_root);

   destination = std::move(source);
   EXPECT_FALSE(destination_lookup.valid());
   EXPECT_FALSE(destination_view.valid());
   EXPECT_FALSE(source_lookup.valid());
   EXPECT_FALSE(source_view.valid());

   auto assigned_lookup = destination.policy_lookup();
   ASSERT_TRUE(assigned_lookup.valid());
   EXPECT_TRUE(assigned_lookup.at< rm::PolicyLabel::current >(alex_root).valid());
}
