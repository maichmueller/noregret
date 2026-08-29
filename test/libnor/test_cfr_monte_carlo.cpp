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
#include "nor/nor.hpp"
#include "rm_specific_testing_utils.hpp"

using namespace nor;

namespace {

constexpr rm::MCCFRConfig operations_config{
   .update_mode = rm::UpdateMode::alternating,
   .algorithm = rm::MCCFRAlgorithmMode::chance_sampling,
   .weighting = rm::MCCFRWeightingMode::none};

template < auto config >
auto make_rps_operation_solver()
{
   auto setup = setup_rps_test();
   return factory::make_mccfr< config >(
      std::move(std::get< 0 >(setup)),
      std::make_unique< games::rps::State >(),
      std::unordered_map{
         std::pair{Player::alex, std::move(std::get< 3 >(setup))},
         std::pair{Player::bob, std::move(std::get< 4 >(setup))}},
      std::unordered_map{
         std::pair{Player::alex, std::move(std::get< 1 >(setup))},
         std::pair{Player::bob, std::move(std::get< 2 >(setup))}},
      0.6,
      17
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

template < auto config >
void run_mccfr_on_kuhn_poker(
   size_t max_iters = 2e5,
   size_t update_freq = 500,
   double epsilon = 0.6,
   size_t seed = 0
)
{
   run_cfr_on_kuhn_poker< config >(max_iters, update_freq, epsilon, seed);
}
template < auto config >
void run_mccfr_on_rockpaperscissors(
   size_t max_iters = 1e5,
   size_t update_freq = 10,
   double epsilon = 0.6,
   size_t seed = 0
)
{
   run_cfr_on_rps< config >(max_iters, update_freq, epsilon, seed);
}
TEST(KuhnPoker, MCCFR_OS_optimistic_alternating)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::optimistic};
   run_mccfr_on_kuhn_poker< config >();
}

TEST(KuhnPoker, MCCFR_OS_optimistic_simultaneous)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::simultaneous,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::optimistic};
   run_mccfr_on_kuhn_poker< config >();
}

TEST(KuhnPoker, MCCFR_OS_lazy_alternating)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::lazy};
   run_mccfr_on_kuhn_poker< config >();
}

TEST(KuhnPoker, MCCFR_OS_lazy_simultaneous)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::simultaneous,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::lazy};
   run_mccfr_on_kuhn_poker< config >();
}

TEST(KuhnPoker, MCCFR_OS_stochastic_alternating)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::stochastic};
   run_mccfr_on_kuhn_poker< config >();
}

TEST(KuhnPoker, MCCFR_OS_stochastic_simultaneous)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::simultaneous,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::stochastic};
   run_mccfr_on_kuhn_poker< config >();
}

TEST(KuhnPoker, MCCFR_ES_stochastic)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::external_sampling,
      .weighting = rm::MCCFRWeightingMode::stochastic};
   run_mccfr_on_kuhn_poker< config >();
}

TEST(KuhnPoker, MCCFR_CS_alternating)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::chance_sampling,
      .weighting = rm::MCCFRWeightingMode::none};
   run_mccfr_on_kuhn_poker< config >();
}

TEST(KuhnPoker, MCCFR_CS_simultaneous)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::simultaneous,
      .algorithm = rm::MCCFRAlgorithmMode::chance_sampling,
      .weighting = rm::MCCFRWeightingMode::none};
   run_mccfr_on_kuhn_poker< config >();
}

TEST(KuhnPoker, CFR_PURE_alternating)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::pure_cfr,
      .weighting = rm::MCCFRWeightingMode::none};
   run_mccfr_on_kuhn_poker< config >();
}

TEST(KuhnPoker, CFR_PURE_simultaneous)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::simultaneous,
      .algorithm = rm::MCCFRAlgorithmMode::pure_cfr,
      .weighting = rm::MCCFRWeightingMode::none};
   run_mccfr_on_kuhn_poker< config >();
}

TEST(RockPaperScissors, MCCFR_OS_optimistic_alternating)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::optimistic};
   run_mccfr_on_rockpaperscissors< config >();
}

TEST(RockPaperScissors, MCCFR_OS_optimistic_simultaneous)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::simultaneous,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::optimistic};
   run_mccfr_on_rockpaperscissors< config >();
}

TEST(RockPaperScissors, MCCFR_OS_lazy_alternating)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::lazy};
   run_mccfr_on_rockpaperscissors< config >();
}

TEST(RockPaperScissors, MCCFR_OS_lazy_simultaneous)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::simultaneous,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::lazy};
   run_mccfr_on_rockpaperscissors< config >();
}

TEST(RockPaperScissors, MCCFR_OS_stochastic_alternating)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::stochastic};
   run_mccfr_on_rockpaperscissors< config >();
}

TEST(RockPaperScissors, MCCFR_OS_stochastic_simultaneous)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::simultaneous,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::stochastic};
   run_mccfr_on_rockpaperscissors< config >();
}

TEST(RockPaperScissors, MCCFR_ES_stochastic)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::external_sampling,
      .weighting = rm::MCCFRWeightingMode::stochastic};
   run_mccfr_on_rockpaperscissors< config >();
}

TEST(RockPaperScissors, MCCFR_CS_alternating)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::chance_sampling,
      .weighting = rm::MCCFRWeightingMode::none};
   run_mccfr_on_rockpaperscissors< config >();
}

TEST(RockPaperScissors, MCCFR_CS_simultaneous)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::simultaneous,
      .algorithm = rm::MCCFRAlgorithmMode::chance_sampling,
      .weighting = rm::MCCFRWeightingMode::none};
   run_mccfr_on_rockpaperscissors< config >();
}

TEST(RockPaperScissors, CFR_PURE_alternating)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::pure_cfr,
      .weighting = rm::MCCFRWeightingMode::none};
   run_mccfr_on_rockpaperscissors< config >();
}

TEST(RockPaperScissors, CFR_PURE_simultaneous)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::simultaneous,
      .algorithm = rm::MCCFRAlgorithmMode::pure_cfr,
      .weighting = rm::MCCFRWeightingMode::none};
   run_mccfr_on_rockpaperscissors< config >();
}

TEST(TabularSolverOperations, MCCFRStepTraceAndAdvance)
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
   static_assert(std::same_as<
                 decltype(std::declval< solver_type& >().iterate(std::nullopt)),
                 std::vector< std::pair< Player, double > > >);

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
   for(auto&& [legacy, one_step] : std::views::zip(legacy_values, one_step_values)) {
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

   auto compatibility_solver = make_rps_operation_solver< operations_config >();
   auto one_step_compatibility_solver = make_rps_operation_solver< operations_config >();
   const auto compatibility_values = compatibility_solver.iterate(std::nullopt);
   const auto one_step_compatibility_value = one_step_compatibility_solver.iterate();
   ASSERT_EQ(compatibility_values.size(), 1u);
   ASSERT_EQ(one_step_compatibility_value.get().count(compatibility_values.front().first), 1u);
   EXPECT_DOUBLE_EQ(
      compatibility_values.front().second,
      one_step_compatibility_value.get().at(compatibility_values.front().first)
   );
}

TEST(TabularSolverOperations, MCCFRDirectPolicyLookupAndInvalidation)
{
   auto solver = make_rps_operation_solver< operations_config >();
   const games::rps::Infostate alex_root{Player::alex};
   auto missing = games::rps::Infostate{Player::alex};
   missing.update("missing", "missing");

   EXPECT_FALSE(solver.current_policy_at(alex_root).has_value());
   EXPECT_FALSE(solver.average_policy_at(alex_root).has_value());
   EXPECT_FALSE(solver.current_policy_at(missing).has_value());

   const auto first_value = solver.iterate();
   EXPECT_EQ(first_value.get().size(), 2u);
   auto lookup = solver.policy_lookup();
   std::optional< games::rps::Infostate > averaged_infostate;
   ASSERT_EQ(
      solver.visit_average_policy([&](const auto& infostate, const auto&) {
         if(not averaged_infostate.has_value()) {
            averaged_infostate = infostate;
         }
      }),
      2u
   );
   ASSERT_TRUE(averaged_infostate.has_value());
   const auto& sampled_infostate = *averaged_infostate;
   auto current = lookup.at< rm::PolicyLabel::current >(sampled_infostate);
   auto average = lookup.at< rm::PolicyLabel::average >(sampled_infostate);
   ASSERT_TRUE(current.valid());
   ASSERT_TRUE(average.valid());
   EXPECT_EQ(current.size(), 3u);
   EXPECT_EQ(average.size(), 3u);

   EXPECT_EQ(
      solver.visit_current_policy([](const auto&, const auto& view) { EXPECT_TRUE(view.valid()); }),
      2u
   );

   const auto& materialized_current = solver.policy()
                                         .at(sampled_infostate.player())
                                         .table()
                                         .at(sampled_infostate);
   const auto& materialized_average = solver.average_policy()
                                         .at(sampled_infostate.player())
                                         .table()
                                         .at(sampled_infostate);
   for(const auto action :
       {games::rps::Action::paper, games::rps::Action::rock, games::rps::Action::scissors}) {
      EXPECT_DOUBLE_EQ(current.at(action), materialized_current.at(action));
      EXPECT_DOUBLE_EQ(average.at(action), materialized_average.at(action));
   }

   EXPECT_EQ(
      solver.visit_current_policy([](const auto&, const auto& view) { EXPECT_TRUE(view.valid()); }),
      2u
   );
   EXPECT_EQ(
      solver.visit_average_policy([](const auto& view) {
         view.for_each([](const auto&, double value) { EXPECT_GE(value, 0.); });
      }),
      2u
   );

   auto mutation_solver = make_rps_operation_solver< operations_config >();
   (void) mutation_solver.iterate();
   auto mutation_lookup = mutation_solver.policy_lookup();
   auto mutation_view = mutation_lookup.at< rm::PolicyLabel::current >(alex_root);
   const std::vector< games::rps::Action > actions{
      games::rps::Action::paper, games::rps::Action::rock, games::rps::Action::scissors};
   auto& legacy_action_policy = mutation_solver.fetch_policy< true >(alex_root, actions);
   (void) legacy_action_policy;
   EXPECT_FALSE(mutation_lookup.valid());
   EXPECT_FALSE(mutation_view.valid());

   auto stale_lookup = solver.policy_lookup();
   auto stale_view = stale_lookup.at< rm::PolicyLabel::current >(sampled_infostate);
   solver.advance(1);
   EXPECT_FALSE(stale_lookup.valid());
   EXPECT_FALSE(stale_view.valid());
   EXPECT_THROW(
      (void) stale_lookup.find< rm::PolicyLabel::current >(sampled_infostate), std::logic_error
   );
   EXPECT_THROW((void) stale_view.size(), std::logic_error);
}

TEST(TabularSolverOperations, MCCFRMoveTransfersPolicyGeneration)
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

      // The moved-from MCCFR solver lazily acquires an independent token on its first lookup.
      auto moved_from_lookup = solver.policy_lookup();
      EXPECT_TRUE(moved_from_lookup.valid());
      return view;
   }();

   EXPECT_FALSE(moved_view.valid());
   EXPECT_THROW((void) moved_view.size(), std::logic_error);
}
