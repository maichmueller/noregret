#include <gtest/gtest.h>

#include <unordered_map>

#include "../games/stratego/fixtures.hpp"
#include "nor/env.hpp"
#include "nor/nor.hpp"
#include "rm_specific_testing_utils.hpp"

using namespace nor;

TEST(KuhnPoker, MCCFR_OS_optimistic_alternating)
{
   games::kuhn::Environment env{};

   auto root_state = std::make_unique< games::kuhn::State >();
   auto players = env.players(*root_state);

   auto avg_tabular_policy = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      factory::
         make_zero_policy< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >());

   auto tabular_policy = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      factory::make_uniform_policy<
         games::kuhn::Infostate,
         HashmapActionPolicy< games::kuhn::Action > >());

   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::optimistic};

   auto solver = factory::make_mccfr< config, true >(
      std::move(env),
      std::make_unique< games::kuhn::State >(),
      tabular_policy,
      avg_tabular_policy,
      0.6,
      0);

   auto initial_curr_policy_profile = std::unordered_map{
      std::pair{Player::alex, normalize_state_policy(solver.policy().at(Player::alex).table())},
      std::pair{Player::bob, normalize_state_policy(solver.policy().at(Player::bob).table())}};

   auto initial_policy_profile = std::unordered_map{
      std::pair{
         Player::alex,
         normalize_state_policy(solver.average_policy().at(Player::alex).table())},
      std::pair{
         Player::bob, normalize_state_policy(solver.average_policy().at(Player::bob).table())}};

//      size_t n_iters = 200000;
      size_t n_iters = 1;
   for(size_t i = 0; i < n_iters; i++) {
      solver.iterate(1);
#ifndef NDEBUG
      evaluate_policies< true >(solver, initial_curr_policy_profile, i, "Current Policy");
      evaluate_policies< false >(solver, initial_policy_profile, i);
#endif
   }
   evaluate_policies< false >(
      solver, players | utils::is_actual_player_filter, n_iters, "Final Policy");
   assert_optimal_policy_kuhn(solver, env, 0.05);
}

TEST(KuhnPoker, MCCFR_OS_optimistic_simultaneous)
{
   games::kuhn::Environment env{};

   auto root_state = std::make_unique< games::kuhn::State >();
   auto players = env.players(*root_state);

   auto avg_tabular_policy = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      factory::
         make_zero_policy< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >());

   auto tabular_policy = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      factory::make_uniform_policy<
         games::kuhn::Infostate,
         HashmapActionPolicy< games::kuhn::Action > >());

   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::simultaneous,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::optimistic};

   auto solver = factory::make_mccfr< config, true >(
      std::move(env),
      std::make_unique< games::kuhn::State >(),
      tabular_policy,
      avg_tabular_policy,
      0.6,
      0);

   auto initial_curr_policy_profile = std::unordered_map{
      std::pair{Player::alex, normalize_state_policy(solver.policy().at(Player::alex).table())},
      std::pair{Player::bob, normalize_state_policy(solver.policy().at(Player::bob).table())}};

   auto initial_policy_profile = std::unordered_map{
      std::pair{
         Player::alex,
         normalize_state_policy(solver.average_policy().at(Player::alex).table())},
      std::pair{
         Player::bob, normalize_state_policy(solver.average_policy().at(Player::bob).table())}};

   size_t n_iters = 200000;
   for(size_t i = 0; i < n_iters; i++) {
      solver.iterate(1);
#ifndef NDEBUG
      evaluate_policies< true >(solver, initial_curr_policy_profile, i, "Current Policy");
      evaluate_policies< false >(solver, initial_policy_profile, i);
#endif
   }
   evaluate_policies< false >(
      solver, players | utils::is_actual_player_filter, n_iters, "Final Policy");
   assert_optimal_policy_kuhn(solver, env, 0.05);
}

TEST(KuhnPoker, MCCFR_OS_lazy_alternating)
{
   games::kuhn::Environment env{};

   auto root_state = std::make_unique< games::kuhn::State >();
   auto players = env.players(*root_state);

   auto avg_tabular_policy = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      factory::
         make_zero_policy< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >());

   auto tabular_policy = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      factory::make_uniform_policy<
         games::kuhn::Infostate,
         HashmapActionPolicy< games::kuhn::Action > >());

   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::lazy};

   auto solver = factory::make_mccfr< config, true >(
      std::move(env),
      std::make_unique< games::kuhn::State >(),
      tabular_policy,
      avg_tabular_policy,
      0.6,
      0);

   auto initial_curr_policy_profile = std::unordered_map{
      std::pair{Player::alex, normalize_state_policy(solver.policy().at(Player::alex).table())},
      std::pair{Player::bob, normalize_state_policy(solver.policy().at(Player::bob).table())}};

   auto initial_policy_profile = std::unordered_map{
      std::pair{
         Player::alex,
         normalize_state_policy(solver.average_policy().at(Player::alex).table())},
      std::pair{
         Player::bob, normalize_state_policy(solver.average_policy().at(Player::bob).table())}};

   size_t n_iters = 200000;
   for(size_t i = 0; i < n_iters; i++) {
      solver.iterate(1);
#ifndef NDEBUG
      evaluate_policies< true >(solver, initial_curr_policy_profile, i, "Current Policy");
      evaluate_policies< false >(solver, initial_policy_profile, i);
#endif
   }
   evaluate_policies< false >(
      solver, players | utils::is_actual_player_filter, n_iters, "Final Policy");
   assert_optimal_policy_kuhn(solver, env, 0.05);
}

TEST(KuhnPoker, MCCFR_OS_lazy_simultaneous)
{
   games::kuhn::Environment env{};

   auto root_state = std::make_unique< games::kuhn::State >();
   auto players = env.players(*root_state);

   auto avg_tabular_policy = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      factory::
         make_zero_policy< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >());

   auto tabular_policy = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      factory::make_uniform_policy<
         games::kuhn::Infostate,
         HashmapActionPolicy< games::kuhn::Action > >());

   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::simultaneous,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::lazy};

   auto solver = factory::make_mccfr< config, true >(
      std::move(env),
      std::make_unique< games::kuhn::State >(),
      tabular_policy,
      avg_tabular_policy,
      0.6,
      0);

   auto initial_curr_policy_profile = std::unordered_map{
      std::pair{Player::alex, normalize_state_policy(solver.policy().at(Player::alex).table())},
      std::pair{Player::bob, normalize_state_policy(solver.policy().at(Player::bob).table())}};

   auto initial_policy_profile = std::unordered_map{
      std::pair{
         Player::alex,
         normalize_state_policy(solver.average_policy().at(Player::alex).table())},
      std::pair{
         Player::bob, normalize_state_policy(solver.average_policy().at(Player::bob).table())}};

   size_t n_iters = 200000;
   for(size_t i = 0; i < n_iters; i++) {
      solver.iterate(1);
#ifndef NDEBUG
      evaluate_policies< true >(solver, initial_curr_policy_profile, i, "Current Policy");
      evaluate_policies< false >(solver, initial_policy_profile, i);
#endif
   }
   evaluate_policies< false >(
      solver, players | utils::is_actual_player_filter, n_iters, "Final Policy");
   assert_optimal_policy_kuhn(solver, env, 0.05);
}

TEST(KuhnPoker, MCCFR_OS_stochastic_alternating)
{
   games::kuhn::Environment env{};

   auto root_state = std::make_unique< games::kuhn::State >();
   auto players = env.players(*root_state);

   auto avg_tabular_policy = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      factory::
         make_zero_policy< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >());

   auto tabular_policy = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      factory::make_uniform_policy<
         games::kuhn::Infostate,
         HashmapActionPolicy< games::kuhn::Action > >());

   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::stochastic};

   auto solver = factory::make_mccfr< config, true >(
      std::move(env),
      std::make_unique< games::kuhn::State >(),
      tabular_policy,
      avg_tabular_policy,
      0.6,
      0);

   auto initial_curr_policy_profile = std::unordered_map{
      std::pair{Player::alex, normalize_state_policy(solver.policy().at(Player::alex).table())},
      std::pair{Player::bob, normalize_state_policy(solver.policy().at(Player::bob).table())}};

   auto initial_policy_profile = std::unordered_map{
      std::pair{
         Player::alex,
         normalize_state_policy(solver.average_policy().at(Player::alex).table())},
      std::pair{
         Player::bob, normalize_state_policy(solver.average_policy().at(Player::bob).table())}};

   size_t n_iters = 200000;
   for(size_t i = 0; i < n_iters; i++) {
      solver.iterate(1);
#ifndef NDEBUG
      evaluate_policies< true >(solver, initial_curr_policy_profile, i, "Current Policy");
      evaluate_policies< false >(solver, initial_policy_profile, i);
#endif
   }
   evaluate_policies< false >(
      solver, players | utils::is_actual_player_filter, n_iters, "Final Policy");
   assert_optimal_policy_kuhn(solver, env, 0.05);
}

TEST(KuhnPoker, MCCFR_OS_stochastic_simultaneous)
{
   games::kuhn::Environment env{};

   auto root_state = std::make_unique< games::kuhn::State >();
   auto players = env.players(*root_state);

   auto avg_tabular_policy = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      factory::
         make_zero_policy< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >());

   auto tabular_policy = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      factory::make_uniform_policy<
         games::kuhn::Infostate,
         HashmapActionPolicy< games::kuhn::Action > >());

   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::simultaneous,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::stochastic};

   auto solver = factory::make_mccfr< config, true >(
      std::move(env),
      std::make_unique< games::kuhn::State >(),
      tabular_policy,
      avg_tabular_policy,
      0.6,
      0);

   auto initial_curr_policy_profile = std::unordered_map{
      std::pair{Player::alex, normalize_state_policy(solver.policy().at(Player::alex).table())},
      std::pair{Player::bob, normalize_state_policy(solver.policy().at(Player::bob).table())}};

   auto initial_policy_profile = std::unordered_map{
      std::pair{
         Player::alex,
         normalize_state_policy(solver.average_policy().at(Player::alex).table())},
      std::pair{
         Player::bob, normalize_state_policy(solver.average_policy().at(Player::bob).table())}};

   size_t n_iters = 200000;
   for(size_t i = 0; i < n_iters; i++) {
      solver.iterate(1);
#ifndef NDEBUG
      evaluate_policies< true >(solver, initial_curr_policy_profile, i, "Current Policy");
      evaluate_policies< false >(solver, initial_policy_profile, i);
#endif
   }
   evaluate_policies< false >(
      solver, players | utils::is_actual_player_filter, n_iters, "Final Policy");
   assert_optimal_policy_kuhn(solver, env, 0.05);
}

TEST(KuhnPoker, MCCFR_ES_stochastic)
{
   games::kuhn::Environment env{};

   auto root_state = std::make_unique< games::kuhn::State >();
   auto players = env.players(*root_state);

   auto avg_tabular_policy = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      factory::
         make_zero_policy< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >());

   auto tabular_policy = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      factory::make_uniform_policy<
         games::kuhn::Infostate,
         HashmapActionPolicy< games::kuhn::Action > >());

   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::external_sampling,
      .weighting = rm::MCCFRWeightingMode::stochastic};

   auto solver = factory::make_mccfr< config, true >(
      std::move(env),
      std::make_unique< games::kuhn::State >(),
      tabular_policy,
      avg_tabular_policy,
      0.6,
      0);

   auto initial_curr_policy_profile = std::unordered_map{
      std::pair{Player::alex, normalize_state_policy(solver.policy().at(Player::alex).table())},
      std::pair{Player::bob, normalize_state_policy(solver.policy().at(Player::bob).table())}};

   auto initial_policy_profile = std::unordered_map{
      std::pair{
         Player::alex,
         normalize_state_policy(solver.average_policy().at(Player::alex).table())},
      std::pair{
         Player::bob, normalize_state_policy(solver.average_policy().at(Player::bob).table())}};

   size_t n_iters = 200000;
   for(size_t i = 0; i < n_iters; i++) {
      solver.iterate(1);
#ifndef NDEBUG
      evaluate_policies< true >(solver, initial_curr_policy_profile, i, "Current Policy");
      evaluate_policies< false >(solver, initial_policy_profile, i);
#endif
   }
   evaluate_policies< false >(
      solver, players | utils::is_actual_player_filter, n_iters, "Final Policy");
   assert_optimal_policy_kuhn(solver, env, 0.05);
}

TEST(RockPaperScissors, MCCFR_OS_optimistic_alternating)
{
   auto
      [env,
       avg_tabular_policy_alex,
       avg_tabular_policy_bob,
       tabular_policy_alex,
       tabular_policy_bob,
       infostate_alex,
       infostate_bob,
       init_state] = setup_rps_test();

   auto root_state = std::make_unique< games::rps::State >();
   auto players = env.players(*root_state);

   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::optimistic};

   auto solver = factory::make_mccfr< config >(
      std::move(env),
      std::move(root_state),
      std::unordered_map{
         std::pair{Player::alex, tabular_policy_alex}, std::pair{Player::bob, tabular_policy_bob}},
      std::unordered_map{
         std::pair{Player::alex, avg_tabular_policy_alex},
         std::pair{Player::bob, avg_tabular_policy_bob}},
      0.6,
      0);

   auto initial_curr_policy_profile = std::unordered_map{
      std::pair{Player::alex, normalize_state_policy(solver.policy().at(Player::alex).table())},
      std::pair{Player::bob, normalize_state_policy(solver.policy().at(Player::bob).table())}};

   auto initial_policy_profile = std::unordered_map{
      std::pair{
         Player::alex,
         normalize_state_policy(solver.average_policy().at(Player::alex).table())},
      std::pair{
         Player::bob, normalize_state_policy(solver.average_policy().at(Player::bob).table())}};

   size_t n_iters = 40000;
   for(size_t i = 0; i < n_iters; i++) {
      solver.iterate(1);
#ifndef NDEBUG
      evaluate_policies< true >(solver, initial_curr_policy_profile, i, "Current Policy");
      evaluate_policies< false >(solver, initial_policy_profile, i);
#endif
   }
   evaluate_policies< false >(
      solver, players | utils::is_actual_player_filter, n_iters, "Final Policy");
   assert_optimal_policy_rps(solver);
}

TEST(RockPaperScissors, MCCFR_OS_optimistic_simultaneous)
{
   auto
      [env,
       avg_tabular_policy_alex,
       avg_tabular_policy_bob,
       tabular_policy_alex,
       tabular_policy_bob,
       infostate_alex,
       infostate_bob,
       init_state] = setup_rps_test();

   auto root_state = std::make_unique< games::rps::State >();
   auto players = env.players(*root_state);

   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::simultaneous,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::optimistic};

   auto solver = factory::make_mccfr< config >(
      std::move(env),
      std::move(root_state),
      std::unordered_map{
         std::pair{Player::alex, tabular_policy_alex}, std::pair{Player::bob, tabular_policy_bob}},
      std::unordered_map{
         std::pair{Player::alex, avg_tabular_policy_alex},
         std::pair{Player::bob, avg_tabular_policy_bob}},
      0.6,
      0);

   auto initial_curr_policy_profile = std::unordered_map{
      std::pair{Player::alex, normalize_state_policy(solver.policy().at(Player::alex).table())},
      std::pair{Player::bob, normalize_state_policy(solver.policy().at(Player::bob).table())}};

   auto initial_policy_profile = std::unordered_map{
      std::pair{
         Player::alex,
         normalize_state_policy(solver.average_policy().at(Player::alex).table())},
      std::pair{
         Player::bob, normalize_state_policy(solver.average_policy().at(Player::bob).table())}};

   size_t n_iters = 40000;
   for(size_t i = 0; i < n_iters; i++) {
      solver.iterate(1);
#ifndef NDEBUG
      evaluate_policies< true >(solver, initial_curr_policy_profile, i, "Current Policy");
      evaluate_policies< false >(solver, initial_policy_profile, i);
#endif
   }
   evaluate_policies< false >(
      solver, players | utils::is_actual_player_filter, n_iters, "Final Policy");
   assert_optimal_policy_rps(solver);
}

TEST(RockPaperScissors, MCCFR_OS_lazy_alternating)
{
   auto
      [env,
       avg_tabular_policy_alex,
       avg_tabular_policy_bob,
       tabular_policy_alex,
       tabular_policy_bob,
       infostate_alex,
       infostate_bob,
       init_state] = setup_rps_test();

   auto root_state = std::make_unique< games::rps::State >();
   auto players = env.players(*root_state);

   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::lazy};

   auto solver = factory::make_mccfr< config >(
      std::move(env),
      std::move(root_state),
      std::unordered_map{
         std::pair{Player::alex, tabular_policy_alex}, std::pair{Player::bob, tabular_policy_bob}},
      std::unordered_map{
         std::pair{Player::alex, avg_tabular_policy_alex},
         std::pair{Player::bob, avg_tabular_policy_bob}},
      0.6,
      0);

   auto initial_curr_policy_profile = std::unordered_map{
      std::pair{Player::alex, normalize_state_policy(solver.policy().at(Player::alex).table())},
      std::pair{Player::bob, normalize_state_policy(solver.policy().at(Player::bob).table())}};

   auto initial_policy_profile = std::unordered_map{
      std::pair{
         Player::alex,
         normalize_state_policy(solver.average_policy().at(Player::alex).table())},
      std::pair{
         Player::bob, normalize_state_policy(solver.average_policy().at(Player::bob).table())}};

   size_t n_iters = 40000;
   for(size_t i = 0; i < n_iters; i++) {
      solver.iterate(1);
#ifndef NDEBUG
      evaluate_policies< true >(solver, initial_curr_policy_profile, i, "Current Policy");
      evaluate_policies< false >(solver, initial_policy_profile, i);
#endif
   }
   evaluate_policies< false >(
      solver, players | utils::is_actual_player_filter, n_iters, "Final Policy");
   assert_optimal_policy_rps(solver);
}

TEST(RockPaperScissors, MCCFR_OS_lazy_simultaneous)
{
   auto
      [env,
       avg_tabular_policy_alex,
       avg_tabular_policy_bob,
       tabular_policy_alex,
       tabular_policy_bob,
       infostate_alex,
       infostate_bob,
       init_state] = setup_rps_test();

   auto root_state = std::make_unique< games::rps::State >();
   auto players = env.players(*root_state);

   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::simultaneous,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::lazy};

   auto solver = factory::make_mccfr< config >(
      std::move(env),
      std::move(root_state),
      std::unordered_map{
         std::pair{Player::alex, tabular_policy_alex}, std::pair{Player::bob, tabular_policy_bob}},
      std::unordered_map{
         std::pair{Player::alex, avg_tabular_policy_alex},
         std::pair{Player::bob, avg_tabular_policy_bob}},
      0.6,
      0);

   auto initial_curr_policy_profile = std::unordered_map{
      std::pair{Player::alex, normalize_state_policy(solver.policy().at(Player::alex).table())},
      std::pair{Player::bob, normalize_state_policy(solver.policy().at(Player::bob).table())}};

   auto initial_policy_profile = std::unordered_map{
      std::pair{
         Player::alex,
         normalize_state_policy(solver.average_policy().at(Player::alex).table())},
      std::pair{
         Player::bob, normalize_state_policy(solver.average_policy().at(Player::bob).table())}};

   size_t n_iters = 40000;
   for(size_t i = 0; i < n_iters; i++) {
      solver.iterate(1);
#ifndef NDEBUG
      evaluate_policies< true >(solver, initial_curr_policy_profile, i, "Current Policy");
      evaluate_policies< false >(solver, initial_policy_profile, i);
#endif
   }
   evaluate_policies< false >(
      solver, players | utils::is_actual_player_filter, n_iters, "Final Policy");
   assert_optimal_policy_rps(solver);
}

TEST(RockPaperScissors, MCCFR_OS_stochastic_alternating)
{
   auto
      [env,
       avg_tabular_policy_alex,
       avg_tabular_policy_bob,
       tabular_policy_alex,
       tabular_policy_bob,
       infostate_alex,
       infostate_bob,
       init_state] = setup_rps_test();

   auto root_state = std::make_unique< games::rps::State >();
   auto players = env.players(*root_state);

   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::stochastic};

   auto solver = factory::make_mccfr< config >(
      std::move(env),
      std::move(root_state),
      std::unordered_map{
         std::pair{Player::alex, tabular_policy_alex}, std::pair{Player::bob, tabular_policy_bob}},
      std::unordered_map{
         std::pair{Player::alex, avg_tabular_policy_alex},
         std::pair{Player::bob, avg_tabular_policy_bob}},
      0.6,
      0);

   auto initial_curr_policy_profile = std::unordered_map{
      std::pair{Player::alex, normalize_state_policy(solver.policy().at(Player::alex).table())},
      std::pair{Player::bob, normalize_state_policy(solver.policy().at(Player::bob).table())}};

   auto initial_policy_profile = std::unordered_map{
      std::pair{
         Player::alex,
         normalize_state_policy(solver.average_policy().at(Player::alex).table())},
      std::pair{
         Player::bob, normalize_state_policy(solver.average_policy().at(Player::bob).table())}};

   size_t n_iters = 40000;
   for(size_t i = 0; i < n_iters; i++) {
      solver.iterate(1);
#ifndef NDEBUG
      evaluate_policies< true >(solver, initial_curr_policy_profile, i, "Current Policy");
      evaluate_policies< false >(solver, initial_policy_profile, i);
#endif
   }
   evaluate_policies< false >(
      solver, players | utils::is_actual_player_filter, n_iters, "Final Policy");
   assert_optimal_policy_rps(solver);
}

TEST(RockPaperScissors, MCCFR_OS_stochastic_simultaneous)
{
   auto
      [env,
       avg_tabular_policy_alex,
       avg_tabular_policy_bob,
       tabular_policy_alex,
       tabular_policy_bob,
       infostate_alex,
       infostate_bob,
       init_state] = setup_rps_test();

   auto root_state = std::make_unique< games::rps::State >();
   auto players = env.players(*root_state);

   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::simultaneous,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::stochastic};

   auto solver = factory::make_mccfr< config >(
      std::move(env),
      std::move(root_state),
      std::unordered_map{
         std::pair{Player::alex, tabular_policy_alex}, std::pair{Player::bob, tabular_policy_bob}},
      std::unordered_map{
         std::pair{Player::alex, avg_tabular_policy_alex},
         std::pair{Player::bob, avg_tabular_policy_bob}},
      0.6,
      0);

   auto initial_curr_policy_profile = std::unordered_map{
      std::pair{Player::alex, normalize_state_policy(solver.policy().at(Player::alex).table())},
      std::pair{Player::bob, normalize_state_policy(solver.policy().at(Player::bob).table())}};

   auto initial_policy_profile = std::unordered_map{
      std::pair{
         Player::alex,
         normalize_state_policy(solver.average_policy().at(Player::alex).table())},
      std::pair{
         Player::bob, normalize_state_policy(solver.average_policy().at(Player::bob).table())}};

   size_t n_iters = 40000;
   for(size_t i = 0; i < n_iters; i++) {
      solver.iterate(1);
#ifndef NDEBUG
      evaluate_policies< true >(solver, initial_curr_policy_profile, i, "Current Policy");
      evaluate_policies< false >(solver, initial_policy_profile, i);
#endif
   }
   evaluate_policies< false >(
      solver, players | utils::is_actual_player_filter, n_iters, "Final Policy");
   assert_optimal_policy_rps(solver);
}

TEST(RockPaperScissors, MCCFR_ES_stochastic)
{
   auto
      [env,
       avg_tabular_policy_alex,
       avg_tabular_policy_bob,
       tabular_policy_alex,
       tabular_policy_bob,
       infostate_alex,
       infostate_bob,
       init_state] = setup_rps_test();

   auto root_state = std::make_unique< games::rps::State >();
   auto players = env.players(*root_state);

   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::external_sampling,
      .weighting = rm::MCCFRWeightingMode::stochastic};

   auto solver = factory::make_mccfr< config >(
      std::move(env),
      std::move(root_state),
      std::unordered_map{
         std::pair{Player::alex, tabular_policy_alex}, std::pair{Player::bob, tabular_policy_bob}},
      std::unordered_map{
         std::pair{Player::alex, avg_tabular_policy_alex},
         std::pair{Player::bob, avg_tabular_policy_bob}},
      0.6,
      0);

   auto initial_curr_policy_profile = std::unordered_map{
      std::pair{Player::alex, normalize_state_policy(solver.policy().at(Player::alex).table())},
      std::pair{Player::bob, normalize_state_policy(solver.policy().at(Player::bob).table())}};

   auto initial_policy_profile = std::unordered_map{
      std::pair{
         Player::alex,
         normalize_state_policy(solver.average_policy().at(Player::alex).table())},
      std::pair{
         Player::bob, normalize_state_policy(solver.average_policy().at(Player::bob).table())}};

   size_t n_iters = 20000;
   for(size_t i = 0; i < n_iters; i++) {
      solver.iterate(1);
#ifndef NDEBUG
      evaluate_policies< true >(solver, initial_curr_policy_profile, i, "Current Policy");
      evaluate_policies< false >(solver, initial_policy_profile, i);
#endif
   }
   evaluate_policies< false >(
      solver, players | utils::is_actual_player_filter, n_iters, "Final Policy");
   assert_optimal_policy_rps(solver);
}

// TEST_F(StrategoState3x3, vanilla_cfr_usage_stratego)
//{
//    //      auto env = std::make_shared< Environment >(std::make_unique< Logic >());
//    games::stratego::Environment env{std::make_unique< games::stratego::Logic >()};
//    UniformPolicy uniform_policy = factory::make_uniform_policy<
//       games::stratego::InfoState,
//       HashmapActionPolicy< games::stratego::Action > >();
//
//    auto tabular_policy = factory::make_tabular_policy(
//       std::unordered_map<
//          games::stratego::InfoState,
//          HashmapActionPolicy< games::stratego::Action > >{});
//
//    constexpr rm::CFRConfig cfr_config{.alternating_updates = false, .store_public_states =
//    false};
//
//    auto solver = factory::make_cfr_vanilla< cfr_config, true >(
//       std::move(env),
//       std::make_unique< State >(std::move(state)),
//       tabular_policy,
//       std::move(uniform_policy));
//    solver.initialize();
//    const auto& curr_policy = *solver.iterate(100);
//    std::cout << curr_policy.at(Player::alex).table().size();
//    LOGD2("Table size", curr_policy.at(Player::alex).table().size());
// }

// template < typename IntegralConstantType >
// class MCCFRTest: public testing::Test {
//   public:
//    constexpr static auto cfg = IntegralConstantType::value;
// };
//
// using MCCFRConfigurations = ::testing::Types<
//    std::integral_constant<
//       rm::MCCFRConfig,
//       rm::MCCFRConfig{
//          .update_mode = rm::UpdateMode::alternating,
//          .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
//          .weighting = rm::MCCFRWeightingMode::optimistic} >,
//    std::integral_constant<
//       rm::MCCFRConfig,
//       rm::MCCFRConfig{
//          .update_mode = rm::UpdateMode::simultaneous,
//          .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
//          .weighting = rm::MCCFRWeightingMode::optimistic} >,
//    std::integral_constant<
//       rm::MCCFRConfig,
//       rm::MCCFRConfig{
//          .update_mode = rm::UpdateMode::alternating,
//          .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
//          .weighting = rm::MCCFRWeightingMode::lazy} >,
//    std::integral_constant<
//       rm::MCCFRConfig,
//       rm::MCCFRConfig{
//          .update_mode = rm::UpdateMode::simultaneous,
//          .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
//          .weighting = rm::MCCFRWeightingMode::lazy} >,
//    std::integral_constant<
//       rm::MCCFRConfig,
//       rm::MCCFRConfig{
//          .update_mode = rm::UpdateMode::alternating,
//          .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
//          .weighting = rm::MCCFRWeightingMode::stochastic} >,
//    std::integral_constant<
//       rm::MCCFRConfig,
//       rm::MCCFRConfig{
//          .update_mode = rm::UpdateMode::simultaneous,
//          .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
//          .weighting = rm::MCCFRWeightingMode::stochastic} >,
//    std::integral_constant<
//       rm::MCCFRConfig,
//       rm::MCCFRConfig{
//          .update_mode = rm::UpdateMode::alternating,
//          .algorithm = rm::MCCFRAlgorithmMode::external_sampling,
//          .weighting = rm::MCCFRWeightingMode::stochastic} > >;
//
// TYPED_TEST_SUITE(MCCFRTest, MCCFRConfigurations);
//
//
// TYPED_TEST(MCCFRTest, DoesBlah) {
//    ASSERT_TRUE(true);
// }
