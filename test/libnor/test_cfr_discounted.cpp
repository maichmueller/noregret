#include <gtest/gtest.h>

#include <unordered_map>

#include "../games/stratego/fixtures.hpp"
#include "nor/env.hpp"
#include "nor/nor.hpp"
#include "utils_for_testing.hpp"

using namespace nor;

TEST(KuhnPoker, CFR_DISCOUNTED_alternating)
{
   games::kuhn::Environment env{};

   auto root_state = std::make_unique< games::kuhn::State >();
   auto players = env.players(*root_state);

   auto avg_tabular_policy = rm::factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      rm::factory::
         make_zero_policy< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >());

   auto tabular_policy = rm::factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      rm::factory::make_uniform_policy<
         games::kuhn::Infostate,
         HashmapActionPolicy< games::kuhn::Action > >());

   constexpr rm::CFRDiscountedConfig cfr_config{.update_mode = rm::UpdateMode::alternating};

   auto solver = rm::factory::make_cfr_discounted< cfr_config, true >(
      std::move(env), std::make_unique< games::kuhn::State >(), tabular_policy, avg_tabular_policy);

   auto initial_curr_policy_profile = std::unordered_map{
      std::pair{Player::alex, normalize_state_policy(solver.policy().at(Player::alex).table())},
      std::pair{Player::bob, normalize_state_policy(solver.policy().at(Player::bob).table())}};

   auto initial_policy_profile = std::unordered_map{
      std::pair{
         Player::alex,
         normalize_state_policy(solver.average_policy().at(Player::alex).table())},
      std::pair{
         Player::bob, normalize_state_policy(solver.average_policy().at(Player::bob).table())}};

   size_t n_iters = 500;
   for(size_t i = 0; i < n_iters; i++) {
      solver.iterate(1);
#ifndef NDEBUG
      evaluate_policies< true >(solver, initial_curr_policy_profile, i, "Current Policy");
      evaluate_policies< false >(solver, initial_policy_profile, i);
#endif
   }
   evaluate_policies< false >(
      solver, players | utils::is_actual_player_filter, n_iters, "Final Policy");
   auto game_value_map = solver.game_value();
   double alex_true_game_value = -1. / 18.;
   ASSERT_NEAR(game_value_map.get()[Player::alex], alex_true_game_value, 1e-3);
   assert_optimal_policy_kuhn(solver, env);
}

TEST(KuhnPoker, CFR_DISCOUNTED_simultaneous)
{
   games::kuhn::Environment env{};

   auto root_state = std::make_unique< games::kuhn::State >();
   auto players = env.players(*root_state);

   auto avg_tabular_policy = rm::factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      rm::factory::
         make_zero_policy< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >());

   auto tabular_policy = rm::factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      rm::factory::make_uniform_policy<
         games::kuhn::Infostate,
         HashmapActionPolicy< games::kuhn::Action > >());

   constexpr rm::CFRDiscountedConfig cfr_config{.update_mode = rm::UpdateMode::simultaneous};

   auto solver = rm::factory::make_cfr_discounted< cfr_config, true >(
      std::move(env), std::make_unique< games::kuhn::State >(), tabular_policy, avg_tabular_policy);

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
   auto game_value_map = solver.game_value();
   double alex_true_game_value = -1. / 18.;
   ASSERT_NEAR(game_value_map.get()[Player::alex], alex_true_game_value, 1e-3);
   assert_optimal_policy_kuhn(solver, env);
}

TEST(RockPaperScissors, CFR_DISCOUNTED_alternating)
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

   constexpr rm::CFRDiscountedConfig cfr_config{.update_mode = rm::UpdateMode::alternating};

   auto solver = rm::factory::make_cfr_discounted< cfr_config >(
      std::move(env),
      std::move(root_state),
      std::unordered_map{
         std::pair{Player::alex, tabular_policy_alex}, std::pair{Player::bob, tabular_policy_bob}},
      std::unordered_map{
         std::pair{Player::alex, avg_tabular_policy_alex},
         std::pair{Player::bob, avg_tabular_policy_bob}});

   auto initial_curr_policy_profile = std::unordered_map{
      std::pair{Player::alex, normalize_state_policy(solver.policy().at(Player::alex).table())},
      std::pair{Player::bob, normalize_state_policy(solver.policy().at(Player::bob).table())}};
   auto initial_policy_profile = std::unordered_map{
      std::pair{
         Player::alex,
         normalize_state_policy(solver.average_policy().at(Player::alex).table())},
      std::pair{
         Player::bob, normalize_state_policy(solver.average_policy().at(Player::bob).table())}};

   size_t n_iters = 150;
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

TEST(RockPaperScissors, CFR_DISCOUNTED_simultaneous)
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

   constexpr rm::CFRDiscountedConfig cfr_config{.update_mode = rm::UpdateMode::simultaneous};

   auto solver = rm::factory::make_cfr_discounted< cfr_config >(
      std::move(env),
      std::move(root_state),
      std::unordered_map{
         std::pair{Player::alex, tabular_policy_alex}, std::pair{Player::bob, tabular_policy_bob}},
      std::unordered_map{
         std::pair{Player::alex, avg_tabular_policy_alex},
         std::pair{Player::bob, avg_tabular_policy_bob}});

   auto initial_curr_policy_profile = std::unordered_map{
      std::pair{Player::alex, normalize_state_policy(solver.policy().at(Player::alex).table())},
      std::pair{Player::bob, normalize_state_policy(solver.policy().at(Player::bob).table())}};
   auto initial_policy_profile = std::unordered_map{
      std::pair{
         Player::alex,
         normalize_state_policy(solver.average_policy().at(Player::alex).table())},
      std::pair{
         Player::bob, normalize_state_policy(solver.average_policy().at(Player::bob).table())}};

   size_t n_iters = 70;
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

// TEST_F(StrategoState3x3, CFR_DISCOUNTED)
//{
//
//    std::cout << "Before anything...\n" << std::endl;
//    //      auto env = std::make_shared< Environment >(std::make_unique< Logic >());
//    games::stratego::Environment env{std::make_unique< games::stratego::Logic >()};
//    auto players = env.players();
//
//    auto avg_tabular_policy = rm::factory::make_tabular_policy(
//       std::unordered_map<
//          games::stratego::Infostate,
//          HashmapActionPolicy< games::stratego::Action > >{},
//       rm::factory::make_zero_policy<
//          games::stratego::Infostate,
//          HashmapActionPolicy< games::stratego::Action > >());
//
//    auto tabular_policy = rm::factory::make_tabular_policy(
//       std::unordered_map<
//          games::stratego::Infostate,
//          HashmapActionPolicy< games::stratego::Action > >{},
//       rm::factory::make_uniform_policy<
//          games::stratego::Infostate,
//          HashmapActionPolicy< games::stratego::Action > >());
//
//    constexpr rm::CFRDiscountedConfig cfr_config{.update_mode = rm::UpdateMode::alternating};
//
//    auto solver = rm::factory::make_cfr_discounted< cfr_config, true >(
//       std::move(env),
//       std::make_unique< games::stratego::State >(std::move(state)),
//       tabular_policy,
//       avg_tabular_policy);
//
//    auto initial_curr_policy_profile = std::unordered_map{
//       std::pair{Player::alex,
//       rm::normalize_state_policy(solver.policy().at(Player::alex).table())},
//       std::pair{Player::bob,
//       rm::normalize_state_policy(solver.policy().at(Player::bob).table())}};
//
//    auto initial_policy_profile = std::unordered_map{
//       std::pair{
//          Player::alex,
//          rm::normalize_state_policy(solver.average_policy().at(Player::alex).table())},
//       std::pair{
//          Player::bob,
//          rm::normalize_state_policy(solver.average_policy().at(Player::bob).table())}};
//
//       std::cout << "Before Iterations\n";
//    size_t n_iters = 15000;
//    for(size_t i = 0; i < n_iters; i++) {
//       std::cout << "Iteration " << i << std::endl;
//       solver.iterate(1);
//#ifndef NDEBUG
//       evaluate_policies< true >(solver, initial_curr_policy_profile, i, "Current Policy");
//       evaluate_policies< false >(solver, initial_policy_profile, i);
//#endif
//    }
//    evaluate_policies< false >(
//       solver, players | utils::is_nonchance_player_filter, n_iters, "Final Policy");
// }
//
//
// TEST_F(StrategoState5x5, CFR_DISCOUNTED)
//{
//    //      auto env = std::make_shared< Environment >(std::make_unique< Logic >());
//    games::stratego::Environment env{std::make_unique< games::stratego::Logic >()};
//    auto players = env.players();
//
//    auto avg_tabular_policy = rm::factory::make_tabular_policy(
//       std::unordered_map<
//          games::stratego::Infostate,
//          HashmapActionPolicy< games::stratego::Action > >{},
//       rm::factory::make_zero_policy<
//          games::stratego::Infostate,
//          HashmapActionPolicy< games::stratego::Action > >());
//
//    auto tabular_policy = rm::factory::make_tabular_policy(
//       std::unordered_map<
//          games::stratego::Infostate,
//          HashmapActionPolicy< games::stratego::Action > >{},
//       rm::factory::make_uniform_policy<
//          games::stratego::Infostate,
//          HashmapActionPolicy< games::stratego::Action > >());
//
//    constexpr rm::CFRDiscountedConfig cfr_config{.update_mode = rm::UpdateMode::alternating};
//
//    auto solver = rm::factory::make_cfr_discounted< cfr_config, true >(
//       std::move(env),
//       std::make_unique< games::stratego::State >(std::move(state)),
//       tabular_policy,
//       avg_tabular_policy);
//
//    auto initial_curr_policy_profile = std::unordered_map{
//       std::pair{Player::alex,
//       rm::normalize_state_policy(solver.policy().at(Player::alex).table())},
//       std::pair{Player::bob,
//       rm::normalize_state_policy(solver.policy().at(Player::bob).table())}};
//
//    auto initial_policy_profile = std::unordered_map{
//       std::pair{
//          Player::alex,
//          rm::normalize_state_policy(solver.average_policy().at(Player::alex).table())},
//       std::pair{
//          Player::bob,
//          rm::normalize_state_policy(solver.average_policy().at(Player::bob).table())}};
//
//    size_t n_iters = 15000;
//    for(size_t i = 0; i < n_iters; i++) {
//       std::cout << "Iteration " << i << std::endl;
//       solver.iterate(1);
//#ifndef NDEBUG
//       evaluate_policies< true >(solver, initial_curr_policy_profile, i, "Current Policy");
//       evaluate_policies< false >(solver, initial_policy_profile, i);
//#endif
//    }
//    evaluate_policies< false >(
//       solver, players | utils::is_nonchance_player_filter, n_iters, "Final Policy");
// }
