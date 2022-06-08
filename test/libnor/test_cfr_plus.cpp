#include <gtest/gtest.h>

#include <unordered_map>

#include "../games/stratego/fixtures.hpp"
#include "nor/nor.hpp"
#include "nor/wrappers.hpp"
#include "utils_for_testing.hpp"

using namespace nor;

TEST(KuhnPoker, CFR_PLUS)
{
   games::kuhn::Environment env{};
   auto players = env.players();
   auto avg_tabular_policy = rm::factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      rm::factory::
         make_zero_policy< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >());

   auto tabular_policy = rm::factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      rm::factory::make_uniform_policy<
         games::kuhn::Infostate,
         HashmapActionPolicy< games::kuhn::Action > >());

   auto solver = rm::factory::make_cfrplus< true >(
      std::move(env), std::make_unique< games::kuhn::State >(), tabular_policy, avg_tabular_policy);

   auto initial_curr_policy_profile = std::unordered_map{
      std::pair{Player::alex, rm::normalize_state_policy(solver.policy().at(Player::alex).table())},
      std::pair{Player::bob, rm::normalize_state_policy(solver.policy().at(Player::bob).table())}};

   auto initial_policy_profile = std::unordered_map{
      std::pair{
         Player::alex,
         rm::normalize_state_policy(solver.average_policy().at(Player::alex).table())},
      std::pair{
         Player::bob, rm::normalize_state_policy(solver.average_policy().at(Player::bob).table())}};

   size_t n_iters = 10000;
   for(size_t i = 0; i < n_iters; i++) {
      solver.iterate(1);
#ifndef NDEBUG
      evaluate_policies< true >(solver, initial_curr_policy_profile, i, "Current Policy");
      evaluate_policies< false >(solver, initial_policy_profile, i);
#endif
   }
   evaluate_policies< false >(
      solver, players | utils::is_nonchance_player_filter, n_iters, "Final Policy");
   auto game_value_map = solver.game_value();
   double alex_true_game_value = -1. / 18.;
   ASSERT_NEAR(game_value_map.get()[Player::alex], alex_true_game_value, 1e-3);
   assert_optimal_policy_kuhn(solver, env);
}

TEST(RockPaperScissors, CFR_PLUS)
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

   auto players = env.players();

   auto solver = rm::factory::make_cfrplus(
      std::move(env),
      std::make_unique< games::rps::State >(),
      std::unordered_map{
         std::pair{Player::alex, tabular_policy_alex}, std::pair{Player::bob, tabular_policy_bob}},
      std::unordered_map{
         std::pair{Player::alex, avg_tabular_policy_alex},
         std::pair{Player::bob, avg_tabular_policy_bob}});

   auto initial_curr_policy_profile = std::unordered_map{
      std::pair{Player::alex, rm::normalize_state_policy(solver.policy().at(Player::alex).table())},
      std::pair{Player::bob, rm::normalize_state_policy(solver.policy().at(Player::bob).table())}};

   auto initial_policy_profile = std::unordered_map{
      std::pair{
         Player::alex,
         rm::normalize_state_policy(solver.average_policy().at(Player::alex).table())},
      std::pair{
         Player::bob, rm::normalize_state_policy(solver.average_policy().at(Player::bob).table())}};

   size_t n_iters = 20000;
   for(size_t i = 0; i < n_iters; i++) {
      solver.iterate(1);
#ifndef NDEBUG
      evaluate_policies< true >(solver, initial_curr_policy_profile, i, "Current Policy");
      evaluate_policies< false >(solver, initial_policy_profile, i);
#endif
   }
   evaluate_policies< false >(
      solver, players | utils::is_nonchance_player_filter, n_iters, "Final Policy");
   ASSERT_NEAR(solver.game_value().get()[Player::alex], 0., 1e-3);
   auto final_policy = solver.average_policy().at(Player::alex).table();
   for(const auto& [state, action_policy] : final_policy) {
      for(const auto& [action, prob] : rm::normalize_action_policy(action_policy)) {
         ASSERT_NEAR(prob, 1. / 3., 1e-3);
      }
   }
   final_policy = solver.average_policy().at(Player::bob).table();
   for(const auto& [state, action_policy] : final_policy) {
      for(const auto& [action, prob] : rm::normalize_action_policy(action_policy)) {
         ASSERT_NEAR(prob, 1. / 3., 1e-3);
      }
   }
}

// TEST_F(StrategoState3x3, vanilla_cfr)
//{
//    //      auto env = std::make_shared< Environment >(std::make_unique< Logic >());
//    games::stratego::Environment env{std::make_unique< games::stratego::Logic >()};
//    UniformPolicy uniform_policy = rm::factory::make_uniform_policy<
//       games::stratego::InfoState,
//       HashmapActionPolicy< games::stratego::Action > >();
//
//    auto tabular_policy = rm::factory::make_tabular_policy(
//       std::unordered_map<
//          games::stratego::InfoState,
//          HashmapActionPolicy< games::stratego::Action > >{});
//
//    constexpr rm::CFRConfig cfr_config{.update_mode = rm::UpdateMode::alternating};
//
//    auto solver = rm::factory::make_vanilla< cfr_config, true >(
//       std::move(env),
//       std::make_unique< State >(std::move(state)),
//       tabular_policy,
//       std::move(uniform_policy));
//    solver.initialize();
//    const auto& curr_policy = *solver.iterate(100);
//    std::cout << curr_policy.at(Player::alex).table().size();
//    LOGD2("Table size", curr_policy.at(Player::alex).table().size());
// }