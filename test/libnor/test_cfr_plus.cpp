#include <gtest/gtest.h>

#include <unordered_map>

#include "../games/stratego/fixtures.hpp"
#include "nor/env.hpp"
#include "nor/nor.hpp"
#include "rm_specific_testing_utils.hpp"

using namespace nor;

TEST(KuhnPoker, CFR_PLUS)
{
   games::kuhn::Environment env{};
   auto root_state = std::make_unique< games::kuhn::State >();
   auto players = env.players(*root_state);

   auto avg_tabular_policy = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{}
   );

   auto tabular_policy = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{}
   );

   auto solver = factory::make_cfr_plus< true >(
      std::move(env), std::move(root_state), tabular_policy, avg_tabular_policy
   );

   auto initial_curr_policy_profile = std::unordered_map{
      std::pair{Player::alex, normalize_state_policy(solver.policy().at(Player::alex).table())},
      std::pair{Player::bob, normalize_state_policy(solver.policy().at(Player::bob).table())}};

   auto initial_policy_profile = std::unordered_map{
      std::pair{
         Player::alex, normalize_state_policy(solver.average_policy().at(Player::alex).table())},
      std::pair{
         Player::bob, normalize_state_policy(solver.average_policy().at(Player::bob).table())}};

   size_t max_iters = 1e5;
   size_t n_iters = 0;
   double expl = std::numeric_limits< double >::max();
   while(expl > EXPLOITABILITY_THRESHOLD or n_iters >= max_iters) {
      solver.iterate(1);
      n_iters++;
#ifndef NDEBUG
      evaluate_policies< true >(solver, initial_curr_policy_profile, i, "Current Policy");
      evaluate_policies< false >(solver, initial_policy_profile, i);
#endif
      if(n_iters % 10 == 0) {
         expl = exploitability(
            env,
            games::kuhn::State{},
            std::unordered_map{
               std::pair{
                  Player::alex, normalize_state_policy(solver.average_policy().at(Player::alex))},
               std::pair{
                  Player::bob, normalize_state_policy(solver.average_policy().at(Player::bob))}}
         );
      }
   }
   evaluate_policies< false >(
      solver, players | utils::is_actual_player_filter, n_iters, "Final Policy"
   );
   EXPECT_TRUE(expl <= EXPLOITABILITY_THRESHOLD);
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

   auto root_state = std::make_unique< games::rps::State >();
   auto players = env.players(*root_state);

   auto solver = factory::make_cfr_plus(
      std::move(env),
      std::move(root_state),
      std::unordered_map{
         std::pair{Player::alex, tabular_policy_alex}, std::pair{Player::bob, tabular_policy_bob}},
      std::unordered_map{
         std::pair{Player::alex, avg_tabular_policy_alex},
         std::pair{Player::bob, avg_tabular_policy_bob}}
   );

   auto initial_curr_policy_profile = std::unordered_map{
      std::pair{Player::alex, normalize_state_policy(solver.policy().at(Player::alex).table())},
      std::pair{Player::bob, normalize_state_policy(solver.policy().at(Player::bob).table())}};

   auto initial_policy_profile = std::unordered_map{
      std::pair{
         Player::alex, normalize_state_policy(solver.average_policy().at(Player::alex).table())},
      std::pair{
         Player::bob, normalize_state_policy(solver.average_policy().at(Player::bob).table())}};

   size_t max_iters = 1e5;
   size_t n_iters = 0;
   double expl = std::numeric_limits< double >::max();
   while(expl > EXPLOITABILITY_THRESHOLD or n_iters >= max_iters) {
      solver.iterate(1);
      n_iters++;
#ifndef NDEBUG
      evaluate_policies< true >(solver, initial_curr_policy_profile, i, "Current Policy");
      evaluate_policies< false >(solver, initial_policy_profile, i);
#endif
      if(n_iters % 10 == 0) {
         expl = exploitability(
            env,
            games::rps::State{},
            std::unordered_map{
               std::pair{
                  Player::alex, normalize_state_policy(solver.average_policy().at(Player::alex))},
               std::pair{
                  Player::bob, normalize_state_policy(solver.average_policy().at(Player::bob))}}
         );
      }
   }
   evaluate_policies< false >(
      solver, players | utils::is_actual_player_filter, n_iters, "Final Policy"
   );
   EXPECT_TRUE(expl <= EXPLOITABILITY_THRESHOLD);
}

// TEST_F(StrategoState3x3, vanilla_cfr)
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
//    constexpr rm::CFRConfig cfr_config{.update_mode = rm::UpdateMode::alternating};
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
