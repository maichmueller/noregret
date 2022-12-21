#include <gtest/gtest.h>

#include <unordered_map>

#include "../games/stratego/fixtures.hpp"
#include "nor/env.hpp"
#include "nor/nor.hpp"
#include "rm_specific_testing_utils.hpp"

using namespace nor;

template < rm::CFRDiscountedConfig config >
void run_kuhn_poker()
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

   auto solver = factory::make_cfr_linear< config, true >(
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

TEST(KuhnPoker, CFR_LINEAR_alternating)
{
   constexpr rm::CFRDiscountedConfig cfr_config{.update_mode = rm::UpdateMode::alternating};
   run_kuhn_poker< cfr_config >();
}

TEST(KuhnPoker, CFR_LINEAR_simultaneous)
{
   constexpr rm::CFRDiscountedConfig cfr_config{.update_mode = rm::UpdateMode::simultaneous};
   run_kuhn_poker< cfr_config >();
}

template < rm::CFRDiscountedConfig config >
void run_rockpaperscissors()
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

   auto solver = factory::make_cfr_linear< config >(
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

TEST(RockPaperScissors, CFR_LINEAR_alternating)
{
   constexpr rm::CFRDiscountedConfig cfr_config{.update_mode = rm::UpdateMode::alternating};
   run_rockpaperscissors< cfr_config >();
}

TEST(RockPaperScissors, CFR_LINEAR_simultaneous)
{
   constexpr rm::CFRDiscountedConfig cfr_config{.update_mode = rm::UpdateMode::simultaneous};
   run_rockpaperscissors< cfr_config >();
}

// TEST_F(StrategoState3x3, CFR_LINEAR)
//{
//
//    std::cout << "Before anything...\n" << std::endl;
//    //      auto env = std::make_shared< Environment >(std::make_unique< Logic >());
//    games::stratego::Environment env{std::make_unique< games::stratego::Logic >()};
//
//    auto root_state = std::make_unique< games::kuhn::State >();
//    auto players = env.players(*root_state);
//
//    auto avg_tabular_policy = factory::make_tabular_policy(
//       std::unordered_map<
//          games::stratego::Infostate,
//          HashmapActionPolicy< games::stratego::Action > >{},
//       factory::make_zero_policy<
//          games::stratego::Infostate,
//          HashmapActionPolicy< games::stratego::Action > >());
//
//    auto tabular_policy = factory::make_tabular_policy(
//       std::unordered_map<
//          games::stratego::Infostate,
//          HashmapActionPolicy< games::stratego::Action > >{},
//       factory::make_uniform_policy<
//          games::stratego::Infostate,
//          HashmapActionPolicy< games::stratego::Action > >());
//
//    constexpr rm::CFRDiscountedConfig cfr_config{.update_mode = rm::UpdateMode::alternating};
//
//    auto solver = factory::make_cfr_linear< cfr_config, true >(
//       std::move(env),
//       std::make_unique< games::stratego::State >(std::move(state)),
//       tabular_policy,
//       avg_tabular_policy);
//
//    auto initial_curr_policy_profile = std::unordered_map{
//       std::pair{Player::alex, normalize_state_policy(solver.policy().at(Player::alex).table())},
//       std::pair{Player::bob, normalize_state_policy(solver.policy().at(Player::bob).table())}};
//
//    auto initial_policy_profile = std::unordered_map{
//       std::pair{
//          Player::alex,
//          normalize_state_policy(solver.average_policy().at(Player::alex).table())},
//       std::pair{
//          Player::bob, normalize_state_policy(solver.average_policy().at(Player::bob).table())}};
//
//       std::cout << "Before Iterations\n";
//    size_t n_iters = 15000;
//    for(size_t i = 0; i < n_iters; i++) {
//       std::cout << "Iteration " << i << std::endl;
//       solver.iterate(1);
// #ifndef NDEBUG
//       evaluate_policies< true >(solver, initial_curr_policy_profile, i, "Current Policy");
//       evaluate_policies< false >(solver, initial_policy_profile, i);
// #endif
//    }
//    evaluate_policies< false >(
//       solver, players | utils::is_nonchance_player_filter, n_iters, "Final Policy");
// }
//
//
// TEST_F(StrategoState5x5, CFR_LINEAR)
//{
//    //      auto env = std::make_shared< Environment >(std::make_unique< Logic >());
//    games::stratego::Environment env{std::make_unique< games::stratego::Logic >()};
//
//    auto root_state = std::make_unique< games::stratego::State >();
//    auto players = env.players(*root_state);
//
//    auto avg_tabular_policy = factory::make_tabular_policy(
//       std::unordered_map<
//          games::stratego::Infostate,
//          HashmapActionPolicy< games::stratego::Action > >{},
//       factory::make_zero_policy<
//          games::stratego::Infostate,
//          HashmapActionPolicy< games::stratego::Action > >());
//
//    auto tabular_policy = factory::make_tabular_policy(
//       std::unordered_map<
//          games::stratego::Infostate,
//          HashmapActionPolicy< games::stratego::Action > >{},
//       factory::make_uniform_policy<
//          games::stratego::Infostate,
//          HashmapActionPolicy< games::stratego::Action > >());
//
//    constexpr rm::CFRDiscountedConfig cfr_config{.update_mode = rm::UpdateMode::alternating};
//
//    auto solver = factory::make_cfr_linear< cfr_config, true >(
//       std::move(env),
//       std::make_unique< games::stratego::State >(std::move(state)),
//       tabular_policy,
//       avg_tabular_policy);
//
//    auto initial_curr_policy_profile = std::unordered_map{
//       std::pair{Player::alex, normalize_state_policy(solver.policy().at(Player::alex).table())},
//       std::pair{Player::bob, normalize_state_policy(solver.policy().at(Player::bob).table())}};
//
//    auto initial_policy_profile = std::unordered_map{
//       std::pair{
//          Player::alex,
//          normalize_state_policy(solver.average_policy().at(Player::alex).table())},
//       std::pair{
//          Player::bob, normalize_state_policy(solver.average_policy().at(Player::bob).table())}};
//
//    size_t n_iters = 15000;
//    for(size_t i = 0; i < n_iters; i++) {
//       std::cout << "Iteration " << i << std::endl;
//       solver.iterate(1);
// #ifndef NDEBUG
//       evaluate_policies< true >(solver, initial_curr_policy_profile, i, "Current Policy");
//       evaluate_policies< false >(solver, initial_policy_profile, i);
// #endif
//    }
//    evaluate_policies< false >(
//       solver, players | utils::is_nonchance_player_filter, n_iters, "Final Policy");
// }
