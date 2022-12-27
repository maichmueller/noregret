#include <gtest/gtest.h>

#include <unordered_map>

#include "../games/stratego/fixtures.hpp"
#include "cfr_run_funcs.hpp"
#include "nor/env.hpp"
#include "nor/nor.hpp"
#include "rm_specific_testing_utils.hpp"
using namespace nor;

TEST(KuhnPoker, CFR_DISCOUNTED_alternating)
{
   run_cfr_on_kuhn_poker< rm::CFRDiscountedConfig{.update_mode = rm::UpdateMode::alternating} >();
}

TEST(KuhnPoker, CFR_DISCOUNTED_simultaneous)
{
   run_cfr_on_kuhn_poker< rm::CFRDiscountedConfig{.update_mode = rm::UpdateMode::simultaneous} >();
}

TEST(RockPaperScissors, CFR_DISCOUNTED_alternating)
{
   run_cfr_on_rps< rm::CFRDiscountedConfig{.update_mode = rm::UpdateMode::alternating} >();
}

TEST(RockPaperScissors, CFR_DISCOUNTED_simultaneous)
{
   run_cfr_on_rps< rm::CFRDiscountedConfig{.update_mode = rm::UpdateMode::simultaneous} >();
}

// TEST_F(StrategoState3x3, CFR_DISCOUNTED)
//{
//
//    std::cout << "Before anything...\n" << std::endl;
//    //      auto env = std::make_shared< Environment >(std::make_unique< Logic >());
//    games::stratego::Environment env{std::make_unique< games::stratego::Logic >()};
//    auto players = env.players();
//
//    auto avg_tabular_policy = factory::make_tabular_policy(
//       std::unordered_map<
//          games::stratego::Infostate,
//          HashmapActionPolicy< games::stratego::Action > >{});
//
//    auto tabular_policy = factory::make_tabular_policy(
//       std::unordered_map<
//          games::stratego::Infostate,
//          HashmapActionPolicy< games::stratego::Action > >{});
//
//    constexpr rm::CFRDiscountedConfig cfr_config{.update_mode = rm::UpdateMode::alternating};
//
//    auto solver = factory::make_cfr_discounted< cfr_config, true >(
//       std::move(env),
//       std::make_unique< games::stratego::State >(std::move(state)),
//       tabular_policy,
//       avg_tabular_policy);
//
//    auto initial_curr_policy_profile = std::unordered_map{
//       std::pair{Player::alex,
//       normalize_state_policy(solver.policy().at(Player::alex).table())},
//       std::pair{Player::bob,
//       normalize_state_policy(solver.policy().at(Player::bob).table())}};
//
//    auto initial_policy_profile = std::unordered_map{
//       std::pair{
//          Player::alex,
//          normalize_state_policy(solver.average_policy().at(Player::alex).table())},
//       std::pair{
//          Player::bob,
//          normalize_state_policy(solver.average_policy().at(Player::bob).table())}};
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
// TEST_F(StrategoState5x5, CFR_DISCOUNTED)
//{
//    //      auto env = std::make_shared< Environment >(std::make_unique< Logic >());
//    games::stratego::Environment env{std::make_unique< games::stratego::Logic >()};
//    auto players = env.players();
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
//    auto solver = factory::make_cfr_discounted< cfr_config, true >(
//       std::move(env),
//       std::make_unique< games::stratego::State >(std::move(state)),
//       tabular_policy,
//       avg_tabular_policy);
//
//    auto initial_curr_policy_profile = std::unordered_map{
//       std::pair{Player::alex,
//       normalize_state_policy(solver.policy().at(Player::alex).table())},
//       std::pair{Player::bob,
//       normalize_state_policy(solver.policy().at(Player::bob).table())}};
//
//    auto initial_policy_profile = std::unordered_map{
//       std::pair{
//          Player::alex,
//          normalize_state_policy(solver.average_policy().at(Player::alex).table())},
//       std::pair{
//          Player::bob,
//          normalize_state_policy(solver.average_policy().at(Player::bob).table())}};
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
