#include <gtest/gtest.h>

#include <unordered_map>

#include "../games/stratego/fixtures.hpp"
#include "nor/nor.hpp"
#include "nor/wrappers.hpp"
#include "utils_for_testing.hpp"

using namespace nor;

TEST(KuhnPoker, OS_MCCFR_optimistic_alternating)
{
   games::kuhn::Environment env{};

   auto avg_tabular_policy = rm::factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      rm::factory::
         make_zero_policy< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >());

   auto tabular_policy = rm::factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      rm::factory::make_uniform_policy<
         games::kuhn::Infostate,
         HashmapActionPolicy< games::kuhn::Action > >());

   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::optimistic};

   auto mccfr_runner = rm::factory::make_mccfr< config, true >(
      std::move(env),
      std::make_unique< games::kuhn::State >(),
      tabular_policy,
      avg_tabular_policy,
      0.6,
      0);

   auto initial_curr_policy_profile = std::unordered_map{
      std::pair{
         Player::alex, rm::normalize_state_policy(mccfr_runner.policy().at(Player::alex).table())},
      std::pair{
         Player::bob, rm::normalize_state_policy(mccfr_runner.policy().at(Player::bob).table())}};

   auto initial_policy_profile = std::unordered_map{
      std::pair{
         Player::alex,
         rm::normalize_state_policy(mccfr_runner.average_policy().at(Player::alex).table())},
      std::pair{
         Player::bob,
         rm::normalize_state_policy(mccfr_runner.average_policy().at(Player::bob).table())}};

   size_t n_iters = 200000;
   for(size_t i = 0; i < n_iters; i++) {
      mccfr_runner.iterate(1);
#ifndef NDEBUG
      evaluate_policies< true >(mccfr_runner, initial_curr_policy_profile, i, "Current Policy");
      evaluate_policies< false >(mccfr_runner, initial_policy_profile, i);
#endif
   }
   evaluate_policies< false >(mccfr_runner, initial_policy_profile, n_iters);
   assert_optimal_policy_kuhn(mccfr_runner, env, 0.05);
}

TEST(KuhnPoker, OS_MCCFR_optimistic_simultaneous)
{
   games::kuhn::Environment env{};

   auto avg_tabular_policy = rm::factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      rm::factory::
         make_zero_policy< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >());

   auto tabular_policy = rm::factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      rm::factory::make_uniform_policy<
         games::kuhn::Infostate,
         HashmapActionPolicy< games::kuhn::Action > >());

   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::simultaneous,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::optimistic};

   auto mccfr_runner = rm::factory::make_mccfr< config, true >(
      std::move(env),
      std::make_unique< games::kuhn::State >(),
      tabular_policy,
      avg_tabular_policy,
      0.6,
      0);

   auto initial_curr_policy_profile = std::unordered_map{
      std::pair{
         Player::alex, rm::normalize_state_policy(mccfr_runner.policy().at(Player::alex).table())},
      std::pair{
         Player::bob, rm::normalize_state_policy(mccfr_runner.policy().at(Player::bob).table())}};

   auto initial_policy_profile = std::unordered_map{
      std::pair{
         Player::alex,
         rm::normalize_state_policy(mccfr_runner.average_policy().at(Player::alex).table())},
      std::pair{
         Player::bob,
         rm::normalize_state_policy(mccfr_runner.average_policy().at(Player::bob).table())}};

   size_t n_iters = 200000;
   for(size_t i = 0; i < n_iters; i++) {
      mccfr_runner.iterate(1);
#ifndef NDEBUG
      evaluate_policies< true >(mccfr_runner, initial_curr_policy_profile, i, "Current Policy");
      evaluate_policies< false >(mccfr_runner, initial_policy_profile, i);
#endif
   }
   evaluate_policies< false >(mccfr_runner, initial_policy_profile, n_iters);
   assert_optimal_policy_kuhn(mccfr_runner, env, 0.05);
}

TEST(KuhnPoker, OS_MCCFR_lazy_alternating)
{
   games::kuhn::Environment env{};

   auto avg_tabular_policy = rm::factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      rm::factory::
         make_zero_policy< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >());

   auto tabular_policy = rm::factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      rm::factory::make_uniform_policy<
         games::kuhn::Infostate,
         HashmapActionPolicy< games::kuhn::Action > >());

   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::lazy};

   auto mccfr_runner = rm::factory::make_mccfr< config, true >(
      std::move(env),
      std::make_unique< games::kuhn::State >(),
      tabular_policy,
      avg_tabular_policy,
      0.6,
      0);

   auto initial_curr_policy_profile = std::unordered_map{
      std::pair{
         Player::alex, rm::normalize_state_policy(mccfr_runner.policy().at(Player::alex).table())},
      std::pair{
         Player::bob, rm::normalize_state_policy(mccfr_runner.policy().at(Player::bob).table())}};

   auto initial_policy_profile = std::unordered_map{
      std::pair{
         Player::alex,
         rm::normalize_state_policy(mccfr_runner.average_policy().at(Player::alex).table())},
      std::pair{
         Player::bob,
         rm::normalize_state_policy(mccfr_runner.average_policy().at(Player::bob).table())}};

   size_t n_iters = 200000;
   for(size_t i = 0; i < n_iters; i++) {
      mccfr_runner.iterate(1);
#ifndef NDEBUG
      evaluate_policies< true >(mccfr_runner, initial_curr_policy_profile, i, "Current Policy");
      evaluate_policies< false >(mccfr_runner, initial_policy_profile, i);
#endif
   }
   evaluate_policies< false >(mccfr_runner, initial_policy_profile, n_iters);
   assert_optimal_policy_kuhn(mccfr_runner, env, 0.05);
}

TEST(KuhnPoker, OS_MCCFR_lazy_simultaneous)
{
   games::kuhn::Environment env{};

   auto avg_tabular_policy = rm::factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      rm::factory::
         make_zero_policy< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >());

   auto tabular_policy = rm::factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      rm::factory::make_uniform_policy<
         games::kuhn::Infostate,
         HashmapActionPolicy< games::kuhn::Action > >());

   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::simultaneous,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::lazy};

   auto mccfr_runner = rm::factory::make_mccfr< config, true >(
      std::move(env),
      std::make_unique< games::kuhn::State >(),
      tabular_policy,
      avg_tabular_policy,
      0.6,
      0);

   auto initial_curr_policy_profile = std::unordered_map{
      std::pair{
         Player::alex, rm::normalize_state_policy(mccfr_runner.policy().at(Player::alex).table())},
      std::pair{
         Player::bob, rm::normalize_state_policy(mccfr_runner.policy().at(Player::bob).table())}};

   auto initial_policy_profile = std::unordered_map{
      std::pair{
         Player::alex,
         rm::normalize_state_policy(mccfr_runner.average_policy().at(Player::alex).table())},
      std::pair{
         Player::bob,
         rm::normalize_state_policy(mccfr_runner.average_policy().at(Player::bob).table())}};

   size_t n_iters = 200000;
   for(size_t i = 0; i < n_iters; i++) {
      mccfr_runner.iterate(1);
#ifndef NDEBUG
      evaluate_policies< true >(mccfr_runner, initial_curr_policy_profile, i, "Current Policy");
      evaluate_policies< false >(mccfr_runner, initial_policy_profile, i);
#endif
   }
   evaluate_policies< false >(mccfr_runner, initial_policy_profile, n_iters);
   assert_optimal_policy_kuhn(mccfr_runner, env, 0.05);
}

TEST(KuhnPoker, OS_MCCFR_stochastic_alternating)
{
   games::kuhn::Environment env{};

   auto avg_tabular_policy = rm::factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      rm::factory::
         make_zero_policy< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >());

   auto tabular_policy = rm::factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      rm::factory::make_uniform_policy<
         games::kuhn::Infostate,
         HashmapActionPolicy< games::kuhn::Action > >());

   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::stochastic};

   auto mccfr_runner = rm::factory::make_mccfr< config, true >(
      std::move(env),
      std::make_unique< games::kuhn::State >(),
      tabular_policy,
      avg_tabular_policy,
      0.6,
      0);

   auto initial_curr_policy_profile = std::unordered_map{
      std::pair{
         Player::alex, rm::normalize_state_policy(mccfr_runner.policy().at(Player::alex).table())},
      std::pair{
         Player::bob, rm::normalize_state_policy(mccfr_runner.policy().at(Player::bob).table())}};

   auto initial_policy_profile = std::unordered_map{
      std::pair{
         Player::alex,
         rm::normalize_state_policy(mccfr_runner.average_policy().at(Player::alex).table())},
      std::pair{
         Player::bob,
         rm::normalize_state_policy(mccfr_runner.average_policy().at(Player::bob).table())}};

   size_t n_iters = 200000;
   for(size_t i = 0; i < n_iters; i++) {
      mccfr_runner.iterate(1);
#ifndef NDEBUG
      evaluate_policies< true >(mccfr_runner, initial_curr_policy_profile, i, "Current Policy");
      evaluate_policies< false >(mccfr_runner, initial_policy_profile, i);
#endif
   }
   evaluate_policies< false >(mccfr_runner, initial_policy_profile, n_iters);
   assert_optimal_policy_kuhn(mccfr_runner, env, 0.05);
}


TEST(KuhnPoker, OS_MCCFR_stochastic_simultaneous)
{
   games::kuhn::Environment env{};

   auto avg_tabular_policy = rm::factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      rm::factory::
         make_zero_policy< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >());

   auto tabular_policy = rm::factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      rm::factory::make_uniform_policy<
         games::kuhn::Infostate,
         HashmapActionPolicy< games::kuhn::Action > >());

   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::simultaneous,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::stochastic};

   auto mccfr_runner = rm::factory::make_mccfr< config, true >(
      std::move(env),
      std::make_unique< games::kuhn::State >(),
      tabular_policy,
      avg_tabular_policy,
      0.6,
      0);

   auto initial_curr_policy_profile = std::unordered_map{
      std::pair{
         Player::alex, rm::normalize_state_policy(mccfr_runner.policy().at(Player::alex).table())},
      std::pair{
         Player::bob, rm::normalize_state_policy(mccfr_runner.policy().at(Player::bob).table())}};

   auto initial_policy_profile = std::unordered_map{
      std::pair{
         Player::alex,
         rm::normalize_state_policy(mccfr_runner.average_policy().at(Player::alex).table())},
      std::pair{
         Player::bob,
         rm::normalize_state_policy(mccfr_runner.average_policy().at(Player::bob).table())}};

   size_t n_iters = 200000;
   for(size_t i = 0; i < n_iters; i++) {
      mccfr_runner.iterate(1);
#ifndef NDEBUG
      evaluate_policies< true >(mccfr_runner, initial_curr_policy_profile, i, "Current Policy");
      evaluate_policies< false >(mccfr_runner, initial_policy_profile, i);
#endif
   }
   evaluate_policies< false >(mccfr_runner, initial_policy_profile, n_iters);
   assert_optimal_policy_kuhn(mccfr_runner, env, 0.05);
}

TEST(KuhnPoker, ES_MCCFR_stochastic)
{
   games::kuhn::Environment env{};

   auto avg_tabular_policy = rm::factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      rm::factory::
         make_zero_policy< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >());

   auto tabular_policy = rm::factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      rm::factory::make_uniform_policy<
         games::kuhn::Infostate,
         HashmapActionPolicy< games::kuhn::Action > >());

   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::external_sampling,
      .weighting = rm::MCCFRWeightingMode::stochastic};

   auto mccfr_runner = rm::factory::make_mccfr< config, true >(
      std::move(env),
      std::make_unique< games::kuhn::State >(),
      tabular_policy,
      avg_tabular_policy,
      0.6,
      0);

   auto initial_curr_policy_profile = std::unordered_map{
      std::pair{
         Player::alex, rm::normalize_state_policy(mccfr_runner.policy().at(Player::alex).table())},
      std::pair{
         Player::bob, rm::normalize_state_policy(mccfr_runner.policy().at(Player::bob).table())}};

   auto initial_policy_profile = std::unordered_map{
      std::pair{
         Player::alex,
         rm::normalize_state_policy(mccfr_runner.average_policy().at(Player::alex).table())},
      std::pair{
         Player::bob,
         rm::normalize_state_policy(mccfr_runner.average_policy().at(Player::bob).table())}};

   size_t n_iters = 200000;
   for(size_t i = 0; i < n_iters; i++) {
      mccfr_runner.iterate(1);
//      evaluate_policies< true >(mccfr_runner, initial_curr_policy_profile, i, "CurrentPolicy");
//      evaluate_policies< false >(mccfr_runner, initial_policy_profile, i);
   }
   evaluate_policies< false >(mccfr_runner, initial_policy_profile, n_iters);
   assert_optimal_policy_kuhn(mccfr_runner, env, 0.05);
}

// TEST(RockPaperScissors, vanilla_cfr_usage_rockpaperscissors)
//{
//    //      auto env = std::make_shared< Environment >(std::make_unique< Logic >());
//    games::rps::Environment env{};
//
//    auto avg_tabular_policy = rm::factory::make_tabular_policy(
//       std::unordered_map< games::rps::InfoState, HashmapActionPolicy< games::rps::Action > >{},
//       rm::factory::
//          make_zero_policy< games::rps::InfoState, HashmapActionPolicy< games::rps::Action > >());
//
//    auto tabular_policy_alex = rm::factory::make_tabular_policy(
//       std::unordered_map< games::rps::InfoState, HashmapActionPolicy< games::rps::Action > >{},
//       rm::factory::
//          make_uniform_policy< games::rps::InfoState, HashmapActionPolicy< games::rps::Action >
//          >());
//
//    auto tabular_policy_bob = rm::factory::make_tabular_policy(
//       std::unordered_map< games::rps::InfoState, HashmapActionPolicy< games::rps::Action > >{},
//       rm::factory::
//          make_uniform_policy< games::rps::InfoState, HashmapActionPolicy< games::rps::Action >
//          >());
//
//    auto infostate_alex = games::rps::InfoState{Player::alex};
//    auto infostate_bob = games::rps::InfoState{Player::alex};
//    auto init_state = games::rps::State();
//    infostate_alex.append(env.private_observation(Player::alex, init_state));
//    infostate_bob.append(env.private_observation(Player::bob, init_state));
//    auto action_alex = games::rps::Action{games::rps::Team::one, games::rps::Hand::rock};
//
//    env.transition(init_state, action_alex);
//
//    infostate_bob.append(env.private_observation(Player::bob, action_alex));
//    infostate_bob.append(env.private_observation(Player::bob, init_state));
//
//    // off-set the given policy by very bad initial values to test the algorithm bouncing back
//    tabular_policy_alex.emplace(
//       infostate_alex,
//       HashmapActionPolicy< games::rps::Action >{std::unordered_map{
//          std::pair{games::rps::Action{games::rps::Team::one, games::rps::Hand::rock}, 1. / 10.},
//          std::pair{games::rps::Action{games::rps::Team::one, games::rps::Hand::paper}, 2. / 10.},
//          std::pair{
//             games::rps::Action{games::rps::Team::one, games::rps::Hand::scissors}, 7. / 10.}}});
//
//    // off-set the given policy by very bad initial values to test the algorithm bouncing back
//    tabular_policy_bob.emplace(
//       infostate_bob,
//       HashmapActionPolicy< games::rps::Action >{std::unordered_map{
//          std::pair{games::rps::Action{games::rps::Team::two, games::rps::Hand::rock}, 5. / 10.},
//          std::pair{games::rps::Action{games::rps::Team::two, games::rps::Hand::paper}, 4. / 10.},
//          std::pair{
//             games::rps::Action{games::rps::Team::two, games::rps::Hand::scissors}, 1. / 10.}}});
//
//    constexpr rm::CFRConfig cfr_config{.alternating_updates = false};
//
//    auto cfr_runner = rm::factory::make_vanilla< cfr_config >(
//       std::move(env),
//       std::make_unique< games::rps::State >(),
//       std::unordered_map{
//          std::pair{Player::alex, tabular_policy_alex}, std::pair{Player::bob,
//          tabular_policy_bob}},
//       std::unordered_map{
//          std::pair{Player::alex, avg_tabular_policy}, std::pair{Player::bob,
//          avg_tabular_policy}});
//
//    auto player = Player::alex;
//
//    auto initial_policy_profile = rm::normalize_state_policy(
//       cfr_runner.average_policy().at(player).table());
//
//    for(size_t i = 0; i < 20000; i++) {
//       cfr_runner.iterate(1);
//       evaluate_policies(player, cfr_runner, initial_policy_profile, i);
//    }
// }

// TEST_F(StrategoState3x3, vanilla_cfr_usage_stratego)
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
//    constexpr rm::CFRConfig cfr_config{.alternating_updates = false, .store_public_states =
//    false};
//
//    auto cfr_runner = rm::factory::make_vanilla< cfr_config, true >(
//       std::move(env),
//       std::make_unique< State >(std::move(state)),
//       tabular_policy,
//       std::move(uniform_policy));
//    cfr_runner.initialize();
//    const auto& curr_policy = *cfr_runner.iterate(100);
//    std::cout << curr_policy.at(Player::alex).table().size();
//    LOGD2("Table size", curr_policy.at(Player::alex).table().size());
// }
