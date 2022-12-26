
#ifndef NOR_CFR_RUN_FUNCS_HPP
#define NOR_CFR_RUN_FUNCS_HPP

#include <gtest/gtest.h>

#include "nor/env.hpp"
#include "nor/nor.hpp"
#include "rm_specific_testing_utils.hpp"

constexpr double EXPLOITABILITY_THRESHOLD = 5e-3;
constexpr double KUHN_POKER_GAME_VALUE_ALEX = -1. / 18.;

template < auto config, bool as_map, typename... Args >
decltype(auto) cfr_factory_func(Args&&... args)
{
   using namespace nor;
   using ConfigType = std::decay_t< decltype(config) >;
   if constexpr(std::same_as< ConfigType, rm::CFRConfig >) {
      return factory::make_cfr_vanilla< config, as_map >(std::forward< Args >(args)...);
   } else if constexpr(std::same_as< ConfigType, rm::CFRDiscountedConfig >) {
      return factory::make_cfr_discounted< config, as_map >(std::forward< Args >(args)...);
   } else if constexpr(std::same_as< ConfigType, rm::CFRLinearConfig >) {
      return factory::make_cfr_linear< config, as_map >(std::forward< Args >(args)...);
   } else if constexpr(std::same_as< ConfigType, rm::CFRPlusConfig >) {
      return factory::make_cfr_plus< as_map >(std::forward< Args >(args)...);
   } else if constexpr(std::same_as< ConfigType, rm::CFRExponentialConfig >) {
      return factory::make_cfr_exponential< config, as_map >(std::forward< Args >(args)...);
   } else if constexpr(std::same_as< ConfigType, rm::CFRPlusConfig >) {
      return factory::make_cfr_plus< config, as_map >(std::forward< Args >(args)...);
   } else if constexpr(std::same_as< ConfigType, rm::MCCFRConfig >) {
      return factory::make_mccfr< config, as_map >(std::forward< Args >(args)...);
   }
};

template < auto config, typename... Args >
decltype(auto) cfr_factory_func(Args&&... args)
{
   using namespace nor;
   using ConfigType = std::decay_t< decltype(config) >;
   if constexpr(std::same_as< ConfigType, rm::CFRConfig >) {
      return factory::make_cfr_vanilla< config >(std::forward< Args >(args)...);
   } else if constexpr(std::same_as< ConfigType, rm::CFRDiscountedConfig >) {
      return factory::make_cfr_discounted< config >(std::forward< Args >(args)...);
   } else if constexpr(std::same_as< ConfigType, rm::CFRLinearConfig >) {
      return factory::make_cfr_linear< config >(std::forward< Args >(args)...);
   } else if constexpr(std::same_as< ConfigType, rm::CFRPlusConfig >) {
      return factory::make_cfr_plus(std::forward< Args >(args)...);
   } else if constexpr(std::same_as< ConfigType, rm::CFRExponentialConfig >) {
      return factory::make_cfr_exponential< config >(std::forward< Args >(args)...);
   } else if constexpr(std::same_as< ConfigType, rm::CFRPlusConfig >) {
      return factory::make_cfr_plus< config >(std::forward< Args >(args)...);
   } else if constexpr(std::same_as< ConfigType, rm::MCCFRConfig >) {
      return factory::make_mccfr< config >(std::forward< Args >(args)...);
   }
};

template < auto config, typename... ExtraFactoryParams >
void run_cfr_on_kuhn_poker(
   size_t max_iters = 1e5,
   size_t update_freq = 10,
   ExtraFactoryParams&&... extra_args
)
{
   using namespace nor;
   games::kuhn::Environment env{};

   auto root_state = std::make_unique< games::kuhn::State >();
   auto players = env.players(*root_state);

   auto avg_tabular_policy = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{}
   );

   auto tabular_policy = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{}
   );

   auto solver = cfr_factory_func< config, true >(
      std::move(env),
      std::move(root_state),
      tabular_policy,
      avg_tabular_policy,
      std::forward< ExtraFactoryParams >(extra_args)...
   );

   auto initial_curr_policy_profile = std::unordered_map{
      std::pair{Player::alex, normalize_state_policy(solver.policy().at(Player::alex).table())},
      std::pair{Player::bob, normalize_state_policy(solver.policy().at(Player::bob).table())}};

   auto initial_policy_profile = std::unordered_map{
      std::pair{
         Player::alex, normalize_state_policy(solver.average_policy().at(Player::alex).table())},
      std::pair{
         Player::bob, normalize_state_policy(solver.average_policy().at(Player::bob).table())}};

   constexpr size_t n_infostates = 6;
   size_t n_iters = 0;
   double expl = std::numeric_limits< double >::max();
   while(expl > EXPLOITABILITY_THRESHOLD or n_iters >= max_iters) {
      solver.iterate(1);
      n_iters++;
#ifndef NDEBUG
      evaluate_policies< true >(solver, initial_curr_policy_profile, i, "Current Policy");
      evaluate_policies< false >(solver, initial_policy_profile, i);
#endif
      if(ranges::all_of(
            solver.average_policy() | ranges::views::values,
            [](const auto& policy) { return policy.size() == n_infostates; }
         )
         and n_iters % update_freq == 0) {
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

template < auto config, typename... ExtraFactoryParams >
void run_cfr_on_rockpaperscissors(
   size_t max_iters = 1e5,
   size_t update_freq = 10,
   ExtraFactoryParams... extra_args
)
{
   using namespace nor;
   auto
      [env,
       avg_tabular_policy_alex,
       avg_tabular_policy_bob,
       curr_tabular_policy_alex,
       curr_tabular_policy_bob,
       infostate_alex,
       infostate_bob,
       init_state] = setup_rps_test();

   auto root_state = std::make_unique< games::rps::State >();
   auto players = env.players(*root_state);

   auto solver = cfr_factory_func< config >(
      std::move(env),
      std::move(root_state),
      std::unordered_map{
         std::pair{Player::alex, curr_tabular_policy_alex},
         std::pair{Player::bob, curr_tabular_policy_bob}},
      std::unordered_map{
         std::pair{Player::alex, avg_tabular_policy_alex},
         std::pair{Player::bob, avg_tabular_policy_bob}},
      extra_args...
   );

   auto initial_curr_policy_profile = std::unordered_map{
      std::pair{Player::alex, normalize_state_policy(solver.policy().at(Player::alex).table())},
      std::pair{Player::bob, normalize_state_policy(solver.policy().at(Player::bob).table())}};

   auto initial_policy_profile = std::unordered_map{
      std::pair{
         Player::alex, normalize_state_policy(solver.average_policy().at(Player::alex).table())},
      std::pair{
         Player::bob, normalize_state_policy(solver.average_policy().at(Player::bob).table())}};

   constexpr size_t n_infostates = 1;
   size_t n_iters = 0;
   double expl = std::numeric_limits< double >::max();
   while(expl > EXPLOITABILITY_THRESHOLD and n_iters < max_iters) {
      solver.iterate(1);
      n_iters++;
#ifndef NDEBUG
      evaluate_policies< true >(solver, initial_curr_policy_profile, i, "Current Policy");
      evaluate_policies< false >(solver, initial_policy_profile, i);
#endif
      if(ranges::all_of(
            solver.average_policy() | ranges::views::values,
            [](const auto& policy) { return policy.size() == n_infostates; }
         )
         and n_iters % update_freq == 0) {
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

#endif  // NOR_CFR_RUN_FUNCS_HPP
