#ifndef NOR_UTILS_FOR_TESTING_HPP
#define NOR_UTILS_FOR_TESTING_HPP

#include <gtest/gtest.h>

#include <unordered_map>

#include "nor/env.hpp"
#include "nor/nor.hpp"

template < typename Policy >
inline void print_policy(const Policy& policy)
{
   auto policy_vec = ranges::to_vector(policy | ranges::views::transform([](const auto& kv) {
                                          return std::pair{std::get< 0 >(kv), std::get< 1 >(kv)};
                                       }));
   std::sort(policy_vec.begin(), policy_vec.end(), [](const auto& kv, const auto& other_kv) {
      return std::get< 0 >(kv).private_history().back().size()
             < std::get< 0 >(other_kv).private_history().back().size();
   });
   auto action_policy_printer = [&](const auto& action_policy) {
      std::stringstream ss;
      ss << "[ ";
      for(const auto& [key, value] : action_policy) {
         ss << std::setw(5) << common::to_string(key) + ": " << std::setw(6) << std::setprecision(3)
            << value << " ";
      }
      ss << "]";
      return ss.str();
   };
   for(const auto& [istate, action_policy] : policy_vec) {
      std::cout << std::setw(5) << istate.player() << " | " << std::setw(8)
                << common::left(istate.private_history().back(), 5, " ") << " -> ";
      std::cout << action_policy_printer(action_policy) << "\n";
   }
}

template < bool current_policy, typename CFRRunner, typename Policy >
inline void evaluate_policies(
   CFRRunner& solver,
   std::unordered_map< nor::Player, Policy >& prev_policy_profile,
   size_t iteration,
   std::string policy_name = "Average Policy")
{
   using namespace nor;
   auto policy_fetcher = [&](Player this_player) {
      if constexpr(current_policy) {
         return rm::normalize_state_policy(solver.policy().at(this_player).table());
      } else {
         return rm::normalize_state_policy(solver.average_policy().at(this_player).table());
      }
   };

   std::unordered_map< Player, Policy > policy_profile_this_iter;
   for(const auto& [p, policy] : prev_policy_profile) {
      policy_profile_this_iter[p] = policy_fetcher(p);
   }

   double total_dev = 0.;
   for(auto p : prev_policy_profile | ranges::views::keys) {
      for(const auto& [curr_pol, prev_pol] :
          ranges::views::zip(policy_profile_this_iter[p], prev_policy_profile[p])) {
         const auto& [curr_istate, curr_istate_pol] = curr_pol;
         const auto& [prev_istate, prev_istate_pol] = prev_pol;
         for(const auto& [curr_pol_state_pol, prev_pol_state_pol] :
             ranges::views::zip(curr_istate_pol, prev_istate_pol)) {
            total_dev = std::abs(
               std::get< 1 >(curr_pol_state_pol) - std::get< 1 >(prev_pol_state_pol));
         }
      }
   }

   std::cout << policy_name + ":\n";
   for(const auto& [p, policy] : policy_profile_this_iter) {
      print_policy(policy);
   }

   prev_policy_profile = std::move(policy_profile_this_iter);
   if constexpr(requires { solver.game_value(); }) {
      if(solver.iteration() > 1) {
         auto game_value_map = solver.game_value();
         for(auto [p, value] : game_value_map.get()) {
            std::cout << "iteration: " << iteration << " | game value for player " << p << ": "
                      << value << "\n";
         }
      }
   }
   std::cout << "total policy change to previous policy: " << total_dev << "\n";
}

template < bool current_policy, typename CFRRunner >
inline void evaluate_policies(
   CFRRunner& solver,
   ranges::range auto players,
   size_t iteration,
   std::string policy_name = "Average Policy")
{
   using namespace nor;
   auto policy_fetcher = [&](Player this_player) {
      if constexpr(current_policy) {
         return rm::normalize_state_policy(solver.policy().at(this_player).table());
      } else {
         return rm::normalize_state_policy(solver.average_policy().at(this_player).table());
      }
   };

   std::unordered_map< Player, decltype(policy_fetcher(Player::alex)) > policy_profile_this_iter;
   std::cout << policy_name + ":\n";
   for(auto player : players) {
      policy_profile_this_iter[player] = policy_fetcher(player);
      print_policy(policy_profile_this_iter[player]);
   }
   if constexpr(requires { solver.game_value(); }) {
      if(solver.iteration() > 1) {
         auto game_value_map = solver.game_value();
         for(auto [p, value] : game_value_map.get()) {
            std::cout << "iteration: " << iteration << " | game value for player " << p << ": "
                      << value << "\n";
         }
      }
   }
}

inline auto kuhn_optimal(double alpha)
{
   using namespace nor;
   using namespace games::kuhn;
   std::unordered_map< std::string, HashmapActionPolicy< Action > > alex_policy;
   alex_policy.emplace(
      "j?",
      HashmapActionPolicy{
         std::unordered_map{std::pair{Action::check, 1. - alpha}, std::pair{Action::bet, alpha}}});
   alex_policy.emplace(
      "j?|cb",
      HashmapActionPolicy{
         std::unordered_map{std::pair{Action::check, 1.}, std::pair{Action::bet, 0.}}});
   alex_policy.emplace(
      "q?",
      HashmapActionPolicy{
         std::unordered_map{std::pair{Action::check, 1.}, std::pair{Action::bet, 0.}}});
   alex_policy.emplace(
      "q?|cb",
      HashmapActionPolicy{std::unordered_map{
         std::pair{Action::check, 2. / 3. - alpha}, std::pair{Action::bet, 1. / 3. + alpha}}});
   alex_policy.emplace(
      "k?",
      HashmapActionPolicy{std::unordered_map{
         std::pair{Action::check, 1. - 3. * alpha}, std::pair{Action::bet, 3. * alpha}}});
   alex_policy.emplace(
      "k?|cb",
      HashmapActionPolicy{
         std::unordered_map{std::pair{Action::check, 0.}, std::pair{Action::bet, 1.}}});

   std::unordered_map< std::string, HashmapActionPolicy< Action > > bob_policy;
   bob_policy.emplace(
      "?j|c",
      HashmapActionPolicy{
         std::unordered_map{std::pair{Action::check, 2. / 3.}, std::pair{Action::bet, 1. / 3.}}});
   bob_policy.emplace(
      "?j|b",
      HashmapActionPolicy{
         std::unordered_map{std::pair{Action::check, 1.}, std::pair{Action::bet, 0.}}});
   bob_policy.emplace(
      "?q|c",
      HashmapActionPolicy{
         std::unordered_map{std::pair{Action::check, 1.}, std::pair{Action::bet, 0.}}});
   bob_policy.emplace(
      "?q|b",
      HashmapActionPolicy{
         std::unordered_map{std::pair{Action::check, 2. / 3.}, std::pair{Action::bet, 1. / 3.}}});
   bob_policy.emplace(
      "?k|c",
      HashmapActionPolicy{
         std::unordered_map{std::pair{Action::check, 0.}, std::pair{Action::bet, 1.}}});
   bob_policy.emplace(
      "?k|b",
      HashmapActionPolicy{
         std::unordered_map{std::pair{Action::check, 0.}, std::pair{Action::bet, 1.}}});

   return std::tuple{alex_policy, bob_policy};
}

inline void assert_optimal_policy_rps(const auto& solver, double precision = 1e-2)
{
   using namespace nor;
   if constexpr(requires { solver.game_value(); }) {
      ASSERT_NEAR(solver.game_value().get()[Player::alex], 0., 1e-4);
   }
   auto final_policy = solver.average_policy().at(Player::alex).table();
   for(const auto& [state, action_policy] : final_policy) {
      for(const auto& [action, prob] : rm::normalize_action_policy(action_policy)) {
         ASSERT_NEAR(prob, 1. / 3., precision);
      }
   }
   final_policy = solver.average_policy().at(Player::bob).table();
   for(const auto& [state, action_policy] : final_policy) {
      for(const auto& [action, prob] : rm::normalize_action_policy(action_policy)) {
         ASSERT_NEAR(prob, 1. / 3., precision);
      }
   }
}

inline void assert_optimal_policy_kuhn(const auto& solver, auto& env, double precision = 1e-2)
{
   using namespace nor;

   games::kuhn::State state;

   games::kuhn::Infostate infostate_alex{Player::alex};

   infostate_alex.append(env.private_observation(Player::alex, state));

   auto chance_action = games::kuhn::ChanceOutcome{kuhn::Player::one, kuhn::Card::jack};
   env.transition(state, chance_action);

   infostate_alex.append(env.private_observation(Player::alex, chance_action));
   infostate_alex.append(env.private_observation(Player::alex, state));

   chance_action = games::kuhn::ChanceOutcome{kuhn::Player::two, kuhn::Card::queen};
   env.transition(state, chance_action);

   infostate_alex.append(env.private_observation(Player::alex, chance_action));
   infostate_alex.append(env.private_observation(Player::alex, state));
   auto policy_tables = std::vector{
      solver.average_policy().at(Player::alex).table(),
      solver.average_policy().at(Player::bob).table()};
   double alpha = rm::normalize_action_policy(
      policy_tables[0].at(infostate_alex))[kuhn::Action::bet];
   auto [alex_optimal_table, bob_optimal_table] = kuhn_optimal(alpha);
   auto optimal_tables = std::vector{std::move(alex_optimal_table), std::move(bob_optimal_table)};

   for(const auto& [computed_table, optimal_table] :
       ranges::views::zip(policy_tables, optimal_tables)) {
      for(const auto& computed_state_policy : computed_table) {
         const auto& [istate, action_policy] = computed_state_policy;
         auto normalized_ap = rm::normalize_action_policy(action_policy);
         for(const auto& [optim_action_and_prob, action_and_prob] :
             ranges::views::zip(optimal_table.at(istate.private_history().back()), normalized_ap)) {
            auto found_action_prob = std::get< 1 >(action_and_prob);
            auto optimal_action_prob = std::get< 1 >(optim_action_and_prob);
            ASSERT_NEAR(found_action_prob, optimal_action_prob, precision);
         }
      }
   }
}

inline auto setup_rps_test()
{
   using namespace nor;
   games::rps::Environment env{};

   auto avg_tabular_policy = rm::factory::make_tabular_policy(
      std::unordered_map< games::rps::InfoState, HashmapActionPolicy< games::rps::Action > >{},
      rm::factory::
         make_zero_policy< games::rps::InfoState, HashmapActionPolicy< games::rps::Action > >());

   auto tabular_policy_alex = rm::factory::make_tabular_policy(
      std::unordered_map< games::rps::InfoState, HashmapActionPolicy< games::rps::Action > >{},
      rm::factory::
         make_uniform_policy< games::rps::InfoState, HashmapActionPolicy< games::rps::Action > >());

   auto tabular_policy_bob = rm::factory::make_tabular_policy(
      std::unordered_map< games::rps::InfoState, HashmapActionPolicy< games::rps::Action > >{},
      rm::factory::
         make_uniform_policy< games::rps::InfoState, HashmapActionPolicy< games::rps::Action > >());

   auto infostate_alex = games::rps::InfoState{Player::alex};
   auto infostate_bob = games::rps::InfoState{Player::bob};
   auto init_state = games::rps::State();
   infostate_alex.append(env.private_observation(Player::alex, init_state));
   infostate_bob.append(env.private_observation(Player::bob, init_state));

   auto action_alex = games::rps::Action{games::rps::Team::one, games::rps::Hand::rock};

   env.transition(init_state, action_alex);

   infostate_bob.append(env.private_observation(Player::bob, action_alex));
   infostate_bob.append(env.private_observation(Player::bob, init_state));

   // off-set the given policy by very bad initial values to test the algorithm bouncing back
   tabular_policy_alex.emplace(
      infostate_alex,
      HashmapActionPolicy< games::rps::Action >{std::unordered_map{
         std::pair{games::rps::Action{games::rps::Team::one, games::rps::Hand::rock}, 1. / 10.},
         std::pair{games::rps::Action{games::rps::Team::one, games::rps::Hand::paper}, 2. / 10.},
         std::pair{
            games::rps::Action{games::rps::Team::one, games::rps::Hand::scissors}, 7. / 10.}}});

   // off-set the given policy by very bad initial values to test the algorithm bouncing back
   tabular_policy_bob.emplace(
      infostate_bob,
      HashmapActionPolicy< games::rps::Action >{std::unordered_map{
         std::pair{games::rps::Action{games::rps::Team::two, games::rps::Hand::rock}, 9. / 10.},
         std::pair{games::rps::Action{games::rps::Team::two, games::rps::Hand::paper}, .5 / 10.},
         std::pair{
            games::rps::Action{games::rps::Team::two, games::rps::Hand::scissors}, .5 / 10.}}});

   return std::tuple{
      std::move(env),
      avg_tabular_policy,
      avg_tabular_policy,
      std::move(tabular_policy_alex),
      std::move(tabular_policy_bob),
      std::move(infostate_alex),
      std::move(infostate_bob),
      std::move(init_state)};
}

#endif  // NOR_UTILS_FOR_TESTING_HPP
