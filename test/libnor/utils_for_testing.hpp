#ifndef NOR_UTILS_FOR_TESTING_HPP
#define NOR_UTILS_FOR_TESTING_HPP

#include <gtest/gtest.h>

#include <unordered_map>

#include "nor/nor.hpp"
#include "nor/wrappers.hpp"


template < typename Policy >
inline void print_policy(const Policy& policy)
{
   auto policy_vec = ranges::to_vector(policy | ranges::views::transform([](const auto& kv) {
                                          return std::pair{std::get< 0 >(kv), std::get< 1 >(kv)};
                                       }));
   std::sort(policy_vec.begin(), policy_vec.end(), [](const auto& kv, const auto& other_kv) {
      return std::get< 0 >(kv).history().back().size()
             < std::get< 0 >(other_kv).history().back().size();
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
                << common::left(istate.history().back(), 5, " ") << " -> ";
      std::cout << action_policy_printer(action_policy) << "\n";
   }
}

template < typename CFRRunner, typename Policy >
inline void evaluate_policies(
   nor::Player player,
   CFRRunner& cfr_runner,
   Policy& prev_policy_profile,
   size_t iteration)
{
   using namespace nor;
   auto curr_policy_profile = rm::normalize_state_policy(
      cfr_runner.average_policy().at(player).table());

   double total_dev = 0.;
   for(const auto& [curr_pol, prev_pol] :
       ranges::views::zip(curr_policy_profile, prev_policy_profile)) {
      const auto& [curr_istate, curr_istate_pol] = curr_pol;
      const auto& [prev_istate, prev_istate_pol] = prev_pol;
      for(const auto& [curr_pol_state_pol, prev_pol_state_pol] :
          ranges::views::zip(curr_istate_pol, prev_istate_pol)) {
         total_dev = std::abs(
            std::get< 1 >(curr_pol_state_pol) - std::get< 1 >(prev_pol_state_pol));
      }
   }
   auto normalized_current_policy = rm::normalize_state_policy(
      cfr_runner.policy().at(player).table());

   std::cout << "Average policy:\n";
   print_policy(rm::normalize_state_policy(cfr_runner.average_policy().at(Player::alex).table()));
   print_policy(rm::normalize_state_policy(cfr_runner.average_policy().at(Player::bob).table()));

   prev_policy_profile = std::move(curr_policy_profile);
   if(cfr_runner.iteration() > 1) {
      auto game_value_map = cfr_runner.game_value();
      std::cout << "iteration: " << iteration << " | game value for player " << player << ": "
                << game_value_map.get()[player] << "\n";
   }
   //   std::cout << "total deviation: " << total_dev << "\n";
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

inline void assert_optimal_policy_kuhn(const auto& cfr_runner, auto& env)
{
   using namespace nor;

   games::kuhn::State state;

   games::kuhn::Infostate infostate_alex{Player::alex};
   games::kuhn::Infostate infostate_bob{Player::bob};

   infostate_alex.append(env.private_observation(Player::alex, state));
   infostate_bob.append(env.private_observation(Player::bob, state));

   auto chance_action = games::kuhn::ChanceOutcome{kuhn::Player::one, kuhn::Card::jack};
   env.transition(state, chance_action);

   infostate_alex.append(env.private_observation(Player::alex, chance_action));
   infostate_alex.append(env.private_observation(Player::alex, state));
   infostate_bob.append(env.private_observation(Player::bob, chance_action));
   infostate_bob.append(env.private_observation(Player::bob, state));

   chance_action = games::kuhn::ChanceOutcome{kuhn::Player::two, kuhn::Card::queen};
   env.transition(state, chance_action);

   infostate_alex.append(env.private_observation(Player::alex, chance_action));
   infostate_alex.append(env.private_observation(Player::alex, state));
   infostate_bob.append(env.private_observation(Player::bob, chance_action));
   infostate_bob.append(env.private_observation(Player::bob, state));

   auto policy_tables = std::vector{
      cfr_runner.average_policy().at(Player::alex).table(),
      cfr_runner.average_policy().at(Player::bob).table()};
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
             ranges::views::zip(optimal_table.at(istate.history().back()), normalized_ap)) {
            auto action_prob = std::get< 1 >(action_and_prob);
            auto optim_action_prob = std::get< 1 >(optim_action_and_prob);
            ASSERT_NEAR(action_prob, optim_action_prob, 1e-2);
         }
      }
   }
}

#endif  // NOR_UTILS_FOR_TESTING_HPP
