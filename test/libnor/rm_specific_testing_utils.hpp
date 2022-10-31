#ifndef NOR_RM_SPECIFIC_TESTING_UTILS_HPP
#define NOR_RM_SPECIFIC_TESTING_UTILS_HPP

#include <gtest/gtest.h>

#include <unordered_map>

#include "nor/env.hpp"
#include "nor/nor.hpp"

template < typename ActionPolicy >
inline std::string print_action_policy(const ActionPolicy& action_policy)
{
   size_t max_len_action = ranges::max(
      action_policy | ranges::views::keys
      | ranges::views::transform([](auto p) { return common::to_string(p).size(); })
   );

   std::stringstream ss;
   ss << "[ ";
   for(const auto& [key, value] : action_policy) {
      ss << std::setw(static_cast< int >(max_len_action)) << common::to_string(key) + ": ";
      ss << std::setw(6) << std::setprecision(3) << value << " ";
   }
   ss << "]";
   return ss.str();
}

template < typename Policy >
inline std::string print_policy(
   const Policy& policy,
   int max_len_player_str,
   int max_len_istate_str,
   std::string istate_to_string_delim = "|"
)
{
   auto policy_vec = ranges::to_vector(policy | ranges::views::transform([](const auto& kv) {
                                          return std::pair{std::get< 0 >(kv), std::get< 1 >(kv)};
                                       }));
   ranges::actions::sort(policy_vec, [](const auto& kv, const auto& other_kv) {
      const auto& istate_0 = std::get< 0 >(kv);
      const auto& istate_1 = std::get< 0 >(other_kv);
      return istate_0.to_string("|").size() < istate_1.to_string("|").size();
   });

   std::stringstream ss;
   for(const auto& [istate, action_policy] : policy_vec) {
      ss << std::setw(max_len_player_str) << istate.player() << " | "
         << std::setw(max_len_istate_str)
         << common::left(istate.to_string(istate_to_string_delim), max_len_istate_str, " ")
         << " -> " << print_action_policy(action_policy) << "\n";
   }
   return ss.str();
}

template < typename PolicyMap >
inline void print_policy_profile(const PolicyMap& policy_map)
{
   // we expect the policy profile to be a map of the type:
   //    nor::Player --> state policy

   std::string to_string_delim = "|";

   auto players = ranges::to_vector(policy_map | ranges::views::keys);
   ranges::actions::sort(players, [](auto p1, auto p2) {
      return static_cast< int >(p1) < static_cast< int >(p2);
   });

   size_t max_len_names = ranges::max(players | ranges::views::transform([](auto p) {
                                         return common::to_string(p).size();
                                      }));

   size_t max_len_istate_str = ranges::max(
      policy_map
      | ranges::views::values /*the actual policy of the player, i.e. istate -> action_policy*/
      | ranges::views::transform([&](const auto& kv) {
           if(kv.begin() == kv.end()) {
              return size_t(0);
           }
           return ranges::max(
              kv | ranges::views::keys /*the actual policy of the player, i.e.
                                          istate -> action_policy*/
              | ranges::views::transform([&](const auto& istate) {
                   return istate.to_string(to_string_delim).size();
                })
           );
        })
   );

   for(auto player : players) {
      std::cout << print_policy(
         policy_map.at(player),
         static_cast< int >(max_len_names),
         static_cast< int >(max_len_istate_str),
         to_string_delim
      );
   }
}

template < bool current_policy, typename CFRRunner, typename Policy >
inline void evaluate_policies(
   CFRRunner& solver,
   std::unordered_map< nor::Player, Policy >& prev_policy_profile,
   size_t iteration,
   std::string policy_name = "Average Policy"
)
{
   using namespace nor;
   auto policy_fetcher = [&](Player this_player) {
      if constexpr(current_policy) {
         return normalize_state_policy(solver.policy().at(this_player).table());
      } else {
         return normalize_state_policy(solver.average_policy().at(this_player).table());
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
               std::get< 1 >(curr_pol_state_pol) - std::get< 1 >(prev_pol_state_pol)
            );
         }
      }
   }

   std::cout << policy_name + ":\n";
   print_policy_profile(policy_profile_this_iter);

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
   std::string policy_name = "Average Policy"
)
{
   using namespace nor;
   auto policy_fetcher = [&](Player this_player) {
      if constexpr(current_policy) {
         return normalize_state_policy(solver.policy().at(this_player).table());
      } else {
         return normalize_state_policy(solver.average_policy().at(this_player).table());
      }
   };

   std::unordered_map< Player, decltype(policy_fetcher(Player::alex)) > policy_profile_this_iter;
   std::cout << policy_name + ":\n";
   for(auto player : players) {
      policy_profile_this_iter[player] = policy_fetcher(player);
   }
   print_policy_profile(policy_profile_this_iter);

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

   std::unordered_map< Infostate, HashmapActionPolicy< Action > > alex_policy;

   auto env = Environment{};
   auto state = State{};
   auto [_, history_to_istate, __] = map_histories_to_infostates(env, state);

   using action_variant_type = typename fosg_auto_traits< Environment >::action_variant_type;

   alex_policy.emplace(
      history_to_istate
         .find(std::vector< action_variant_type >{
            ChanceOutcome{
               .player = kuhn::Player::one,
               .card = Card::jack,
            },
            ChanceOutcome{
               .player = kuhn::Player::two,
               .card = Card::queen,  // which card here does not matter
            }})
         ->second,
      HashmapActionPolicy{std::pair{Action::check, 1. - alpha}, std::pair{Action::bet, alpha}}
   );
   alex_policy.emplace(
      history_to_istate
         .find(std::vector< action_variant_type >{
            ChanceOutcome{
               .player = kuhn::Player::one,
               .card = Card::jack,
            },
            ChanceOutcome{
               .player = kuhn::Player::two,
               .card = Card::queen,  // which card here does not matter
            },
            Action::check,
            Action::bet})
         ->second,
      HashmapActionPolicy{std::pair{Action::check, 1.}, std::pair{Action::bet, 0.}}
   );
   alex_policy.emplace(
      history_to_istate
         .find(std::vector< action_variant_type >{
            ChanceOutcome{
               .player = kuhn::Player::one,
               .card = Card::queen,
            },
            ChanceOutcome{
               .player = kuhn::Player::two,
               .card = Card::king,  // which card here does not matter
            }})
         ->second,
      HashmapActionPolicy{std::pair{Action::check, 1.}, std::pair{Action::bet, 0.}}
   );
   alex_policy.emplace(
      history_to_istate
         .find(std::vector< action_variant_type >{
            ChanceOutcome{
               .player = kuhn::Player::one,
               .card = Card::queen,
            },
            ChanceOutcome{
               .player = kuhn::Player::two,
               .card = Card::king,  // which card here does not matter
            },
            Action::check,
            Action::bet})
         ->second,
      HashmapActionPolicy{
         std::pair{Action::check, 2. / 3. - alpha}, std::pair{Action::bet, 1. / 3. + alpha}}
   );
   alex_policy.emplace(
      history_to_istate
         .find(std::vector< action_variant_type >{
            ChanceOutcome{
               .player = kuhn::Player::one,
               .card = Card::king,
            },
            ChanceOutcome{
               .player = kuhn::Player::two,
               .card = Card::jack,  // which card here does not matter
            }})
         ->second,
      HashmapActionPolicy{
         std::pair{Action::check, 1. - 3. * alpha}, std::pair{Action::bet, 3. * alpha}}
   );
   alex_policy.emplace(
      history_to_istate
         .find(std::vector< action_variant_type >{
            ChanceOutcome{
               .player = kuhn::Player::one,
               .card = Card::king,
            },
            ChanceOutcome{
               .player = kuhn::Player::two,
               .card = Card::queen,  // which card here does not matter
            },
            Action::check,
            Action::bet})
         ->second,
      HashmapActionPolicy{std::pair{Action::check, 0.}, std::pair{Action::bet, 1.}}
   );

   std::unordered_map< Infostate, HashmapActionPolicy< Action > > bob_policy;
   bob_policy.emplace(
      history_to_istate
         .find(std::vector< action_variant_type >{
            ChanceOutcome{
               .player = kuhn::Player::one,
               .card = Card::king,  // which card here does not matter
            },
            ChanceOutcome{
               .player = kuhn::Player::two,
               .card = Card::jack,
            },
            Action::check})
         ->second,
      HashmapActionPolicy{std::pair{Action::check, 2. / 3.}, std::pair{Action::bet, 1. / 3.}}
   );
   bob_policy.emplace(
      history_to_istate
         .find(std::vector< action_variant_type >{
            ChanceOutcome{
               .player = kuhn::Player::one,
               .card = Card::king,  // which card here does not matter
            },
            ChanceOutcome{
               .player = kuhn::Player::two,
               .card = Card::jack,
            },
            Action::bet})
         ->second,
      HashmapActionPolicy{std::pair{Action::check, 1.}, std::pair{Action::bet, 0.}}
   );
   bob_policy.emplace(
      history_to_istate
         .find(std::vector< action_variant_type >{
            ChanceOutcome{
               .player = kuhn::Player::one,
               .card = Card::king,  // which card here does not matter
            },
            ChanceOutcome{
               .player = kuhn::Player::two,
               .card = Card::queen,
            },
            Action::check})
         ->second,
      HashmapActionPolicy{std::pair{Action::check, 1.}, std::pair{Action::bet, 0.}}
   );
   bob_policy.emplace(
      history_to_istate
         .find(std::vector< action_variant_type >{
            ChanceOutcome{
               .player = kuhn::Player::one,
               .card = Card::king,  // which card here does not matter
            },
            ChanceOutcome{
               .player = kuhn::Player::two,
               .card = Card::queen,
            },
            Action::bet})
         ->second,
      HashmapActionPolicy{std::pair{Action::check, 2. / 3.}, std::pair{Action::bet, 1. / 3.}}
   );
   bob_policy.emplace(
      history_to_istate
         .find(std::vector< action_variant_type >{
            ChanceOutcome{
               .player = kuhn::Player::one,
               .card = Card::queen,  // which card here does not matter
            },
            ChanceOutcome{
               .player = kuhn::Player::two,
               .card = Card::king,
            },
            Action::check})
         ->second,
      HashmapActionPolicy{std::pair{Action::check, 0.}, std::pair{Action::bet, 1.}}
   );
   bob_policy.emplace(
      history_to_istate
         .find(std::vector< action_variant_type >{
            ChanceOutcome{
               .player = kuhn::Player::one,
               .card = Card::queen,  // which card here does not matter
            },
            ChanceOutcome{
               .player = kuhn::Player::two,
               .card = Card::king,
            },
            Action::bet})
         ->second,
      HashmapActionPolicy{std::pair{Action::check, 0.}, std::pair{Action::bet, 1.}}
   );

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
      for(const auto& [action, prob] : normalize_action_policy(action_policy)) {
         ASSERT_NEAR(prob, 1. / 3., precision);
      }
   }
   final_policy = solver.average_policy().at(Player::bob).table();
   for(const auto& [state, action_policy] : final_policy) {
      for(const auto& [action, prob] : normalize_action_policy(action_policy)) {
         ASSERT_NEAR(prob, 1. / 3., precision);
      }
   }
}

inline void assert_optimal_policy_kuhn(const auto& solver, auto& env, double precision = 1e-2)
{
   using namespace nor;

   games::kuhn::State state{}, next_state{};

   games::kuhn::Infostate infostate_alex{Player::alex};

   auto chance_action = games::kuhn::ChanceOutcome{kuhn::Player::one, kuhn::Card::jack};

   env.transition(next_state, chance_action);

   infostate_alex.update(
      env.public_observation(state, chance_action, next_state),
      env.private_observation(Player::alex, state, chance_action, next_state)
   );

   chance_action = games::kuhn::ChanceOutcome{kuhn::Player::two, kuhn::Card::queen};

   state = next_state;
   env.transition(next_state, chance_action);

   infostate_alex.update(
      env.public_observation(state, chance_action, next_state),
      env.private_observation(Player::alex, state, chance_action, next_state)
   );

   auto policy_tables = std::vector{
      solver.average_policy().at(Player::alex).table(),
      solver.average_policy().at(Player::bob).table()};
   double alpha = normalize_action_policy(policy_tables[0].at(infostate_alex))[kuhn::Action::bet];
   auto [alex_optimal_table, bob_optimal_table] = kuhn_optimal(alpha);
   auto optimal_tables = std::vector{std::move(alex_optimal_table), std::move(bob_optimal_table)};

   for(const auto& [computed_table, optimal_table] :
       ranges::views::zip(policy_tables, optimal_tables)) {
      for(const auto& computed_state_policy : computed_table) {
         const auto& [istate, action_policy] = computed_state_policy;
         auto normalized_ap = normalize_action_policy(action_policy);
         for(const auto& [optim_action_and_prob, action_and_prob] :
             ranges::views::zip(optimal_table.at(istate), normalized_ap)) {
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

   auto avg_tabular_policy = factory::make_tabular_policy(
      std::unordered_map< games::rps::Infostate, HashmapActionPolicy< games::rps::Action > >{}
   );

   auto tabular_policy_alex = factory::make_tabular_policy(
      std::unordered_map< games::rps::Infostate, HashmapActionPolicy< games::rps::Action > >{}
   );

   auto tabular_policy_bob = factory::make_tabular_policy(
      std::unordered_map< games::rps::Infostate, HashmapActionPolicy< games::rps::Action > >{}
   );

   auto infostate_alex = games::rps::Infostate{Player::alex};
   auto infostate_bob = games::rps::Infostate{Player::bob};
   games::rps::State state{}, next_state{};

   auto action_alex = games::rps::Action::rock;

   state = next_state;
   env.transition(next_state, action_alex);

   infostate_bob.update(
      env.public_observation(state, action_alex, next_state),
      env.private_observation(Player::bob, state, action_alex, next_state)
   );

   // off-set the given policy by very bad initial values to test the algorithm bouncing back
   tabular_policy_alex.emplace(
      infostate_alex,
         std::pair{games::rps::Action::rock, 1. / 10.},
         std::pair{games::rps::Action::paper, 2. / 10.},
         std::pair{games::rps::Action::scissors, 7. / 10.}
   );

   // off-set the given policy by very bad initial values to test the algorithm bouncing back
   tabular_policy_bob.emplace(
      infostate_bob,
         std::pair{games::rps::Action::rock, 9. / 10.},
         std::pair{games::rps::Action::paper, .5 / 10.},
         std::pair{games::rps::Action::scissors, .5 / 10.}
   );

   return std::tuple{
      std::move(env),
      avg_tabular_policy,
      avg_tabular_policy,
      std::move(tabular_policy_alex),
      std::move(tabular_policy_bob),
      std::move(infostate_alex),
      std::move(infostate_bob),
      std::move(next_state)};
}

#endif  // NOR_RM_SPECIFIC_TESTING_UTILS_HPP
