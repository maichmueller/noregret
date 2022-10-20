#include <gtest/gtest.h>

#include <random>
#include <string>

#include "common/common.hpp"
#include "nor/env.hpp"
#include "nor/factory.hpp"
#include "nor/fosg_states.hpp"
#include "nor/game_defs.hpp"
#include "nor/policy/policy.hpp"

class TestInfostate: public nor::DefaultInfostate< TestInfostate, std::string > {};

namespace std {
template <>
struct hash< TestInfostate > {
   size_t operator()(const TestInfostate& istate) const
   {
      return std::hash< std::string >{}(istate.to_string());
   }
};
}  // namespace std

TEST(TabularPolicy, uniform_default)
{
   auto tabular_policy = nor::factory::make_tabular_policy(
      std::unordered_map< TestInfostate, nor::HashmapActionPolicy< int > >{},
      nor::factory::make_uniform_policy< TestInfostate, nor::HashmapActionPolicy< int > >()
   );

   auto istate1 = TestInfostate{nor::Player::alex};
   istate1.update("case1", "case1priv");
   auto actions = std::vector< int >{1, 2, 3, 4, 5};
   auto& initial_policy = tabular_policy[std::pair{istate1, actions}];
   for(auto i : actions) {
      ASSERT_NEAR(initial_policy[i], .2, 1e-10);
   }
   initial_policy[3] += 5;
   for(auto i : actions) {
      if(i != 3) {
         ASSERT_NEAR(initial_policy[i], .2, 1e-10);
      } else {
         ASSERT_NEAR(initial_policy[3], 5.2, 1e-10);
      }
   }
}

TEST(TabularPolicy, kuhn_poker_states)
{
   using namespace nor;
   using namespace nor::games::kuhn;
   auto tabular_policy = factory::make_tabular_policy(
      std::unordered_map< Infostate, HashmapActionPolicy< int > >{},
      factory::make_uniform_policy< Infostate, HashmapActionPolicy< int > >()
   );

   Environment env{};
   State state{}, next_state{};

   auto istate_alex = Infostate{nor::Player::alex};
   auto istate_bob = Infostate{nor::Player::bob};
   for(auto action : std::vector< std::variant< ChanceOutcome, Action > >{
          ChanceOutcome{kuhn::Player::one, Card::queen},
          ChanceOutcome{kuhn::Player::two, Card::king},
          Action::check,
          Action::bet}) {
      std::visit(
         [&](const auto& a) {
            state.apply_action(a);

            istate_alex.update(
               env.public_observation(state, Action::bet, next_state),
               env.private_observation(nor::Player::alex, state, a, next_state)
            );
            istate_bob.update(
               env.public_observation(state, Action::bet, next_state),
               env.private_observation(nor::Player::bob, state, a, next_state)
            );
         },
         action
      );
   }

   auto actions = std::vector{1, 2, 3, 4, 5};
   auto& policy = tabular_policy[std::pair{istate_alex, actions}];
   for(auto i : actions) {
      ASSERT_NEAR(policy[i], .2, 1e-10);
   }
   policy[3] += 5;
   for(auto i : actions) {
      if(i != 3) {
         ASSERT_NEAR(policy[i], .2, 1e-10);
      } else {
         ASSERT_NEAR(policy[3], 5.2, 1e-10);
      }
   }
   actions = std::vector{10, 11, 12, 13, 14};
   policy = tabular_policy[std::pair{istate_bob, actions}];
   for(auto i : actions) {
      ASSERT_NEAR(policy[i], .2, 1e-10);
   }
   policy[12] += -1;
   for(auto i : actions) {
      if(i != 12) {
         ASSERT_NEAR(policy[i], .2, 1e-10);
      } else {
         ASSERT_NEAR(policy[12], -.8, 1e-10);
      }
   }
}

TEST(StatePolicyView, from_tabular_policy)
{
   using namespace nor::games::rps;

   auto rng = common::create_rng();
   std::uniform_real_distribution< double > dist{0., 1.};

   auto policy_alex = nor::factory::make_tabular_policy(
      std::unordered_map< Infostate, nor::HashmapActionPolicy< Action > >{},
      nor::factory::make_uniform_policy< Infostate, nor::HashmapActionPolicy< Action > >()
   );
   auto policy_bob = nor::factory::make_tabular_policy(
      std::unordered_map< Infostate, nor::HashmapActionPolicy< Action > >{},
      nor::factory::make_uniform_policy< Infostate, nor::HashmapActionPolicy< Action > >()
   );

   Environment env{};
   State state{};
   auto istate_alex = Infostate{nor::Player::alex};
   auto istate_bob = Infostate{nor::Player::bob};
   auto actions = env.actions(nor::Player::alex, state);
   auto& action_policy_alex = policy_alex[std::tie(istate_alex, actions)];
   action_policy_alex[Action::paper] = dist(rng);
   action_policy_alex[Action::scissors] = dist(rng);
   action_policy_alex[Action::rock] = dist(rng);
   nor::normalize_action_policy_inplace(action_policy_alex);
   auto& action_policy_bob = policy_bob[std::tie(istate_bob, actions)];
   action_policy_bob[Action::paper] = dist(rng);
   action_policy_bob[Action::scissors] = dist(rng);
   action_policy_bob[Action::rock] = dist(rng);
   nor::normalize_action_policy_inplace(action_policy_bob);

   for(auto hand : {Action::paper, Action::scissors, Action::rock}) {
      ASSERT_EQ(
         (nor::StatePolicyView< Infostate, Action >{policy_bob}.at(istate_bob)[hand]),
         (action_policy_bob[hand])
      );
   }
}

TEST(BestResponsePolicy, rock_paper_scissors)
{
   using namespace nor::games::rps;
   auto policy_alex = nor::factory::make_tabular_policy(
      std::unordered_map< Infostate, nor::HashmapActionPolicy< Action > >{},
      nor::factory::make_uniform_policy< Infostate, nor::HashmapActionPolicy< Action > >()
   );
   auto policy_bob = nor::factory::make_tabular_policy(
      std::unordered_map< Infostate, nor::HashmapActionPolicy< Action > >{},
      nor::factory::make_uniform_policy< Infostate, nor::HashmapActionPolicy< Action > >()
   );

   Environment env{};
   State state{};
   auto istate_alex = Infostate{nor::Player::alex};
   auto istate_bob = Infostate{nor::Player::bob};
   auto actions = env.actions(nor::Player::alex, state);
   auto& action_policy_alex = policy_alex[std::tie(istate_alex, actions)];
   action_policy_alex[Action::paper] = 1.;
   action_policy_alex[Action::scissors] = 0.;
   action_policy_alex[Action::rock] = 0.;
   auto& action_policy_bob = policy_bob[std::tie(istate_bob, actions)];
   action_policy_bob[Action::paper] = 1.;
   action_policy_bob[Action::scissors] = 0.;
   action_policy_bob[Action::rock] = 0.;

   auto best_response_alex = nor::factory::make_best_response_policy< Infostate, Action >(
      nor::Player::alex
   );
   best_response_alex.allocate(
      env,
      std::unordered_map< nor::Player, nor::StatePolicyView< Infostate, Action > >{
         {nor::Player::bob, nor::StatePolicyView< Infostate, Action >{policy_bob}}},
      std::make_unique< State >()
   );
   EXPECT_NEAR(best_response_alex.root_value(), 1., 1e-5);
   //   policy_alex(auto i : actions)
   //   {
   //      ASSERT_NEAR(policy[i], .2, 1e-10);
   //   }
   //   policy[3] += 5;
   //   for(auto i : actions) {
   //      if(i != 3) {
   //         ASSERT_NEAR(policy[i], .2, 1e-10);
   //      } else {
   //         ASSERT_NEAR(policy[3], 5.2, 1e-10);
   //      }
   //   }
   //   actions = std::vector{10, 11, 12, 13, 14};
   //   policy = tabular_policy[std::pair{istate_bob, actions}];
   //   for(auto i : actions) {
   //      ASSERT_NEAR(policy[i], .2, 1e-10);
   //   }
   //   policy[12] += -1;
   //   for(auto i : actions) {
   //      if(i != 12) {
   //         ASSERT_NEAR(policy[i], .2, 1e-10);
   //      } else {
   //         ASSERT_NEAR(policy[12], -.8, 1e-10);
   //      }
   //   }
}