#include <gtest/gtest.h>

#include <string>

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
   auto tabular_policy = nor::rm::factory::make_tabular_policy(
      std::unordered_map< TestInfostate, nor::HashmapActionPolicy< int > >{},
      nor::rm::factory::make_uniform_policy< TestInfostate, nor::HashmapActionPolicy< int > >());

   auto istate1 = TestInfostate{nor::Player::alex};
   istate1.append("case1");
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
   auto tabular_policy = nor::rm::factory::make_tabular_policy(
      std::unordered_map< nor::games::kuhn::Infostate, nor::HashmapActionPolicy< int > >{},
      nor::rm::factory::
         make_uniform_policy< nor::games::kuhn::Infostate, nor::HashmapActionPolicy< int > >());

   nor::games::kuhn::Environment env{};
   nor::games::kuhn::State state{};
   auto istate_alex = nor::games::kuhn::Infostate{nor::Player::alex};
   auto istate_bob = nor::games::kuhn::Infostate{nor::Player::bob};
   state.apply_action(nor::games::kuhn::ChanceOutcome{
      nor::games::kuhn::Player::one, nor::games::kuhn::Card::queen});
   state.apply_action(
      nor::games::kuhn::ChanceOutcome{nor::games::kuhn::Player::two, nor::games::kuhn::Card::king});
   state.apply_action(nor::games::kuhn::Action::check);
   state.apply_action(nor::games::kuhn::Action::bet);

   istate_alex.append(env.private_observation(nor::Player::alex, state));
   istate_bob.append(env.private_observation(nor::Player::bob, state));
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

TEST(BestResponsePolicy, rock_paper_scissors)
{
   using namespace nor::games::rps;
   auto tabular_policy = nor::rm::factory::make_tabular_policy(
      std::unordered_map< Infostate, nor::HashmapActionPolicy< int > >{},
      nor::rm::factory::make_uniform_policy< Infostate, nor::HashmapActionPolicy< int > >());

   Environment env{};
   State state{};
   auto istate_alex = Infostate{nor::Player::alex};
   auto istate_bob = Infostate{nor::Player::bob};
   auto actions = env.actions(nor::Player::alex, state);
   auto& policy_alex = tabular_policy[std::pair{istate_alex, actions}];
   policy_alex[Action{Team::one, Hand::paper}] = 1.;
   policy_alex[Action{Team::one, Hand::scissors}] = 0.;
   policy_alex[Action{Team::one, Hand::rock}] = 0.;
   auto& policy_bob = tabular_policy[std::pair{istate_bob, actions}];
   policy_bob[Action{Team::two, Hand::paper}] = 1.;
   policy_bob[Action{Team::two, Hand::scissors}] = 0.;
   policy_bob[Action{Team::two, Hand::rock}] = 0.;

   auto best_response_bob = nor::rm::BestResponsePolicy{nor::Player::bob, env, };


   policy_alex(auto i : actions) {
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