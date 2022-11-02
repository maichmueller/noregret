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
      std::unordered_map< TestInfostate, nor::HashmapActionPolicy< int > >{}
   );

   auto istate1 = TestInfostate{nor::Player::alex};
   istate1.update("case1", "case1priv");
   auto actions = std::vector< int >{1, 2, 3, 4, 5};
   auto& initial_policy = tabular_policy(
      istate1,
      actions,
      nor::factory::make_uniform_policy< TestInfostate, nor::HashmapActionPolicy< int > >()
   );
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
      std::unordered_map< Infostate, HashmapActionPolicy< int > >{}
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
   auto& policy = tabular_policy(istate_alex, actions);
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
   policy = tabular_policy(istate_bob, actions);
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
      std::unordered_map< Infostate, nor::HashmapActionPolicy< Action > >{}
   );
   auto policy_bob = nor::factory::make_tabular_policy(
      std::unordered_map< Infostate, nor::HashmapActionPolicy< Action > >{}
   );

   Environment env{};
   State state{};
   auto istate_alex = Infostate{nor::Player::alex};
   auto istate_bob = Infostate{nor::Player::bob};
   auto actions = env.actions(nor::Player::alex, state);
   auto& action_policy_alex = policy_alex(istate_alex, actions);
   action_policy_alex[Action::paper] = dist(rng);
   action_policy_alex[Action::scissors] = dist(rng);
   action_policy_alex[Action::rock] = dist(rng);
   nor::normalize_action_policy_inplace(action_policy_alex);
   auto& action_policy_bob = policy_bob(istate_bob, actions);
   action_policy_bob[Action::paper] = dist(rng);
   action_policy_bob[Action::scissors] = dist(rng);
   action_policy_bob[Action::rock] = dist(rng);
   nor::normalize_action_policy_inplace(action_policy_bob);

   for(auto hand : {Action::paper, Action::scissors, Action::rock}) {
      EXPECT_EQ(
         (nor::StatePolicyView< Infostate, Action >{policy_bob}.at(istate_bob).at(hand)),
         (action_policy_bob[hand])
      );
   }
}

class BestResponse_RPS_ParamsF:
    public ::testing::TestWithParam< std::tuple<
       nor::Player,  // the best responder
       nor::HashmapActionPolicy< nor::games::rps::Action >,  // the input policy
       std::vector< nor::games::rps::Action >,  // list of probable best response actions
       double  // the br value
       > > {
  public:
   BestResponse_RPS_ParamsF() : istate_alex(nor::Player::alex), istate_bob(nor::Player::bob) {}

   void SetUp() override
   {
      using namespace nor::games::rps;
      auto policy_alex = nor::factory::make_tabular_policy(
         std::unordered_map< Infostate, nor::HashmapActionPolicy< Action > >{}
      );
      auto policy_bob = nor::factory::make_tabular_policy(
         std::unordered_map< Infostate, nor::HashmapActionPolicy< Action > >{}
      );

      State state{}, next_state{};
      istate_alex = Infostate{nor::Player::alex};
      istate_bob = Infostate{nor::Player::bob};
      actions = env.actions(nor::Player::alex, state);
      next_state.apply_action(actions[0]);
      istate_bob.update(
         env.public_observation(state, actions[0], next_state),
         env.private_observation(nor::Player::bob, state, actions[0], next_state)
      );
   }

  protected:
   nor::games::rps::Infostate istate_alex;
   nor::games::rps::Infostate istate_bob;
   nor::TabularPolicy<
      nor::games::rps::Infostate,
      nor::HashmapActionPolicy< nor::games::rps::Action > >
      state_policy_alex{};
   nor::TabularPolicy<
      nor::games::rps::Infostate,
      nor::HashmapActionPolicy< nor::games::rps::Action > >
      state_policy_bob{};
   std::vector< nor::games::rps::Action > actions{};
   nor::games::rps::Environment env{};
};

TEST_P(BestResponse_RPS_ParamsF, rock_paper_scissors)
{
   auto [best_responder, input_policy, probable_br_actions, br_value] = GetParam();
   using namespace nor::games::rps;

   auto player_policy = [&](nor::Player player) -> auto&
   {
      if(player == nor::Player::alex) {
         return state_policy_alex;
      } else {
         return state_policy_bob;
      }
   };
   auto player_infostate = [&](nor::Player player) -> auto&
   {
      if(player == nor::Player::alex) {
         return istate_alex;
      } else {
         return istate_bob;
      }
   };
   auto opp = [](nor::Player player) {
      if(player == nor::Player::alex) {
         return nor::Player::bob;
      } else {
         return nor::Player::alex;
      }
   };

   nor::Player opponent = opp(best_responder);
   auto& infostate = player_infostate(best_responder);
   auto& opp_policy = player_policy(opponent);
   opp_policy(player_infostate(opponent), actions) = input_policy;

   auto best_response = nor::factory::make_best_response_policy< Infostate, Action >(best_responder
   );
   auto view = nor::StatePolicyView< Infostate, Action >{opp_policy};
   auto res = view.at(player_infostate(opponent));
   best_response.allocate(
      env,
      std::unordered_map{
         std::pair{opponent, nor::StatePolicyView< Infostate, Action >{opp_policy}}},
      std::make_unique< State >()
   );
   EXPECT_NEAR(best_response.value(infostate), br_value, 1e-5);
   auto br_map = best_response(infostate);
   EXPECT_TRUE(ranges::any_of(br_map, [&](const auto& action_prob_pair) {
      return ranges::contains(probable_br_actions, std::get< 0 >(action_prob_pair));
   }));
   for(auto action : {Action::scissors, Action::rock, Action::paper}) {
      EXPECT_EQ(br_map[action], 1. * ranges::contains(probable_br_actions, action));
   }
}

namespace {
using namespace nor::games::rps;

using param_tuple = typename BestResponse_RPS_ParamsF::ParamType;

INSTANTIATE_TEST_SUITE_P(
   best_responses_in_rock_paper_scissors_alex,
   BestResponse_RPS_ParamsF,
   ::testing::Values(
      param_tuple{
         nor::Player::alex,
         {std::pair{Action::rock, 1.},
          std::pair{Action::paper, 0.},
          std::pair{Action::scissors, 0.}},
         std::vector{Action::paper},
         1.},
      param_tuple{
         nor::Player::alex,
         {std::pair{Action::rock, 0.},
          std::pair{Action::paper, 1.},
          std::pair{Action::scissors, 0.}},
         std::vector{Action::scissors},
         1.},
      param_tuple{
         nor::Player::alex,
         {std::pair{Action::rock, 0.},
          std::pair{Action::paper, 0.},
          std::pair{Action::scissors, 1.}},
         std::vector{Action::rock},
         1.},
      param_tuple{
         nor::Player::alex,
         {std::pair{Action::rock, .5},
          std::pair{Action::paper, .5},
          std::pair{Action::scissors, 0.}},
         std::vector{Action::paper},
         .5},
      param_tuple{
         nor::Player::alex,
         {std::pair{Action::rock, .3},
          std::pair{Action::paper, .7},
          std::pair{Action::scissors, 0.}},
         std::vector{Action::scissors},
         .4},
      param_tuple{
         nor::Player::alex,
         {std::pair{Action::rock, .2},
          std::pair{Action::paper, .2},
          std::pair{Action::scissors, 0.6}},
         std::vector{Action::rock},
         .4},
      param_tuple{
         nor::Player::alex,
         {std::pair{Action::rock, .3},
          std::pair{Action::paper, .3},
          std::pair{Action::scissors, 0.4}},
         std::vector{Action::rock},
         .1},
      param_tuple{
         nor::Player::alex,
         {std::pair{Action::rock, .5},
          std::pair{Action::paper, .25},
          std::pair{Action::scissors, 0.25}},
         std::vector{Action::paper},
         .25}
   )
);


INSTANTIATE_TEST_SUITE_P(
   best_responses_in_rock_paper_scissors_bob,
   BestResponse_RPS_ParamsF,
   ::testing::Values(
      param_tuple{
         nor::Player::bob,
         {std::pair{Action::rock, 1.},
          std::pair{Action::paper, 0.},
          std::pair{Action::scissors, 0.}},
         std::vector{Action::paper},
         1.},
      param_tuple{
         nor::Player::bob,
         {std::pair{Action::rock, 0.},
          std::pair{Action::paper, 1.},
          std::pair{Action::scissors, 0.}},
         std::vector{Action::scissors},
         1.},
      param_tuple{
         nor::Player::bob,
         {std::pair{Action::rock, 0.},
          std::pair{Action::paper, 0.},
          std::pair{Action::scissors, 1.}},
         std::vector{Action::rock},
         1.},
      param_tuple{
         nor::Player::bob,
         {std::pair{Action::rock, .5},
          std::pair{Action::paper, .5},
          std::pair{Action::scissors, 0.}},
         std::vector{Action::paper},
         .5},
      param_tuple{
         nor::Player::bob,
         {std::pair{Action::rock, .3},
          std::pair{Action::paper, .7},
          std::pair{Action::scissors, 0.}},
         std::vector{Action::scissors},
         .4},
      param_tuple{
         nor::Player::bob,
         {std::pair{Action::rock, .2},
          std::pair{Action::paper, .2},
          std::pair{Action::scissors, 0.6}},
         std::vector{Action::rock},
         .4},
      param_tuple{
         nor::Player::bob,
         {std::pair{Action::rock, .3},
          std::pair{Action::paper, .3},
          std::pair{Action::scissors, 0.4}},
         std::vector{Action::rock},
         .1},
      param_tuple{
         nor::Player::bob,
         {std::pair{Action::rock, .5},
          std::pair{Action::paper, .25},
          std::pair{Action::scissors, 0.25}},
         std::vector{Action::paper},
         .25}
   )
);

}  // namespace