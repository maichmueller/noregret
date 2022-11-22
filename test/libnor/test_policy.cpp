#include <gtest/gtest.h>

#include <random>
#include <string>

#include "common/common.hpp"
#include "nor/env.hpp"
#include "nor/factory.hpp"
#include "nor/fosg_helpers.hpp"
#include "nor/fosg_states.hpp"
#include "nor/game_defs.hpp"
#include "nor/policy/policy.hpp"
#include "nor/rm/policy_value.hpp"
#include "nor/game_defs.hpp"
#include "rm_specific_testing_utils.hpp"

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

   best_response.allocate(
      env,
      std::unordered_map{std::pair{opponent, nor::StatePolicyView{opp_policy}}},
      std::make_unique< State >()
   );
   // check if the BR value of the computed policy is close to the expected value
   EXPECT_NEAR(best_response.value(infostate), br_value, 1e-5);
   // check if the BR action is one of the expected ones and whether the returned probabilities make
   // sense
   auto br_map = best_response(infostate);
   if(probable_br_actions.size() == 1) {
      // for better failure output we distinguish here
      auto br_action_expected = *probable_br_actions.begin();
      auto [br_action_actual, br_prob_actual] = *br_map.begin();
      EXPECT_TRUE(br_map.size() == 1);
      EXPECT_EQ(br_prob_actual, 1.);
      EXPECT_EQ(br_action_actual, br_action_expected);
   } else {
      EXPECT_TRUE(ranges::any_of(br_map, [&](const auto& action_prob_pair) {
         return ranges::contains(probable_br_actions, std::get< 0 >(action_prob_pair));
      }));
   }
}

namespace rps_tests {
using namespace nor::games::rps;

template < typename T1, typename... Ts >
std::tuple< Ts... > leftshift_tuple(std::tuple< T1, Ts... > tuple);

using param_tuple_type = typename BestResponse_RPS_ParamsF::ParamType;
using no_player_param_tuple = decltype(leftshift_tuple(std::declval< param_tuple_type >()));

INSTANTIATE_TEST_SUITE_P(all, BestResponse_RPS_ParamsF, testing::ValuesIn(std::invoke([] {
                            auto br_rps_values = std::array{
                               no_player_param_tuple{
                                  {std::pair{Action::rock, 1.},
                                   std::pair{Action::paper, 0.},
                                   std::pair{Action::scissors, 0.}},
                                  std::vector{Action::paper},
                                  1.},
                               no_player_param_tuple{
                                  {std::pair{Action::rock, 0.},
                                   std::pair{Action::paper, 1.},
                                   std::pair{Action::scissors, 0.}},
                                  std::vector{Action::scissors},
                                  1.},
                               no_player_param_tuple{
                                  {std::pair{Action::rock, 0.},
                                   std::pair{Action::paper, 0.},
                                   std::pair{Action::scissors, 1.}},
                                  std::vector{Action::rock},
                                  1.},
                               no_player_param_tuple{
                                  {std::pair{Action::rock, .5},
                                   std::pair{Action::paper, .5},
                                   std::pair{Action::scissors, 0.}},
                                  std::vector{Action::paper},
                                  .5},
                               no_player_param_tuple{
                                  {std::pair{Action::rock, .3},
                                   std::pair{Action::paper, .7},
                                   std::pair{Action::scissors, 0.}},
                                  std::vector{Action::scissors},
                                  .4},
                               no_player_param_tuple{
                                  {std::pair{Action::rock, .2},
                                   std::pair{Action::paper, .2},
                                   std::pair{Action::scissors, 0.6}},
                                  std::vector{Action::rock},
                                  .4},
                               no_player_param_tuple{
                                  {std::pair{Action::rock, .3},
                                   std::pair{Action::paper, .3},
                                   std::pair{Action::scissors, 0.4}},
                                  std::vector{Action::rock},
                                  .1},
                               no_player_param_tuple{
                                  {std::pair{Action::rock, 1. / .3},
                                   std::pair{Action::paper, 1. / .3},
                                   std::pair{Action::scissors, 1. / .3}},
                                  std::vector{Action::rock, Action::paper, Action::scissors},
                                  0.},
                               no_player_param_tuple{
                                  {std::pair{Action::rock, .5},
                                   std::pair{Action::paper, .25},
                                   std::pair{Action::scissors, 0.25}},
                                  std::vector{Action::paper},
                                  .25}};

                            std::vector< param_tuple_type > vec_out;
                            for(const auto& param_tuple : br_rps_values) {
                               for(auto player : {nor::Player::alex, nor::Player::bob}) {
                                  vec_out.emplace_back(
                                     std::tuple_cat(std::forward_as_tuple(player), param_tuple)
                                  );
                               }
                            }
                            return vec_out;
                         })));

}  // namespace rps_tests

class BestResponse_KuhnPoker_ParamsF:
    public ::testing::TestWithParam< std::tuple<
       nor::Player,  // the best responder
       nor::TabularPolicy<
          nor::games::kuhn::Infostate,
          nor::HashmapActionPolicy< nor::games::kuhn::Action > >,  // the input policy
       nor::TabularPolicy<
          nor::games::kuhn::Infostate,
          nor::HashmapActionPolicy< nor::games::kuhn::Action > >,  // expected best response policy
       double  // the br value at root
       > > {
  public:
   BestResponse_KuhnPoker_ParamsF() = default;

  protected:
   nor::games::kuhn::Environment env{};
};

TEST_P(BestResponse_KuhnPoker_ParamsF, kuhn_poker)
{
   auto [best_responder, opp_policy, expected_br_policy, br_root_value] = GetParam();
   using namespace nor::games::kuhn;

   nor::Player opponent = nor::Player(1 - static_cast< int >(best_responder));

   auto root_state = State{};
   auto [terminals, infostate_map] = nor::map_histories_to_infostates(env, root_state);
   using history_type = typename decltype(terminals)::value_type;
   auto best_response = nor::factory::make_best_response_policy< Infostate, Action >(best_responder
   );

   best_response.allocate(
      env,
      std::unordered_map{std::pair{opponent, nor::StatePolicyView{opp_policy}}},
      std::make_unique< State >()
   );

   for(const auto& [history, active_player_vec_infostate_map] : infostate_map) {
      const auto& [active_players, infostates] = active_player_vec_infostate_map;
      if(common::contains(active_players, best_responder)) {
         const auto& infostate = *infostates.at(best_responder);
         EXPECT_EQ(best_response(infostate), expected_br_policy(infostate));
      }
   }
   auto value = policy_value(
      env,
      State{},
      nor::player_hash_map{
         std::pair{best_responder, nor::StatePolicyView{best_response}},
         std::pair{opponent, nor::StatePolicyView{opp_policy}}}
   );
   // check if the BR value of the computed policy is close to the expected value
   EXPECT_NEAR(
      value,
      br_root_value,
      1e-5
   );
}

namespace kuhn_tests {
using namespace nor::games::kuhn;

using param_tuple_type = typename BestResponse_KuhnPoker_ParamsF::ParamType;

INSTANTIATE_TEST_SUITE_P(all, BestResponse_KuhnPoker_ParamsF, testing::ValuesIn(std::invoke([] {
                            std::vector< param_tuple_type > vec_out;

                            auto uniform_policy = kuhn_policy_always_mix_like(0.5, 0.5);
                            auto always_check_policy = kuhn_policy_always_mix_like(1.0, 0.);
                            auto always_bet_policy = kuhn_policy_always_mix_like(0., 1.);

                            return vec_out;
                         })));

}  // namespace kuhn_tests