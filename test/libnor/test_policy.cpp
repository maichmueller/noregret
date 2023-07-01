#include <gtest/gtest.h>

#include <random>
#include <string>

#include "common/common.hpp"
#include "nor/env.hpp"
#include "nor/factory.hpp"
#include "nor/fosg_helpers.hpp"
#include "nor/fosg_states.hpp"
#include "nor/fosg_traits.hpp"
#include "nor/game_defs.hpp"
#include "nor/holder.hpp"
#include "nor/policy/policy.hpp"
#include "nor/rm/policy_value.hpp"
#include "nor/utils/utils.hpp"
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

template < typename T >
   requires nor::concepts::map< T >
void concept_tabular_policy_map_check()
{
}

TEST(TabularPolicy, concept_fulfillment)
{
   using namespace nor;
   concept_tabular_policy_map_check< TabularPolicy< TestInfostate, HashmapActionPolicy< int > > >();
   EXPECT_TRUE((nor::concepts::map< TabularPolicy< TestInfostate, HashmapActionPolicy< int > > >) );
}

TEST(TabularPolicy, uniform_default)
{
   using namespace nor;
   auto tabular_policy = factory::make_tabular_policy< TestInfostate, HashmapActionPolicy< int > >(
   );

   auto istate1 = TestInfostate{Player::alex};
   istate1.update(std::string{"case1"}, std::string{"case1priv"});
   auto actions = std::vector< ActionHolder< int > >{1, 2, 3, 4, 5};
   auto& initial_policy = tabular_policy(
      istate1, actions, factory::make_uniform_policy< TestInfostate, HashmapActionPolicy< int > >()
   );
   for(auto i : actions) {
      ASSERT_NEAR(initial_policy[i], .2, 1e-10);
   }
   initial_policy[3] += 5;
   for(auto i : actions) {
      if(i.uneqals(3)) {
         ASSERT_NEAR(initial_policy[i], .2, 1e-10);
      } else {
         ASSERT_NEAR(initial_policy[3], 5.2, 1e-10);
      }
   }
}

TEST(TabularPolicy, kuhn_poker_states)
{
   using namespace nor;
   using namespace games::kuhn;
   auto tabular_policy = factory::make_tabular_policy< Infostate, HashmapActionPolicy< Action > >();

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

   auto actions = std::vector< ActionHolder< Action > >{
      Action{1}, Action{2}, Action{3}, Action{4}, Action{5}};
   auto& policy = tabular_policy(istate_alex, actions);
   for(auto a : actions) {
      ASSERT_NEAR(policy[a], .2, 1e-10);
   }
   policy[Action{3}] += 5;
   for(Action a : actions) {
      if(a != Action{3}) {
         ASSERT_NEAR(policy[a], .2, 1e-10);
      } else {
         ASSERT_NEAR(policy[Action{3}], 5.2, 1e-10);
      }
   }
   actions = std::vector< ActionHolder< Action > >{
      Action{10}, Action{11}, Action{12}, Action{13}, Action{14}};
   policy = tabular_policy(istate_bob, actions);
   for(Action a : actions) {
      ASSERT_NEAR(policy[a], .2, 1e-10);
   }
   policy[Action{12}] += -1;
   for(Action a : actions) {
      if(a != Action{12}) {
         ASSERT_NEAR(policy[a], .2, 1e-10);
      } else {
         ASSERT_NEAR(policy[Action{12}], -.8, 1e-10);
      }
   }
}

TEST(StatePolicyView, from_tabular_policy)
{
   using namespace nor;
   using namespace games::rps;

   auto rng = common::create_rng();
   std::uniform_real_distribution< double > dist{0., 1.};

   auto policy_alex = factory::make_tabular_policy< Infostate, HashmapActionPolicy< Action > >();
   auto policy_bob = factory::make_tabular_policy< Infostate, HashmapActionPolicy< Action > >();

   Environment env{};
   State state{};
   auto istate_alex = Infostate{Player::alex};
   auto istate_bob = Infostate{Player::bob};
   auto actions = env.actions(Player::alex, state);
   auto& action_policy_alex = policy_alex(istate_alex, actions);
   action_policy_alex[Action::paper] = dist(rng);
   action_policy_alex[Action::scissors] = dist(rng);
   action_policy_alex[Action::rock] = dist(rng);
   normalize_action_policy_inplace(action_policy_alex);
   auto& action_policy_bob = policy_bob(istate_bob, actions);
   action_policy_bob[Action::paper] = dist(rng);
   action_policy_bob[Action::scissors] = dist(rng);
   action_policy_bob[Action::rock] = dist(rng);
   normalize_action_policy_inplace(action_policy_bob);

   for(auto hand : {Action::paper, Action::scissors, Action::rock}) {
      EXPECT_EQ(
         (StatePolicyView< Infostate, Action >{policy_bob}.at(istate_bob).at(hand)),
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
      auto policy_alex = nor::factory::
         make_tabular_policy< Infostate, nor::HashmapActionPolicy< Action > >();
      auto policy_bob = nor::factory::
         make_tabular_policy< Infostate, nor::HashmapActionPolicy< Action > >();

      State state{}, next_state{};
      istate_alex = Infostate{nor::Player::alex};
      istate_bob = Infostate{nor::Player::bob};
      actions = std::invoke([&] {
         auto acts = env.actions(nor::Player::alex, state);
         return ranges::to_vector(acts | ranges::views::transform([](auto action) {
                                     return action.get();
                                  }));
      });
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

   auto player_policy = [&](nor::Player player) -> auto& {
      if(player == nor::Player::alex) {
         return state_policy_alex;
      } else {
         return state_policy_bob;
      }
   };
   auto player_infostate = [&](nor::Player player) -> auto& {
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
   opp_policy(
      player_infostate(opponent), nor::to_holder_vector< Action >(actions, nor::tag::action{})
   ) = input_policy;

   auto best_response = nor::factory::
      make_best_response_policy< Infostate, Action, nor::BRConfig{.store_infostate_values = true} >(
         best_responder
      );

   best_response.allocate(
      env,
      nor::WorldstateHolder< State >{},
      std::unordered_map{std::pair{opponent, nor::StatePolicyView{opp_policy}}}
   );
   // check if the BR value of the computed policy is close to the expected value
   EXPECT_NEAR(best_response.value(infostate), br_value, 1e-5);
   // check if the BR action is one of the expected ones and whether the returned probabilities
   // make sense
   auto br_map = best_response(infostate);
   if(probable_br_actions.size() == 1) {
      // for better failure output we distinguish here
      auto br_action_expected = *probable_br_actions.begin();
      auto [br_action_actual, br_prob_actual] = *br_map.begin();
      EXPECT_TRUE(br_map.size() == 1);
      EXPECT_EQ(br_prob_actual, 1.);
      EXPECT_EQ(br_action_actual, br_action_expected);
   } else {
      EXPECT_EQ(br_map.size(), 1);
      EXPECT_TRUE(ranges::contains(probable_br_actions, br_map.begin()->first.get()));
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
       std::string,  // test case description
       nor::TabularPolicy<
          nor::games::kuhn::Infostate,
          nor::HashmapActionPolicy< nor::games::kuhn::Action > >,  // the input policy
       double  // the br value at root
       > > {
  public:
   BestResponse_KuhnPoker_ParamsF() = default;

  protected:
   nor::games::kuhn::Environment env{};
};

TEST_P(BestResponse_KuhnPoker_ParamsF, kuhn_poker)
{
   using namespace nor;
   using namespace games::kuhn;

   auto [best_responder, _, opp_policy, br_root_value] = GetParam();

   nor::Player opponent = nor::Player(1 - nor::static_to< int >(best_responder));

   auto root_state = State{};
   auto [terminals, infostate_map] = map_histories_to_infostates(env, root_state);

   auto best_response = factory::make_best_response_policy< Infostate, Action >(best_responder);

   best_response.allocate(
      env, WorldstateHolder< State >{}, std::unordered_map{std::pair{opponent, opp_policy}}
   );
   auto value_map = rm::policy_value(
      env,
      WorldstateHolder< State >{},
      std::unordered_map{
         std::pair{best_responder, StatePolicyView{best_response}},
         std::pair{opponent, StatePolicyView{opp_policy}}}
   );

   // check if the BR value of the computed policy is close to the expected value
   EXPECT_NEAR(value_map.get().at(best_responder), br_root_value, 1e-5);
}

namespace kuhn_tests {
using namespace nor::games::kuhn;

using param_tuple_type = typename BestResponse_KuhnPoker_ParamsF::ParamType;

auto uniform_br_expected(nor::Player br_player, const auto& istate_map, auto& policy_map)
{
   auto fetch_infostate = [&](std::string infostate_str, nor::Player player) {
      return *(
         istate_map.find(kuhn_istate_to_history_rep.at(infostate_str))->second.second.at(player)
      );
   };

   using action_holder = nor::ActionHolder< Action >;
   auto& policy = policy_map[br_player];
   if(br_player == nor::Player::alex) {
      policy.emplace(
         fetch_infostate("j?", nor::Player::alex),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{0.}},
            std::pair{action_holder{Action::bet}, ValueChecker{1.}}}
      );
      policy.emplace(
         fetch_infostate("j?cb", nor::Player::alex),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{}},
            std::pair{action_holder{Action::bet}, ValueChecker{}}}
      );
      policy.emplace(
         fetch_infostate("q?", nor::Player::alex),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{0.}},
            std::pair{action_holder{Action::bet}, ValueChecker{1.}}}
      );
      policy.emplace(
         fetch_infostate("q?cb", nor::Player::alex),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{}},
            std::pair{action_holder{Action::bet}, ValueChecker{}}}
      );
      policy.emplace(
         fetch_infostate("k?", nor::Player::alex),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{1.}},
            std::pair{action_holder{Action::bet}, ValueChecker{0.}}}
      );
      policy.emplace(
         fetch_infostate("k?cb", nor::Player::alex),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{}},
            std::pair{action_holder{Action::bet}, ValueChecker{}}}
      );
   } else {
      policy.emplace(
         fetch_infostate("?jc", nor::Player::bob),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{0.}},
            std::pair{action_holder{Action::bet}, ValueChecker{1.}}}
      );
      policy.emplace(
         fetch_infostate("?jb", nor::Player::bob),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{1.}},
            std::pair{action_holder{Action::bet}, ValueChecker{0.}}}
      );
      policy.emplace(
         fetch_infostate("?qc", nor::Player::bob),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{0.}},
            std::pair{action_holder{Action::bet}, ValueChecker{1.}}}
      );
      policy.emplace(
         fetch_infostate("?qb", nor::Player::bob),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{0.}},
            std::pair{action_holder{Action::bet}, ValueChecker{1.}}}
      );
      policy.emplace(
         fetch_infostate("?kc", nor::Player::bob),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{0.}},
            std::pair{action_holder{Action::bet}, ValueChecker{1.}}}
      );
      policy.emplace(
         fetch_infostate("?kb", nor::Player::bob),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{0.}},
            std::pair{action_holder{Action::bet}, ValueChecker{1.}}}
      );
   }
}

auto always_check_br_expected(nor::Player br_player, const auto& istate_map, auto& policy_map)
{
   auto fetch_infostate = [&](std::string infostate_str, nor::Player player) {
      return *(
         istate_map.find(kuhn_istate_to_history_rep.at(infostate_str))->second.second.at(player)
      );
   };

   using action_holder = nor::ActionHolder< Action >;
   auto& policy = policy_map[br_player];
   if(br_player == nor::Player::alex) {
      policy.emplace(
         fetch_infostate("j?", nor::Player::alex),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{0.}},
            std::pair{action_holder{Action::bet}, ValueChecker{1.}}}
      );
      policy.emplace(
         fetch_infostate("j?cb", nor::Player::alex),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{0.}},
            std::pair{action_holder{Action::bet}, ValueChecker{1.}}}
      );
      policy.emplace(
         fetch_infostate("q?", nor::Player::alex),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{0.}},
            std::pair{action_holder{Action::bet}, ValueChecker{1.}}}
      );
      policy.emplace(
         fetch_infostate("q?cb", nor::Player::alex),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{}},
            std::pair{action_holder{Action::bet}, ValueChecker{}}}
      );
      policy.emplace(
         fetch_infostate("k?", nor::Player::alex),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{1.}},
            std::pair{action_holder{Action::bet}, ValueChecker{0.}}}
      );
      policy.emplace(
         fetch_infostate("k?cb", nor::Player::alex),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{}},
            std::pair{action_holder{Action::bet}, ValueChecker{}}}
      );
   } else {
      policy.emplace(
         fetch_infostate("?jc", nor::Player::bob),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{0.}},
            std::pair{action_holder{Action::bet}, ValueChecker{1.}}}
      );
      policy.emplace(
         fetch_infostate("?jb", nor::Player::bob),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{}},
            std::pair{action_holder{Action::bet}, ValueChecker{}}}
      );
      policy.emplace(
         fetch_infostate("?qc", nor::Player::bob),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{0.}},
            std::pair{action_holder{Action::bet}, ValueChecker{1.}}}
      );
      policy.emplace(
         fetch_infostate("?qb", nor::Player::bob),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{}},
            std::pair{action_holder{Action::bet}, ValueChecker{}}}
      );
      policy.emplace(
         fetch_infostate("?kc", nor::Player::bob),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{1.}},
            std::pair{action_holder{Action::bet}, ValueChecker{0.}}}
      );
      policy.emplace(
         fetch_infostate("?kb", nor::Player::bob),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{}},
            std::pair{action_holder{Action::bet}, ValueChecker{}}}
      );
   }
}

auto always_bet_br_expected(nor::Player br_player, const auto& istate_map, auto& policy_map)
{
   auto fetch_infostate = [&](std::string infostate_str, nor::Player player) {
      return *(
         istate_map.find(kuhn_istate_to_history_rep.at(infostate_str))->second.second.at(player)
      );
   };
   using action_holder = nor::ActionHolder< Action >;
   auto& policy = policy_map[br_player];

   if(br_player == nor::Player::alex) {
      policy.emplace(
         fetch_infostate("j?", nor::Player::alex),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{1.}},
            std::pair{action_holder{Action::bet}, ValueChecker{0.}}}
      );
      policy.emplace(
         fetch_infostate("j?cb", nor::Player::alex),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{1.}},
            std::pair{action_holder{Action::bet}, ValueChecker{0.}}}
      );
      policy.emplace(
         fetch_infostate("q?", nor::Player::alex),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{1.}},
            std::pair{action_holder{Action::bet}, ValueChecker{0.}}}
      );
      policy.emplace(
         fetch_infostate("q?cb", nor::Player::alex),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{0.}},
            std::pair{action_holder{Action::bet}, ValueChecker{1.}}}
      );
      policy.emplace(
         fetch_infostate("k?", nor::Player::alex),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{1.}},
            std::pair{action_holder{Action::bet}, ValueChecker{0.}}}
      );
      policy.emplace(
         fetch_infostate("k?cb", nor::Player::alex),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{0.}},
            std::pair{action_holder{Action::bet}, ValueChecker{1.}}}
      );
   } else {
      policy.emplace(
         fetch_infostate("?jc", nor::Player::bob),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{}},
            std::pair{action_holder{Action::bet}, ValueChecker{}}}
      );
      policy.emplace(
         fetch_infostate("?jb", nor::Player::bob),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{1.}},
            std::pair{action_holder{Action::bet}, ValueChecker{0.}}}
      );
      policy.emplace(
         fetch_infostate("?qc", nor::Player::bob),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{}},
            std::pair{action_holder{Action::bet}, ValueChecker{}}}
      );
      policy.emplace(
         fetch_infostate("?qb", nor::Player::bob),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{0.}},
            std::pair{action_holder{Action::bet}, ValueChecker{1.}}}
      );
      policy.emplace(
         fetch_infostate("?kc", nor::Player::bob),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{}},
            std::pair{action_holder{Action::bet}, ValueChecker{}}}
      );
      policy.emplace(
         fetch_infostate("?kb", nor::Player::bob),
         std::unordered_map{
            std::pair{action_holder{Action::check}, ValueChecker{0.}},
            std::pair{action_holder{Action::bet}, ValueChecker{1.}}}
      );
   }
}

INSTANTIATE_TEST_SUITE_P(
   all,
   BestResponse_KuhnPoker_ParamsF,
   testing::ValuesIn(std::invoke([] {
      std::vector< param_tuple_type > vec_out;
      vec_out.reserve(3);

      auto [uniform_policy_alex, uniform_policy_bob] = kuhn_policy_always_mix_like(0.5, 0.5);
      auto [always_check_policy_alex, always_check_policy_bob] = kuhn_policy_always_mix_like(
         1., 0.
      );
      auto [always_bet_policy_alex, always_bet_policy_bob] = kuhn_policy_always_mix_like(0., 1.);

      vec_out.emplace_back(
         nor::Player::alex,  // best responder
         "opponent_policy_uniform",  // policy test name (modified by string functor below)
         uniform_policy_bob,  // opponent policy
         0.5  // br policy value
      );
      vec_out.emplace_back(
         nor::Player::alex, "opponent_policy_always_check", always_check_policy_bob, 1.
      );
      vec_out.emplace_back(
         nor::Player::alex, "opponent_policy_always_bet", always_bet_policy_bob, 1. / 3.
      );

      vec_out.emplace_back(
         nor::Player::bob, "opponent_policy_uniform", uniform_policy_alex, 0.4 + 1. / 60.
      );
      vec_out.emplace_back(
         nor::Player::bob, "opponent_policy_always_check", always_check_policy_alex, 1.
      );
      vec_out.emplace_back(
         nor::Player::bob, "opponent_policy_always_bet", always_bet_policy_alex, 1. / 3.
      );

      return vec_out;
   })),
   [](const auto& params) {
      return common::replace_all(
         common::to_string(std::get< 0 >(params.param)) + "_" + std::get< 1 >(params.param),
         " ",
         "_"
      );
   }
);

}  // namespace kuhn_tests
