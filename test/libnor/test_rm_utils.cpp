
#include <gtest/gtest.h>

#include "nor/env.hpp"
#include "nor/nor.hpp"
#include "nor/policy/policy.hpp"
#include "nor/rm/rm_utils.hpp"

using namespace nor;

TEST(InfostateNode, storage_correctness)
{
   using namespace nor::games::kuhn;

   std::unordered_map<
      sptr< Infostate >,
      rm::InfostateNodeData< Action >,
      decltype([](const sptr< Infostate >& ptr) { return std::hash< Infostate >{}(*ptr); }),
      decltype([](const sptr< Infostate >& ptr1, const sptr< Infostate >& ptr2) {
         return *ptr1 == *ptr2;
      }) >
      infonode_map{};

   Environment env{};
   State state{}, next_state{};
   auto first_istate_alex = std::make_shared< Infostate >(nor::Player::alex);
   auto istate_bob = std::make_shared< Infostate >(nor::Player::bob);
   auto chance_action = ChanceOutcome{kuhn::Player::one, Card::queen};
   auto player_action = Action::bet;

   next_state.apply_action(chance_action);
   first_istate_alex->update(
      env.public_observation(state, chance_action, next_state),
      env.private_observation(nor::Player::alex, state, chance_action, next_state)
   );
   istate_bob->update(
      env.public_observation(state, chance_action, next_state),
      env.private_observation(nor::Player::alex, state, chance_action, next_state)
   );
   state = next_state;

   chance_action = ChanceOutcome{kuhn::Player::two, Card::king};
   next_state.apply_action(chance_action);
   first_istate_alex->update(
      env.public_observation(state, chance_action, next_state),
      env.private_observation(nor::Player::alex, state, chance_action, next_state)
   );
   istate_bob->update(
      env.public_observation(state, chance_action, next_state),
      env.private_observation(nor::Player::alex, state, chance_action, next_state)
   );
   state = next_state;

   infonode_map.emplace(
      first_istate_alex, rm::InfostateNodeData< Action >{env.actions(nor::Player::alex, state)}
   );

   next_state.apply_action(player_action);
   first_istate_alex->update(
      env.public_observation(state, player_action, next_state),
      env.private_observation(nor::Player::alex, state, player_action, next_state)
   );
   istate_bob->update(
      env.public_observation(state, player_action, next_state),
      env.private_observation(nor::Player::alex, state, player_action, next_state)
   );
   state = next_state;

   infonode_map.emplace(
      istate_bob, rm::InfostateNodeData< Action >{env.actions(nor::Player::bob, state)}
   );

   auto second_istate_alex = std::make_shared< Infostate >(*first_istate_alex);
   next_state.apply_action(player_action);
   second_istate_alex->update(
      env.public_observation(state, player_action, next_state),
      env.private_observation(nor::Player::alex, state, player_action, next_state)
   );
   istate_bob->update(
      env.public_observation(state, player_action, next_state),
      env.private_observation(nor::Player::alex, state, player_action, next_state)
   );
   state = next_state;

   auto& second_alex_node_data = infonode_map
                                    .emplace(
                                       second_istate_alex,
                                       rm::InfostateNodeData< Action >{
                                          env.actions(nor::Player::alex, state)}
                                    )
                                    .first->second;
   second_alex_node_data.regret(games::kuhn::Action::check) += 5;
   second_alex_node_data.regret(games::kuhn::Action::bet) -= 10;

   auto& second_alex_node_data_other_ref = infonode_map.at(
      std::make_shared< games::kuhn::Infostate >(*second_istate_alex)
   );

   ASSERT_EQ(
      second_alex_node_data.regret(games::kuhn::Action::check),
      second_alex_node_data_other_ref.regret(games::kuhn::Action::check)
   );
   ASSERT_EQ(
      second_alex_node_data.regret(games::kuhn::Action::bet),
      second_alex_node_data_other_ref.regret(games::kuhn::Action::bet)
   );
}

class RegretMatchingParamsF:
    public ::testing::TestWithParam< std::tuple<
       std::vector< double >,  // regret values
       std::unordered_map< int, double >,  // the expected policy
       nor::HashmapActionPolicy< int >  // the policy
       > > {
  protected:
   std::vector< int > actions{1, 2, 3, 4, 5};
};

TEST_P(RegretMatchingParamsF, integer_actions)
{
   auto [regret, expected, policy] = GetParam();

   std::unordered_map< int, double > regret_map;
   for(auto [a, r] : ranges::views::zip(actions, regret)) {
      regret_map[a] = r;
   }
   rm::regret_matching(policy, regret_map);
   ASSERT_EQ(policy, HashmapActionPolicy< int >{expected});
}

template < size_t N >
auto value_pack();

template <>
auto value_pack< 0 >()
{
   std::vector< double > r{1., 2., 3., 4., 5.};
   std::unordered_map< int, double > exp{
      {1, 1. / 15.}, {2, 2. / 15.}, {3, 3. / 15.}, {4, 4. / 15.}, {5, 5. / 15.}};
   nor::HashmapActionPolicy< int > pol{std::unordered_map< int, double >{
      {1, 1. / 15.}, {2, 2. / 15.}, {3, 3. / 15.}, {4, 4. / 15.}, {5, 5. / 15.}}};
   return std::tuple{r, exp, pol};
}

template <>
auto value_pack< 1 >()
{
   std::vector< double > r{1, -1, 1, -1, 1};
   std::unordered_map< int, double > exp{
      {1, 1. / 3.}, {2, 0.}, {3, 1. / 3.}, {4, 0.}, {5, 1. / 3.}};
   nor::HashmapActionPolicy< int > pol{std::unordered_map< int, double >{
      {1, 1. / 15.}, {2, 2. / 15.}, {3, 3. / 15.}, {4, 4. / 15.}, {5, 5. / 15.}}};
   return std::tuple{r, exp, pol};
}

template <>
auto value_pack< 2 >()
{
   std::vector< double > r{-1, -1, 0, -1, -1};
   std::unordered_map< int, double > exp{{1, 0.2}, {2, 0.2}, {3, 0.2}, {4, 0.2}, {5, 0.2}};
   nor::HashmapActionPolicy< int > pol{std::unordered_map< int, double >{
      {1, 1. / 15.}, {2, 2. / 15.}, {3, 3. / 15.}, {4, 4. / 15.}, {5, 5. / 15.}}};
   return std::tuple{r, exp, pol};
}

INSTANTIATE_TEST_SUITE_P(
   integer_actions_simple_test,
   RegretMatchingParamsF,
   ::testing::Values(value_pack< 0 >(), value_pack< 1 >(), value_pack< 2 >())
);
