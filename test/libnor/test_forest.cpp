#include <gtest/gtest.h>

#include <memory>
#include <utility>
#include <variant>

#include "nor/env.hpp"
#include "nor/policy/action_policy.hpp"
#include "nor/rm/forest.hpp"

using namespace nor;

namespace {

struct RpsPolicy {
   using info_state_type = games::rps::Infostate;
   using action_type = games::rps::Action;
   using action_policy_type = HashmapActionPolicy< action_type >;

   RpsPolicy()
   {
      policy.emplace(action_type::rock, .25);
      policy.emplace(action_type::paper, .5);
      policy.emplace(action_type::scissors, .25);
   }

   action_policy_type operator()(const info_state_type&) const { return policy; }
   action_policy_type at(const info_state_type&) const { return policy; }
   size_t size() const { return policy.size(); }

   action_policy_type policy;
};

struct KuhnPolicy {
   using info_state_type = games::kuhn::Infostate;
   using action_type = games::kuhn::Action;
   using action_policy_type = HashmapActionPolicy< action_type >;

   KuhnPolicy()
   {
      policy.emplace(action_type::check, .25);
      policy.emplace(action_type::bet, .75);
   }

   action_policy_type operator()(const info_state_type&) const { return policy; }
   action_policy_type at(const info_state_type&) const { return policy; }
   size_t size() const { return policy.size(); }

   action_policy_type policy;
};

template < typename Tree >
typename Tree::player_policies_type kuhn_policies()
{
   typename Tree::player_policies_type policies;
   policies.emplace(Player::alex, typename Tree::player_policy_type{KuhnPolicy{}});
   policies.emplace(Player::bob, typename Tree::player_policy_type{KuhnPolicy{}});
   return policies;
}

}  // namespace

TEST(InfostateTree, deterministicOwnerReachAndNodeOwnership)
{
   using Environment = games::rps::Environment;
   using Tree = forest::InfostateTree< Environment >;

   Environment env{};
   auto root_state = std::make_unique< games::rps::State >();
   auto* root_address = root_state.get();
   Tree tree{env, std::move(root_state)};

   ASSERT_EQ(&tree.root_state(), root_address);
   ASSERT_EQ(tree.root_node().active_player, Player::alex);
   ASSERT_EQ(tree.root_node().children.size(), size_t{3});
   ASSERT_TRUE(tree.root_node().infostate);
   EXPECT_EQ(tree.root_node().infostate->player(), Player::alex);

   // Moving the owner must not leave child parent pointers aimed at the moved-from root object.
   Tree moved_tree{std::move(tree)};
   ASSERT_EQ(moved_tree.root_node().children.size(), size_t{3});

   typename Tree::player_policies_type policies;
   policies.emplace(Player::bob, typename Tree::player_policy_type{RpsPolicy{}});
   moved_tree.build(Player::alex, std::move(policies));

   for(auto& [action_variant, child_tuple] : moved_tree.root_node().children) {
      ASSERT_TRUE(std::holds_alternative< games::rps::Action >(action_variant));
      auto& [child_node, action_probability, action_value] = child_tuple;
      ASSERT_TRUE(child_node);
      EXPECT_EQ(child_node->parent, &moved_tree.root_node());
      EXPECT_EQ(child_node->active_player, Player::bob);
      ASSERT_EQ(child_node->children.size(), size_t{3});

      EXPECT_TRUE(action_probability);
      EXPECT_DOUBLE_EQ(action_probability->get(), 1.);
      EXPECT_FALSE(action_value);

      for(auto& [bob_action_variant, bob_child_tuple] : child_node->children) {
         ASSERT_TRUE(std::holds_alternative< games::rps::Action >(bob_action_variant));
         auto& [terminal_node, bob_action_probability, bob_action_value] = bob_child_tuple;
         EXPECT_FALSE(terminal_node);
         ASSERT_TRUE(bob_action_probability);
         const auto bob_action = std::get< games::rps::Action >(bob_action_variant);
         const auto expected_probability = bob_action == games::rps::Action::paper ? .5 : .25;
         EXPECT_DOUBLE_EQ(bob_action_probability->get(), expected_probability);
         ASSERT_TRUE(bob_action_value);
      }
   }

   EXPECT_THROW(moved_tree.build(Player::chance, {}), std::invalid_argument);
}

TEST(InfostateTree, stochasticChanceProbabilitiesAndBufferedObservations)
{
   using Environment = games::kuhn::Environment;
   using ChanceOutcome = games::kuhn::ChanceOutcome;
   using Tree = forest::InfostateTree< Environment >;

   Environment env{};
   Tree tree{env, std::make_unique< games::kuhn::State >()};
   ASSERT_EQ(tree.root_node().active_player, Player::chance);
   ASSERT_FALSE(tree.root_node().infostate);
   ASSERT_FALSE(tree.root_node().children.empty());

   auto root_entry = tree.root_node().children.begin();
   const auto first_outcome = std::get< ChanceOutcome >(root_entry->first);
   auto first_state = child_state(env, tree.root_state(), first_outcome);
   const auto second_outcome = env.chance_actions(*first_state).front();
   auto second_state = child_state(env, *first_state, second_outcome);
   const auto decision_player = env.active_player(*second_state);
   ASSERT_TRUE(utils::is_actual_player_pred(decision_player));

   auto expected_infostate = games::kuhn::Infostate{decision_player};
   expected_infostate.update(
      env.public_observation(tree.root_state(), first_outcome, *first_state),
      env.private_observation(decision_player, tree.root_state(), first_outcome, *first_state)
   );
   expected_infostate.update(
      env.public_observation(*first_state, second_outcome, *second_state),
      env.private_observation(decision_player, *first_state, second_outcome, *second_state)
   );

   tree.build(Player::alex, kuhn_policies< Tree >());

   auto& [first_node, first_probability, first_value] = root_entry->second;
   ASSERT_TRUE(first_node);
   ASSERT_EQ(first_node->active_player, Player::chance);
   EXPECT_FALSE(first_value);

   auto second_entry = first_node->children.find(Tree::action_variant_type{second_outcome});
   ASSERT_NE(second_entry, first_node->children.end());
   auto& [second_node, second_probability, second_value] = second_entry->second;
   ASSERT_TRUE(second_node);
   ASSERT_EQ(second_node->active_player, decision_player);
   ASSERT_TRUE(second_node->infostate);
   EXPECT_EQ(*second_node->infostate, expected_infostate);
   EXPECT_FALSE(second_value);

   ASSERT_TRUE(first_probability);
   EXPECT_DOUBLE_EQ(
      first_probability->get(), env.chance_probability(tree.root_state(), first_outcome)
   );
   ASSERT_TRUE(second_probability);
   EXPECT_DOUBLE_EQ(
      second_probability->get(), env.chance_probability(*first_state, second_outcome)
   );

   // Chance edges retain the environment probabilities, and the first decision player's edges
   // are all one when that player is the owner of the counterfactual tree.
   for(auto& [root_action, root_tuple] : tree.root_node().children) {
      auto& [root_child, root_probability, root_value] = root_tuple;
      ASSERT_TRUE(root_child);
      ASSERT_TRUE(root_probability);
      const auto& first_deal = std::get< ChanceOutcome >(root_action);
      auto state_after_first_deal = child_state(env, tree.root_state(), first_deal);
      EXPECT_DOUBLE_EQ(
         root_probability->get(), env.chance_probability(tree.root_state(), first_deal)
      );
      EXPECT_FALSE(root_value);
      for(auto& [outcome_variant, outcome_tuple] : root_child->children) {
         const auto& outcome = std::get< ChanceOutcome >(outcome_variant);
         auto& [decision_node, outcome_probability, outcome_value] = outcome_tuple;
         ASSERT_TRUE(decision_node);
         ASSERT_TRUE(outcome_probability);
         EXPECT_DOUBLE_EQ(
            outcome_probability->get(), env.chance_probability(*state_after_first_deal, outcome)
         );
         EXPECT_FALSE(outcome_value);
         if(decision_node->active_player == Player::alex) {
            for(auto& [action_variant, action_tuple] : decision_node->children) {
               ASSERT_TRUE(std::holds_alternative< games::kuhn::Action >(action_variant));
               auto& [child_node, action_probability, action_value] = action_tuple;
               EXPECT_TRUE(action_probability);
               EXPECT_DOUBLE_EQ(action_probability->get(), 1.);
               if(child_node) {
                  EXPECT_TRUE(child_node->infostate);
               } else {
                  EXPECT_TRUE(action_value);
               }
            }
         }
      }
   }
}

TEST(InfostateTree, rejectsInvalidRootState)
{
   using Tree = forest::InfostateTree< games::rps::Environment >;
   games::rps::Environment env{};
   EXPECT_THROW(([&] { Tree tree{env, nullptr}; }()), std::invalid_argument);
}
