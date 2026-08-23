
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <optional>
#include <random>
#include <unordered_set>
#include <vector>

#include "fixtures.hpp"
#include "liars_dice/liars_dice.hpp"
#include "nor/nor.hpp"

using namespace liars_dice;

// concept compliance: liar's dice is a stochastic FOSG (the dice rolls are chance events)
static_assert(nor::concepts::fosg< nor::games::liars_dice::Environment >);
static_assert(nor::concepts::stochastic_fosg< nor::games::liars_dice::Environment >);
static_assert(not nor::concepts::deterministic_fosg< nor::games::liars_dice::Environment >);

// ##################################################################################################################
// Chance phase: roll dealing, probabilities, exhaustion, determinism
// ##################################################################################################################

TEST_F(LiarsDiceState, chance_deals_in_seat_order_and_exhausts)
{
   EXPECT_EQ(state.active_player(), Player::chance);
   ASSERT_EQ(state.chance_actions().size(), test_n_faces);
   // deal order is deterministic: player one first, faces ascending
   for(uint8_t f = 1; f <= test_n_faces; ++f) {
      EXPECT_EQ(state.chance_actions()[f - 1], (Roll{Player::one, f}));
   }
   state.apply_action(Roll{Player::one, 2});
   ASSERT_TRUE(state.die(Player::one).has_value());
   ASSERT_EQ(state.chance_actions().size(), test_n_faces);
   for(uint8_t f = 1; f <= test_n_faces; ++f) {
      EXPECT_EQ(state.chance_actions()[f - 1], (Roll{Player::two, f}));
   }
   // player one's die can not be rolled twice
   EXPECT_THROW(state.apply_action(Roll{Player::one, 1}), std::logic_error);
   // out-of-range faces are rejected
   EXPECT_THROW(state.apply_action(Roll{Player::two, 0}), std::logic_error);
   EXPECT_THROW(state.apply_action(Roll{Player::two, uint8_t(test_n_faces + 1)}), std::logic_error);

   state.apply_action(Roll{Player::two, 3});
   EXPECT_TRUE(state.chance_actions().empty());
   EXPECT_EQ(state.rolls_done(), 2u);
   // both dice dealt --> the opener acts
   EXPECT_EQ(state.active_player(), Player::one);
   EXPECT_EQ(*state.die(Player::one), 2u);
   EXPECT_EQ(*state.die(Player::two), 3u);
   // no further chance outcomes are accepted
   EXPECT_THROW(state.apply_action(Roll{Player::one, 1}), std::logic_error);
}

TEST_F(LiarsDiceState, chance_probabilities_are_uniform)
{
   const double expected_prob = 1. / double(test_n_faces);
   double sum = 0.;
   for(auto outcome : state.chance_actions()) {
      auto prob = state.chance_probability(outcome);
      EXPECT_DOUBLE_EQ(prob, expected_prob);
      EXPECT_TRUE(state.is_valid(outcome));
      sum += prob;
   }
   EXPECT_NEAR(sum, 1., 1e-12);

   state.apply_action(Roll{Player::one, 1});
   sum = 0.;
   for(auto outcome : state.chance_actions()) {
      auto prob = state.chance_probability(outcome);
      EXPECT_DOUBLE_EQ(prob, expected_prob);
      sum += prob;
   }
   EXPECT_NEAR(sum, 1., 1e-12);

   state.apply_action(Roll{Player::two, 2});
   // once both dice are out, no chance remains
   for(auto face : {uint8_t(1), test_n_faces}) {
      EXPECT_DOUBLE_EQ(state.chance_probability(Roll{Player::one, face}), 0.);
      EXPECT_DOUBLE_EQ(state.chance_probability(Roll{Player::two, face}), 0.);
      EXPECT_FALSE(state.is_valid(Roll{Player::one, face}));
   }
}

TEST_F(LiarsDiceState, deal_order_determinism)
{
   auto clone = state;
   ASSERT_EQ(clone, state);
   state.apply_action(Roll{Player::one, 3});
   state.apply_action(Roll{Player::two, 1});
   state.apply_action(Action{ActionType::bid, Bid{1, 2}});

   clone.apply_action(Roll{Player::one, 3});
   clone.apply_action(Roll{Player::two, 1});
   clone.apply_action(Action{ActionType::bid, Bid{1, 2}});

   EXPECT_EQ(state, clone);
   EXPECT_EQ(state.chance_actions(), clone.chance_actions());
   EXPECT_EQ(state.actions(), clone.actions());
   EXPECT_EQ(state.actions_history(), clone.actions_history());
}

// ##################################################################################################################
// Bidding legality: bounds + lexicographic dominance
// ##################################################################################################################

TEST_F(LiarsDiceState, opening_bid_bounds)
{
   apply_rolls(state, 1, 2);
   // in-range openings are legal ...
   for(uint8_t count = 1; count <= state.config().max_bid_count(); ++count) {
      for(uint8_t face = 1; face <= test_n_faces; ++face) {
         EXPECT_TRUE(state.is_valid(Action{ActionType::bid, Bid{count, face}}));
      }
   }
   // ... and everything out of range is not
   EXPECT_FALSE(state.is_valid(Action{ActionType::bid, Bid{0, 1}}));
   EXPECT_FALSE(state.is_valid(Action{ActionType::bid, Bid{3, 1}}));  // only two dice in play
   EXPECT_FALSE(state.is_valid(Action{ActionType::bid, Bid{1, 0}}));
   EXPECT_FALSE(state.is_valid(Action{ActionType::bid, Bid{1, uint8_t(test_n_faces + 1)}}));
}

TEST_F(LiarsDiceState, actions_before_deal_empty)
{
   EXPECT_TRUE(state.actions().empty());
   state.apply_action(Roll{Player::one, 2});
   EXPECT_TRUE(state.actions().empty());
}

TEST_P(BidLegalityParamsF, bid_dominance_table)
{
   const auto& [standing_opt, candidate, expected_legal] = GetParam();
   apply_rolls(state, 2, 3);
   if(standing_opt.has_value()) {
      state.apply_action(Action{ActionType::bid, *standing_opt});
   }
   Action candidate_action{ActionType::bid, candidate};
   EXPECT_EQ(state.is_valid(candidate_action), expected_legal)
      << "standing bid: " << common::to_string(candidate);
   // consistency between is_valid and the generated action list
   auto legal_actions = state.actions();
   bool contained = ranges::contains(legal_actions, candidate_action);
   EXPECT_EQ(contained, expected_legal);
   // challenge availability mirrors the standing bid
   bool challenge_available = ranges::contains(legal_actions, Action{ActionType::challenge, Bid{}});
   EXPECT_EQ(challenge_available, standing_opt.has_value());
   if(standing_opt.has_value()) {
      // the challenge is always enumerated last
      EXPECT_EQ(legal_actions.back(), (Action{ActionType::challenge, Bid{}}));
   }
}

INSTANTIATE_TEST_SUITE_P(
   bid_legality_tests,
   BidLegalityParamsF,
   ::testing::Values(
      // no standing bid: any in-range bid may open
      BidLegalityCase{std::nullopt, Bid{1, 1}, true},
      BidLegalityCase{std::nullopt, Bid{2, 3}, true},
      BidLegalityCase{std::nullopt, Bid{1, 4}, false},
      BidLegalityCase{std::nullopt, Bid{3, 1}, false},
      // facing (1,1): equal is illegal, anything lexicographically greater is legal
      BidLegalityCase{Bid{1, 1}, Bid{1, 1}, false},
      BidLegalityCase{Bid{1, 1}, Bid{1, 2}, true},
      BidLegalityCase{Bid{1, 1}, Bid{1, 3}, true},
      BidLegalityCase{Bid{1, 1}, Bid{2, 1}, true},
      BidLegalityCase{Bid{1, 1}, Bid{2, 3}, true},
      // facing (1,3): no same-count raise exists anymore
      BidLegalityCase{Bid{1, 3}, Bid{1, 1}, false},
      BidLegalityCase{Bid{1, 3}, Bid{1, 3}, false},
      BidLegalityCase{Bid{1, 3}, Bid{2, 1}, true},
      // facing (2,1)
      BidLegalityCase{Bid{2, 1}, Bid{1, 3}, false},
      BidLegalityCase{Bid{2, 1}, Bid{2, 1}, false},
      BidLegalityCase{Bid{2, 1}, Bid{2, 2}, true},
      BidLegalityCase{Bid{2, 1}, Bid{2, 3}, true},
      // facing the maximal bid (2,3): nothing dominates it anymore
      BidLegalityCase{Bid{2, 3}, Bid{1, 1}, false},
      BidLegalityCase{Bid{2, 3}, Bid{2, 2}, false},
      BidLegalityCase{Bid{2, 3}, Bid{2, 3}, false}
   )
);

// ##################################################################################################################
// Challenge resolution truth table incl. boundary equality (actual >= count wins for bidder)
// ##################################################################################################################

TEST_P(ChallengeResolutionParamsF, challenge_truth_table)
{
   const auto& [dice, bid, expected_winner] = GetParam();
   apply_rolls(state, dice[0], dice[1]);
   state.apply_action(Action{ActionType::bid, bid});
   state.apply_action(Action{ActionType::challenge, Bid{}});

   ASSERT_TRUE(state.is_terminal());
   ASSERT_TRUE(state.challenge_outcome().has_value());
   uint8_t actual = state.actual_count(bid.face);
   auto expected_outcome = actual >= bid.count ? Outcome::bidder_wins : Outcome::challenger_wins;
   EXPECT_EQ(*state.challenge_outcome(), expected_outcome);

   Player loser = expected_winner == Player::one ? Player::two : Player::one;
   EXPECT_DOUBLE_EQ(state.payoff(expected_winner), 1.);
   EXPECT_DOUBLE_EQ(state.payoff(loser), -1.);
   auto payoffs = state.payoffs();
   EXPECT_DOUBLE_EQ(payoffs[0] + payoffs[1], 0.);
}

INSTANTIATE_TEST_SUITE_P(
   challenge_resolution_tests,
   ChallengeResolutionParamsF,
   ::testing::Values(
      ChallengeResolutionCase{{2, 3}, Bid{1, 1}, Player::two},  // actual 0 < 1
      ChallengeResolutionCase{{1, 3}, Bid{1, 1}, Player::one},  // boundary equality: 1 >= 1
      ChallengeResolutionCase{{1, 1}, Bid{1, 1}, Player::one},  // actual 2 >= 1
      ChallengeResolutionCase{{2, 3}, Bid{2, 1}, Player::two},  // actual 0 < 2
      ChallengeResolutionCase{{1, 3}, Bid{2, 1}, Player::two},  // actual 1 < 2
      ChallengeResolutionCase{{1, 1}, Bid{2, 1}, Player::one},  // boundary equality: 2 >= 2
      ChallengeResolutionCase{{1, 3}, Bid{1, 2}, Player::two},  // actual 0
      ChallengeResolutionCase{{2, 1}, Bid{1, 2}, Player::one},  // boundary equality
      ChallengeResolutionCase{{2, 2}, Bid{1, 2}, Player::one},
      ChallengeResolutionCase{{1, 3}, Bid{2, 2}, Player::two},
      ChallengeResolutionCase{{2, 1}, Bid{2, 2}, Player::two},
      ChallengeResolutionCase{{2, 2}, Bid{2, 2}, Player::one},  // boundary equality
      ChallengeResolutionCase{{1, 2}, Bid{1, 3}, Player::two},
      ChallengeResolutionCase{{3, 1}, Bid{1, 3}, Player::one},  // boundary equality
      ChallengeResolutionCase{{3, 2}, Bid{1, 3}, Player::one},
      ChallengeResolutionCase{{1, 2}, Bid{2, 3}, Player::two},
      ChallengeResolutionCase{{3, 1}, Bid{2, 3}, Player::two},
      ChallengeResolutionCase{{3, 3}, Bid{2, 3}, Player::one}  // boundary equality
   )
);

// ##################################################################################################################
// Terminality: the game ends iff a challenge was made
// ##################################################################################################################

TEST_P(TerminalityParamsF, scripted_terminality)
{
   const auto& [dice, action_seq, expected_terminal] = GetParam();
   apply_rolls(state, dice[0], dice[1]);
   for(const auto& action : action_seq) {
      state.apply_action(action);
   }
   EXPECT_EQ(state.is_terminal(), expected_terminal);
}

INSTANTIATE_TEST_SUITE_P(
   terminality_tests,
   TerminalityParamsF,
   ::testing::Values(
      TerminalityCase{{1, 2}, {}, false},
      TerminalityCase{{1, 2}, {Action{ActionType::bid, Bid{1, 1}}}, false},
      TerminalityCase{
         {1, 2},
         {Action{ActionType::bid, Bid{1, 1}}, Action{ActionType::challenge, Bid{}}},
         true},
      TerminalityCase{
         {1, 2},
         {Action{ActionType::bid, Bid{1, 1}},
          Action{ActionType::bid, Bid{2, 1}},
          Action{ActionType::challenge, Bid{}}},
         true},
      TerminalityCase{
         {1, 2},
         {Action{ActionType::bid, Bid{1, 1}}, Action{ActionType::bid, Bid{2, 1}}},
         false}
   )
);

TEST_F(LiarsDiceState, terminal_state_rejects_everything)
{
   apply_rolls(state, 1, 2);
   state.apply_action(Action{ActionType::bid, Bid{1, 1}});
   state.apply_action(Action{ActionType::challenge, Bid{}});
   EXPECT_TRUE(state.is_terminal());
   EXPECT_THROW(state.apply_action(Action{ActionType::bid, Bid{1, 1}}), std::logic_error);
   EXPECT_THROW(state.apply_action(Action{ActionType::challenge, Bid{}}), std::logic_error);
   EXPECT_TRUE(state.actions().empty());
   EXPECT_TRUE(state.chance_actions().empty());
}

TEST_F(LiarsDiceState, zero_sum_over_random_playouts)
{
   std::mt19937 rng(42);
   for(size_t playout = 0; playout < 300; ++playout) {
      liars_dice::State rollout_state{liars_dice::DiceConfig(test_n_faces)};
      while(not rollout_state.is_terminal()) {
         if(rollout_state.active_player() == Player::chance) {
            auto outcomes = rollout_state.chance_actions();
            rollout_state.apply_action(outcomes[rng() % outcomes.size()]);
         } else {
            auto legal = rollout_state.actions();
            rollout_state.apply_action(legal[rng() % legal.size()]);
         }
      }
      auto p1 = rollout_state.payoff(Player::one);
      auto p2 = rollout_state.payoff(Player::two);
      EXPECT_DOUBLE_EQ(p1 + p2, 0.);
      EXPECT_TRUE(std::abs(p1) == 1.);
   }
}

// ##################################################################################################################
// Observation correctness: opponent's die hidden pre-challenge, revealed post-challenge
// ##################################################################################################################

namespace {

/// builds `istate` of `observer` by walking `world` and appending env observations per transition
template < typename World >
void observe_transition(
   const nor::games::liars_dice::Environment& env,
   nor::games::liars_dice::Infostate& istate,
   nor::Player observer,
   const World& pre,
   const auto& action_or_outcome,
   const World& next
)
{
   istate.update(
      env.public_observation(pre, action_or_outcome, next),
      env.private_observation(observer, pre, action_or_outcome, next)
   );
}

}  // namespace

TEST(LiarsDiceEnvironment, observations_of_chance_and_bids)
{
   using namespace nor::games::liars_dice;
   Environment env = Environment(DiceConfig(test_n_faces));

   auto wstate = env.initial_world_state();
   // chance + both seat players (mirrors the kuhn poker convention)
   EXPECT_EQ(env.players(wstate).size(), 3u);
   EXPECT_EQ(env.active_player(wstate), nor::Player::chance);
   EXPECT_FALSE(env.is_terminal(wstate));
   EXPECT_TRUE(env.is_partaking(wstate, nor::Player::alex));

   auto pre = wstate;
   auto outcome = Roll{Player::one, 2};
   env.transition(wstate, outcome);

   // private observation: only the recipient learns his own face
   auto alex_priv = env.private_observation(nor::Player::alex, pre, outcome, wstate);
   auto bob_priv = env.private_observation(nor::Player::bob, pre, outcome, wstate);
   ASSERT_TRUE(alex_priv.roll.has_value());
   EXPECT_EQ(*alex_priv.roll, outcome);
   EXPECT_EQ(bob_priv, Observation{});

   // public observation: only the recipient's identity is revealed
   auto pub = env.public_observation(pre, outcome, wstate);
   ASSERT_TRUE(pub.hidden_roll_to.has_value());
   EXPECT_EQ(to_nor_player(*pub.hidden_roll_to), nor::Player::alex);
   EXPECT_FALSE(pub.bid.has_value());

   // a bid announcement is fully public and carries no private information
   env.transition(wstate, Roll{Player::two, 3});
   auto action_pre = wstate;
   Action bid_action{ActionType::bid, Bid{1, 2}};
   env.transition(wstate, bid_action);
   auto bid_pub = env.public_observation(action_pre, bid_action, wstate);
   ASSERT_TRUE(bid_pub.bid.has_value());
   EXPECT_EQ(*bid_pub.bid, (Bid{1, 2}));
   for(auto observer : {nor::Player::alex, nor::Player::bob}) {
      EXPECT_EQ(env.private_observation(observer, action_pre, bid_action, wstate), Observation{});
   }
}

TEST(LiarsDiceEnvironment, opponent_die_invisible_until_challenge)
{
   using namespace nor::games::liars_dice;
   constexpr auto observer = nor::Player::alex;
   Environment env = Environment(DiceConfig(test_n_faces));

   // world A: dice (2, 3); world B: dice (2, 1) --> differ only in bob's die
   std::array worlds = {env.initial_world_state(), env.initial_world_state()};
   Infostate istates[2] = {Infostate(observer), Infostate(observer)};
   std::array second_die = {uint8_t(3), uint8_t(1)};

   // identical own roll
   for(auto i : {0u, 1u}) {
      auto pre = worlds[i];
      auto outcome = Roll{Player::one, 2};
      env.transition(worlds[i], outcome);
      observe_transition(env, istates[i], observer, pre, outcome, worlds[i]);
   }

   // differing opponent rolls are publicly indistinguishable
   for(auto i : {0u, 1u}) {
      auto pre = worlds[i];
      auto outcome = Roll{Player::two, second_die[i]};
      env.transition(worlds[i], outcome);
      observe_transition(env, istates[i], observer, pre, outcome, worlds[i]);
   }

   // an identical opening bid keeps them indistinguishable
   for(auto i : {0u, 1u}) {
      auto pre = worlds[i];
      Action bid_action{ActionType::bid, Bid{1, 2}};
      env.transition(worlds[i], bid_action);
      observe_transition(env, istates[i], observer, pre, bid_action, worlds[i]);
   }

   EXPECT_EQ(istates[0], istates[1]);
   EXPECT_NE(istates[0].hash(), 0u);
   EXPECT_EQ(istates[0].hash(), istates[1].hash());

   std::unordered_set< Infostate > set;
   set.emplace(istates[0]);
   set.emplace(istates[1]);
   EXPECT_EQ(set.size(), 1u);

   // bob challenges --> the reveal distinguishes both worlds
   for(auto i : {0u, 1u}) {
      auto pre = worlds[i];
      Action challenge{ActionType::challenge, Bid{}};
      env.transition(worlds[i], challenge);
      observe_transition(env, istates[i], observer, pre, challenge, worlds[i]);
      EXPECT_TRUE(env.is_terminal(worlds[i]));
      // public observation carries both dice + resolution
      auto pub = env.public_observation(pre, challenge, worlds[i]);
      ASSERT_TRUE(pub.reveal.has_value());
      EXPECT_EQ(pub.reveal->die_one, 2u);
      EXPECT_EQ(pub.reveal->die_two, second_die[i]);
      auto expected_outcome = worlds[i].actual_count(2) >= 1 ? Outcome::bidder_wins
                                                             : Outcome::challenger_wins;
      EXPECT_EQ(pub.reveal->outcome, expected_outcome);
      // challenges carry no private payload
      EXPECT_EQ(env.private_observation(observer, pre, challenge, worlds[i]), Observation{});
   }

   EXPECT_NE(istates[0], istates[1]);
   EXPECT_NE(istates[0].hash(), istates[1].hash());
   set.emplace(istates[0]);
   set.emplace(istates[1]);
   EXPECT_EQ(set.size(), 3u);
}

// ##################################################################################################################
// Convergence smoke: VanillaCFR on d=3 drops exploitability well below its starting value
// ##################################################################################################################

TEST(LiarsDiceCFR, vanilla_cfr_converges_on_three_faces)
{
   using namespace nor;
   using Env = games::liars_dice::Environment;
   using StateT = games::liars_dice::State;
   using InfostateT = games::liars_dice::Infostate;
   using ActionT = games::liars_dice::Action;

   constexpr size_t warmup_iters = 10;
   constexpr size_t max_iters = 300;

   Env env = Env(games::liars_dice::DiceConfig(test_n_faces));

   auto avg_tabular_policy = factory::make_tabular_policy(
      std::unordered_map< InfostateT, HashmapActionPolicy< ActionT > >{}
   );
   auto tabular_policy = factory::make_tabular_policy(
      std::unordered_map< InfostateT, HashmapActionPolicy< ActionT > >{}
   );

   auto solver = factory::
      make_cfr< rm::CFRDiscountedConfig{.update_mode = rm::UpdateMode::alternating}, true >(
         Env(env),
         std::make_unique< StateT >(games::liars_dice::DiceConfig(test_n_faces)),
         tabular_policy,
         avg_tabular_policy
      );

   auto exploitability_of_avg = [&]() {
      const auto& avg_policies = solver.average_policy();
      return exploitability(
         env,
         games::liars_dice::State(games::liars_dice::DiceConfig(test_n_faces)),
         player_hashmap< std::decay_t< decltype(avg_policies.at(nor::Player::alex)) > >{
            std::pair{
               nor::Player::alex, normalize_state_policy(avg_policies.at(nor::Player::alex))},
            std::pair{nor::Player::bob, normalize_state_policy(avg_policies.at(nor::Player::bob))}}
      );
   };

   for(size_t iter = 1; iter <= warmup_iters; ++iter) {
      solver.iterate(1);
   }
   double expl_start = exploitability_of_avg();

   for(size_t iter = warmup_iters + 1; iter <= max_iters; ++iter) {
      solver.iterate(1);
   }
   double expl_final = exploitability_of_avg();

   std::cout << "liar's dice (d=" << int(test_n_faces) << ") vanilla CFR alternating:\n"
             << "  exploitability after " << warmup_iters << " iters: " << expl_start << "\n"
             << "  exploitability after " << max_iters << " iters: " << expl_final << "\n";

   EXPECT_LT(expl_final, 0.05);
   EXPECT_LT(expl_final, 0.2 * expl_start);
}
