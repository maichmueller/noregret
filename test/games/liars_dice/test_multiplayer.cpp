
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <unordered_set>
#include <vector>

#include "liars_dice/liars_dice.hpp"
#include "nor/exploitability.hpp"
#include "nor/nor.hpp"

// concept compliance: multiplayer liar's dice is a stochastic FOSG (the dice rolls are chance
// events) for any configuration within the supported parameter ranges
static_assert(nor::concepts::fosg< nor::games::liars_dice::Environment >);
static_assert(nor::concepts::stochastic_fosg< nor::games::liars_dice::Environment >);
static_assert(not nor::concepts::deterministic_fosg< nor::games::liars_dice::Environment >);

namespace {

using namespace liars_dice;

constexpr uint8_t mp_n_faces = 3;

/// the seat order in which the chance player deals the running round (alive seats, slot asc.)
std::vector< Roll > expected_roll_order(const liars_dice::State& state)
{
   std::vector< Roll > out;
   for(uint8_t seat = 0; seat < state.config().n_players; ++seat) {
      auto p = Player(static_cast< int >(seat));
      if(not state.alive(p)) {
         continue;
      }
      for(uint8_t slot = 0; slot < state.dice_left(p); ++slot) {
         out.emplace_back(Roll{p, 1, slot});
      }
   }
   return out;
}

/// applies one full chance phase, drawing the given faces in the deterministic deal order
void deal_round(liars_dice::State& state, const std::vector< uint8_t >& faces)
{
   for(auto face : faces) {
      auto legal = state.chance_actions();
      ASSERT_FALSE(legal.empty());
      auto outcome = legal.front();
      outcome.face = face;
      state.apply_action(outcome);
   }
}

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

// ##################################################################################################################
// Multi-round elimination flow: who-loses-die-when, elimination order, last-standing payoffs
// ##################################################################################################################

TEST(LiarsDiceMpElimination, scripted_three_player_match_with_two_challenges)
{
   liars_dice::State state{DiceConfig(3, 1, mp_n_faces)};

   // ----- round 0: all three seats roll (seat order), opener is seat zero -----
   ASSERT_EQ(state.active_player(), Player::chance);
   deal_round(state, {2, 3, 3});
   EXPECT_TRUE(state.chance_actions().empty());
   EXPECT_EQ(state.round_index(), 0u);
   EXPECT_EQ(state.opener(), Player::one);
   EXPECT_EQ(state.active_player(), Player::one);

   // one truthfully claims both threes; two challenges anyway and loses his last die
   state.apply_action(Action{ActionType::bid, Bid{2, 3}});
   EXPECT_EQ(state.active_player(), Player::two);
   state.apply_action(Action{ActionType::challenge, Bid{}});
   EXPECT_EQ(*state.challenge_outcome(), Outcome::bidder_wins);
   EXPECT_FALSE(state.alive(Player::two));
   EXPECT_EQ(state.dice_left(Player::two), 0);
   EXPECT_FALSE(state.is_terminal()) << "two players still stand";
   EXPECT_EQ(state.round_index(), 1u);
   EXPECT_EQ(state.round_outcomes().size(), 1u);

   // ----- round 1: re-roll among the alive seats only; opener skips eliminated seat two -----
   EXPECT_EQ(state.active_player(), Player::chance);
   {
      auto actual_order = state.chance_actions();
      auto expected_order = expected_roll_order(state);
      ASSERT_EQ(actual_order.size(), expected_order.size());
      for(size_t i = 0; i < actual_order.size(); ++i) {
         EXPECT_EQ(actual_order[i].player, expected_order[i].player);
         EXPECT_EQ(actual_order[i].slot, expected_order[i].slot);
      }
   }
   EXPECT_EQ(state.opener(), Player::three) << "nominal seat 1 is dead -> next alive clockwise";
   deal_round(state, {1, 1});
   EXPECT_EQ(state.active_player(), Player::three);

   // bidding cycles three -> one (skipping the eliminated seat two)
   state.apply_action(Action{ActionType::bid, Bid{1, 1}});
   EXPECT_EQ(state.active_player(), Player::one);
   state.apply_action(Action{ActionType::bid, Bid{2, 1}});
   EXPECT_EQ(state.active_player(), Player::three);

   // actual ones = 2 >= 2 -> bidder (one) survives, challenger (three) is eliminated
   state.apply_action(Action{ActionType::challenge, Bid{}});
   ASSERT_TRUE(state.is_terminal());
   EXPECT_FALSE(state.alive(Player::three));
   EXPECT_TRUE(state.alive(Player::one));

   // elimination order was two then three; the last standing takes +1, the rest split -1
   EXPECT_EQ(state.round_outcomes().size(), 2u);
   EXPECT_DOUBLE_EQ(state.payoff(Player::one), 1.);
   EXPECT_DOUBLE_EQ(state.payoff(Player::two), -0.5);
   EXPECT_DOUBLE_EQ(state.payoff(Player::three), -0.5);
   auto payoffs = state.payoffs();
   ASSERT_EQ(payoffs.size(), 3u);
   EXPECT_NEAR(payoffs[0] + payoffs[1] + payoffs[2], 0., 1e-12);
}

TEST(LiarsDiceMpElimination, damaged_player_rolls_fewer_dice_and_top_bid_forces_challenge)
{
   liars_dice::State state{DiceConfig(3, 2, mp_n_faces)};

   // round 0: every seat rolls two dice (slots ascending per seat)
   auto order0 = expected_roll_order(state);
   ASSERT_EQ(order0.size(), 6u);
   EXPECT_EQ((order0[0]), (Roll{Player::one, 1, 0}));
   EXPECT_EQ((order0[1]), (Roll{Player::one, 1, 1}));
   EXPECT_EQ((order0[2]), (Roll{Player::two, 1, 0}));
   deal_round(state, {3, 3, 1, 1, 2, 2});

   // one claims two threes (true), two bluffs three threes, three calls the bluff:
   // bidder (two) loses ONE of his two dice and stays alive
   state.apply_action(Action{ActionType::bid, Bid{2, 3}});
   state.apply_action(Action{ActionType::bid, Bid{3, 3}});
   state.apply_action(Action{ActionType::challenge, Bid{}});
   EXPECT_EQ(*state.challenge_outcome(), Outcome::challenger_wins);
   EXPECT_TRUE(state.alive(Player::two));
   EXPECT_EQ(state.dice_left(Player::two), 1);
   EXPECT_EQ(state.round_index(), 1u);
   EXPECT_EQ(state.opener(), Player::two) << "nominal round-1 opener seat 1 is still alive";

   // round 1: one rolls two dice, damaged two rolls only one, three rolls two
   auto order1 = expected_roll_order(state);
   ASSERT_EQ(order1.size(), 5u);
   EXPECT_EQ((order1[2]), (Roll{Player::two, 1, 0})) << "damaged seat contributes a single die";
   EXPECT_EQ((order1[3]), (Roll{Player::three, 1, 0}));
   deal_round(state, {1, 3, 2, 3, 3});
   EXPECT_EQ(state.active_player(), Player::two);
   EXPECT_EQ(state.max_bid_count(), 5) << "bids may cover all five alive dice";

   // two opens, three raises to the top bid {5,3}: one can only challenge now
   state.apply_action(Action{ActionType::bid, Bid{1, 2}});
   state.apply_action(Action{ActionType::bid, Bid{5, 3}});
   auto forced = state.actions();
   ASSERT_EQ(forced.size(), 1u);
   EXPECT_EQ(forced.front(), (Action{ActionType::challenge, Bid{}}));

   // actual threes = 3 < 5 -> challenger (one) wins the round, bidder (three) drops to one die
   state.apply_action(Action{ActionType::challenge, Bid{}});
   EXPECT_FALSE(state.is_terminal());
   EXPECT_EQ(*state.challenge_outcome(), Outcome::challenger_wins);
   EXPECT_EQ(state.dice_left(Player::three), 1);
   EXPECT_EQ(state.round_outcomes().size(), 2u);
   EXPECT_EQ(state.round_index(), 2u);
}

// ##################################################################################################################
// Opener rotation incl. skipping eliminated seats (4 players over a full match)
// ##################################################################################################################

TEST(LiarsDiceMpOpenerRotation, rotates_over_seats_and_skips_eliminated_in_four_player_match)
{
   liars_dice::State state{DiceConfig(4, 1, mp_n_faces)};
   const auto bid_challenge = [&state](uint8_t count, uint8_t face) {
      state.apply_action(Action{ActionType::bid, Bid{count, face}});
      state.apply_action(Action{ActionType::challenge, Bid{}});
   };

   // ----- round 0: opener seat zero -----
   deal_round(state, {3, 1, 3, 3});
   EXPECT_EQ(state.opener(), Player::one);
   EXPECT_EQ(state.active_player(), Player::one);
   // one truthfully bids all three threes; two challenges and busts
   bid_challenge(3, 3);
   EXPECT_FALSE(state.alive(Player::two));

   // ----- round 1: nominal opener seat 1 is dead -> seat 2 (three) opens -----
   EXPECT_EQ(state.opener(), Player::three);
   deal_round(state, {1, 1, 2});  // alive seats one, three, four in seat order
   EXPECT_EQ(state.active_player(), Player::three);
   // three truthfully bids both ones; four challenges and busts
   bid_challenge(2, 1);
   EXPECT_FALSE(state.alive(Player::four));

   // ----- round 2: nominal opener seat 2 is alive -> three opens again -----
   EXPECT_EQ(state.opener(), Player::three);
   deal_round(state, {2, 1});  // alive seats one and three
   EXPECT_EQ(state.active_player(), Player::three);
   // three bids the single two; one challenges but 1 >= 1 holds -> bidder wins, one busts
   bid_challenge(1, 2);
   EXPECT_FALSE(state.alive(Player::one));
   EXPECT_TRUE(state.is_terminal());

   // elimination order two -> four -> one; three is the last standing
   EXPECT_TRUE(state.alive(Player::three));
   EXPECT_DOUBLE_EQ(state.payoff(Player::three), 1.);
   for(auto loser : {Player::one, Player::two, Player::four}) {
      EXPECT_NEAR(state.payoff(loser), -1. / 3., 1e-12);
   }
   double sum = 0.;
   for(auto value : state.payoffs()) {
      sum += value;
   }
   EXPECT_NEAR(sum, 0., 1e-12);
}

// ##################################################################################################################
// Challenge resolution truth table against ALL alive dice incl. boundary equality (>=)
// ##################################################################################################################

namespace {

/// challenge scenario: (dice of seats 0..2, standing bid, whether the bidder holds)
using MpChallengeCase = std::
   tuple< std::array< uint8_t, 3 >, Bid, Outcome, std::array< double, 3 > >;

class LiarsDiceMpChallengeParamsF: public ::testing::TestWithParam< MpChallengeCase > {
  protected:
   liars_dice::DiceConfig config = liars_dice::DiceConfig(3, 1, mp_n_faces);
   liars_dice::State state = liars_dice::State(config);
};

}  // namespace

TEST_P(LiarsDiceMpChallengeParamsF, challenge_truth_table_over_all_alive_dice)
{
   const auto& [dice, bid, expected_outcome, expected_payoffs] = GetParam();
   deal_round(state, {dice[0], dice[1], dice[2]});
   state.apply_action(Action{ActionType::bid, bid});
   state.apply_action(Action{ActionType::challenge, Bid{}});

   ASSERT_TRUE(state.is_terminal());
   uint8_t actual = state.actual_count(bid.face);
   auto derived_outcome = actual >= bid.count ? Outcome::bidder_wins : Outcome::challenger_wins;
   EXPECT_EQ(*state.challenge_outcome(), derived_outcome);
   EXPECT_EQ(*state.challenge_outcome(), expected_outcome);

   auto payoffs = state.payoffs();
   for(size_t seat = 0; seat < 3; ++seat) {
      EXPECT_DOUBLE_EQ(state.payoff(Player(seat)), expected_payoffs[seat]);
      EXPECT_DOUBLE_EQ(payoffs[seat], expected_payoffs[seat]);
   }
   EXPECT_NEAR(payoffs[0] + payoffs[1] + payoffs[2], 0., 1e-12);
}

INSTANTIATE_TEST_SUITE_P(
   mp_challenge_resolution_tests,
   LiarsDiceMpChallengeParamsF,
   ::testing::Values(
      // boundary equality: two ones on the table satisfy the claim of two
      MpChallengeCase{{1, 1, 3}, Bid{2, 1}, Outcome::bidder_wins, {1., -0.5, -0.5}},
      // one short: only one two on the table vs claim of two
      MpChallengeCase{{1, 2, 3}, Bid{2, 2}, Outcome::challenger_wins, {-0.5, 1., -0.5}},
      // claim of three threes holds exactly at the boundary
      MpChallengeCase{{3, 3, 3}, Bid{3, 3}, Outcome::bidder_wins, {1., -0.5, -0.5}},
      // bluffing an impossible count with everyone alive
      MpChallengeCase{{2, 3, 3}, Bid{3, 2}, Outcome::challenger_wins, {-0.5, 1., -0.5}},
      // single-face sweep: all dice show ones
      MpChallengeCase{{1, 1, 1}, Bid{3, 1}, Outcome::bidder_wins, {1., -0.5, -0.5}},
      // zero occurrences lose against any positive claim
      MpChallengeCase{{2, 2, 3}, Bid{1, 1}, Outcome::challenger_wins, {-0.5, 1., -0.5}}
   )
);

// ##################################################################################################################
// Re-roll probabilities: uniform per face per die, exhaustion after all alive dice are dealt
// ##################################################################################################################

TEST(LiarsDiceMpChancePhase, roll_probabilities_are_uniform_and_phase_exhausts)
{
   liars_dice::State state{DiceConfig(3, 2, mp_n_faces)};
   const double expected_prob = 1. / double(mp_n_faces);

   size_t total_rolls = 0;
   while(not state.chance_actions().empty()) {
      double sum = 0.;
      for(auto outcome : state.chance_actions()) {
         EXPECT_DOUBLE_EQ(state.chance_probability(outcome), expected_prob);
         EXPECT_TRUE(state.is_valid(outcome));
         sum += state.chance_probability(outcome);
      }
      EXPECT_NEAR(sum, 1., 1e-12);
      auto outcomes = state.chance_actions();
      state.apply_action(outcomes.front());
      ++total_rolls;
   }
   EXPECT_EQ(total_rolls, 6u) << "three seats x two starting dice";
   EXPECT_EQ(state.rolls_done(), 6u);

   // nothing is dealable anymore, not even stale outcomes
   EXPECT_DOUBLE_EQ(state.chance_probability(Roll{Player::one, 1, 0}), 0.);
   EXPECT_FALSE(state.is_valid(Roll{Player::three, 1, 1}));
   EXPECT_THROW(state.apply_action(Roll{Player::one, 1, 0}), std::logic_error);
}

TEST(LiarsDiceMpChancePhase, reroll_after_challenge_only_deals_alive_players_remaining_dice)
{
   liars_dice::State state{DiceConfig(3, 2, mp_n_faces)};
   deal_round(state, {3, 3, 1, 1, 2, 2});
   state.apply_action(Action{ActionType::bid, Bid{2, 3}});
   state.apply_action(Action{ActionType::bid, Bid{3, 3}});
   state.apply_action(Action{ActionType::challenge, Bid{}});
   ASSERT_EQ(state.dice_left(Player::two), 1);  // two lost one die on the challenge

   // each individual roll remains uniform regardless of seat or slot; the deal order visits
   // the alive seats in seat order with slots ascending
   const double expected_prob = 1. / double(mp_n_faces);
   size_t total_rolls = 0;
   while(not state.chance_actions().empty()) {
      auto outcomes = state.chance_actions();
      ASSERT_EQ(outcomes.size(), size_t(mp_n_faces));
      auto next_expected = expected_roll_order(state).front();
      for(auto outcome : outcomes) {
         EXPECT_DOUBLE_EQ(state.chance_probability(outcome), expected_prob);
         EXPECT_EQ(outcome.player, next_expected.player);
         EXPECT_EQ(outcome.slot, next_expected.slot);
      }
      state.apply_action(outcomes.front());
      ++total_rolls;
   }
   EXPECT_EQ(total_rolls, 5u) << "two + one + two remaining dice across the alive seats";
}

// ##################################################################################################################
// Zero-sum invariant over random playouts with N=3
// ##################################################################################################################

TEST(LiarsDiceMpPlayouts, random_playouts_preserve_zero_sum_and_single_winner)
{
   std::mt19937 rng(7);
   auto draw = [&rng](size_t n) { return std::uniform_int_distribution< size_t >(0, n - 1)(rng); };

   for(size_t playout = 0; playout < 200; ++playout) {
      liars_dice::State rollout_state{DiceConfig(3, 1, mp_n_faces)};
      size_t guard = 0;
      while(not rollout_state.is_terminal()) {
         ASSERT_LT(guard++, 32);
         if(rollout_state.active_player() == Player::chance) {
            auto outcomes = rollout_state.chance_actions();
            rollout_state.apply_action(outcomes[draw(outcomes.size())]);
         } else {
            auto legal = rollout_state.actions();
            rollout_state.apply_action(legal[draw(legal.size())]);
         }
      }
      auto payoffs = rollout_state.payoffs();
      ASSERT_EQ(payoffs.size(), 3u);
      EXPECT_NEAR(payoffs[0] + payoffs[1] + payoffs[2], 0., 1e-12) << "playout " << playout;
      size_t winners = static_cast< size_t >(ranges::count_if(payoffs, [](double v) {
         return std::abs(v - 1.) < 1e-12;
      }));
      EXPECT_EQ(winners, 1u);
      for(auto value : payoffs) {
         EXPECT_TRUE(std::abs(value) == 1. or std::abs(value + 0.5) < 1e-12)
            << "payoffs must follow the +1 / -1/(N-1) convention";
      }
   }
}

// ##################################################################################################################
// Information hiding: other players' dice stay invisible between challenges
// ##################################################################################################################

TEST(LiarsDiceMpObservations, opponent_dice_indistinguishable_until_reveal_across_rounds)
{
   using namespace nor::games::liars_dice;
   constexpr auto observer = nor::Player::alex;  // seat one
   Environment env = Environment(DiceConfig(3, 1, mp_n_faces));

   // world A: dice (2, 1, 3); world B: dice (2, 1, 1) --> differ only in cedric's (seat three's)
   // die, which alex must not be able to distinguish before the reveal
   std::array worlds = {env.initial_world_state(), env.initial_world_state()};
   Infostate istates[2] = {Infostate(observer), Infostate(observer)};
   std::array third_die = {uint8_t(3), uint8_t(1)};
   std::array first_two = {
      std::array< uint8_t, 2 >{uint8_t(2), uint8_t(1)},
      std::array< uint8_t, 2 >{uint8_t(2), uint8_t(1)}};

   for(auto i : {0u, 1u}) {
      // identical own + bob's rolls ...
      for(auto [seat, face] : ranges::views::enumerate(first_two[i])) {
         auto pre = worlds[i];
         auto outcome = Roll{Player(uint8_t(seat)), face, 0};
         env.transition(worlds[i], outcome);
         observe_transition(env, istates[i], observer, pre, outcome, worlds[i]);
      }
      // ... differing cedric roll is publicly indistinguishable ...
      {
         auto pre = worlds[i];
         auto outcome = Roll{Player::three, third_die[i], 0};
         env.transition(worlds[i], outcome);
         observe_transition(env, istates[i], observer, pre, outcome, worlds[i]);
      }
      // ... and so are identical public bid announcements
      for(auto action :
          std::array{Action{ActionType::bid, Bid{1, 2}}, Action{ActionType::bid, Bid{1, 3}}}) {
         auto pre = worlds[i];
         env.transition(worlds[i], action);
         observe_transition(env, istates[i], observer, pre, action, worlds[i]);
      }
   }

   EXPECT_EQ(istates[0], istates[1]);
   EXPECT_NE(istates[0].hash(), 0u);
   EXPECT_EQ(istates[0].hash(), istates[1].hash());
   {
      std::unordered_set< Infostate > set;
      set.emplace(istates[0]);
      set.emplace(istates[1]);
      EXPECT_EQ(set.size(), 1u);
   }

   // bob challenges --> the reveal carries ALL alive dice and separates both worlds
   for(auto i : {0u, 1u}) {
      auto pre = worlds[i];
      Action challenge{ActionType::challenge, Bid{}};
      env.transition(worlds[i], challenge);
      observe_transition(env, istates[i], observer, pre, challenge, worlds[i]);
      // standing bid was {1,3}: world A holds it (actual threes 1 >= 1), world B does not
      auto expected_outcome = i == 0 ? Outcome::bidder_wins : Outcome::challenger_wins;
      auto pub = env.public_observation(pre, challenge, worlds[i]);
      ASSERT_TRUE(pub.reveal.has_value());
      EXPECT_EQ(pub.reveal->die_one, 2u);
      EXPECT_EQ(pub.reveal->die_two, 1u);
      ASSERT_EQ(pub.reveal->further_dice.size(), 1u);
      EXPECT_EQ(pub.reveal->further_dice.front(), third_die[i]);
      EXPECT_EQ(pub.reveal->outcome, expected_outcome);
      EXPECT_EQ(env.private_observation(observer, pre, challenge, worlds[i]), Observation{});
   }

   EXPECT_NE(istates[0], istates[1]);
   EXPECT_NE(istates[0].hash(), istates[1].hash());
   std::unordered_set< Infostate > set;
   set.emplace(istates[0]);
   set.emplace(istates[1]);
   EXPECT_EQ(set.size(), 3u);
}

// ##################################################################################################################
// Convergence smoke: MCCFR external sampling + vanilla CFR on N=3, d=4, one die per player
// ##################################################################################################################

namespace {

using MpEnv = nor::games::liars_dice::Environment;
using MpState = nor::games::liars_dice::State;
using MpInfostate = nor::games::liars_dice::Infostate;
using MpAction = nor::games::liars_dice::Action;

liars_dice::DiceConfig mp_cfr_config()
{
   return liars_dice::DiceConfig(3, 1, 4);
}

template < typename Solver >
double exploitability_of_average_policies(Solver& solver, MpEnv& env)
{
   using namespace nor::games::liars_dice;
   const auto& avg_policies = solver.average_policy();
   return nor::exploitability(
      env,
      MpState(mp_cfr_config()),
      nor::player_hashmap< std::decay_t< decltype(avg_policies.at(nor::Player::alex)) > >{
         std::pair{
            nor::Player::alex, nor::normalize_state_policy(avg_policies.at(nor::Player::alex))},
         std::pair{
            nor::Player::bob, nor::normalize_state_policy(avg_policies.at(nor::Player::bob))},
         std::pair{
            nor::Player::cedric, nor::normalize_state_policy(avg_policies.at(nor::Player::cedric))}}
   );
}

}  // namespace

TEST(LiarsDiceMpCFR, mccfr_external_sampling_exploitability_decreases)
{
   using namespace nor;
   MpEnv env = MpEnv(mp_cfr_config());

   auto curr_policy = factory::make_tabular_policy(
      std::unordered_map< MpInfostate, HashmapActionPolicy< MpAction > >{}
   );
   auto avg_policy = factory::make_tabular_policy(
      std::unordered_map< MpInfostate, HashmapActionPolicy< MpAction > >{}
   );

   auto solver = factory::make_mccfr<
      rm::MCCFRConfig{
         .update_mode = rm::UpdateMode::alternating,
         .algorithm = rm::MCCFRAlgorithmMode::external_sampling,
         .weighting = rm::MCCFRWeightingMode::stochastic},
      true >(
      MpEnv(env),
      std::make_unique< MpState >(mp_cfr_config()),
      curr_policy,
      avg_policy,
      /*epsilon=*/0.,
      12345
   );

   constexpr size_t n_iterations = 500;
   double expl_mid = 0.;
   double expl_end = 0.;
   for(size_t iter = 1; iter <= n_iterations; ++iter) {
      solver.iterate(1);
      if(iter == 250) {
         expl_mid = exploitability_of_average_policies(solver, env);
      }
      if(iter == n_iterations) {
         expl_end = exploitability_of_average_policies(solver, env);
      }
   }
   std::cout << "LiarsDice MP (N=3,d=4) MCCFR external-sampling exploitability: iter250="
             << expl_mid << " iter500=" << expl_end << "\n";
   // sampled trajectories are noisy; only assert that training moved the profile substantially
   EXPECT_LT(expl_end, 0.9 * expl_mid + 1e-9);
   EXPECT_GT(expl_mid, 0.);
}

TEST(LiarsDiceMpCFR, vanilla_cfr_smoke_on_three_players)
{
   using namespace nor;
   MpEnv env = MpEnv(mp_cfr_config());

   auto avg_tabular_policy = factory::make_tabular_policy(
      std::unordered_map< MpInfostate, HashmapActionPolicy< MpAction > >{}
   );
   auto tabular_policy = factory::make_tabular_policy(
      std::unordered_map< MpInfostate, HashmapActionPolicy< MpAction > >{}
   );

   auto solver = factory::make_cfr<
      rm::CFRDiscountedConfig{.update_mode = rm::UpdateMode::alternating},
      true >(
      MpEnv(env), std::make_unique< MpState >(mp_cfr_config()), tabular_policy, avg_tabular_policy
   );

   auto exploitability_of_avg = [&] { return exploitability_of_average_policies(solver, env); };

   double expl_start = 0.;
   double expl_final = 0.;
   for(size_t iter = 1; iter <= 100; ++iter) {
      solver.iterate(1);
      if(iter == 10) {
         expl_start = exploitability_of_avg();
      }
      if(iter == 100) {
         expl_final = exploitability_of_avg();
      }
   }
   std::cout << "LiarsDice MP (N=3,d=4) vanilla CFR alternating:\n"
             << "  exploitability after 10 iters: " << expl_start << "\n"
             << "  exploitability after 100 iters: " << expl_final << "\n";
   EXPECT_GT(expl_start, 0.);
   EXPECT_LT(expl_final, expl_start);
}
