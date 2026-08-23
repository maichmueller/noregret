
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <unordered_set>
#include <variant>
#include <vector>

#include "fixtures.hpp"
#include "nor/concepts/concrete.hpp"
#include "nor/exploitability.hpp"
#include "nor/nor.hpp"

using namespace nor::games::goofspiel;
// NOTE: 'Player' stays unambiguously bound to ::goofspiel::Player; nor's player enum is always
// spelled 'nor::Player' in this file.

// ##################################################################################################################
// FOSG concept compliance
// ##################################################################################################################

static_assert(nor::concepts::fosg< Environment >);
static_assert(nor::concepts::stochastic_fosg< Environment >);
static_assert(nor::concepts::supports_all_histories< Environment >);

// ##################################################################################################################
// Helpers
// ##################################################################################################################

namespace {

/// plays the given (prize, bid_one, bid_two) sequence through the environment while accumulating
/// the observation stream of `observer` into an infostate
template < typename Prizes, typename BidsOne, typename BidsTwo >
Infostate observed_infoset(
   const Environment& env,
   nor::Player observer,
   const Prizes& prizes,
   const BidsOne& bids_one,
   const BidsTwo& bids_two
)
{
   auto wstate = env.initial_world_state();
   Infostate istate{observer};
   for(size_t r = 0; r < prizes.size(); ++r) {
      auto step = [&](auto outcome_or_action) {
         auto pre = wstate;
         env.transition(wstate, outcome_or_action);
         istate.update(
            env.public_observation(pre, outcome_or_action, wstate),
            env.private_observation(observer, pre, outcome_or_action, wstate)
         );
      };
      step(PrizeCard{prizes[r]});
      step(Bid{bids_one[r]});
      step(Bid{bids_two[r]});
      step(PrizeCard{uint8_t(0)});
   }
   return istate;
}

}  // namespace

// ##################################################################################################################
// World state: scoring & ties
// ##################################################################################################################

TEST_F(GoofspielState, mirrored_bids_tie_every_round)
{
   play_round(state, 1, 1, 1);
   play_round(state, 2, 2, 2);
   play_round(state, 3, 3, 3);

   ASSERT_TRUE(state.is_terminal());
   EXPECT_EQ(state.score(Player::one), 0);
   EXPECT_EQ(state.score(Player::two), 0);
   EXPECT_EQ(state.payoff(Player::one), 0.);
   EXPECT_EQ(state.payoff(Player::two), 0.);
}

TEST_F(GoofspielState, alternating_wins_balance_out)
{
   // r1: 3>2 -> one takes prize 1 | r2: 1<3 -> two takes prize 2 | r3: 2>1 -> one takes prize 3
   play_round(state, 1, 3, 2);
   play_round(state, 2, 1, 3);
   play_round(state, 3, 2, 1);

   ASSERT_TRUE(state.is_terminal());
   EXPECT_EQ(state.score(Player::one), 4);
   EXPECT_EQ(state.score(Player::two), 2);
   EXPECT_DOUBLE_EQ(state.payoff(Player::one), 2.);
   EXPECT_DOUBLE_EQ(state.payoff(Player::two), -2.);
}

TEST_F(GoofspielState, equal_bids_discard_the_prize_mid_game)
{
   play_round(state, 1, 2, 2);  // tie on prize 1 --> discarded
   EXPECT_EQ(state.score(Player::one), 0);
   EXPECT_EQ(state.score(Player::two), 0);
   EXPECT_EQ(state.round(), 1u);
   // the tied-out prize is gone from the deck: only {2, 3} remain for the next reveal
   auto remaining = state.chance_actions();
   ASSERT_EQ(remaining.size(), 2u);

   play_round(state, 3, 1, 3);  // two takes prize 3
   play_round(state, 2, 3, 1);  // one takes prize 2

   ASSERT_TRUE(state.is_terminal());
   EXPECT_EQ(state.score(Player::one), 2);
   EXPECT_EQ(state.score(Player::two), 3);
}

TEST_F(GoofspielState, single_card_deck_always_ties)
{
   goofspiel::State tiny{GoofspielConfig{.deck_size = 1}};
   play_round(tiny, 1, 1, 1);
   ASSERT_TRUE(tiny.is_terminal());
   EXPECT_DOUBLE_EQ(tiny.payoff(Player::one), 0.);
   EXPECT_DOUBLE_EQ(tiny.payoff(Player::two), 0.);
}

TEST_P(GoofspielPayoffParamsF, scripted_payoffs_are_zero_sum)
{
   const auto& [cfg, rounds, expected_one, expected_two] = GetParam();
   State scripted{cfg};
   for(const auto& [prize, bid_one, bid_two] : rounds) {
      ASSERT_FALSE(scripted.is_terminal());
      play_round(scripted, prize, bid_one, bid_two);
   }
   ASSERT_TRUE(scripted.is_terminal());
   auto payoffs = scripted.payoffs();
   EXPECT_NEAR(payoffs[0], expected_one, 1e-12) << "player one";
   EXPECT_NEAR(payoffs[1], expected_two, 1e-12) << "player two";
   EXPECT_NEAR(payoffs[0] + payoffs[1], 0., 1e-12);
}

INSTANTIATE_TEST_SUITE_P(
   goofspiel_payoffs,
   GoofspielPayoffParamsF,
   ::testing::Values(
      // mirrored play: every round tied, every prize discarded
      std::tuple{
         GoofspielConfig{},
         std::vector< std::tuple< uint8_t, uint8_t, uint8_t > >{{1, 1, 1}, {2, 2, 2}, {3, 3, 3}},
         0.,
         0.},
      // alternating wins balance out (see alternating_wins_balance_out above)
      std::tuple{
         GoofspielConfig{},
         std::vector< std::tuple< uint8_t, uint8_t, uint8_t > >{{1, 3, 2}, {2, 1, 3}, {3, 2, 1}},
         2.,
         -2.},
      // a discarded middle prize plus split wins: r1: 1v2 -> two +1 | r2: 3v3 -> discarded |
      // r3: 2v1 -> one +3
      std::tuple{
         GoofspielConfig{},
         std::vector< std::tuple< uint8_t, uint8_t, uint8_t > >{{1, 1, 2}, {2, 3, 3}, {3, 2, 1}},
         2.,
         -2.},
      // five cards: one wins prizes 2 and 4, two wins prizes 1, 3 and 5
      std::tuple{
         GoofspielConfig{.deck_size = 5},
         std::vector< std::tuple< uint8_t, uint8_t, uint8_t > >{
            {1, 2, 3},
            {2, 4, 1},
            {3, 1, 2},
            {4, 5, 4},
            {5, 3, 5}},
         -3.,
         3.},
      // limited-information mode shares the very same payoff structure
      std::tuple{
         GoofspielConfig{.imp_info = true},
         std::vector< std::tuple< uint8_t, uint8_t, uint8_t > >{{1, 3, 2}, {2, 1, 3}, {3, 2, 1}},
         2.,
         -2.}
   )
);

// ##################################################################################################################
// World state: deck / chance probabilities
// ##################################################################################################################

TEST_F(GoofspielState, reveal_probabilities_sum_to_one_per_round)
{
   for(size_t round = 0; round < state.deck_size(); ++round) {
      ASSERT_EQ(state.phase(), Phase::prize_reveal);
      auto outcomes = state.chance_actions();
      ASSERT_EQ(outcomes.size(), state.deck_size() - round);
      double prob_sum = 0.;
      for(const auto& outcome : outcomes) {
         double p = state.chance_probability(outcome);
         EXPECT_DOUBLE_EQ(p, 1. / double(outcomes.size()));
         prob_sum += p;
      }
      EXPECT_NEAR(prob_sum, 1., 1e-12);
      // consume one round via arbitrary legal moves
      state.apply_action(outcomes.front());
      state.apply_action(state.actions().front());
      state.apply_action(state.actions().front());
      state.apply_action(PrizeCard{0});
   }
   EXPECT_TRUE(state.is_terminal());
}

TEST_F(GoofspielState, deck_exhaustion_and_illegal_redraw)
{
   play_round(state, 1, 1, 2);
   play_round(state, 2, 2, 3);
   play_round(state, 3, 3, 1);

   ASSERT_TRUE(state.is_terminal());
   EXPECT_TRUE(state.chance_actions().empty());
   EXPECT_TRUE(state.actions().empty());
   EXPECT_DOUBLE_EQ(state.chance_probability(PrizeCard{1}), 0.);
}

// ##################################################################################################################
// World state: phase machine legality
// ##################################################################################################################

TEST_F(GoofspielState, illegal_transitions_rejected)
{
   // bidding before any prize reveal
   EXPECT_THROW(state.apply_action(Bid{1}), std::invalid_argument);
   EXPECT_FALSE(state.is_valid(Bid{1}));
   // resolving before anything was committed
   EXPECT_THROW(state.apply_action(PrizeCard{0}), std::invalid_argument);
   EXPECT_FALSE(state.is_valid(PrizeCard{0}));
   // revealing a prize not in the deck
   EXPECT_THROW(state.apply_action(PrizeCard{7}), std::invalid_argument);
   EXPECT_THROW(state.apply_action(PrizeCard{0}), std::invalid_argument);

   state.apply_action(PrizeCard{2});
   ASSERT_EQ(state.phase(), Phase::commit_p1);
   // a second reveal during the commit phases
   EXPECT_THROW(state.apply_action(PrizeCard{1}), std::invalid_argument);
   EXPECT_FALSE(state.is_valid(PrizeCard{1}));
   // resolving before both players committed
   EXPECT_THROW(state.apply_action(PrizeCard{0}), std::invalid_argument);

   state.apply_action(Bid{2});
   ASSERT_EQ(state.phase(), Phase::commit_p2);
   // player one cannot bid twice: his committed card left his hand, so any further bid attempt
   // through the machine would target player two whose hand still holds '2' -- but player two IS
   // the legitimate committer now, hence verify the hand bookkeeping directly
   EXPECT_EQ(state.hand_mask(Player::one) & goofspiel::card_bit(2), 0);
   EXPECT_NE(state.hand_mask(Player::two) & goofspiel::card_bit(2), 0);
   // the resolve confirmation stays illegal until the second commitment arrived
   EXPECT_THROW(state.apply_action(PrizeCard{0}), std::invalid_argument);

   state.apply_action(Bid{3});
   ASSERT_EQ(state.phase(), Phase::resolve);
   // only the sentinel confirm outcome is legal here
   EXPECT_THROW(state.apply_action(PrizeCard{1}), std::invalid_argument);
   EXPECT_FALSE(state.is_valid(PrizeCard{1}));
   EXPECT_TRUE(state.is_valid(PrizeCard{0}));

   state.apply_action(PrizeCard{0});
   EXPECT_EQ(state.round(), 1u);
   EXPECT_EQ(state.phase(), Phase::prize_reveal);
}

TEST_F(GoofspielState, cards_cannot_be_reused_across_rounds)
{
   play_round(state, 1, 2, 3);

   state.apply_action(PrizeCard{2});
   ASSERT_EQ(state.phase(), Phase::commit_p1);
   auto legal = state.actions();
   // card 2 was spent in round one
   EXPECT_FALSE(std::ranges::contains(legal, Bid{2}));
   EXPECT_TRUE(std::ranges::contains(legal, Bid{1}));
   EXPECT_THROW(state.apply_action(Bid{2}), std::invalid_argument);
   EXPECT_FALSE(state.is_valid(Bid{2}));

   // player two analogously cannot reuse his spent card
   state.apply_action(Bid{1});
   ASSERT_EQ(state.phase(), Phase::commit_p2);
   EXPECT_FALSE(std::ranges::contains(state.actions(), Bid{3}));
   EXPECT_THROW(state.apply_action(Bid{3}), std::invalid_argument);
}

TEST_F(GoofspielState, no_actions_beyond_terminality)
{
   play_round(state, 1, 1, 1);
   play_round(state, 2, 2, 2);
   play_round(state, 3, 3, 3);

   ASSERT_TRUE(state.is_terminal());
   EXPECT_THROW(state.apply_action(Bid{1}), std::logic_error);
   EXPECT_THROW(state.apply_action(PrizeCard{1}), std::logic_error);
   EXPECT_THROW(state.apply_action(PrizeCard{0}), std::logic_error);
}

TEST_F(GoofspielState, state_equality_and_copy)
{
   auto untouched = state;
   play_round(state, 1, 2, 3);
   EXPECT_FALSE(state == untouched);
   auto copy = state;
   EXPECT_TRUE(copy == state);
   state.apply_action(PrizeCard{2});
   EXPECT_FALSE(copy == state);
}

// ##################################################################################################################
// Chance-driven terminality & zero-sumness across random playouts
// ##################################################################################################################

TEST(TerminalityPlayouts, always_terminal_at_round_k_and_zero_sum)
{
   std::mt19937 rng{12345};
   for(size_t k : {size_t(1), size_t(3), size_t(5)}) {
      SCOPED_TRACE(::testing::Message() << "deck_size=" << k);
      for(int trial = 0; trial < 50; ++trial) {
         State s{GoofspielConfig{.deck_size = k}};
         size_t reveals = 0;
         std::array< int32_t, 2 > final_scores{};
         ASSERT_TRUE(random_playout(s, rng, &final_scores, &reveals));

         ASSERT_TRUE(s.is_terminal());
         EXPECT_EQ(s.round(), k);
         EXPECT_EQ(reveals, k);
         // zero-sum rewards
         EXPECT_NEAR(s.payoff(Player::one) + s.payoff(Player::two), 0., 1e-12);
         // scores are consistent with the recorded rounds' outcomes
         int32_t sum_one = 0, sum_two = 0;
         int32_t awarded_total = 0;
         for(size_t r = 0; r < k; ++r) {
            const auto& rec = s.rounds()[r];
            switch(rec.outcome) {
               case RoundOutcome::p1_wins:
                  sum_one += rec.prize;
                  awarded_total += rec.prize;
                  break;
               case RoundOutcome::p2_wins:
                  sum_two += rec.prize;
                  awarded_total += rec.prize;
                  break;
               case RoundOutcome::tie: break;
            }
         }
         EXPECT_EQ(final_scores[0], sum_one);
         EXPECT_EQ(final_scores[1], sum_two);
         // zero-sum scoreboard: awarded points cancel exactly
         EXPECT_EQ(sum_one - sum_two, s.payoff(Player::one));
         (void) awarded_total;
      }
   }
}

// ##################################################################################################################
// FOSG environment adapter: observation policies
// ##################################################################################################################

TEST(EnvironmentObservations, public_reveal_exposes_bids_after_resolve)
{
   Environment env{GoofspielConfig{.deck_size = 3, .imp_info = false}};
   auto wstate = env.initial_world_state();

   // chance reveals prize 2
   auto pre = wstate;
   env.transition(wstate, PrizeCard{2});
   auto pub_prize = env.public_observation(pre, PrizeCard{2}, wstate);
   ASSERT_TRUE(pub_prize.prize.has_value());
   EXPECT_EQ(*pub_prize.prize, 2);
   EXPECT_EQ(pub_prize, Observation{.prize = uint8_t(2)});
   // nobody learns anything private from a reveal
   EXPECT_EQ(env.private_observation(nor::Player::alex, pre, PrizeCard{2}, wstate), Observation{});

   // player one commits bid 3: the commitment event is public, its value withheld
   pre = wstate;
   env.transition(wstate, Bid{3});
   auto pub_commit = env.public_observation(pre, Bid{3}, wstate);
   ASSERT_TRUE(pub_commit.committed_by.has_value());
   EXPECT_EQ(*pub_commit.committed_by, ::goofspiel::Player::one);
   EXPECT_FALSE(pub_commit.revealed_bids.has_value());
   EXPECT_FALSE(pub_commit.own_bid.has_value());
   // the committer privately knows his own bid, the opponent does not
   auto own = env.private_observation(nor::Player::alex, pre, Bid{3}, wstate);
   ASSERT_TRUE(own.own_bid.has_value());
   EXPECT_EQ(own.own_bid->card, 3);
   EXPECT_EQ(env.private_observation(nor::Player::bob, pre, Bid{3}, wstate), Observation{});

   // player two commits bid 1 and the resolve publishes both bids right afterwards
   env.transition(wstate, Bid{1});
   pre = wstate;
   env.transition(wstate, PrizeCard{0});
   auto pub_resolve = env.public_observation(pre, PrizeCard{0}, wstate);
   ASSERT_TRUE(pub_resolve.revealed_bids.has_value());
   EXPECT_EQ(pub_resolve.revealed_bids->first.card, 3);
   EXPECT_EQ(pub_resolve.revealed_bids->second.card, 1);
   ASSERT_TRUE(pub_resolve.outcome.has_value());
   EXPECT_EQ(*pub_resolve.outcome, RoundOutcome::p1_wins);
}

TEST(EnvironmentObservations, limited_info_announces_only_outcomes)
{
   Environment env{GoofspielConfig{.deck_size = 3, .imp_info = true}};
   auto wstate = env.initial_world_state();

   env.transition(wstate, PrizeCard{2});
   auto pre = wstate;
   env.transition(wstate, Bid{3});
   // commitment events stay observable (the sequentialized turn order implies them anyway)
   auto pub_commit = env.public_observation(pre, Bid{3}, wstate);
   ASSERT_TRUE(pub_commit.committed_by.has_value());

   env.transition(wstate, Bid{1});
   pre = wstate;
   env.transition(wstate, PrizeCard{0});
   auto pub_resolve = env.public_observation(pre, PrizeCard{0}, wstate);
   ASSERT_TRUE(pub_resolve.outcome.has_value());
   EXPECT_EQ(*pub_resolve.outcome, RoundOutcome::p1_wins);
   // but neither bid value leaks
   EXPECT_FALSE(pub_resolve.revealed_bids.has_value());
}

TEST(EnvironmentObservations, opponent_bid_hidden_in_limited_info_only)
{
   // two games identical except for player two's second-round bid; in limited-information mode the
   // resulting infostate of player one must be indistinguishable
   const std::array< uint8_t, 5 > prizes{1, 2, 3, 4, 5};
   const std::array< uint8_t, 5 > bids_one{2, 5, 1, 4, 3};
   // both of two's plays win/lose exactly the same rounds (outcome pattern W2,W1,W2,W1,W2) while
   // the bid values genuinely differ in three of them
   const std::array< uint8_t, 5 > bids_two_a{4, 1, 3, 2, 5};
   const std::array< uint8_t, 5 > bids_two_b{3, 1, 5, 2, 4};

   Environment reveal_env{GoofspielConfig{.deck_size = 5, .imp_info = false}};
   auto reveal_a = observed_infoset(reveal_env, nor::Player::alex, prizes, bids_one, bids_two_a);
   auto reveal_b = observed_infoset(reveal_env, nor::Player::alex, prizes, bids_one, bids_two_b);
   EXPECT_NE(reveal_a, reveal_b);
   EXPECT_NE(reveal_a.hash(), reveal_b.hash());
   std::unordered_set< Infostate > reveal_set;
   reveal_set.emplace(reveal_a);
   reveal_set.emplace(reveal_b);
   EXPECT_EQ(reveal_set.size(), 2u);

   Environment lim_env{GoofspielConfig{.deck_size = 5, .imp_info = true}};
   auto lim_a = observed_infoset(lim_env, nor::Player::alex, prizes, bids_one, bids_two_a);
   auto lim_b = observed_infoset(lim_env, nor::Player::alex, prizes, bids_one, bids_two_b);
   EXPECT_EQ(lim_a, lim_b);
   EXPECT_EQ(lim_a.hash(), lim_b.hash());
   std::unordered_set< Infostate > lim_set;
   lim_set.emplace(lim_a);
   lim_set.emplace(lim_b);
   EXPECT_EQ(lim_set.size(), 1u);

   // sanity: player two's own view DOES distinguish his diverging bids even in limited-info mode
   auto lim_two_a = observed_infoset(lim_env, nor::Player::bob, prizes, bids_one, bids_two_a);
   auto lim_two_b = observed_infoset(lim_env, nor::Player::bob, prizes, bids_one, bids_two_b);
   EXPECT_NE(lim_two_a, lim_two_b);
}

TEST(EnvironmentObservations, infostate_hash_and_equality_sanity)
{
   Infostate istate(nor::Player::alex);
   Infostate twin(nor::Player::alex);
   Infostate other_player(nor::Player::bob);

   Observation prize_obs{.prize = uint8_t(3)};
   Observation bid_obs{.own_bid = Bid{3}};

   istate.update(prize_obs, Observation{});
   twin.update(prize_obs, Observation{});
   other_player.update(prize_obs, bid_obs);

   EXPECT_EQ(istate, twin);
   EXPECT_NE(istate.hash(), 0u);
   EXPECT_EQ(istate.hash(), twin.hash());
   EXPECT_NE(istate, other_player);

   std::unordered_set< Infostate > set;
   set.emplace(istate);
   set.emplace(twin);
   EXPECT_EQ(set.size(), 1u);
   set.emplace(other_player);
   EXPECT_EQ(set.size(), 2u);
}

TEST(EnvironmentHistories, open_history_covers_every_transition)
{
   Environment env{GoofspielConfig{.deck_size = 3}};
   auto wstate = env.initial_world_state();
   play_round(wstate, 2, 1, 3);
   // mid second round: prize revealed, player one committed
   wstate.apply_action(PrizeCard{1});
   wstate.apply_action(Bid{2});

   auto open = env.open_history(wstate);
   // round one: prize + bid + bid + resolve | round two: prize + bid
   ASSERT_EQ(open.size(), 6u);
   EXPECT_TRUE(std::holds_alternative< PrizeCard >(open[0].value()));
   EXPECT_EQ(std::get< PrizeCard >(open[0].value()).value, 2);
   EXPECT_EQ(std::get< Bid >(open[1].value()).card, 1);
   EXPECT_EQ(open[1].player(), nor::Player::alex);
   EXPECT_EQ(std::get< Bid >(open[2].value()).card, 3);
   EXPECT_EQ(open[2].player(), nor::Player::bob);
   EXPECT_EQ(std::get< PrizeCard >(open[3].value()).value, 0);
   EXPECT_EQ(std::get< PrizeCard >(open[4].value()).value, 1);
   EXPECT_EQ(std::get< Bid >(open[5].value()).card, 2);
}

TEST(EnvironmentHistories, private_history_masks_opponent_bids)
{
   {
      Environment env{GoofspielConfig{.deck_size = 3, .imp_info = false}};
      auto wstate = env.initial_world_state();
      play_round(wstate, 2, 1, 3);

      auto hist = env.private_history(nor::Player::bob, wstate);
      ASSERT_EQ(hist.size(), 4u);
      // resolved bids are public knowledge in public-reveal mode
      EXPECT_TRUE(std::get< Bid >(hist[1].value().value()).card == 1);
      EXPECT_TRUE(std::get< Bid >(hist[2].value().value()).card == 3);
   }
   {
      Environment env{GoofspielConfig{.deck_size = 3, .imp_info = true}};
      auto wstate = env.initial_world_state();
      play_round(wstate, 2, 1, 3);

      auto hist = env.private_history(nor::Player::bob, wstate);
      ASSERT_EQ(hist.size(), 4u);
      // player one's bid never reached bob
      EXPECT_FALSE(hist[1].value().has_value());
      // bob's own bid stays visible to himself
      EXPECT_TRUE(std::get< Bid >(hist[2].value().value()).card == 3);

      auto hist_alex = env.private_history(nor::Player::alex, wstate);
      ASSERT_EQ(hist_alex.size(), 4u);
      EXPECT_TRUE(std::get< Bid >(hist_alex[1].value().value()).card == 1);
      EXPECT_FALSE(hist_alex[2].value().has_value());
   }
}

// ##################################################################################################################
// Convergence smoke: Vanilla CFR on k=3 goofspiel
// ##################################################################################################################

TEST(GoofspielCFR, VanillaCFRAlternatingConvergesK3)
{
   using ActionT = Bid;

   Environment env{};
   auto root_state = std::make_unique< State >();

   auto avg_tabular_policy = nor::factory::make_tabular_policy(
      std::unordered_map< Infostate, nor::HashmapActionPolicy< ActionT > >{}
   );
   auto tabular_policy = nor::factory::make_tabular_policy(
      std::unordered_map< Infostate, nor::HashmapActionPolicy< ActionT > >{}
   );

   auto solver = nor::factory::
      make_cfr< nor::rm::CFRConfig{.update_mode = nor::rm::UpdateMode::alternating}, true >(
         std::move(env), std::move(root_state), tabular_policy, avg_tabular_policy
      );

   constexpr size_t max_iters = 200;
   constexpr size_t eval_freq = 10;
   // generous absolute threshold in score units (payoffs range in [-8, 8] for k=3)
   constexpr double expl_threshold = 0.25;

   double first_expl = std::numeric_limits< double >::max();
   double last_expl = std::numeric_limits< double >::max();
   double prev_expl = std::numeric_limits< double >::max();
   size_t monotone_violations = 0;

   for(size_t iter = 1; iter <= max_iters; ++iter) {
      solver.iterate(1);
      if(iter % eval_freq == 0) {
         const auto& avg_policies = solver.average_policy();
         last_expl = nor::exploitability(
            env,
            State{},
            nor::player_hashmap< std::decay_t< decltype(avg_policies.at(nor::Player::alex)) > >{
               std::pair{
                  nor::Player::alex,
                  nor::normalize_state_policy(avg_policies.at(nor::Player::alex))},
               std::pair{
                  nor::Player::bob,
                  nor::normalize_state_policy(avg_policies.at(nor::Player::bob))}},
            /*constant_sum=*/true
         );
         if(first_expl == std::numeric_limits< double >::max()) {
            first_expl = last_expl;
         }
         if(last_expl > prev_expl + 1e-9) {
            ++monotone_violations;
         }
         prev_expl = last_expl;
      }
   }

   std::cout << "[  INFO  ] goofspiel k=3 vanilla-CFR exploitability: start=" << first_expl
             << " end=" << last_expl << " (non-monotone eval steps: " << monotone_violations << ")"
             << "\n";

   EXPECT_LT(last_expl, first_expl);
   EXPECT_LE(monotone_violations, 6u);  // decreasing "monotonically-ish"
   EXPECT_LT(last_expl, expl_threshold);
}
