
#include <gtest/gtest.h>

#include <array>
#include <unordered_set>
#include <variant>

#include "fixtures.hpp"
#include "nor/concepts/concrete.hpp"
#include "texas_holdem_poker/environment.hpp"
#include "texas_holdem_poker/hand_evaluator.hpp"

using namespace texholdem;

// ##################################################################################################################
// Helpers
// ##################################################################################################################

namespace {

Card C(Rank rank, Suit suit)
{
   return Card{rank, suit};
}

uint32_t score(std::initializer_list< Card > cards)
{
   return eval::evaluate(cards.begin(), cards.size());
}

auto default_config()
{
   return PokerConfig{};
}

/// three players with stacks {10, 25, 50}: dealer seat 0, sb seat 1, bb seat 2
auto three_player_config()
{
   return PokerConfig(3, 200., 1., 2., {1., 2., 4., 8.}, std::vector< double >{10., 25., 50.});
}

/// the four hole card deals of a heads-up hand in chronological deal order
std::vector< HoldemStep >
heads_up_holes(Card first_one, Card first_two, Card second_one, Card second_two)
{
   return {
      HoldemStep{first_one}, HoldemStep{first_two}, HoldemStep{second_one}, HoldemStep{second_two}};
}

}  // namespace

// ##################################################################################################################
// Hand evaluation
// ##################################################################################################################

TEST(EvaluatorTest, category_ordering)
{
   auto high_card = score(
      {C(Rank::two, Suit::clubs),
       C(Rank::five, Suit::diamonds),
       C(Rank::nine, Suit::hearts),
       C(Rank::jack, Suit::diamonds),
       C(Rank::king, Suit::spades)}
   );
   auto one_pair = score(
      {C(Rank::two, Suit::clubs),
       C(Rank::two, Suit::diamonds),
       C(Rank::nine, Suit::hearts),
       C(Rank::jack, Suit::diamonds),
       C(Rank::king, Suit::spades)}
   );
   auto two_pair = score(
      {C(Rank::two, Suit::clubs),
       C(Rank::two, Suit::diamonds),
       C(Rank::nine, Suit::hearts),
       C(Rank::nine, Suit::diamonds),
       C(Rank::king, Suit::spades)}
   );
   auto trips = score(
      {C(Rank::two, Suit::clubs),
       C(Rank::two, Suit::diamonds),
       C(Rank::two, Suit::hearts),
       C(Rank::jack, Suit::diamonds),
       C(Rank::king, Suit::spades)}
   );
   auto straight = score(
      {C(Rank::five, Suit::clubs),
       C(Rank::six, Suit::diamonds),
       C(Rank::seven, Suit::hearts),
       C(Rank::eight, Suit::spades),
       C(Rank::nine, Suit::clubs)}
   );
   auto flush = score(
      {C(Rank::ace, Suit::hearts),
       C(Rank::ten, Suit::hearts),
       C(Rank::seven, Suit::hearts),
       C(Rank::four, Suit::hearts),
       C(Rank::two, Suit::hearts)}
   );
   auto full_house = score(
      {C(Rank::queen, Suit::clubs),
       C(Rank::queen, Suit::diamonds),
       C(Rank::queen, Suit::hearts),
       C(Rank::seven, Suit::spades),
       C(Rank::seven, Suit::clubs)}
   );
   auto quads = score(
      {C(Rank::king, Suit::clubs),
       C(Rank::king, Suit::diamonds),
       C(Rank::king, Suit::hearts),
       C(Rank::king, Suit::spades),
       C(Rank::two, Suit::diamonds)}
   );
   auto royal = score(
      {C(Rank::ace, Suit::clubs),
       C(Rank::king, Suit::clubs),
       C(Rank::queen, Suit::clubs),
       C(Rank::jack, Suit::clubs),
       C(Rank::ten, Suit::clubs)}
   );

   EXPECT_GT(one_pair, high_card);
   EXPECT_GT(two_pair, one_pair);
   EXPECT_GT(trips, two_pair);
   EXPECT_GT(straight, trips);
   EXPECT_GT(flush, straight);
   EXPECT_GT(full_house, flush);
   EXPECT_GT(quads, full_house);
   EXPECT_GT(royal, quads);
}

TEST(EvaluatorTest, kickers_decide_within_category)
{
   // pair of kings with ace kicker beats pair of kings with queen kicker
   auto higher = score(
      {C(Rank::king, Suit::clubs),
       C(Rank::king, Suit::diamonds),
       C(Rank::ace, Suit::hearts),
       C(Rank::seven, Suit::hearts),
       C(Rank::three, Suit::clubs)}
   );
   auto lower = score(
      {C(Rank::king, Suit::hearts),
       C(Rank::king, Suit::spades),
       C(Rank::queen, Suit::hearts),
       C(Rank::seven, Suit::hearts),
       C(Rank::four, Suit::diamonds)}
   );
   EXPECT_GT(higher, lower);

   // two pair: the second pair decides before the side card
   auto two_pair_higher = score(
      {C(Rank::ace, Suit::clubs),
       C(Rank::ace, Suit::diamonds),
       C(Rank::three, Suit::hearts),
       C(Rank::three, Suit::spades),
       C(Rank::queen, Suit::clubs)}
   );
   auto two_pair_lower = score(
      {C(Rank::ace, Suit::hearts),
       C(Rank::ace, Suit::spades),
       C(Rank::two, Suit::hearts),
       C(Rank::two, Suit::spades),
       C(Rank::king, Suit::clubs)}
   );
   EXPECT_GT(two_pair_higher, two_pair_lower);
}

TEST(EvaluatorTest, straights_including_wheel)
{
   auto wheel = score(
      {C(Rank::ace, Suit::clubs),
       C(Rank::two, Suit::diamonds),
       C(Rank::three, Suit::hearts),
       C(Rank::four, Suit::spades),
       C(Rank::five, Suit::clubs)}
   );
   auto six_high = score(
      {C(Rank::two, Suit::clubs),
       C(Rank::three, Suit::diamonds),
       C(Rank::four, Suit::hearts),
       C(Rank::five, Suit::spades),
       C(Rank::six, Suit::clubs)}
   );
   auto broadway = score(
      {C(Rank::ten, Suit::clubs),
       C(Rank::jack, Suit::diamonds),
       C(Rank::queen, Suit::hearts),
       C(Rank::king, Suit::spades),
       C(Rank::ace, Suit::clubs)}
   );
   // the wheel is the weakest straight but still beats any trips
   auto trips = score(
      {C(Rank::eight, Suit::clubs),
       C(Rank::eight, Suit::diamonds),
       C(Rank::eight, Suit::hearts),
       C(Rank::four, Suit::spades),
       C(Rank::two, Suit::clubs)}
   );
   EXPECT_GT(wheel, trips);
   EXPECT_GT(six_high, wheel);
   EXPECT_GT(broadway, six_high);
}

TEST(EvaluatorTest, best_five_of_seven)
{
   // seven cards containing an ace-high heart flush buried among noise + a redundant pair
   std::array< Card, 7 > cards{
      C(Rank::ace, Suit::hearts),
      C(Rank::ten, Suit::hearts),
      C(Rank::seven, Suit::hearts),
      C(Rank::four, Suit::hearts),
      C(Rank::two, Suit::hearts),
      C(Rank::king, Suit::clubs),
      C(Rank::king, Suit::diamonds)};
   uint32_t seven_card_score = eval::evaluate(cards.data(), cards.size());
   // the evaluator must find the pure ace-high heart flush among the seven cards
   auto plain_flush = score(
      {C(Rank::ace, Suit::hearts),
       C(Rank::ten, Suit::hearts),
       C(Rank::seven, Suit::hearts),
       C(Rank::four, Suit::hearts),
       C(Rank::two, Suit::hearts)}
   );
   EXPECT_EQ(seven_card_score, plain_flush);

   // board-play tie: both players' best five is the same broadway straight
   std::array< Card, 7 > first{
      C(Rank::ace, Suit::hearts),
      C(Rank::king, Suit::diamonds),
      C(Rank::queen, Suit::spades),
      C(Rank::jack, Suit::diamonds),
      C(Rank::ten, Suit::clubs),
      C(Rank::two, Suit::clubs),
      C(Rank::three, Suit::diamonds)};
   std::array< Card, 7 > second{
      C(Rank::ace, Suit::spades),
      C(Rank::king, Suit::clubs),
      C(Rank::queen, Suit::hearts),
      C(Rank::jack, Suit::clubs),
      C(Rank::ten, Suit::spades),
      C(Rank::five, Suit::clubs),
      C(Rank::six, Suit::diamonds)};
   EXPECT_EQ(
      eval::evaluate(first.data(), first.size()), eval::evaluate(second.data(), second.size())
   );
}

// ##################################################################################################################
// World state: blinds, dealing, betting, payoffs
// ##################################################################################################################

TEST_F(TexasHoldemState, blinds_are_posted_at_hand_start)
{
   // heads-up default: dealer seat 0 posts the small blind, seat 1 the big blind
   EXPECT_EQ(state.dealer_pos(), 0);
   EXPECT_EQ(state.small_blind_pos(), 0);
   EXPECT_EQ(state.big_blind_pos(), 1);
   EXPECT_DOUBLE_EQ(state.stack(Player::one), 199.);
   EXPECT_DOUBLE_EQ(state.stack(Player::two), 198.);
   EXPECT_DOUBLE_EQ(state.street_contribution(Player::one), 1.);
   EXPECT_DOUBLE_EQ(state.street_contribution(Player::two), 2.);
   EXPECT_DOUBLE_EQ(state.current_total_bet(), 2.);
   EXPECT_DOUBLE_EQ(state.last_increment(), 2.);
   EXPECT_DOUBLE_EQ(state.pot(), 3.);
   EXPECT_EQ(state.active_player(), Player::chance);
   EXPECT_FALSE(state.is_terminal());
   EXPECT_TRUE(state.actions().empty());
   EXPECT_EQ(state.chance_actions().size(), 52u);
}

TEST_F(TexasHoldemState, chance_dealing_order_and_deck)
{
   auto deck = state.chance_actions();
   ASSERT_EQ(deck.size(), 52u);
   EXPECT_DOUBLE_EQ(state.chance_probability(deck.front()), 1. / 52.);

   // deal order heads-up: SB(seat 0), BB(seat 1), SB, BB
   auto c1 = C(Rank::three, Suit::clubs);
   auto c2 = C(Rank::four, Suit::spades);
   auto c3 = C(Rank::ace, Suit::diamonds);
   auto c4 = C(Rank::ace, Suit::spades);

   state.apply_action(c1);
   EXPECT_EQ(state.next_deal_recipient(), Player::two);
   state.apply_action(c2);
   EXPECT_EQ(state.next_deal_recipient(), Player::one);
   state.apply_action(c3);
   state.apply_action(c4);

   EXPECT_EQ(state.hole_card(Player::one, 0), c1);
   EXPECT_EQ(state.hole_card(Player::two, 0), c2);
   EXPECT_EQ(state.hole_card(Player::one, 1), c3);
   EXPECT_EQ(state.hole_card(Player::two, 1), c4);

   EXPECT_EQ(state.active_player(), Player::one);  // preflop starts at the button heads-up
   EXPECT_EQ(state.deck_size(), 48u);
   EXPECT_DOUBLE_EQ(state.chance_probability(c1), 0.);  // already dealt
   EXPECT_THROW(state.apply_action(c1), std::invalid_argument);
   // no further chance outcomes are due while players act
   EXPECT_TRUE(state.chance_actions().empty());
}

TEST_F(TexasHoldemState, betting_legality_and_min_raise)
{
   apply_steps(
      state,
      {HoldemStep{C(Rank::two, Suit::clubs)},
       HoldemStep{C(Rank::king, Suit::diamonds)},
       HoldemStep{C(Rank::ace, Suit::hearts)},
       HoldemStep{C(Rank::queen, Suit::spades)}}
   );

   const double bb = 2.;
   auto sb_actions = state.actions();
   // fold & call (SB owes 1), raise-to targets bb + increments of bb*mult for mult {1,2,4,8},
   // plus all-in
   EXPECT_TRUE(ranges::contains(sb_actions, Action{ActionType::fold}));
   EXPECT_TRUE(ranges::contains(sb_actions, Action{ActionType::call}));
   EXPECT_FALSE(ranges::contains(sb_actions, Action{ActionType::check}));
   for(double target : {4., 6., 10., 18.}) {
      EXPECT_TRUE(ranges::contains(sb_actions, Action{ActionType::raise, target}));
   }
   EXPECT_TRUE(ranges::contains(sb_actions, Action{ActionType::all_in}));

   // after a raise to 6 (increment 4 over the big blind) the minimum re-raise-to is 10:
   state.apply_action(Action{ActionType::raise, 6.});
   auto bb_actions = state.actions();
   EXPECT_FALSE(ranges::contains(bb_actions, Action{ActionType::raise, 8.}));  // below min-raise
   EXPECT_TRUE(ranges::contains(bb_actions, Action{ActionType::raise, 10.}));
   EXPECT_TRUE(ranges::contains(bb_actions, Action{ActionType::raise, 14.}));
   EXPECT_TRUE(ranges::contains(bb_actions, Action{ActionType::raise, 22.}));
   EXPECT_FALSE(ranges::contains(bb_actions, Action{ActionType::bet, 6.})
   );  // no bets when facing one
   EXPECT_TRUE(ranges::contains(bb_actions, Action{ActionType::fold}));
   EXPECT_TRUE(ranges::contains(bb_actions, Action{ActionType::call}));
   EXPECT_DOUBLE_EQ(state.last_increment(), 4.);
   EXPECT_EQ(state.last_aggressor(), 0);

   // illegal amounts are rejected by is_valid
   EXPECT_FALSE(state.is_valid(Action{ActionType::raise, 8.}));
   EXPECT_TRUE(state.is_valid(Action{ActionType::raise, 10.}));

   // calling equalizes the wager and concludes the street --> the flop is due from chance
   state.apply_action(Action{ActionType::call});
   EXPECT_EQ(state.active_player(), Player::chance);
   EXPECT_EQ(state.street(), Street::preflop);
   state.apply_action(C(Rank::two, Suit::hearts));
   state.apply_action(C(Rank::seven, Suit::diamonds));
   state.apply_action(C(Rank::nine, Suit::diamonds));
   EXPECT_EQ(state.street(), Street::flop);
   EXPECT_EQ(state.n_community_cards(), 3u);
   // postflop action starts left of the button (= the big blind in heads-up)
   EXPECT_EQ(state.active_player(), Player::two);
   // postflop betting opens at zero
   auto flop_actions = state.actions();
   EXPECT_TRUE(ranges::contains(flop_actions, Action{ActionType::check}));
   EXPECT_TRUE(ranges::contains(flop_actions, Action{ActionType::bet, bb * 8.}));
   EXPECT_FALSE(ranges::contains(flop_actions, Action{ActionType::call}));
   EXPECT_DOUBLE_EQ(state.current_total_bet(), 0.);
}

TEST_F(TexasHoldemState, short_stack_all_in_below_min_raise)
{
   texholdem::State short_state{PokerConfig(2, /*stack=*/3.)};
   apply_steps(
      short_state,
      {HoldemStep{C(Rank::two, Suit::clubs)},
       HoldemStep{C(Rank::king, Suit::diamonds)},
       HoldemStep{C(Rank::ace, Suit::hearts)},
       HoldemStep{C(Rank::queen, Suit::spades)}}
   );
   auto actions = short_state.actions();
   // a jam to 3 is below the min-raise-to of 4 but still offered as an explicit all-in
   EXPECT_FALSE(ranges::contains(actions, Action{ActionType::raise, 3.}));
   EXPECT_TRUE(ranges::contains(actions, Action{ActionType::all_in}));

   short_state.apply_action(Action{ActionType::all_in});
   EXPECT_TRUE(short_state.is_allin(Player::one));
   EXPECT_DOUBLE_EQ(short_state.total_contribution(Player::one), 3.);
   EXPECT_DOUBLE_EQ(short_state.current_total_bet(), 3.);
   EXPECT_TRUE(short_state.is_valid(Action{ActionType::call}));  // BB can call off his last chip

   short_state.apply_action(Action{ActionType::call});
   EXPECT_TRUE(short_state.is_allin(Player::two));
   // nobody can act anymore: the board runs out without any betting
   EXPECT_EQ(short_state.active_player(), Player::chance);
   for(auto card :
       {C(Rank::two, Suit::hearts),
        C(Rank::seven, Suit::diamonds),
        C(Rank::nine, Suit::spades),
        C(Rank::three, Suit::diamonds),
        C(Rank::four, Suit::clubs)}) {
      ASSERT_FALSE(short_state.is_terminal());
      short_state.apply_action(card);
   }
   ASSERT_TRUE(short_state.is_terminal());
   // aces win: pot 6, each player contributed 3
   auto payoffs = short_state.payoffs();
   EXPECT_DOUBLE_EQ(payoffs[0], 3.);
   EXPECT_DOUBLE_EQ(payoffs[1], -3.);
}

TEST_F(TexasHoldemState, check_down_showdown_payoffs)
{
   // P1: Ah Ad -- P2: Ks Kd; board 2c 7h 9s | 3d | 4c
   apply_steps(
      state,
      {HoldemStep{C(Rank::ace, Suit::hearts)},
       HoldemStep{C(Rank::king, Suit::spades)},
       HoldemStep{C(Rank::ace, Suit::diamonds)},
       HoldemStep{C(Rank::king, Suit::diamonds)}}
   );
   // P1 calls the blind, P2 checks back --> each has 2 chips in the pot
   state.apply_action(Action{ActionType::call});
   state.apply_action(Action{ActionType::check});
   for(auto card :
       {C(Rank::two, Suit::clubs), C(Rank::seven, Suit::hearts), C(Rank::nine, Suit::spades)}) {
      state.apply_action(card);
   }
   // postflop: BB acts first in heads-up; check down through all streets
   state.apply_action(Action{ActionType::check});
   state.apply_action(Action{ActionType::check});
   state.apply_action(C(Rank::three, Suit::diamonds));
   state.apply_action(Action{ActionType::check});
   state.apply_action(Action{ActionType::check});
   state.apply_action(C(Rank::four, Suit::clubs));
   EXPECT_FALSE(state.is_terminal());  // river betting still pending
   state.apply_action(Action{ActionType::check});
   state.apply_action(Action{ActionType::check});

   ASSERT_TRUE(state.is_terminal());
   auto payoffs = state.payoffs();
   EXPECT_DOUBLE_EQ(payoffs[0], 2.);  // pair of aces wins the 4-chip pot
   EXPECT_DOUBLE_EQ(payoffs[1], -2.);
   EXPECT_NEAR(payoffs[0] + payoffs[1], 0., 1e-9);  // zero-sum
}

TEST_F(TexasHoldemState, split_pot_when_board_plays)
{
   // royal flush on the board: both players play the board --> exact halves
   apply_steps(
      state,
      {HoldemStep{C(Rank::two, Suit::diamonds)},
       HoldemStep{C(Rank::four, Suit::hearts)},
       HoldemStep{C(Rank::three, Suit::diamonds)},
       HoldemStep{C(Rank::five, Suit::hearts)}}
   );
   state.apply_action(Action{ActionType::call});
   state.apply_action(Action{ActionType::check});
   for(auto card :
       {C(Rank::ace, Suit::clubs), C(Rank::king, Suit::clubs), C(Rank::queen, Suit::clubs)}) {
      state.apply_action(card);
   }
   for(auto card : {C(Rank::jack, Suit::clubs), C(Rank::ten, Suit::clubs)}) {
      state.apply_action(Action{ActionType::check});
      state.apply_action(Action{ActionType::check});
      state.apply_action(card);
   }
   state.apply_action(Action{ActionType::check});
   state.apply_action(Action{ActionType::check});

   ASSERT_TRUE(state.is_terminal());
   auto payoffs = state.payoffs();
   EXPECT_DOUBLE_EQ(payoffs[0], 0.);
   EXPECT_DOUBLE_EQ(payoffs[1], 0.);
}

TEST_F(TexasHoldemState, uncalled_bet_is_returned_on_fold)
{
   apply_steps(
      state,
      {HoldemStep{C(Rank::ace, Suit::hearts)},
       HoldemStep{C(Rank::king, Suit::spades)},
       HoldemStep{C(Rank::ace, Suit::diamonds)},
       HoldemStep{C(Rank::king, Suit::diamonds)}}
   );
   state.apply_action(Action{ActionType::call});  // SB completes
   state.apply_action(Action{ActionType::check});  // BB checks --> flop
   for(auto card :
       {C(Rank::two, Suit::clubs), C(Rank::seven, Suit::hearts), C(Rank::nine, Suit::spades)}) {
      state.apply_action(card);
   }
   state.apply_action(Action{ActionType::check});  // BB checks
   // NOTE: the action space is a discrete ladder (multiples of the big blind) plus an explicit
   // all-in, so a plain oversized 'bet 50' is illegal here; a jam serves the same purpose
   state.apply_action(Action{ActionType::all_in});  // SB jams (uncalled excess)
   state.apply_action(Action{ActionType::fold});  // BB folds

   ASSERT_TRUE(state.is_terminal());
   auto payoffs = state.payoffs();
   // the bettor gets everything back except the opponent's blinds
   EXPECT_DOUBLE_EQ(payoffs[0], 2.);
   EXPECT_DOUBLE_EQ(payoffs[1], -2.);
}

TEST_F(TexasHoldemState, immediate_terminal_on_preflop_fold_out)
{
   apply_steps(
      state,
      {HoldemStep{C(Rank::ace, Suit::hearts)},
       HoldemStep{C(Rank::king, Suit::spades)},
       HoldemStep{C(Rank::ace, Suit::diamonds)},
       HoldemStep{C(Rank::king, Suit::diamonds)}}
   );
   state.apply_action(Action{ActionType::fold});
   ASSERT_TRUE(state.is_terminal());
   auto payoffs = state.payoffs();
   // the big blind wins the small blind's posting
   EXPECT_DOUBLE_EQ(payoffs[0], -1.);
   EXPECT_DOUBLE_EQ(payoffs[1], 1.);
}

TEST_F(TexasHoldemState, side_pots_with_three_players)
{
   texholdem::State multi{three_player_config()};
   // deal order (n=3): seats [sb=1, bb=2, utg=0] twice
   apply_steps(
      multi,
      {HoldemStep{C(Rank::king, Suit::spades)}, /* -> sb (seat 1) */
       HoldemStep{C(Rank::queen, Suit::spades)}, /* -> bb (seat 2) */
       HoldemStep{C(Rank::ace, Suit::spades)}, /* -> utg (seat 0) */
       HoldemStep{C(Rank::king, Suit::diamonds)}, /* -> seat 1 */
       HoldemStep{C(Rank::jack, Suit::diamonds)}, /* -> seat 2 */
       HoldemStep{C(Rank::ace, Suit::hearts)}} /* -> seat 0 */
   );
   // UTG = seat left of the big blind = seat 0
   EXPECT_EQ(multi.active_player(), Player::one);
   // fold/call + raises-to {4, 6, 10} within the 10 chip stack (jam covered by raise-to-10)
   EXPECT_EQ(multi.actions().size(), 5u);

   // UTG (seat 0) jams for 10, SB (seat 1) calls all-in for 25, BB calls 23 more (total 25)
   multi.apply_action(Action{ActionType::raise, 10.});
   multi.apply_action(Action{ActionType::all_in});
   multi.apply_action(Action{ActionType::call});

   EXPECT_TRUE(multi.is_allin(Player::one));
   EXPECT_TRUE(multi.is_allin(Player::two));
   EXPECT_FALSE(multi.is_allin(Player::three));
   EXPECT_EQ(multi.active_player(), Player::chance);

   // runout: no betting on any street
   for(auto card :
       {C(Rank::two, Suit::clubs),
        C(Rank::seven, Suit::hearts),
        C(Rank::nine, Suit::spades),
        C(Rank::three, Suit::diamonds),
        C(Rank::four, Suit::clubs)}) {
      ASSERT_FALSE(multi.is_terminal());
      multi.apply_action(card);
   }
   ASSERT_TRUE(multi.is_terminal());

   // main pot (30) goes to seat 0 (aces), side pot (30) to seat 1 (kings over queen-high)
   auto payoffs = multi.payoffs();
   EXPECT_DOUBLE_EQ(payoffs[0], 20.);  // +30 - 10
   EXPECT_DOUBLE_EQ(payoffs[1], 5.);  // +30 - 25
   EXPECT_DOUBLE_EQ(payoffs[2], -25.);  // 0 - 25
   EXPECT_NEAR(payoffs[0] + payoffs[1] + payoffs[2], 0., 1e-9);
}

TEST_F(TexasHoldemState, state_equality_and_copy)
{
   auto untouched = state;
   apply_steps(
      state,
      {HoldemStep{C(Rank::ace, Suit::hearts)},
       HoldemStep{C(Rank::king, Suit::spades)},
       HoldemStep{C(Rank::ace, Suit::diamonds)},
       HoldemStep{C(Rank::king, Suit::diamonds)}}
   );
   EXPECT_NE(state == untouched, true);
   auto copy = state;
   EXPECT_TRUE(copy == state);
   copy.apply_action(Action{ActionType::call});
   EXPECT_NE(copy == state, true);
}

TEST_F(TexasHoldemState, blind_post_consuming_stack_marks_allin_and_runout_terminates)
{
   // the big blind seat only has exactly one big blind --> his posting is an all-in
   texholdem::State tiny{
      PokerConfig(2, 200., 1., 2., {1., 2., 4., 8.}, std::vector< double >{200., 2.})};
   apply_steps(
      tiny,
      {HoldemStep{C(Rank::two, Suit::clubs)},
       HoldemStep{C(Rank::king, Suit::diamonds)},
       HoldemStep{C(Rank::ace, Suit::hearts)},
       HoldemStep{C(Rank::queen, Suit::spades)}}
   );
   EXPECT_TRUE(tiny.is_allin(Player::two));
   EXPECT_DOUBLE_EQ(tiny.stack(Player::two), 0.);

   // the small blind calls the blind and the hand runs out without any betting: the single
   // remaining player with chips must not block terminality after the river
   tiny.apply_action(Action{ActionType::call});
   EXPECT_EQ(tiny.active_player(), Player::chance);
   for(auto card :
       {C(Rank::two, Suit::hearts),
        C(Rank::seven, Suit::diamonds),
        C(Rank::nine, Suit::spades),
        C(Rank::three, Suit::diamonds),
        C(Rank::four, Suit::clubs)}) {
      ASSERT_FALSE(tiny.is_terminal());
      tiny.apply_action(card);
   }
   ASSERT_TRUE(tiny.is_terminal());
   auto payoffs = tiny.payoffs();
   // aces in the small blind win the 4-chip pot
   EXPECT_DOUBLE_EQ(payoffs[0], 2.);
   EXPECT_DOUBLE_EQ(payoffs[1], -2.);
}

// ##################################################################################################################
// FOSG environment adapter
// ##################################################################################################################

static_assert(nor::concepts::fosg< nor::games::texholdem::Environment >);
static_assert(nor::concepts::stochastic_fosg< nor::games::texholdem::Environment >);
static_assert(nor::concepts::supports_all_histories< nor::games::texholdem::Environment >);

TEST(TexholdemEnvironment, infostate_hash_and_equality)
{
   using namespace nor::games::texholdem;
   Infostate istate(nor::Player::alex);
   Infostate twin(nor::Player::alex);
   Infostate other_player(nor::Player::bob);

   Observation hole_card_obs{.card = Card{Rank::ace, Suit::spades}};
   Observation action_obs{.action = Action{ActionType::raise, 6.}};

   istate.update(action_obs, Observation{});
   twin.update(action_obs, Observation{});
   other_player.update(action_obs, hole_card_obs);

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

TEST(TexholdemEnvironment, observations_of_transitions)
{
   using namespace nor::games::texholdem;
   Environment env;

   auto wstate = env.initial_world_state();
   EXPECT_EQ(env.players(wstate).size(), 2u);
   EXPECT_EQ(env.active_player(wstate), nor::Player::chance);
   EXPECT_FALSE(env.is_terminal(wstate));
   EXPECT_TRUE(env.is_partaking(wstate, nor::Player::alex));

   auto deck = env.chance_actions(wstate);
   auto pre = wstate;
   env.transition(wstate, deck[3]);  // three of clubs --> seat 0 (alex)

   // private observation: only the recipient sees the card identity
   auto alex_priv = env.private_observation(nor::Player::alex, pre, deck[3], wstate);
   auto bob_priv = env.private_observation(nor::Player::bob, pre, deck[3], wstate);
   ASSERT_TRUE(alex_priv.card.has_value());
   EXPECT_EQ(*alex_priv.card, deck[3]);
   EXPECT_FALSE(bob_priv.card.has_value());

   // public observation: only the recipient's identity is revealed
   auto pub = env.public_observation(pre, deck[3], wstate);
   ASSERT_TRUE(pub.hidden_deal_to.has_value());
   EXPECT_EQ(to_nor_player(*pub.hidden_deal_to), nor::Player::alex);
   EXPECT_FALSE(pub.action.has_value());
   EXPECT_FALSE(pub.card.has_value());

   // finish dealing until a player is due to act
   while(env.active_player(wstate) == nor::Player::chance) {
      auto next_deck = env.chance_actions(wstate);
      if(next_deck.empty()) {
         break;
      }
      env.transition(wstate, next_deck.front());
   }
   ASSERT_EQ(env.active_player(wstate), nor::Player::alex);

   // a betting action is fully public and carries no private information
   auto action_pre = wstate;
   auto legal = env.actions(nor::Player::alex, wstate);
   ASSERT_FALSE(legal.empty());
   const auto& action = legal[0];
   env.transition(wstate, action);
   auto action_pub = env.public_observation(action_pre, action, wstate);
   ASSERT_TRUE(action_pub.action.has_value());
   EXPECT_EQ(*action_pub.action, action);
   auto action_priv = env.private_observation(nor::Player::bob, action_pre, action, wstate);
   EXPECT_EQ(action_priv, Observation{});
}

// ##################################################################################################################
// Parametrized terminality / payoff scenarios
// ##################################################################################################################

TEST_P(TexHoldemTerminalParamsF, scripted_terminality)
{
   const auto& [cfg, steps, expected_terminal] = GetParam();
   State scripted{cfg};
   apply_steps(scripted, steps);
   EXPECT_EQ(scripted.is_terminal(), expected_terminal);
}

INSTANTIATE_TEST_SUITE_P(
   texas_holdem_terminality,
   TexHoldemTerminalParamsF,
   ::testing::Values(
      // everyone folds preflop
      std::tuple{
         default_config(),
         [&] {
            auto steps = heads_up_holes(
               C(Rank::ace, Suit::hearts),
               C(Rank::king, Suit::spades),
               C(Rank::ace, Suit::diamonds),
               C(Rank::king, Suit::diamonds)
            );
            steps.emplace_back(HoldemStep{Action{ActionType::fold}});
            return steps;
         }(),
         true},
      // flop just dealt: not terminal yet
      std::tuple{
         default_config(),
         [&] {
            auto steps = heads_up_holes(
               C(Rank::ace, Suit::hearts),
               C(Rank::king, Suit::spades),
               C(Rank::ace, Suit::diamonds),
               C(Rank::king, Suit::diamonds)
            );
            steps.emplace_back(HoldemStep{Action{ActionType::call}});
            steps.emplace_back(HoldemStep{Action{ActionType::check}});
            for(auto card :
                {C(Rank::two, Suit::clubs),
                 C(Rank::seven, Suit::hearts),
                 C(Rank::nine, Suit::spades)}) {
               steps.emplace_back(HoldemStep{card});
            }
            return steps;
         }(),
         false},
      // river dealt but river betting still pending
      std::tuple{
         default_config(),
         [&] {
            auto steps = heads_up_holes(
               C(Rank::ace, Suit::hearts),
               C(Rank::king, Suit::spades),
               C(Rank::ace, Suit::diamonds),
               C(Rank::king, Suit::diamonds)
            );
            steps.emplace_back(HoldemStep{Action{ActionType::call}});
            steps.emplace_back(HoldemStep{Action{ActionType::check}});
            for(auto card :
                {C(Rank::two, Suit::clubs),
                 C(Rank::seven, Suit::hearts),
                 C(Rank::nine, Suit::spades)}) {
               steps.emplace_back(HoldemStep{card});
            }
            steps.emplace_back(HoldemStep{Action{ActionType::check}});
            steps.emplace_back(HoldemStep{Action{ActionType::check}});
            steps.emplace_back(HoldemStep{C(Rank::three, Suit::diamonds)});
            steps.emplace_back(HoldemStep{Action{ActionType::check}});
            steps.emplace_back(HoldemStep{Action{ActionType::check}});
            steps.emplace_back(HoldemStep{C(Rank::four, Suit::clubs)});
            return steps;
         }(),
         false},
      // complete check-down ends in a showdown
      std::tuple{
         default_config(),
         [&] {
            auto steps = heads_up_holes(
               C(Rank::ace, Suit::hearts),
               C(Rank::king, Suit::spades),
               C(Rank::ace, Suit::diamonds),
               C(Rank::king, Suit::diamonds)
            );
            steps.emplace_back(HoldemStep{Action{ActionType::call}});
            steps.emplace_back(HoldemStep{Action{ActionType::check}});
            for(auto card :
                {C(Rank::two, Suit::clubs),
                 C(Rank::seven, Suit::hearts),
                 C(Rank::nine, Suit::spades)}) {
               steps.emplace_back(HoldemStep{card});
            }
            for(auto card : {C(Rank::three, Suit::diamonds), C(Rank::four, Suit::clubs)}) {
               steps.emplace_back(HoldemStep{Action{ActionType::check}});
               steps.emplace_back(HoldemStep{Action{ActionType::check}});
               steps.emplace_back(HoldemStep{card});
            }
            steps.emplace_back(HoldemStep{Action{ActionType::check}});
            steps.emplace_back(HoldemStep{Action{ActionType::check}});
            return steps;
         }(),
         true},
      // triple all-in runout ends in a showdown
      std::tuple{
         three_player_config(),
         [&] {
            std::vector< HoldemStep > steps{
               HoldemStep{C(Rank::king, Suit::spades)},
               HoldemStep{C(Rank::queen, Suit::spades)},
               HoldemStep{C(Rank::ace, Suit::spades)},
               HoldemStep{C(Rank::king, Suit::diamonds)},
               HoldemStep{C(Rank::jack, Suit::diamonds)},
               HoldemStep{C(Rank::ace, Suit::hearts)},
               HoldemStep{Action{ActionType::raise, 10.}},
               HoldemStep{Action{ActionType::all_in}},
               HoldemStep{Action{ActionType::call}}};
            for(auto card :
                {C(Rank::two, Suit::clubs),
                 C(Rank::seven, Suit::hearts),
                 C(Rank::nine, Suit::spades),
                 C(Rank::three, Suit::diamonds),
                 C(Rank::four, Suit::clubs)}) {
               steps.emplace_back(HoldemStep{card});
            }
            return steps;
         }(),
         true}
   )
);

TEST_P(TexHoldemPayoffParamsF, scripted_payoffs)
{
   const auto& [cfg, steps, expected_payoffs] = GetParam();
   State scripted{cfg};
   apply_steps(scripted, steps);
   ASSERT_TRUE(scripted.is_terminal());
   auto payoffs = scripted.payoffs();
   for(size_t p = 0; p < expected_payoffs.size(); ++p) {
      EXPECT_NEAR(payoffs[p], expected_payoffs[p], 1e-9) << "player " << p;
   }
}

INSTANTIATE_TEST_SUITE_P(
   texas_holdem_payoffs,
   TexHoldemPayoffParamsF,
   ::testing::Values(
      // pair of aces wins the checked-down pot
      std::tuple{
         default_config(),
         [&] {
            auto steps = heads_up_holes(
               C(Rank::ace, Suit::hearts),
               C(Rank::king, Suit::spades),
               C(Rank::ace, Suit::diamonds),
               C(Rank::king, Suit::diamonds)
            );
            steps.emplace_back(HoldemStep{Action{ActionType::call}});
            steps.emplace_back(HoldemStep{Action{ActionType::check}});
            for(auto card :
                {C(Rank::two, Suit::clubs),
                 C(Rank::seven, Suit::hearts),
                 C(Rank::nine, Suit::spades)}) {
               steps.emplace_back(HoldemStep{card});
            }
            for(auto card : {C(Rank::three, Suit::diamonds), C(Rank::four, Suit::clubs)}) {
               steps.emplace_back(HoldemStep{Action{ActionType::check}});
               steps.emplace_back(HoldemStep{Action{ActionType::check}});
               steps.emplace_back(HoldemStep{card});
            }
            steps.emplace_back(HoldemStep{Action{ActionType::check}});
            steps.emplace_back(HoldemStep{Action{ActionType::check}});
            return steps;
         }(),
         std::vector< double >{2., -2.}},
      // preflop fold: the big blind takes the small blind
      std::tuple{
         default_config(),
         [&] {
            auto steps = heads_up_holes(
               C(Rank::ace, Suit::hearts),
               C(Rank::king, Suit::spades),
               C(Rank::ace, Suit::diamonds),
               C(Rank::king, Suit::diamonds)
            );
            steps.emplace_back(HoldemStep{Action{ActionType::fold}});
            return steps;
         }(),
         std::vector< double >{-1., 1.}},
      // board plays itself: exact split of a symmetric pot
      std::tuple{
         default_config(),
         [&] {
            auto steps = heads_up_holes(
               C(Rank::two, Suit::diamonds),
               C(Rank::four, Suit::hearts),
               C(Rank::three, Suit::diamonds),
               C(Rank::five, Suit::hearts)
            );
            steps.emplace_back(HoldemStep{Action{ActionType::call}});
            steps.emplace_back(HoldemStep{Action{ActionType::check}});
            for(auto card :
                {C(Rank::ace, Suit::clubs),
                 C(Rank::king, Suit::clubs),
                 C(Rank::queen, Suit::clubs)}) {
               steps.emplace_back(HoldemStep{card});
            }
            for(auto card : {C(Rank::jack, Suit::clubs), C(Rank::ten, Suit::clubs)}) {
               steps.emplace_back(HoldemStep{Action{ActionType::check}});
               steps.emplace_back(HoldemStep{Action{ActionType::check}});
               steps.emplace_back(HoldemStep{card});
            }
            steps.emplace_back(HoldemStep{Action{ActionType::check}});
            steps.emplace_back(HoldemStep{Action{ActionType::check}});
            return steps;
         }(),
         std::vector< double >{0., 0.}},
      // three-way all-in with side pots: {+20, +5, -25}
      std::tuple{
         three_player_config(),
         [&] {
            std::vector< HoldemStep > steps{
               HoldemStep{C(Rank::king, Suit::spades)},
               HoldemStep{C(Rank::queen, Suit::spades)},
               HoldemStep{C(Rank::ace, Suit::spades)},
               HoldemStep{C(Rank::king, Suit::diamonds)},
               HoldemStep{C(Rank::jack, Suit::diamonds)},
               HoldemStep{C(Rank::ace, Suit::hearts)},
               HoldemStep{Action{ActionType::raise, 10.}},
               HoldemStep{Action{ActionType::all_in}},
               HoldemStep{Action{ActionType::call}}};
            for(auto card :
                {C(Rank::two, Suit::clubs),
                 C(Rank::seven, Suit::hearts),
                 C(Rank::nine, Suit::spades),
                 C(Rank::three, Suit::diamonds),
                 C(Rank::four, Suit::clubs)}) {
               steps.emplace_back(HoldemStep{card});
            }
            return steps;
         }(),
         std::vector< double >{20., 5., -25.}}
   )
);
