

#include <gtest/gtest.h>

#include <random>

#include "fixtures.hpp"
#include "kuhn_poker/kuhn_poker.hpp"
#include "nor/env/kuhn.hpp"
#include "nor/exploitability.hpp"
#include "nor/nor.hpp"
#include "nor/utils/player_informed_type.hpp"
#include "testing_utils.hpp"

using namespace kuhn;

// the configured 3-player environment must satisfy the stochastic fosg framework contract
static_assert(nor::concepts::stochastic_fosg< nor::games::kuhn::Environment >);

////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////// Chance dealing ////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(KuhnThreePlayerDealParamsF, deal_probability_uniform_over_ordered_deals)
{
   auto [deal, expected_prob] = GetParam();
   kuhn::State state{{Card::jack, Card::queen, Card::king}, 3};

   double accumulated = 1.;
   for(auto&& [seat, card] : ranges::views::enumerate(deal)) {
      auto outcomes = state.chance_actions();
      double stage_sum = 0.;
      for(auto outcome : outcomes) {
         stage_sum += state.chance_probability(outcome);
      }
      // each dealing stage distributes probability uniformly over the remaining cards
      EXPECT_NEAR(stage_sum, 1., 1e-12);
      auto chosen = ChanceOutcome{static_cast< Player >(seat), card};
      ASSERT_TRUE(state.is_valid(chosen));
      accumulated *= state.chance_probability(chosen);
      state.apply_action(chosen);
   }
   EXPECT_NEAR(accumulated, expected_prob, 1e-12);

   // exhaustion: after all seats received a card there is nothing left to deal
   EXPECT_TRUE(state.chance_actions().empty());
   EXPECT_NEAR(state.chance_probability(ChanceOutcome{Player::one, Card::jack}), 0., 1e-12);
   EXPECT_EQ(state.active_player(), Player::one);
}

INSTANTIATE_TEST_SUITE_P(
   deal_probability_tests,
   KuhnThreePlayerDealParamsF,
   ::testing::Values(
      std::tuple{std::array{Card::jack, Card::queen, Card::king}, 1. / 6.},
      std::tuple{std::array{Card::jack, Card::king, Card::queen}, 1. / 6.},
      std::tuple{std::array{Card::queen, Card::jack, Card::king}, 1. / 6.},
      std::tuple{std::array{Card::queen, Card::king, Card::jack}, 1. / 6.},
      std::tuple{std::array{Card::king, Card::jack, Card::queen}, 1. / 6.},
      std::tuple{std::array{Card::king, Card::queen, Card::jack}, 1. / 6.}
   )
);

TEST_F(KuhnThreePlayerState, chance_deal_order_and_exhaustion)
{
   EXPECT_TRUE(state.history().empty());
   EXPECT_TRUE(cmp_equal_rngs(
      state.chance_actions(),
      std::vector{
         ChanceOutcome{Player::one, Card::jack},
         ChanceOutcome{Player::one, Card::queen},
         ChanceOutcome{Player::one, Card::king}}
   ));
   EXPECT_NEAR(state.chance_probability(ChanceOutcome{Player::one, Card::jack}), 1. / 3., 1e-12);

   state.apply_action(ChanceOutcome{Player::one, Card::jack});

   EXPECT_TRUE(cmp_equal_rngs(
      state.chance_actions(),
      std::vector{ChanceOutcome{Player::two, Card::queen}, ChanceOutcome{Player::two, Card::king}}
   ));
   EXPECT_NEAR(state.chance_probability(ChanceOutcome{Player::two, Card::queen}), 1. / 2., 1e-12);

   state.apply_action(ChanceOutcome{Player::two, Card::queen});

   EXPECT_TRUE(
      cmp_equal_rngs(state.chance_actions(), std::vector{ChanceOutcome{Player::three, Card::king}})
   );
   EXPECT_NEAR(state.chance_probability(ChanceOutcome{Player::three, Card::king}), 1., 1e-12);

   state.apply_action(ChanceOutcome{Player::three, Card::king});

   EXPECT_TRUE(state.chance_actions().empty());
   EXPECT_EQ(state.active_player(), Player::one);
   EXPECT_FALSE(state.is_terminal());

   // oversized decks deal with probability 1/(remaining cards) per seat in order
   kuhn::State big_state{{Card::jack, Card::queen, Card::king, Card::ace, Card::ten}, 3};
   EXPECT_NEAR(big_state.chance_probability(ChanceOutcome{Player::one, Card::ace}), 1. / 5., 1e-12);
}

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// Betting legality ///////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

TEST_F(KuhnThreePlayerState, betting_legality_matrix)
{
   // nothing is legal before all cards are dealt
   EXPECT_FALSE(state.is_valid(Action::check));
   EXPECT_FALSE(state.is_valid(Action::bet));
   EXPECT_TRUE(state.actions().empty());
   EXPECT_FALSE(state.is_terminal());

   state.apply_action(ChanceOutcome{Player::one, Card::jack});
   state.apply_action(ChanceOutcome{Player::two, Card::queen});
   state.apply_action(ChanceOutcome{Player::three, Card::king});

   // after the deal every betting action is legal while the hand has not closed
   EXPECT_TRUE(state.is_valid(Action::check));
   EXPECT_TRUE(state.is_valid(Action::bet));
   EXPECT_TRUE(cmp_equal_rngs(state.actions(), std::vector{Action::check, Action::bet}));
   EXPECT_EQ(state.active_player(), Player::one);

   // cyclic action order one -> two -> three -> one ...
   state.apply_action(Action::check);
   EXPECT_EQ(state.active_player(), Player::two);
   EXPECT_FALSE(state.is_terminal());
   state.apply_action(Action::check);
   EXPECT_EQ(state.active_player(), Player::three);
   EXPECT_FALSE(state.is_terminal());

   // three opens; both one and two still owe a response to the bet (cyclically one first)
   state.apply_action(Action::bet);
   EXPECT_EQ(state.active_player(), Player::one);
   EXPECT_FALSE(state.is_terminal());

   // one folds ('check' against an outstanding bet)
   state.apply_action(Action::check);
   EXPECT_TRUE(state.folded(Player::one));
   EXPECT_FALSE(state.is_terminal());  // two still owes a response
   EXPECT_EQ(state.active_player(), Player::two);

   // two calls ('bet' against an outstanding bet): everyone matched -> closure
   state.apply_action(Action::bet);
   EXPECT_TRUE(state.is_terminal());

   const auto& actors = state.history_actors();
   ASSERT_EQ(actors.size(), state.history().size());
   const std::vector expected_actors{
      Player::one, Player::two, Player::three, Player::one, Player::two};
   EXPECT_TRUE(cmp_equal_rngs(actors, expected_actors));
}

TEST_F(KuhnThreePlayerState, fold_out_short_circuit)
{
   auto state = make_kuhn_three_player_state(Card::king, Card::jack, Card::queen);

   state.apply_action(Action::bet);  // one opens
   EXPECT_FALSE(state.is_terminal());
   state.apply_action(Action::check);  // two folds
   EXPECT_FALSE(state.is_terminal());
   state.apply_action(Action::check);  // three folds

   // only one player remains -> immediate pot award without further betting
   EXPECT_TRUE(state.is_terminal());
   EXPECT_TRUE(state.folded(Player::two));
   EXPECT_TRUE(state.folded(Player::three));
   EXPECT_FALSE(state.folded(Player::one));
   EXPECT_EQ(state.payoff(Player::one), 2);
   EXPECT_EQ(state.payoff(Player::two), -1);
   EXPECT_EQ(state.payoff(Player::three), -1);
}

////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////// Pot awarding //////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(KuhnThreePlayerPayoffParamsF, payoff_combinations)
{
   auto [cards, actions, expected_payoffs] = GetParam();
   auto state = make_kuhn_three_player_state(cards[0], cards[1], cards[2]);
   for(auto action : actions) {
      state.apply_action(action);
   }
   ASSERT_TRUE(state.is_terminal());
   int sum = 0;
   for(size_t seat = 0; seat < 3; seat++) {
      EXPECT_EQ(state.payoff(Player(seat)), expected_payoffs[seat]);
      sum += expected_payoffs[seat];
   }
   EXPECT_EQ(sum, 0);
}

INSTANTIATE_TEST_SUITE_P(
   payoff_combination_tests,
   KuhnThreePlayerPayoffParamsF,
   ::testing::Values(
      // everyone passes: showdown for the ante pot of 3
      std::tuple{
         std::array{Card::jack, Card::queen, Card::king},
         std::vector{Action::check, Action::check, Action::check},
         std::array{-1, -1, 2}},
      // everyone opens/calls: pot of 6 goes to the king at showdown
      std::tuple{
         std::array{Card::jack, Card::queen, Card::king},
         std::vector{Action::bet, Action::bet, Action::bet},
         std::array{-2, -2, 4}},
      // opener wins both folds immediately: antes + single bet = 4
      std::tuple{
         std::array{Card::king, Card::jack, Card::queen},
         std::vector{Action::bet, Action::check, Action::check},
         std::array{2, -1, -1}},
      // check-raise style line: three bets after two passes, one folds, two calls:
      // showdown between two and three over a pot of 5
      std::tuple{
         std::array{Card::queen, Card::king, Card::jack},
         std::vector{Action::check, Action::bet, Action::bet, Action::check},
         std::array{-1, 3, -2}},
      // everyone passes but the ace (here king) holder is not involved in a bet: pure ante split
      std::tuple{
         std::array{Card::queen, Card::king, Card::jack},
         std::vector{Action::check, Action::check, Action::check},
         std::array{-1, 2, -1}}
   )
);

TEST_F(KuhnThreePlayerState, defensive_split_on_tied_showdown)
{
   // duplicated ranks make a true tie possible; the pot must split evenly with any remainder
   // chip going to the tied players in seat order
   kuhn::State tie_state{{Card::king, Card::king, Card::queen}, 3};
   tie_state.apply_action(ChanceOutcome{Player::one, Card::king});
   tie_state.apply_action(ChanceOutcome{Player::two, Card::king});
   tie_state.apply_action(ChanceOutcome{Player::three, Card::queen});
   for(size_t seat = 0; seat < 3; ++seat) {
      tie_state.apply_action(Action::check);
   }
   ASSERT_TRUE(tie_state.is_terminal());
   // pot = 3, two winners: base share 1 each plus remainder chip for the lowest tied seat
   EXPECT_EQ(tie_state.payoff(Player::one), 1);
   EXPECT_EQ(tie_state.payoff(Player::two), 0);
   EXPECT_EQ(tie_state.payoff(Player::three), -1);

   kuhn::State tied_bets{{Card::king, Card::king, Card::queen}, 3};
   tied_bets.apply_action(ChanceOutcome{Player::one, Card::king});
   tied_bets.apply_action(ChanceOutcome{Player::two, Card::king});
   tied_bets.apply_action(ChanceOutcome{Player::three, Card::queen});
   for(size_t seat = 0; seat < 3; ++seat) {
      tied_bets.apply_action(Action::bet);
   }
   ASSERT_TRUE(tied_bets.is_terminal());
   // pot = 6, two winners splitting evenly: +1 net each, caller loses double ante
   EXPECT_EQ(tied_bets.payoff(Player::one), 1);
   EXPECT_EQ(tied_bets.payoff(Player::two), 1);
   EXPECT_EQ(tied_bets.payoff(Player::three), -2);
}

////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// Zero-sum random playouts //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

TEST_F(KuhnThreePlayerState, random_playouts_preserve_zero_sum)
{
   std::mt19937 rng{42};
   auto draw = [&rng](size_t n) { return std::uniform_int_distribution< size_t >(0, n - 1)(rng); };

   for(int trial = 0; trial < 200; ++trial) {
      // deal a uniformly random ordered deal
      auto state = [&] {
         kuhn::State fresh{{Card::jack, Card::queen, Card::king}, 3};
         while(not fresh.chance_actions().empty()) {
            auto outcomes = fresh.chance_actions();
            fresh.apply_action(outcomes[draw(outcomes.size())]);
         }
         return fresh;
      }();

      int guard = 0;
      while(not state.is_terminal()) {
         ASSERT_LT(guard++, 16);
         if(state.active_player() == Player::chance) {
            auto outcomes = state.chance_actions();
            state.apply_action(outcomes[draw(outcomes.size())]);
         } else {
            auto legal = state.actions();
            ASSERT_FALSE(legal.empty());
            state.apply_action(legal[draw(legal.size())]);
         }
      }
      int reward_sum = 0;
      for(size_t seat = 0; seat < 3; ++seat) {
         reward_sum += state.payoff(Player(seat));
      }
      EXPECT_EQ(reward_sum, 0) << "trial " << trial;
   }
}

////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////// Environment adapter: roster & history attribution ////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

TEST_F(KuhnThreePlayerState, env_roster_includes_chance_and_all_seats)
{
   using namespace nor::games::kuhn;
   Environment env{};
   auto roster = env.players(state);
   ASSERT_EQ(roster.size(), 4);
   EXPECT_EQ(roster[0], nor::Player::chance);
   EXPECT_EQ(roster[1], nor::Player::alex);
   EXPECT_EQ(roster[2], nor::Player::bob);
   EXPECT_EQ(roster[3], nor::Player::cedric);
   static_assert(Environment::player_count() == std::dynamic_extent);
   static_assert(Environment::max_player_count() == State::max_player_count);
}

TEST_F(KuhnThreePlayerState, env_history_attributes_entries_to_acting_seats)
{
   using namespace nor::games::kuhn;
   Environment env{};

   state.apply_action(ChanceOutcome{Player::one, Card::jack});
   state.apply_action(ChanceOutcome{Player::two, Card::queen});
   state.apply_action(ChanceOutcome{Player::three, Card::king});

   state.apply_action(Action::check);  // seat one
   state.apply_action(Action::bet);  // seat two opens
   state.apply_action(Action::bet);  // seat three calls
   state.apply_action(Action::bet);  // seat one calls

   ASSERT_TRUE(state.is_terminal());
   const std::vector expected_actor_seats{Player::one, Player::two, Player::three, Player::one};

   auto open_hist = env.open_history(state);
   ASSERT_EQ(open_hist.size(), 4);
   for(auto&& [i, entry] : ranges::views::enumerate(open_hist)) {
      EXPECT_EQ(entry.player(), to_nor_player(expected_actor_seats[i]));
   }

   // private history hides only the owner's own betting actions, but tags every entry with its
   // true actor (index-based attribution would mislabel the closing call of seat one)
   auto private_hist = env.private_history(nor::Player::alex, state);
   ASSERT_EQ(private_hist.size(), 7);  // 3 card entries + 4 actions
   EXPECT_EQ(private_hist[0].player(), nor::Player::chance);
   EXPECT_TRUE(private_hist[0].value().has_value());  // alex sees his own card
   EXPECT_EQ(private_hist[1].player(), nor::Player::chance);
   EXPECT_FALSE(private_hist[1].value().has_value());  // bob's card hidden
   EXPECT_FALSE(private_hist[2].value().has_value());  // cedric's card hidden
   EXPECT_EQ(private_hist[3].value(), std::nullopt);  // own check hidden
   EXPECT_EQ(private_hist[4].player(), nor::Player::bob);
   EXPECT_EQ(private_hist[5].player(), nor::Player::cedric);
   EXPECT_EQ(private_hist[6].player(), nor::Player::alex);

   // public history keeps every card hidden and every action visible with correct attribution
   auto public_hist = env.public_history(state);
   ASSERT_EQ(public_hist.size(), 7);
   for(auto&& [i, entry] : ranges::views::enumerate(public_hist)) {
      if(i < 3) {
         EXPECT_EQ(entry.player(), nor::Player::chance);
         EXPECT_FALSE(entry.value().has_value());
      } else {
         EXPECT_EQ(entry.player(), to_nor_player(expected_actor_seats[i - 3]));
         EXPECT_TRUE(entry.value().has_value());
      }
   }
}

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// Convergence smoke tests /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

template < typename Solver, typename Env >
double exploitability_of_average_policies(Solver& solver, Env& env)
{
   using namespace nor::games::kuhn;
   const auto& avg_policies = solver.average_policy();
   return nor::exploitability(
      env,
      kuhn::State{std::vector< Card >{Card::jack, Card::queen, Card::king}, 3},
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

TEST(KuhnThreePlayerCFR, vanilla_alternating_convergence_smoke)
{
   using namespace nor;
   using namespace nor::games::kuhn;

   Environment env{};
   auto root_state = std::make_unique< kuhn::State >(
      std::vector< Card >{Card::jack, Card::queen, Card::king}, 3
   );

   auto curr_policy = factory::make_tabular_policy(
      std::unordered_map< Infostate, HashmapActionPolicy< Action > >{}
   );
   auto avg_policy = factory::make_tabular_policy(
      std::unordered_map< Infostate, HashmapActionPolicy< Action > >{}
   );

   auto solver = factory::
      make_cfr< rm::CFRDiscountedConfig{.update_mode = rm::UpdateMode::alternating}, true >(
         env, std::move(root_state), curr_policy, avg_policy
      );

   constexpr size_t n_iterations = 300;
   std::vector< double > expl_trace{};
   for(size_t iter = 1; iter <= n_iterations; ++iter) {
      solver.iterate(1);
      if(iter % 50 == 0) {
         expl_trace.emplace_back(exploitability_of_average_policies(solver, env));
      }
   }

   ASSERT_EQ(expl_trace.size(), 6);
   const double expl_start = expl_trace.front();
   const double expl_end = expl_trace.back();
   std::cout << "Kuhn3P VanillaCFR alternating exploitability trace (every 50 iters):\n";
   for(size_t i = 0; i < expl_trace.size(); ++i) {
      std::cout << "  iter " << (i + 1) * 50 << ": " << expl_trace[i] << "\n";
   }
   // substantial decrease against the early-training profile and an overall downward trend
   EXPECT_LT(expl_end, 0.5 * expl_start);
   EXPECT_LT(expl_end, expl_trace[expl_trace.size() - 2]);
}

TEST(KuhnThreePlayerCFR, mccfr_external_sampling_smoke)
{
   using namespace nor;
   using namespace nor::games::kuhn;

   Environment env{};
   auto root_state = std::make_unique< kuhn::State >(
      std::vector< Card >{Card::jack, Card::queen, Card::king}, 3
   );

   auto curr_policy = factory::make_tabular_policy(
      std::unordered_map< Infostate, HashmapActionPolicy< Action > >{}
   );
   auto avg_policy = factory::make_tabular_policy(
      std::unordered_map< Infostate, HashmapActionPolicy< Action > >{}
   );

   auto solver = factory::make_mccfr<
      rm::MCCFRConfig{
         .update_mode = rm::UpdateMode::alternating,
         .algorithm = rm::MCCFRAlgorithmMode::external_sampling,
         .weighting = rm::MCCFRWeightingMode::stochastic},
      true >(env, std::move(root_state), curr_policy, avg_policy, /*epsilon=*/0., 12345);

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
   std::cout << "Kuhn3P MCCFR external-sampling exploitability: iter250=" << expl_mid
             << " iter500=" << expl_end << "\n";
   // sampled trajectories are noisy; only assert that training moved the profile substantially
   EXPECT_LT(expl_end, 0.75 * expl_mid + 1e-9);
}
