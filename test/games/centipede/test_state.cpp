
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "fixtures.hpp"
#include "nor/concepts.hpp"
#include "nor/exploitability.hpp"
#include "nor/nor.hpp"

// NOTE: 'Player' stays unambiguously bound to ::centipede::Player; nor's player enum is always
// spelled 'nor::Player' in this file.

// #####################################################################################################################
// world state basics & config guards
// #####################################################################################################################

TEST_F(CentipedeState, initial_layout_follows_the_transcription)
{
   EXPECT_EQ(state.round(), 0u);
   EXPECT_EQ(state.active_player(), Player::one);
   EXPECT_FALSE(state.terminal());
   EXPECT_EQ(state.terminal_cause(), TerminalCause::none);
   EXPECT_DOUBLE_EQ(state.payoff(Player::one), 0.);
   EXPECT_DOUBLE_EQ(state.payoff(Player::two), 0.);

   auto copy = state;
   EXPECT_TRUE(copy == state);
   EXPECT_FALSE(copy != state);

   // config guards: rounds >= 1, piles m0 > m1 >= 1
   EXPECT_THROW(Config(0, 4, 1), std::invalid_argument);
   EXPECT_THROW(Config(Config::max_rounds + 1, 4, 1), std::invalid_argument);
   EXPECT_THROW(Config(4, 1, 1), std::invalid_argument);  // m0 <= m1
   EXPECT_THROW(Config(4, 4, 0), std::invalid_argument);  // m1 < 1
}

// #####################################################################################################################
// mover alternation & move legality
// #####################################################################################################################

TEST_F(CentipedeMoves, movers_alternate_and_both_moves_are_always_legal)
{
   for(size_t r = 0; r < test_config().rounds; ++r) {
      const auto expected_mover = r % 2 == 0 ? Player::one : Player::two;
      ASSERT_EQ(state.active_player(), expected_mover) << "round " << r;
      auto legal = state.actions(expected_mover);
      ASSERT_EQ(legal.size(), 2u);  // {take, push}
      EXPECT_EQ(legal[0], (Move{true}));
      EXPECT_EQ(legal[1], (Move{false}));
      EXPECT_TRUE(state.is_valid(Move{true}));
      EXPECT_TRUE(state.is_valid(Move{false}));
      // the non-acting player cannot enumerate anything
      EXPECT_TRUE(state.actions(r % 2 == 0 ? Player::two : Player::one).empty());
      state.apply_action(Move{false});
   }
   EXPECT_TRUE(state.terminal());
}

TEST_F(CentipedeMoves, terminal_states_reject_further_moves)
{
   State s{test_config()};
   s.apply_action(Move{true});
   ASSERT_TRUE(s.terminal());
   EXPECT_FALSE(s.is_valid(Move{true}));
   EXPECT_TRUE(s.actions(Player::one).empty());
   EXPECT_THROW(s.apply_action(Move{false}), std::logic_error);
}

// #####################################################################################################################
// termination: taken vs exhausted with exact Rosenthal payoffs G(N, m0, m1)
// #####################################################################################################################

TEST_F(CentipedeTermination, take_at_any_round_pays_the_doubled_piles_to_the_taker)
{
   // G(4,4,1): take at round t gives taker 2^t * 4, opponent 2^t * 1
   constexpr std::array< uint64_t, 4 > taker_share{{4, 8, 16, 32}};
   constexpr std::array< uint64_t, 4 > opp_share{{1, 2, 4, 8}};
   for(size_t t = 0; t < 4; ++t) {
      SCOPED_TRACE(::testing::Message() << "take at round " << t);
      CPScript moves(t, Move{false});
      moves.push_back(Move{true});
      auto final_state = play_script(test_config(), moves);
      ASSERT_TRUE(final_state.terminal());
      EXPECT_EQ(final_state.terminal_cause(), TerminalCause::taken);
      EXPECT_EQ(final_state.round(), t);
      const bool one_moves = t % 2 == 0;
      EXPECT_DOUBLE_EQ(
         final_state.payoff(Player::one), double(one_moves ? taker_share[t] : opp_share[t])
      );
      EXPECT_DOUBLE_EQ(
         final_state.payoff(Player::two), double(one_moves ? opp_share[t] : taker_share[t])
      );
      // coin holdings agree with payoffs and the FOSG adapter's rewards
      EXPECT_EQ(final_state.coin_holdings(Player::one), uint64_t(final_state.payoff(Player::one)));
      cp::Environment env{};
      EXPECT_DOUBLE_EQ(env.reward(nor::Player::alex, final_state), final_state.payoff(Player::one));
      EXPECT_DOUBLE_EQ(env.reward(nor::Player::bob, final_state), final_state.payoff(Player::two));
   }
}

TEST_F(CentipedeTermination, all_push_exhaustion_swaps_the_doubled_pile_owners)
{
   // G(4,4,1): N=4 pushes => last pusher is player two; he pockets 2^4 * m1 = 16 while player one
   // collects the pushed-across big pile 2^4 * m0 = 64
   auto final_state = play_script(test_config(), CPScript(4, Move{false}));
   ASSERT_TRUE(final_state.terminal());
   EXPECT_EQ(final_state.terminal_cause(), TerminalCause::exhausted);
   EXPECT_EQ(final_state.round(), 4u);
   EXPECT_DOUBLE_EQ(final_state.payoff(Player::one), 64.);
   EXPECT_DOUBLE_EQ(final_state.payoff(Player::two), 16.);
}

// #####################################################################################################################
// scripted payoff truth table across several G(N, m0, m1) instantiations
// #####################################################################################################################

TEST_F(CentipedeTruthTable, scripted_payoffs_follow_the_truth_table)
{
   struct Row {
      size_t n;
      uint32_t big;
      uint32_t small;
      std::vector< Move > moves;
      double u_one;
      double u_two;
   };
   const std::array< Row, 6 > rows{{
      // G(1,4,1): immediate take by one: (2^0*m0, 2^0*m1)
      {1, 4, 1, {Move{true}}, 4., 1.},
      // G(1,4,1): single push exhausts immediately; one pushed so two gets the big doubled pile:
      // pusher one pockets 2^1*m1=2, opponent two collects 2^1*m0=8
      {1, 4, 1, {Move{false}}, 2., 8.},
      // G(2,4,1): take at round 1 by two: two pockets 2^1*m0=8, one holds 2^1*m1=2
      {2, 4, 1, {Move{false}, Move{true}}, 2., 8.},
      // G(2,4,1): all-push exhaustion; last pusher two pockets 2^2*m1=4, one collects 2^2*m0=16
      {2, 4, 1, {Move{false}, Move{false}}, 16., 4.},
      // G(3,3,2): take at round 2 by one: 2^2*3=12 vs 2^2*2=8
      {3, 3, 2, {Move{false}, Move{false}, Move{true}}, 12., 8.},
      // G(3,5,2): all-push exhaustion; movers are one,two,one so the LAST pusher is one:
      // he pockets 2^3*m1=16 while two collects 2^3*m0=40
      {3, 5, 2, {Move{false}, Move{false}, Move{false}}, 16., 40.},
   }};
   for(const auto& row : rows) {
      SCOPED_TRACE(
         ::testing::Message() << "G(" << row.n << "," << row.big << "," << row.small << ") with "
                              << row.moves.size() << " scripted moves"
      );
      Config cfg(row.n, row.big, row.small);
      auto final_state = play_script(cfg, row.moves);
      ASSERT_TRUE(final_state.terminal()) << "script did not terminate the game";
      EXPECT_DOUBLE_EQ(final_state.payoff(Player::one), row.u_one);
      EXPECT_DOUBLE_EQ(final_state.payoff(Player::two), row.u_two);
   }
}

// #####################################################################################################################
// backward induction: taking now strictly dominates push-then-opponent-takes in every round
// (growth condition m0 > 2*m1 of the default configuration => unique SPE = immediate take)
// #####################################################################################################################

TEST_F(CentipedeDominance, take_dominates_from_every_decision_node_in_default_config)
{
   State s{test_config()};
   for(size_t r = 0; r < test_config().rounds - 1; ++r) {
      SCOPED_TRACE(::testing::Message() << "round " << r);
      const auto [take_mover, take_opp] = s.take_resolution();
      const auto [push_mover, push_opp] = s.push_then_opponent_takes_resolution();
      // the mover prefers taking over pushing and letting the opponent take next ...
      EXPECT_GT(take_mover, push_mover);
      // ... which is exactly what makes pushing incredible: the opponent will take too
      EXPECT_GT(push_opp, take_opp);
      EXPECT_FALSE(s.terminal());
      s.apply_action(Move{false});
   }
   // final decision node (round 3): taking yields 32 and beats the exhaustion payout of the last
   // pusher (2^4 * m1 = 16)
   const auto [take_mover_final, take_opp_final] = s.take_resolution();
   EXPECT_EQ(take_mover_final, 32ull);
   EXPECT_GT(take_mover_final, 16ull);
   EXPECT_LT(take_opp_final, 64.);
   s.apply_action(Move{true});
   EXPECT_TRUE(s.terminal());
   EXPECT_DOUBLE_EQ(s.payoff(Player::two), 32.);  // player two is the final-round mover
}

// #####################################################################################################################
// random playouts: always terminal, rewards consistent with the state, both payoffs positive
// #####################################################################################################################

TEST_F(CentipedeRandomPlayouts, playouts_always_terminate_with_consistent_payoffs)
{
   const std::array< Config, 4 > configs{
      Config(1, 4, 1), Config(4, 4, 1), Config(6, 10, 3), Config(9, 7, 1)};
   cp::Environment env{};
   for(const auto& config : configs) {
      SCOPED_TRACE(::testing::Message() << common::to_string(config));
      for(unsigned seed = 0; seed < 30; ++seed) {
         State s{config};
         std::mt19937 rng{987654321u + 31 * seed};
         const auto cause = random_playout(s, rng);
         ASSERT_TRUE(s.terminal());
         const auto [u_one, u_two] = s.payoffs();
         // rewards agree with the world-state payoffs
         EXPECT_DOUBLE_EQ(env.reward(nor::Player::alex, s), u_one);
         EXPECT_DOUBLE_EQ(env.reward(nor::Player::bob, s), u_two);
         if(cause == TerminalCause::taken) {
            // the mover at the terminal round pocketed the big doubled pile
            const bool one_took = (s.round() % 2 == 0);
            const auto& winner_coins = one_took ? u_one : u_two;
            const auto& loser_coins = one_took ? u_two : u_one;
            EXPECT_EQ(winner_coins, double(uint64_t(1) << s.round()) * config.pile_big);
            EXPECT_EQ(loser_coins, double(uint64_t(1) << s.round()) * config.pile_small);
         } else if(cause == TerminalCause::exhausted) {
            EXPECT_EQ(s.round(), config.rounds);
            // owner swap: the OPPOSITE player of the final pusher holds the big pile
            const bool one_last_pushed = ((config.rounds - 1) % 2 == 0);
            const auto& big_holder = one_last_pushed ? u_two : u_one;
            EXPECT_EQ(big_holder, double(uint64_t(1) << config.rounds) * config.pile_big);
         } else {
            ADD_FAILURE() << "non-terminal cause after playout";
         }
         // both payoffs strictly positive in any terminal resolution (piles are positive)
         EXPECT_GT(u_one, 0.);
         EXPECT_GT(u_two, 0.);
      }
   }
}

// #####################################################################################################################
// information correctness: perfect information, infosets coincide with public states
// #####################################################################################################################

TEST_F(CentipedeInfo, every_move_is_public_and_histories_coincide)
{
   const Config cfg(4, 4, 1);
   cp::Environment env{cfg};

   State s{cfg};
   auto pre = s;
   env.transition(s, Move{false});
   auto pub = env.public_observation(pre, Move{false}, s);
   EXPECT_EQ(pub.moved_by, nor::Player::alex);
   ASSERT_TRUE(pub.move.has_value());
   EXPECT_EQ(*pub.move, (Move{false}));
   EXPECT_EQ(pub.next_active, nor::Player::bob);
   EXPECT_FALSE(pub.terminal_cause.has_value());
   // private observations are always empty (nothing hidden)
   EXPECT_EQ(env.private_observation(nor::Player::alex, pre, Move{false}, s), cp::Observation{});
   EXPECT_EQ(env.private_observation(nor::Player::bob, pre, Move{false}, s), cp::Observation{});

   // terminal transition announces cause + coins
   auto mid = s;
   env.transition(s, Move{true});
   auto term_pub = env.public_observation(mid, Move{true}, s);
   EXPECT_EQ(term_pub.moved_by, nor::Player::bob);
   ASSERT_TRUE(term_pub.terminal_cause.has_value());
   EXPECT_EQ(*term_pub.terminal_cause, TerminalCause::taken);
   ASSERT_TRUE(term_pub.coins.has_value());
   EXPECT_EQ(term_pub.coins->first, 2ull);  // one pushed once then bob took: one gets 2^1*m1
   EXPECT_EQ(term_pub.coins->second, 8ull);  // ... and bob pockets 2^1*m0

   // streamed histories match the reconstructed ones and have no hidden entries
   auto hist_bob = env.private_history(nor::Player::bob, s);
   auto hist_pub = env.public_history(s);
   auto hist_open = env.open_history(s);
   ASSERT_EQ(hist_bob.size(), hist_pub.size());
   ASSERT_EQ(hist_bob.size(), hist_open.size());
   EXPECT_EQ(hist_bob.size(), 2u);  // push, take
   for(size_t i = 0; i < hist_bob.size(); ++i) {
      ASSERT_TRUE(hist_bob[i].value().has_value());  // nothing is ever masked
      EXPECT_EQ(hist_bob[i].value().value(), hist_open[i].value());
   }
   // attribution alternates alex -> bob
   EXPECT_EQ(hist_open[0].player(), nor::Player::alex);
   EXPECT_EQ(hist_open[1].player(), nor::Player::bob);
}

// #####################################################################################################################
// deterministic-env trait checks
// #####################################################################################################################

static_assert(nor::concepts::deterministic_fosg< cp::Environment >);

TEST_F(CentipedeTraits, deterministic_fosg_concepts)
{
   using Env = cp::Environment;
   static_assert(std::same_as< typename Env::chance_outcome_type, std::monostate >);
   static_assert(std::same_as<
                 typename Env::action_variant_type,
                 std::variant< centipede::Move, std::monostate > >);
   EXPECT_TRUE((nor::concepts::fosg< Env >) );
   EXPECT_TRUE((nor::concepts::deterministic_env< Env >) );
   EXPECT_FALSE((nor::concepts::stochastic_env< Env >) );
   EXPECT_EQ(Env::stochasticity(), nor::Stochasticity::deterministic);
   EXPECT_FALSE(nor::concepts::has::method::chance_actions< Env >);
   EXPECT_FALSE(nor::concepts::has::method::chance_probability< Env >);
}

// #####################################################################################################################
// CFR convergence smoke: vanilla alternating CFR on G(4,4,1)
//
// METRIC NOTE (general-sum handling). Like Shapley's game this is GENERAL-SUM, so we assert on
// nash_conv(..., constant_sum=false) (sum of per-player best-response improvements,
// general-sum-safe per nor/exploitability.hpp) plus per-player BR gap reporting via
// per_player_br_gaps(); exploitability()'s zero-sum normalization is NOT used. Unlike Shapley's
// game the uniform initialization is far from equilibrium here, giving a genuinely decreasing
// metric trajectory towards the unique SPE (immediate take).
// #####################################################################################################################

namespace {

struct CentipedeCFRConvergenceReport {
   double nash_conv_first_checkpoint = std::numeric_limits< double >::quiet_NaN();
   double nash_conv_final = std::numeric_limits< double >::quiet_NaN();
   std::array< double, 2 > gaps_first_checkpoint{
      std::numeric_limits< double >::quiet_NaN(),
      std::numeric_limits< double >::quiet_NaN()};
   std::array< double, 2 > gaps_final{
      std::numeric_limits< double >::quiet_NaN(),
      std::numeric_limits< double >::quiet_NaN()};
   size_t iterations = 0;
};

}  // namespace

TEST_F(CentipedeCFR, vanilla_alternating_nash_conv_decreases_towards_immediate_take)
{
   using namespace nor;
   using Env = games::centipede::Environment;

   auto config = test_config();  // G(4,4,1)
   Env env{config};
   auto root_state = std::make_unique< games::centipede::State >(config);

   auto avg_policy = factory::make_tabular_policy(
      std::unordered_map< games::centipede::Infostate, HashmapActionPolicy< centipede::Move > >{}
   );
   auto curr_policy = factory::make_tabular_policy(
      std::unordered_map< games::centipede::Infostate, HashmapActionPolicy< centipede::Move > >{}
   );

   auto
      solver = factory::make_cfr< rm::CFRConfig{.update_mode = rm::UpdateMode::alternating}, true >(
         std::move(env), std::move(root_state), curr_policy, avg_policy
      );
   Env expl_env{config};

   constexpr size_t kIterations = 300;
   constexpr size_t kFirstCheckpoint = 2;
   constexpr size_t kCheckpoint = 50;

   CentipedeCFRConvergenceReport report{};
   report.iterations = kIterations;
   for(size_t iter = 1; iter <= kIterations; ++iter) {
      solver.iterate(1);
      if(iter != kFirstCheckpoint and (iter < kFirstCheckpoint or iter % kCheckpoint != 0)) {
         continue;
      }
      const auto& avg_policies = solver.average_policy();
      using AvgTablePolicy = std::decay_t< decltype(avg_policies.at(nor::Player::alex)) >;
      auto normalized_profile = player_hashmap< AvgTablePolicy >{
         std::pair{nor::Player::alex, normalize_state_policy(avg_policies.at(nor::Player::alex))},
         std::pair{nor::Player::bob, normalize_state_policy(avg_policies.at(nor::Player::bob))}};
      // constant_sum=false: the general-sum-safe metric
      double nc = nash_conv(expl_env, games::centipede::State{config}, normalized_profile, false);
      const auto gaps = per_player_br_gaps(
         expl_env, games::centipede::State{config}, normalized_profile
      );
      fmt::print(
         "[centipede-cfr-baseline] iter={} nash_conv={:.6e} gap_alex={:.6e} gap_bob={:.6e}\n",
         iter,
         nc,
         gaps.at(0),
         gaps.at(1)
      );
      // cross-check: nash_conv equals the sum of the reported per-player gaps
      EXPECT_NEAR(nc, gaps.at(0) + gaps.at(1), 1e-9);
      if(iter == kFirstCheckpoint) {
         report.nash_conv_first_checkpoint = nc;
         report.gaps_first_checkpoint = gaps;
      }
      if(iter == kIterations) {
         report.nash_conv_final = nc;
         report.gaps_final = gaps;
      }
   }

   fmt::print(
      "[centipede-cfr-baseline] summary iterations={} nash_conv_at_iter_{}={:.6e} "
      "nash_conv_final={:.6e} gaps_final=({:.6e},{:.6e})\n",
      report.iterations,
      kFirstCheckpoint,
      report.nash_conv_first_checkpoint,
      report.nash_conv_final,
      report.gaps_final.at(0),
      report.gaps_final.at(1)
   );

   EXPECT_LT(report.nash_conv_final, report.nash_conv_first_checkpoint);
}
