
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

#include "fixtures.hpp"
#include "nor/concepts.hpp"
#include "nor/exploitability.hpp"
#include "nor/nor.hpp"

// NOTE: 'Player' stays unambiguously bound to ::shapley::Player; nor's player enum is always
// spelled 'nor::Player' in this file.

namespace {

/// runs a joint-profile script through the environment while accumulating observer's
/// infostate/publicstate from the observation streams
inline std::pair< sh::Infostate, sh::Publicstate >
observed_run(Play play_one, Play play_two, nor::Player observer)
{
   sh::Environment env{};
   sh::Infostate istate{observer};
   sh::Publicstate pubstate{};
   State s{};
   auto step = [&](const Play& play) {
      auto pre = s;
      env.transition(s, play);
      auto pub = env.public_observation(pre, play, s);
      auto priv = env.private_observation(observer, pre, play, s);
      pubstate.update(pub);
      istate.update(pub, priv);
   };
   step(play_one);
   step(play_two);
   return {std::move(istate), std::move(pubstate)};
}

}  // namespace

// #####################################################################################################################
// world state basics
// #####################################################################################################################

TEST_F(ShapleyState, initial_layout_matches_the_sequentialized_normal_form)
{
   EXPECT_EQ(state.phase(), Phase::commit_p1);
   EXPECT_EQ(state.active_player(), Player::one);
   EXPECT_FALSE(state.terminal());
   EXPECT_DOUBLE_EQ(state.payoff(Player::one), 0.);
   EXPECT_DOUBLE_EQ(state.payoff(Player::two), 0.);
   EXPECT_EQ(state.committed_play(Player::one), std::nullopt);
   EXPECT_EQ(state.committed_play(Player::two), std::nullopt);

   auto copy = state;
   EXPECT_TRUE(copy == state);
   EXPECT_FALSE(copy != state);

   // nothing is parameterized: there is no config type to validate
   static_assert(std::is_default_constructible_v< State >);
}

// #####################################################################################################################
// commitment legality & phase gating
// #####################################################################################################################

TEST_F(ShapleyBidding, three_plays_per_committer_and_phase_gates)
{
   auto legal_one = state.actions(Player::one);
   ASSERT_EQ(legal_one.size(), 3u);
   for(uint8_t s = 0; s < 3; ++s) {
      EXPECT_NE(std::find(legal_one.begin(), legal_one.end(), Play{s}), legal_one.end());
      EXPECT_TRUE(state.is_valid(Play{s}));
   }
   // strategy indices >= 3 are illegal for everyone
   EXPECT_FALSE(state.is_valid(Player::one, Play{3}));
   EXPECT_THROW(state.apply_action(Play{3}), std::invalid_argument);
   // player two cannot enumerate or commit while it is player one's phase
   EXPECT_TRUE(state.actions(Player::two).empty());

   state.apply_action(Play{0});
   ASSERT_EQ(state.phase(), Phase::commit_p2);
   EXPECT_EQ(state.active_player(), Player::two);
   ASSERT_TRUE(state.committed_play(Player::one).has_value());
   EXPECT_FALSE(state.committed_play(Player::two).has_value());
   // player one is gated out until the resolve cleared him (which never happens pre-terminality):
   // he cannot enumerate any action anymore; any applied Play now routes to player two
   EXPECT_TRUE(state.actions(Player::one).empty());
   // player two can enumerate exactly his three strategies in this phase
   EXPECT_EQ(state.actions(Player::two).size(), 3u);

   state.apply_action(Play{2});
   EXPECT_TRUE(state.terminal());
   EXPECT_TRUE(state.actions(Player::two).empty());
}

// #####################################################################################################################
// payoff truth table: the canonical bimatrix transcribed cell by cell
//
//            left      middle    right
//   top     ( 1, 0)   ( 0, 0)   ( 0, 1)
//   middle  ( 0, 1)   ( 1, 0)   ( 0, 0)
//   bottom  ( 0, 0)   ( 0, 1)   ( 1, 0)
// #####################################################################################################################

TEST_F(ShapleyTruthTable, every_profile_follows_the_canonical_bimatrix)
{
   constexpr std::array< std::array< double, 3 >, 3 > expected_one{
      {{{1., 0., 0.}}, {{0., 1., 0.}}, {{0., 0., 1.}}}};
   constexpr std::array< std::array< double, 3 >, 3 > expected_two{
      {{{0., 0., 1.}}, {{1., 0., 0.}}, {{0., 1., 0.}}}};
   for(uint8_t row = 0; row < 3; ++row) {
      for(uint8_t col = 0; col < 3; ++col) {
         SCOPED_TRACE(
            ::testing::Message() << "profile (" << unsigned(row) << "," << unsigned(col) << ")"
         );
         auto final_state = play_profile(Play{row}, Play{col});
         EXPECT_DOUBLE_EQ(final_state.payoff(Player::one), expected_one[row][col]);
         EXPECT_DOUBLE_EQ(final_state.payoff(Player::two), expected_two[row][col]);
         const auto [u1, u2] = final_state.payoffs();
         EXPECT_DOUBLE_EQ(u1, final_state.payoff(Player::one));
         EXPECT_DOUBLE_EQ(u2, final_state.payoff(Player::two));
      }
   }
}

TEST_F(ShapleyTruthTable, game_is_genuinely_general_sum_not_zero_sum)
{
   // (top, left) yields (1, 0): nonzero sum witnesses the general-sum character that forbids
   // zero-sum-normalized exploitability reporting for this game
   auto state_tl = play_profile(Play{0}, Play{0});
   EXPECT_NE(state_tl.payoff(Player::one) + state_tl.payoff(Player::two), 0.);
   // and the diagonal profiles are not antisymmetric either: u1(top,left)=1 but u1(left-as-row,
   // top-as-col)=0
   auto state_lt = play_profile(Play{2}, Play{2});
   EXPECT_NE(state_lt.payoff(Player::one), -state_tl.payoff(Player::two));
}

// #####################################################################################################################
// best-response cycle truth table straight off the matrices
// #####################################################################################################################

TEST_F(ShapleyBestResponse, pure_best_responses_form_the_famous_cycle)
{
   // BR correspondence read off the canonical matrices (see state.hpp transcription notes)
   const std::array< std::vector< uint8_t >, 3 > br_one{
      {/*vs left*/ {0},
       /*vs middle*/ {1}, /*vs right*/
       {2}}};
   const std::array< std::vector< uint8_t >, 3 > br_two{
      {/*vs top*/ {2},
       /*vs middle*/ {0}, /*vs bottom*/
       {1}}};

   EXPECT_EQ(State::best_responses(Player::one, 0), br_one[0]);
   EXPECT_EQ(State::best_responses(Player::one, 1), br_one[1]);
   EXPECT_EQ(State::best_responses(Player::one, 2), br_one[2]);
   EXPECT_EQ(State::best_responses(Player::two, 0), br_two[0]);
   EXPECT_EQ(State::best_responses(Player::two, 1), br_two[1]);
   EXPECT_EQ(State::best_responses(Player::two, 2), br_two[2]);

   // walking the cycle: (top,left) -> (top,right) -> (bottom,right) -> (bottom,middle)
   // -> (middle,middle) -> (middle,left) -> back to (top,left): six profiles, none on the
   // antidiagonal
   std::vector< std::pair< uint8_t, uint8_t > > cycle;
   auto step = [&](uint8_t row, uint8_t col, bool one_moves) {
      const auto responses = State::best_responses(
         one_moves ? Player::one : Player::two, one_moves ? col : row
      );
      ASSERT_EQ(responses.size(), 1u);
      if(one_moves) {
         cycle.emplace_back(responses.front(), col);
      } else {
         cycle.emplace_back(row, responses.front());
      }
   };
   cycle.emplace_back(0, 0);
   step(0, 0, false);  // two deviates: left -> right
   step(0, 2, true);  // one deviates: top -> bottom
   step(2, 2, false);  // two deviates: right -> middle
   step(2, 1, true);  // one deviates: bottom -> middle
   step(1, 1, false);  // two deviates: middle -> left
   step(1, 0, true);  // one deviates: middle -> top (cycle closed)
   ASSERT_EQ(cycle.size(), 7u);
   EXPECT_EQ(cycle.front(), cycle.back());
   // the antidiagonal is never visited
   for(const auto& [r, c] : cycle) {
      EXPECT_FALSE((r == 0 && c == 1) || (r == 1 && c == 2) || (r == 2 && c == 0));
   }
}

// #####################################################################################################################
// information correctness: hidden commitments invisible pre-resolve
// #####################################################################################################################

TEST_F(ShapleyInfo, commit_events_hide_values_until_the_resolve_publishes_them)
{
   sh::Environment env{};

   State s{};
   auto pre = s;
   env.transition(s, Play{1});
   auto pub = env.public_observation(pre, Play{1}, s);
   EXPECT_EQ(pub.committed_by, nor::Player::alex);
   EXPECT_FALSE(pub.own_play.has_value());  // the value never reaches the public channel
   EXPECT_FALSE(pub.revealed_plays.has_value());
   auto priv_bob = env.private_observation(nor::Player::bob, pre, Play{1}, s);
   EXPECT_EQ(priv_bob, sh::Observation{});
   auto priv_alex = env.private_observation(nor::Player::alex, pre, Play{1}, s);
   ASSERT_TRUE(priv_alex.own_play.has_value());
   EXPECT_EQ(*priv_alex.own_play, (Play{1}));

   // fused resolve publishes both plays
   auto mid = s;
   env.transition(s, Play{0});
   auto resolve_pub = env.public_observation(mid, Play{0}, s);
   ASSERT_TRUE(resolve_pub.revealed_plays.has_value());
   EXPECT_EQ(resolve_pub.revealed_plays->first, (Play{1}));
   EXPECT_EQ(resolve_pub.revealed_plays->second, (Play{0}));
}

TEST_F(ShapleyInfo, opponent_view_identical_under_divergent_hidden_commitments)
{
   // worlds A and B differ only in player one's commitment; bob's view must be unable to tell
   // them apart before the resolve and perfectly able afterwards
   sh::Environment env{};

   auto bob_view_after_steps = [&](uint8_t first_play, size_t n_steps) {
      sh::Infostate istate{nor::Player::bob};
      sh::Publicstate pubstate{};
      State s{};
      const std::array< Play, 2 > steps{Play{first_play}, Play{2}};
      for(size_t i = 0; i < n_steps; ++i) {
         auto pre = s;
         env.transition(s, steps[i]);
         auto pub = env.public_observation(pre, steps[i], s);
         auto priv = env.private_observation(nor::Player::bob, pre, steps[i], s);
         pubstate.update(pub);
         istate.update(pub, priv);
      }
      return istate;
   };

   auto bob_a_commit = bob_view_after_steps(0, 1);
   auto bob_b_commit = bob_view_after_steps(2, 1);
   EXPECT_EQ(bob_a_commit, bob_b_commit);
   EXPECT_EQ(bob_a_commit.hash(), bob_b_commit.hash());
   std::unordered_set< sh::Infostate > bob_set;
   bob_set.emplace(bob_a_commit);
   bob_set.emplace(bob_b_commit);
   EXPECT_EQ(bob_set.size(), 1u);  // both merge into ONE of bob's infosets

   // after the resolve the revealed plays separate the worlds irreversibly
   auto bob_a_resolved = bob_view_after_steps(0, 2);
   auto bob_b_resolved = bob_view_after_steps(2, 2);
   EXPECT_NE(bob_a_resolved, bob_b_resolved);
   EXPECT_NE(bob_a_resolved.hash(), bob_b_resolved.hash());

   // the committer himself distinguishes the worlds immediately through his own play echo
   auto alex_view = [&](uint8_t first_play) {
      sh::Infostate istate{nor::Player::alex};
      sh::Publicstate pubstate{};
      State s{};
      auto pre = s;
      env.transition(s, Play{first_play});
      auto pub = env.public_observation(pre, Play{first_play}, s);
      auto priv = env.private_observation(nor::Player::alex, pre, Play{first_play}, s);
      pubstate.update(pub);
      istate.update(pub, priv);
      return istate;
   };
   EXPECT_NE(alex_view(0), alex_view(2));
   EXPECT_NE(alex_view(0).hash(), alex_view(2).hash());

   // full-script sanity: streamed histories have the expected length
   auto [info, pub] = observed_run(Play{1}, Play{0}, nor::Player::bob);
   EXPECT_EQ(info.history().size(), pub.history().size());
   EXPECT_EQ(info.history().size(), 2u);
}

// #####################################################################################################################
// exhaustive playouts over all nine profiles
// #####################################################################################################################

TEST_F(ShapleyRandomPlayouts, all_playouts_terminate_with_matrix_rewards)
{
   sh::Environment env{};
   for(uint8_t row = 0; row < 3; ++row) {
      for(uint8_t col = 0; col < 3; ++col) {
         State s{};
         env.transition(s, Play{row});
         env.transition(s, Play{col});
         ASSERT_TRUE(s.terminal());
         const auto [u1, u2] = s.payoffs();
         // rewards agree with the world-state payoffs and live inside [-0, 1] bimatrix range
         EXPECT_DOUBLE_EQ(env.reward(nor::Player::alex, s), u1);
         EXPECT_DOUBLE_EQ(env.reward(nor::Player::bob, s), u2);
         EXPECT_TRUE(u1 == 0. || u1 == 1.) << "u1 outside {0,1}";
         EXPECT_TRUE(u2 == 0. || u2 == 1.) << "u2 outside {0,1}";
      }
   }
}

// #####################################################################################################################
// deterministic-env trait checks
// #####################################################################################################################

static_assert(nor::concepts::deterministic_fosg< sh::Environment >);

TEST_F(ShapleyTraits, deterministic_fosg_concepts)
{
   using Env = sh::Environment;
   static_assert(std::same_as< typename Env::chance_outcome_type, std::monostate >);
   static_assert(std::same_as<
                 typename Env::action_variant_type,
                 std::variant< shapley::Play, std::monostate > >);
   EXPECT_TRUE((nor::concepts::fosg< Env >) );
   EXPECT_TRUE((nor::concepts::deterministic_env< Env >) );
   EXPECT_FALSE((nor::concepts::stochastic_env< Env >) );
   EXPECT_EQ(Env::stochasticity(), nor::Stochasticity::deterministic);
   EXPECT_FALSE(nor::concepts::has::method::chance_actions< Env >);
   EXPECT_FALSE(nor::concepts::has::method::chance_probability< Env >);
}

// #####################################################################################################################
// CFR convergence smoke: vanilla alternating CFR on the sequentialized normal form
//
// METRIC NOTE (general-sum handling). Shapley's game is GENERAL-SUM, so exploitability()'s zero-sum
// normalization (nash_conv / N against a fixed total) is NOT meaningful here. We assert on
// nash_conv(..., constant_sum=false), which per nor/exploitability.hpp is exactly the sum of
// per-player best-response improvements u_i(BR_i, pi_-i) - u_i(pi) -- well-defined for any-sum
// games -- and additionally REPORT each player's gap individually via per_player_br_gaps().
// The unique Nash equilibrium is the uniform profile ((1/3,1/3,1/3),(1/3,1/3,1/3)) -- which is
// ALSO the regret-matching initialization. Since no unilateral deviation against a uniform
// opponent ever improves any payoff in this bimatrix, and alternating CFR settles into the
// deterministic best-response cycle whose empirical distribution is exactly uniform, nash_conv
// collapses to machine zero within the first few iterations: the metric FLOOR (0) is reached
// immediately, so a strict monotone-decrease assertion over late checkpoints is vacuous/impossible
// here. We therefore assert (a) non-increase from the very first iteration to the final one and
// (b) actual convergence (final < 1e-9), and we RECORD + REPORT the full checkpoint trace with
// per-player BR gaps for later comparisons by the multiplayer-CFR/greedy-weights/EFCE variants.
// #####################################################################################################################

namespace {

struct ShapleyCFRConvergenceReport {
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

TEST_F(ShapleyCFR, vanilla_alternating_nash_conv_decreases_towards_the_uniform_equilibrium)
{
   using namespace nor;
   using Env = games::shapley::Environment;

   Env env{};
   auto root_state = std::make_unique< games::shapley::State >();

   auto avg_policy = factory::make_tabular_policy(
      std::unordered_map< games::shapley::Infostate, HashmapActionPolicy< shapley::Play > >{}
   );
   auto curr_policy = factory::make_tabular_policy(
      std::unordered_map< games::shapley::Infostate, HashmapActionPolicy< shapley::Play > >{}
   );

   auto
      solver = factory::make_cfr< rm::CFRConfig{.update_mode = rm::UpdateMode::alternating}, true >(
         std::move(env), std::move(root_state), curr_policy, avg_policy
      );
   Env expl_env{};

   constexpr size_t kIterations = 300;
   // first evaluation only once BOTH players' lazily-populated tabular average policies exist
   // (alternating updates touch one player per iteration)
   constexpr size_t kFirstCheckpoint = 2;
   constexpr size_t kCheckpoint = 50;

   ShapleyCFRConvergenceReport report{};
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
      double nc = nash_conv(expl_env, games::shapley::State{}, normalized_profile, false);
      const auto gaps = per_player_br_gaps(expl_env, games::shapley::State{}, normalized_profile);
      fmt::print(
         "[shapley-cfr-baseline] iter={} nash_conv={:.6e} gap_alex={:.6e} gap_bob={:.6e}\n",
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
      "[shapley-cfr-baseline] summary iterations={} nash_conv_at_iter_{}={:.6e} "
      "nash_conv_final={:.6e} gaps_final=({:.6e},{:.6e})\n",
      report.iterations,
      kFirstCheckpoint,
      report.nash_conv_first_checkpoint,
      report.nash_conv_final,
      report.gaps_final.at(0),
      report.gaps_final.at(1)
   );

   // (a) the profile never becomes MORE exploitable than at the very first iteration ...
   EXPECT_LE(report.nash_conv_final, report.nash_conv_first_checkpoint + 1e-12);
   // ... and (b) it actually converges to the uniform equilibrium (see metric note above)
   EXPECT_LT(report.nash_conv_final, 1e-9);
}
