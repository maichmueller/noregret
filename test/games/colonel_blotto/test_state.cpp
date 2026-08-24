
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

// NOTE: 'Player' stays unambiguously bound to ::colonel_blotto::Player; nor's player enum is
// always spelled 'nor::Player' in this file.

// #####################################################################################################################
// world state basics & config guards
// #####################################################################################################################

TEST_F(ColonelBlottoState, initial_layout_follows_the_transcription)
{
   EXPECT_EQ(state.field(), 0u);
   EXPECT_EQ(state.phase(), Phase::commit_p1);
   EXPECT_EQ(state.active_player(), Player::one);
   EXPECT_FALSE(state.terminal());
   EXPECT_EQ(state.terminal_cause(), TerminalCause::none);
   EXPECT_EQ(state.remaining_budget(Player::one), 3u);
   EXPECT_EQ(state.remaining_budget(Player::two), 3u);
   EXPECT_DOUBLE_EQ(state.payoff(Player::one), 0.);
   EXPECT_DOUBLE_EQ(state.payoff(Player::two), 0.);

   auto copy = state;
   EXPECT_TRUE(copy == state);
   EXPECT_FALSE(copy != state);

   // config guards: budget within [1, max_budget]
   EXPECT_THROW(BlottoConfig(0), std::invalid_argument);
   EXPECT_THROW(BlottoConfig(max_budget + 1), std::invalid_argument);
}

// #####################################################################################################################
// commitment legality & per-field phase gating
// #####################################################################################################################

TEST_F(ColonelBlottoCommitments, budget_caps_deployments_and_phases_gate_sides)
{
   auto legal_one = state.actions(Player::one);
   ASSERT_EQ(legal_one.size(), 4u);  // {0, ..., 3}
   for(uint32_t t = 0; t <= 3; ++t) {
      EXPECT_NE(std::find(legal_one.begin(), legal_one.end(), Deploy{t}), legal_one.end());
      EXPECT_TRUE(state.is_valid(Deploy{t}));
   }
   // overspending is illegal for everyone
   EXPECT_FALSE(state.is_valid(Player::one, Deploy{4}));
   EXPECT_THROW(state.apply_action(Deploy{4}), std::invalid_argument);
   // player two cannot enumerate or commit while it is player one's phase on field 0
   EXPECT_TRUE(state.actions(Player::two).empty());

   state.apply_action(Deploy{2});
   ASSERT_EQ(state.phase(), Phase::commit_p2);
   EXPECT_EQ(state.active_player(), Player::two);
   // commitments consume budget immediately, but nothing is revealed publicly before the resolve
   EXPECT_EQ(state.remaining_budget(Player::one), 1u);  // 3 - 2 spent on field 0
   EXPECT_EQ(state.remaining_budget(Player::two), 3u);
   ASSERT_TRUE(state.committed_deploy(Player::one).has_value());
   EXPECT_FALSE(state.committed_deploy(Player::two).has_value());
   // player one is gated out while player two commits field 0
   EXPECT_TRUE(state.actions(Player::one).empty());
   // player two sees exactly his full remaining budget as deployment options
   EXPECT_EQ(state.actions(Player::two).size(), 4u);

   state.apply_action(Deploy{1});
   // both committed field 0 -> advance to field 1 with player one due
   EXPECT_EQ(state.phase(), Phase::commit_p1);
   EXPECT_EQ(state.active_player(), Player::one);
   EXPECT_EQ(state.field(), 1u);
   EXPECT_EQ(state.remaining_budget(Player::two), 2u);  // 3 - 1
   EXPECT_FALSE(state.terminal());
}

// #####################################################################################################################
// fused resolve-all: terminality only after BOTH committed all N battlefields
// #####################################################################################################################

TEST_F(ColonelBlottoResolution, game_ends_only_after_the_last_field_pair_resolves)
{
   State s{test_config()};
   // commit battlefields 0 and 1 fully
   for(size_t j = 0; j < battlefield_count - 1; ++j) {
      s.apply_action(Deploy{1});
      s.apply_action(Deploy{1});
      EXPECT_FALSE(s.terminal()) << "ended early after battlefield " << j;
      EXPECT_EQ(s.field(), j + 1u);
   }
   // final pair resolves everything at once
   s.apply_action(Deploy{0});
   EXPECT_FALSE(s.terminal());  // still awaiting player two
   EXPECT_EQ(s.active_player(), Player::two);
   s.apply_action(Deploy{0});
   ASSERT_TRUE(s.terminal());
   EXPECT_EQ(s.terminal_cause(), TerminalCause::resolved);

   // terminal states reject further deployments and have empty action sets
   EXPECT_FALSE(s.is_valid(Deploy{0}));
   EXPECT_TRUE(s.actions(Player::one).empty());
   EXPECT_THROW(s.apply_action(Deploy{0}), std::logic_error);
}

// #####################################################################################################################
// resolution truth table incl. tie splitting
// #####################################################################################################################

TEST_F(ColonelBlottoTruthTable, scripted_allocations_follow_the_resolution_truth_table)
{
   struct Row {
      std::array< uint32_t, battlefield_count > alloc_one;
      std::array< uint32_t, battlefield_count > alloc_two;
      std::array< FieldOutcome, battlefield_count > outcomes;
      double u_one;
      double u_two;
   };
   const std::array< Row, 5 > rows{{
      // one sweeps every field: fracs (3/3, 0) -> centered (+0.5, -0.5)
      {{{1, 1, 1}},
       {{0, 0, 0}},
       {{FieldOutcome::one_wins, FieldOutcome::one_wins, FieldOutcome::one_wins}},
       0.5,
       -0.5},
      // two sweeps every field: mirrored
      {{{0, 0, 0}},
       {{1, 1, 1}},
       {{FieldOutcome::two_wins, FieldOutcome::two_wins, FieldOutcome::two_wins}},
       -0.5,
       0.5},
      // perfect split: one field each plus a tie => fracs (1.5/3, 1.5/3) -> centered (0, 0)
      {{{1, 1, 0}},
       {{0, 1, 1}},
       {{FieldOutcome::one_wins, FieldOutcome::split, FieldOutcome::two_wins}},
       0.,
       0.},
      // two ties (fields 0 and 2) + two wins field 1: won (1.0, 2.0)/3 -> centered (-1/6, +1/6)
      {{{2, 0, 0}},
       {{2, 1, 0}},
       {{FieldOutcome::split, FieldOutcome::two_wins, FieldOutcome::split}},
       -1. / 6.,
       1. / 6.},
      // leftover-budget play (sums < B): dominated but legal; one wins fields 0+1, two wins 2
      {{{2, 1, 0}},
       {{0, 0, 1}},
       {{FieldOutcome::one_wins, FieldOutcome::one_wins, FieldOutcome::two_wins}},
       1. / 6.,
       -1. / 6.},
   }};
   cb::Environment env{};
   for(const auto& row : rows) {
      SCOPED_TRACE(
         ::testing::Message() << "allocs (" << row.alloc_one[0] << "," << row.alloc_one[1] << ","
                              << row.alloc_one[2] << ") vs (" << row.alloc_two[0] << ","
                              << row.alloc_two[1] << "," << row.alloc_two[2] << ")"
      );
      CBScript script;
      script.config = test_config();
      for(size_t j = 0; j < battlefield_count; ++j) {
         script.fields.emplace_back(Deploy{row.alloc_one.at(j)}, Deploy{row.alloc_two.at(j)});
      }
      auto final_state = play_script(script);
      ASSERT_TRUE(final_state.terminal());
      // per-field outcomes match exactly
      for(size_t j = 0; j < battlefield_count; ++j) {
         EXPECT_EQ(final_state.field_outcomes().at(j), row.outcomes.at(j)) << "field " << j;
      }
      EXPECT_DOUBLE_EQ(final_state.payoff(Player::one), row.u_one);
      EXPECT_DOUBLE_EQ(final_state.payoff(Player::two), row.u_two);
      // constant-sum invariant: centered rewards always sum to zero (up to FP rounding)
      EXPECT_NEAR(final_state.payoff(Player::one) + final_state.payoff(Player::two), 0., 1e-12);
      // the FOSG adapter's rewards agree with the world-state payoffs
      EXPECT_DOUBLE_EQ(env.reward(nor::Player::alex, final_state), row.u_one);
      EXPECT_DOUBLE_EQ(env.reward(nor::Player::bob, final_state), row.u_two);
   }
}

// #####################################################################################################################
// tree-size guard: keep the sequentialized trees small before pinning parameters
// #####################################################################################################################

TEST_F(ColonelBlottoTreeSize, trees_stay_below_the_50k_history_budget)
{
   for(size_t budget : {size_t(2), size_t(3)}) {
      SCOPED_TRACE(::testing::Message() << "B=" << budget);
      const size_t states = count_tree_states(BlottoConfig(budget));
      fmt::print("[blotto-tree-size] B={} total_world_states={}\n", budget, states);
      EXPECT_LT(states, size_t(50000));
      // closed-form upper bound: branching never exceeds (B+1) per stage over 2N stages
      EXPECT_LE(states, size_t(std::pow(double(budget + 1), 2. * double(battlefield_count))));
   }
}

// #####################################################################################################################
// random playouts: constant-sum invariant over the whole reachable space
// #####################################################################################################################

TEST_F(ColonelBlottoRandomPlayouts, playouts_are_always_terminal_and_constant_sum)
{
   const std::array< BlottoConfig, 2 > configs{BlottoConfig(2), BlottoConfig(3)};
   cb::Environment env{};
   for(const auto& config : configs) {
      SCOPED_TRACE(::testing::Message() << common::to_string(config));
      for(unsigned seed = 0; seed < 40; ++seed) {
         State s{config};
         std::mt19937 rng{192837465u + 97 * seed};
         const auto cause = random_playout(s, rng);
         ASSERT_EQ(cause, TerminalCause::resolved);
         const auto [u_one, u_two] = s.payoffs();
         EXPECT_NEAR(u_one + u_two, 0., 1e-12);
         // rewards live in [-0.5, 0.5] under uniform v_j = 1
         EXPECT_GE(u_one, -0.5);
         EXPECT_LE(u_one, 0.5);
         // fractions won are consistent with the raw counters
         EXPECT_DOUBLE_EQ(u_one, s.won_value(Player::one) / double(battlefield_count) - 0.5);
         // budgets fully consistent: allocations never exceed B in total
         const auto [allocs_one, allocs_two] = s.allocations();
         uint32_t sum_one = 0;
         uint32_t sum_two = 0;
         for(size_t j = 0; j < battlefield_count; ++j) {
            sum_one += allocs_one.at(j);
            sum_two += allocs_two.at(j);
         }
         EXPECT_LE(sum_one, config.budget);
         EXPECT_LE(sum_two, config.budget);
         // the FOSG adapter's rewards agree with the world-state payoffs
         EXPECT_DOUBLE_EQ(env.reward(nor::Player::alex, s), u_one);
         EXPECT_DOUBLE_EQ(env.reward(nor::Player::bob, s), u_two);
      }
   }

   // statistical anchor: uniform behavioral strategies yield expected reward ~0 (each battlefield
   // is an independent uniform lottery whose value split is fair in expectation)
   {
      State s{test_config()};
      std::mt19937 rng{1234567u};
      double mean_reward = 0.;
      constexpr unsigned trials = 4000;
      for(unsigned t = 0; t < trials; ++t) {
         State trial{test_config()};
         random_playout(trial, rng);
         mean_reward += trial.payoff(Player::one) / double(trials);
      }
      fmt::print(
         "[blotto-random] mean uniform-strategy reward of player one: {:.6f}\n", mean_reward
      );
      EXPECT_NEAR(mean_reward, 0., 0.02);
   }
}

// #####################################################################################################################
// information correctness: hidden commitments invisible pre-resolve
// #####################################################################################################################

TEST_F(ColonelBlottoInfo, commit_events_hide_values_until_resolve_all_publishes_them)
{
   const BlottoConfig cfg(3);
   cb::Environment env{cfg};

   // plain commit transition: value withheld from every public channel
   State s{cfg};
   auto pre = s;
   env.transition(s, Deploy{2});
   auto pub = env.public_observation(pre, Deploy{2}, s);
   EXPECT_EQ(pub.committed_by, nor::Player::alex);
   EXPECT_EQ(pub.field, size_t(0));
   EXPECT_FALSE(pub.own_deploy.has_value());  // the amount never reaches the public channel
   EXPECT_FALSE(pub.revealed_allocations.has_value());
   EXPECT_FALSE(pub.terminal_cause.has_value());
   auto priv_bob = env.private_observation(nor::Player::bob, pre, Deploy{2}, s);
   EXPECT_EQ(priv_bob, cb::Observation{});
   auto priv_alex = env.private_observation(nor::Player::alex, pre, Deploy{2}, s);
   ASSERT_TRUE(priv_alex.own_deploy.has_value());
   EXPECT_EQ(*priv_alex.own_deploy, (Deploy{2}));

   // mid-game public event on a later battlefield carries no values either
   env.transition(s, Deploy{0});  // bob commits field 0
   auto pre_mid = s;
   env.transition(s, Deploy{1});  // alex commits field 1 (budget 3-2 allows exactly this)
   auto pub_mid = env.public_observation(pre_mid, Deploy{1}, s);
   EXPECT_EQ(pub_mid.committed_by, nor::Player::alex);
   EXPECT_EQ(pub_mid.field, size_t(1));
   EXPECT_FALSE(pub_mid.revealed_allocations.has_value());
   EXPECT_FALSE(pub_mid.outcomes.has_value());

   // resolve-all publishes both allocation vectors and every battlefield outcome
   State fin{cfg};
   const std::array< CBJointField, battlefield_count > rounds{
      CBJointField{Deploy{2}, Deploy{0}},
      CBJointField{Deploy{1}, Deploy{1}},
      CBJointField{Deploy{0}, Deploy{2}}};
   for(const auto& [a, b] : rounds) {
      fin.apply_action(a);
      fin.apply_action(b);
   }
   ASSERT_TRUE(fin.terminal());
   const auto [allocs_one, allocs_two] = fin.allocations();
   EXPECT_EQ(allocs_one, (std::array< uint32_t, 3 >{{2, 1, 0}}));
   EXPECT_EQ(allocs_two, (std::array< uint32_t, 3 >{{0, 1, 2}}));
   EXPECT_EQ(fin.field_outcomes().at(0), FieldOutcome::one_wins);
   EXPECT_EQ(fin.field_outcomes().at(1), FieldOutcome::split);
   EXPECT_EQ(fin.field_outcomes().at(2), FieldOutcome::two_wins);
   EXPECT_DOUBLE_EQ(fin.payoff(Player::one), 0.);  // (1 + 0.5)/3 - 0.5 = 0
   EXPECT_DOUBLE_EQ(fin.payoff(Player::two), 0.);
}

TEST_F(ColonelBlottoInfo, opponent_view_identical_under_divergent_hidden_commitments)
{
   // worlds A and B differ ONLY in player one's hidden first deployment; bob's streamed view must
   // be unable to tell them apart through the subsequent commitment stages ...
   const BlottoConfig cfg(3);
   cb::Environment env{cfg};

   auto bob_view_after_steps = [&](uint32_t first_deploy, size_t n_steps) {
      cb::Infostate istate{nor::Player::bob};
      cb::Publicstate pubstate{};
      State s{cfg};
      // chronological steps: alex f0 (=first_deploy), bob f0, alex f1, bob f1, alex f2, bob f2;
      // legal for BOTH worlds: bob spends (2,1,0), alex spends (first_deploy, 0, 0)
      const std::array< Deploy, 6 > steps{
         Deploy{first_deploy}, Deploy{2}, Deploy{0}, Deploy{1}, Deploy{0}, Deploy{0}};
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

   // four steps in: worlds differ only in alex's hidden first deployment
   auto bob_a_pre_reveal = bob_view_after_steps(3, 4);
   auto bob_b_pre_reveal = bob_view_after_steps(0, 4);
   EXPECT_EQ(bob_a_pre_reveal, bob_b_pre_reveal);
   EXPECT_EQ(bob_a_pre_reveal.hash(), bob_b_pre_reveal.hash());

   // ... but the resolve-all reveal separates them irreversibly
   auto bob_a_revealed = bob_view_after_steps(3, 6);
   auto bob_b_revealed = bob_view_after_steps(0, 6);
   EXPECT_NE(bob_a_revealed, bob_b_revealed);
   EXPECT_NE(bob_a_revealed.hash(), bob_b_revealed.hash());

   // the committer himself distinguishes the worlds immediately through his own deploy echo
   auto alex_view = [&](uint32_t first_deploy) {
      cb::Infostate istate{nor::Player::alex};
      cb::Publicstate pubstate{};
      State s{cfg};
      auto pre = s;
      env.transition(s, Deploy{first_deploy});
      auto pub = env.public_observation(pre, Deploy{first_deploy}, s);
      auto priv = env.private_observation(nor::Player::alex, pre, Deploy{first_deploy}, s);
      pubstate.update(pub);
      istate.update(pub, priv);
      return istate;
   };
   EXPECT_NE(alex_view(0), alex_view(3));
   EXPECT_NE(alex_view(0).hash(), alex_view(3).hash());

   // histories reconstructed from the records agree in length with the streamed views and mask
   // exactly the opponent entries
   State mid{cfg};  // both sides committed fields 0 and 1
   mid.apply_action(Deploy{1});
   mid.apply_action(Deploy{1});
   mid.apply_action(Deploy{1});
   mid.apply_action(Deploy{1});
   auto hist_bob = env.private_history(nor::Player::bob, mid);
   auto hist_pub = env.public_history(mid);
   auto hist_open = env.open_history(mid);
   ASSERT_EQ(hist_bob.size(), hist_pub.size());
   ASSERT_EQ(hist_bob.size(), hist_open.size());
   EXPECT_EQ(hist_bob.size(), 4u);  // two fields fully committed: (p1,p2) x 2
   // bob sees only his own deployments
   ASSERT_TRUE(hist_bob[1].value().has_value());
   EXPECT_TRUE(std::holds_alternative< colonel_blotto::Deploy >(hist_bob[1].value().value()));
   EXPECT_FALSE(hist_bob[0].value().has_value());  // alex's field-0 deploy masked
   EXPECT_FALSE(hist_bob[2].value().has_value());  // alex's field-1 deploy masked
   // open history reveals everything attributed to the right owners
   EXPECT_EQ(hist_open[0].player(), nor::Player::alex);
   EXPECT_EQ(hist_open[1].player(), nor::Player::bob);
}

// #####################################################################################################################
// deterministic-env trait checks
// #####################################################################################################################

static_assert(nor::concepts::deterministic_fosg< cb::Environment >);

TEST_F(ColonelBlottoTraits, deterministic_fosg_concepts)
{
   using Env = cb::Environment;
   static_assert(std::same_as< typename Env::chance_outcome_type, std::monostate >);
   static_assert(std::same_as<
                 typename Env::action_variant_type,
                 std::variant< colonel_blotto::Deploy, std::monostate > >);
   EXPECT_TRUE((nor::concepts::fosg< Env >) );
   EXPECT_TRUE((nor::concepts::deterministic_env< Env >) );
   EXPECT_FALSE((nor::concepts::stochastic_env< Env >) );
   EXPECT_EQ(Env::stochasticity(), nor::Stochasticity::deterministic);
   EXPECT_FALSE(nor::concepts::has::method::chance_actions< Env >);
   EXPECT_FALSE(nor::concepts::has::method::chance_probability< Env >);
}

// #####################################################################################################################
// CFR convergence smoke: vanilla alternating CFR on B=3, N=3, uniform v=1
//
// METRIC NOTE (constant-sum handling). Discretized Colonel Blotto is CONSTANT-SUM (centered
// rewards sum to zero at every terminal history), so the normalized metric
// exploitability(..., constant_sum=true) IS well-defined here -- unlike Shapley/centipede --
// and nash_conv coincides with it since the centered profile value sums to zero already. We
// assert exploitability decreases from the first checkpoint to the final one and record the
// trace.
// #####################################################################################################################

namespace {

struct BlottoCFRConvergenceReport {
   double expl_first_checkpoint = std::numeric_limits< double >::quiet_NaN();
   double expl_final = std::numeric_limits< double >::quiet_NaN();
   size_t iterations = 0;
};

}  // namespace

TEST_F(ColonelBlottoCFR, vanilla_alternating_exploitability_decreases_B3)
{
   using namespace nor;
   using Env = games::colonel_blotto::Environment;

   auto config = test_config();  // B=3
   Env env{config};
   auto root_state = std::make_unique< games::colonel_blotto::State >(config);

   auto avg_policy = factory::make_tabular_policy(std::unordered_map<
                                                  games::colonel_blotto::Infostate,
                                                  HashmapActionPolicy< colonel_blotto::Deploy > >{}
   );
   auto curr_policy = factory::make_tabular_policy(std::unordered_map<
                                                   games::colonel_blotto::Infostate,
                                                   HashmapActionPolicy< colonel_blotto::Deploy > >{}
   );

   auto
      solver = factory::make_cfr< rm::CFRConfig{.update_mode = rm::UpdateMode::alternating}, true >(
         std::move(env), std::move(root_state), curr_policy, avg_policy
      );
   Env expl_env{config};

   constexpr size_t kIterations = 300;
   constexpr size_t kFirstCheckpoint = 2;
   constexpr size_t kCheckpoint = 50;

   BlottoCFRConvergenceReport report{};
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
      // constant_sum=true: valid normalization for this constant-sum game
      double expl = exploitability(
         expl_env, games::colonel_blotto::State{config}, normalized_profile, true
      );
      const auto gaps = per_player_br_gaps(
         expl_env, games::colonel_blotto::State{config}, normalized_profile
      );
      fmt::print(
         "[blotto-cfr-baseline] iter={} exploitability={:.6e} gap_sum={:.6e}\n",
         iter,
         expl,
         gaps.at(0) + gaps.at(1)
      );
      // cross-check: exploitability (= nash_conv / N) equals half the summed BR gaps here since
      // the centered rewards make both metric variants coincide
      EXPECT_NEAR(expl, (gaps.at(0) + gaps.at(1)) / 2., 1e-9);
      if(iter == kFirstCheckpoint) {
         report.expl_first_checkpoint = expl;
      }
      report.expl_final = expl;
   }

   fmt::print(
      "[blotto-cfr-baseline] summary iterations={} expl_at_iter_{}={:.6e} expl_final={:.6e}\n",
      report.iterations,
      kFirstCheckpoint,
      report.expl_first_checkpoint,
      report.expl_final
   );

   EXPECT_LT(report.expl_final, report.expl_first_checkpoint);
}
