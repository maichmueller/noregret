
#include <gtest/gtest.h>

#include <algorithm>
#include <unordered_set>
#include <utility>
#include <vector>

#include "fixtures.hpp"
#include "nor/concepts.hpp"
#include "nor/exploitability.hpp"
#include "nor/nor.hpp"

// NOTE: 'Player' stays unambiguously bound to ::sheriff::Player; nor's player enum is always
// spelled 'nor::Player' in this file.

// #####################################################################################################################
// config validation
// #####################################################################################################################

TEST(SheriffConfig, defaults_transcribe_the_baseline_instance)
{
   // Farina et al. 2019, Section 5.2 baseline: v=5, p=1, s=1, n_max=10, b_max=2, r=2
   const sheriff::Config config{};
   EXPECT_DOUBLE_EQ(config.v, 5.);
   EXPECT_DOUBLE_EQ(config.p, 1.);
   EXPECT_DOUBLE_EQ(config.s, 1.);
   EXPECT_EQ(config.n_max, 10u);
   EXPECT_EQ(config.b_max, 2u);
   EXPECT_EQ(config.rounds, 2u);
   EXPECT_NO_THROW(config.validate());
}

TEST(SheriffConfig, invalid_configurations_are_rejected)
{
   EXPECT_THROW(sheriff::Config(-1., 1., 1., 2, 1, 1), std::invalid_argument);  // negative v
   EXPECT_THROW(sheriff::Config(1., -0.5, 1., 2, 1, 1), std::invalid_argument);  // negative p
   EXPECT_THROW(sheriff::Config(1., 1., -1., 2, 1, 1), std::invalid_argument);  // negative s
   EXPECT_THROW(sheriff::Config(1., 1., 1., 0, 1, 1), std::invalid_argument);  // empty cargo set
   EXPECT_THROW(sheriff::Config(1., 1., 1., 2, 1, 0), std::invalid_argument);  // no rounds
   EXPECT_THROW(
      sheriff::Config(1., 1., 1., 2, 1, sheriff::Config::max_rounds + 1), std::invalid_argument
   );
}

// #####################################################################################################################
// initial layout & legality / phase gating
// #####################################################################################################################

TEST_F(SheriffLegality, initial_layout_matches_the_sequentialization)
{
   EXPECT_EQ(state.phase(), sheriff::Phase::load);
   EXPECT_EQ(state.active_player(), sheriff::Player::one);  // the smuggler acts first
   EXPECT_FALSE(state.terminal());
   EXPECT_EQ(state.cargo(), std::nullopt);
   EXPECT_EQ(state.pending_bribe(), std::nullopt);
   EXPECT_EQ(state.resolved_rounds(), 0u);
   EXPECT_EQ(state.round(), 1u);
   EXPECT_DOUBLE_EQ(state.payoff(sheriff::Player::one), 0.);
   EXPECT_DOUBLE_EQ(state.payoff(sheriff::Player::two), 0.);

   auto copy = state;
   EXPECT_TRUE(copy == state);
   EXPECT_FALSE(copy != state);
}

TEST_F(SheriffLegality, load_phase_enumerates_all_cargo_sizes_for_the_smuggler_only)
{
   auto legal = state.actions(sheriff::Player::one);
   ASSERT_EQ(legal.size(), state.config().n_max + 1);
   for(size_t n : std::views::iota(size_t{0}, state.config().n_max + 1)) {
      EXPECT_NE(
         std::find(legal.begin(), legal.end(), sheriff::Action{sheriff::Load{uint32_t(n)}}),
         legal.end()
      );
      EXPECT_TRUE(state.is_valid(sheriff::Action{sheriff::Load{uint32_t(n)}}));
   }
   // overfull cargo is illegal
   sheriff::Action overfull{sheriff::Load{uint32_t(state.config().n_max + 1)}};
   EXPECT_FALSE(state.is_valid(overfull));
   EXPECT_THROW(state.apply_action(overfull), std::invalid_argument);
   // the sheriff cannot enumerate or apply anything during the loading phase
   EXPECT_TRUE(state.actions(sheriff::Player::two).empty());
}

TEST_F(SheriffLegality, offer_and_respond_phases_gate_by_role)
{
   state.apply_action(sheriff::Action{sheriff::Load{3}});
   ASSERT_EQ(state.phase(), sheriff::Phase::offer);
   ASSERT_EQ(state.cargo(), 3u);

   // offers enumerate {0..b_max} for the smuggler only
   auto legal = state.actions(sheriff::Player::one);
   ASSERT_EQ(legal.size(), state.config().b_max + 1);
   for(const auto& action : legal) {
      EXPECT_TRUE(std::holds_alternative< sheriff::Offer >(action));
   }
   EXPECT_TRUE(state.actions(sheriff::Player::two).empty());

   // wrong action type in the offer phase is rejected
   EXPECT_FALSE(state.is_valid(sheriff::Action{sheriff::Respond{true}}));
   EXPECT_THROW(state.apply_action(sheriff::Action{sheriff::Respond{true}}), std::invalid_argument);

   state.apply_action(sheriff::Action{sheriff::Offer{2}});
   ASSERT_EQ(state.phase(), sheriff::Phase::respond);
   ASSERT_EQ(state.pending_bribe(), 2u);
   // now only the sheriff acts: exactly the two responses
   EXPECT_TRUE(state.actions(sheriff::Player::one).empty());
   auto responses = state.actions(sheriff::Player::two);
   ASSERT_EQ(responses.size(), 2u);
   EXPECT_EQ(responses[0], sheriff::Action{sheriff::Respond{true}});
   EXPECT_EQ(responses[1], sheriff::Action{sheriff::Respond{false}});
   // bribes above b_max are rejected
   sheriff::Action too_much{sheriff::Offer{uint32_t(state.config().b_max + 1)}};
   EXPECT_FALSE(state.is_valid(too_much));

   state.apply_action(sheriff::Action{sheriff::Respond{false}});
   // the baseline instance bargains over r = 2 rounds: this intermediate rejection is
   // non-consequential and loops back to the smuggler's next offer
   ASSERT_FALSE(state.terminal());
   ASSERT_EQ(state.round(), 2u);
   ASSERT_EQ(state.phase(), sheriff::Phase::offer);
   EXPECT_TRUE(state.actions(sheriff::Player::two).empty());
   state.apply_action(sheriff::Action{sheriff::Offer{0}});
   state.apply_action(sheriff::Action{sheriff::Respond{false}});
   // the FINAL rejection on a non-empty cargo triggers an inspection with seized goods
   ASSERT_TRUE(state.terminal());
   EXPECT_EQ(state.terminal_cause(), sheriff::TerminalCause::inspection_goods);
}

TEST_F(SheriffLegality, multi_round_progression_loops_back_to_the_offer)
{
   sheriff::State game{sh_test::baseline_config()};  // r=2
   game.apply_action(sheriff::Action{sheriff::Load{4}});
   for([[maybe_unused]] size_t round : {size_t{1}, size_t{2}}) {
      ASSERT_EQ(game.phase(), sheriff::Phase::offer);
      ASSERT_EQ(game.round(), round);
      game.apply_action(sheriff::Action{sheriff::Offer{1}});
      ASSERT_EQ(game.phase(), sheriff::Phase::respond);
      game.apply_action(sheriff::Action{sheriff::Respond{false}});
      if(round == 1) {
         // intermediate rounds are non-consequential: the game continues with the next offer
         EXPECT_FALSE(game.terminal());
         EXPECT_EQ(game.resolved_rounds(), 1u);
      }
   }
   EXPECT_TRUE(game.terminal());
   EXPECT_EQ(game.resolved_rounds(), 2u);
   // the FINAL bribe (round 2) decides: rejected with goods on board
   EXPECT_EQ(game.terminal_cause(), sheriff::TerminalCause::inspection_goods);
   EXPECT_EQ(game.rounds_log().at(0).bribe, 1u);
   EXPECT_EQ(game.rounds_log().at(1).bribe, 1u);
}

// #####################################################################################################################
// payoff truth table: Appendix F.1 outcomes transcribed scenario by scenario
//
// accept b_r:        smuggler n*v - b_r     | sheriff +b_r
// inspect & found:   smuggler -n*p          | sheriff +n*p
// inspect & clean:   smuggler +s            | sheriff -s
// #####################################################################################################################

TEST_F(SheriffTruthTable, every_outcome_follows_appendix_f1)
{
   const double v = 5., p = 1., s = 1.;
   const sheriff::Config config{v, p, s, /*n_max=*/10, /*b_max=*/2, /*rounds=*/2};

   // accepted final bribe of 2 with three items on board --> (3*5-2, 2) = (13, 2)
   // (exactly the equilibrium-path payoffs quoted in Appendix F.2)
   auto accepted = play_script({config, 3, {0, 2}, {true, true}});
   EXPECT_EQ(accepted.terminal_cause(), sheriff::TerminalCause::bribe_accepted);
   EXPECT_DOUBLE_EQ(accepted.payoff(sheriff::Player::one), 13.);
   EXPECT_DOUBLE_EQ(accepted.payoff(sheriff::Player::two), 2.);

   // inspection finds two items --> (-2*p, +2*p)
   auto found = play_script({config, 2, {1, 0}, {false, false}});
   EXPECT_EQ(found.terminal_cause(), sheriff::TerminalCause::inspection_goods);
   EXPECT_DOUBLE_EQ(found.payoff(sheriff::Player::one), -2.);
   EXPECT_DOUBLE_EQ(found.payoff(sheriff::Player::two), 2.);

   // inspection of an honest (empty) cargo --> (+s, -s) false alarm
   auto clean = play_script({config, 0, {2, 2}, {false, false}});
   EXPECT_EQ(clean.terminal_cause(), sheriff::TerminalCause::inspection_clean);
   EXPECT_DOUBLE_EQ(clean.payoff(sheriff::Player::one), s);
   EXPECT_DOUBLE_EQ(clean.payoff(sheriff::Player::two), -s);

   // accepting a bribe while smuggling nothing costs the smuggler the bribe itself
   auto empty_handed = play_script({config, 0, {0, 1}, {false, true}});
   EXPECT_EQ(empty_handed.terminal_cause(), sheriff::TerminalCause::bribe_accepted);
   EXPECT_DOUBLE_EQ(empty_handed.payoff(sheriff::Player::one), -1.);
   EXPECT_DOUBLE_EQ(empty_handed.payoff(sheriff::Player::two), 1.);

   // intermediate rounds never influence the outcome: identical final rounds must coincide no
   // matter what happened in round one
   auto a = play_script({config, 3, {0, 2}, {false, true}});
   auto b = play_script({config, 3, {2, 2}, {true, true}});
   EXPECT_DOUBLE_EQ(a.payoff(sheriff::Player::one), b.payoff(sheriff::Player::one));
   EXPECT_DOUBLE_EQ(a.payoff(sheriff::Player::two), b.payoff(sheriff::Player::two));
   EXPECT_DOUBLE_EQ(a.payoff(sheriff::Player::one), 13.);
}

TEST_P(SheriffPayoffParamsF, scripted_payoffs_follow_the_truth_table)
{
   const auto& [script, expected] = GetParam();
   auto final_state = play_script(script);
   ASSERT_TRUE(final_state.terminal()) << common::to_string(final_state.phase());
   EXPECT_DOUBLE_EQ(final_state.payoff(sheriff::Player::one), expected.first);
   EXPECT_DOUBLE_EQ(final_state.payoff(sheriff::Player::two), expected.second);
}

INSTANTIATE_TEST_SUITE_P(
   SheriffPayoffs,
   SheriffPayoffParamsF,
   ::testing::Values(
      // single-round instances (r=1) over all four canonical outcomes
      std::make_tuple(
         SheriffScript{{5., 1., 1., 2, 1, 1}, 2, {1}, {true}},
         std::make_pair(9., 1.)  // accept: 2*5-1 | +1
      ),
      std::make_tuple(
         SheriffScript{{5., 1., 1., 2, 1, 1}, 2, {1}, {false}},
         std::make_pair(-2., 2.)  // inspect & find
      ),
      std::make_tuple(
         SheriffScript{{5., 1., 1., 2, 1, 1}, 0, {0}, {false}},
         std::make_pair(1., -1.)  // false alarm
      ),
      std::make_tuple(
         SheriffScript{{5., 1., 1., 2, 1, 1}, 1, {0}, {true}},
         std::make_pair(5., 0.)  // free pass with zero bribe: pure smuggling win
      ),
      // scaled parameters: v=2, p=3, s=4
      std::make_tuple(
         SheriffScript{{2., 3., 4., 3, 2, 1}, 3, {2}, {false}},
         std::make_pair(-9., 9.)  // -3*3 | +3*3
      ),
      std::make_tuple(
         SheriffScript{{2., 3., 4., 2, 2, 1}, 0, {2}, {false}},
         std::make_pair(4., -4.)  // false alarm with s=4
      )
   )
);

TEST_F(SheriffTruthTable, game_is_genuinely_general_sum_not_zero_sum)
{
   const sheriff::Config config{5., 1., 1., 10, 2, 1};
   auto accepted = play_script({config, 3, {0}, {true}});
   // (15, 0): nonzero sum witnesses the general-sum character that forbids zero-sum-normalized
   // exploitability reporting for this game
   EXPECT_NE(accepted.payoffs()[0] + accepted.payoffs()[1], 0.);
   auto found = play_script({config, 3, {1}, {false}});
   // (-3, +3) IS zero-sum while acceptance is not: the sum depends on the terminal
   EXPECT_DOUBLE_EQ(found.payoffs()[0] + found.payoffs()[1], 0.);
}

// #####################################################################################################################
// exhaustive playouts over all cargo/bribe/response combinations
// #####################################################################################################################

TEST_F(SheriffRandomPlayouts, all_playouts_terminate_with_consistent_rewards)
{
   using namespace sheriff;
   const Config config{5., 1., 1., /*n_max=*/2, /*b_max=*/1, /*rounds=*/2};
   sh::Environment env{config};
   for(size_t cargo : std::views::iota(size_t{0}, config.n_max + 1)) {
      for(size_t bribe : std::views::iota(size_t{0}, config.b_max + 1)) {
         for(bool accept : {false, true}) {
            State s{config};
            env.transition(s, Action{Load{uint32_t(cargo)}});
            for([[maybe_unused]] size_t _ : std::views::iota(size_t{0}, config.rounds)) {
               env.transition(s, Action{Offer{uint32_t(bribe)}});
               env.transition(s, Action{Respond{accept}});
            }
            ASSERT_TRUE(s.terminal());
            const auto [u1, u2] = s.payoffs();
            EXPECT_DOUBLE_EQ(env.reward(nor::Player::alex, s), u1);
            EXPECT_DOUBLE_EQ(env.reward(nor::Player::bob, s), u2);
            if(accept) {
               EXPECT_EQ(s.terminal_cause(), TerminalCause::bribe_accepted);
               EXPECT_DOUBLE_EQ(u1, double(cargo) * config.v - double(bribe));
            } else {
               EXPECT_EQ(
                  s.terminal_cause(),
                  cargo > 0 ? TerminalCause::inspection_goods : TerminalCause::inspection_clean
               );
            }
         }
      }
   }
}

// #####################################################################################################################
// information correctness: hidden cargo invisible until an inspection reveal
// #####################################################################################################################

namespace {

/// runs `script` through the environment while accumulating observer's infostate/publicstate from
/// the observation streams
inline std::pair< sh::Infostate, sh::Publicstate > observed_run(
   const SheriffScript& script,
   nor::Player observer,
   bool stop_before_final_response = false
)
{
   sh::Environment env{script.config};
   sh::Infostate istate{observer};
   sh::Publicstate pubstate{};
   State s{script.config};
   auto step = [&](const sheriff::Action& action) {
      auto pre = s;
      env.transition(s, action);
      auto pub = env.public_observation(pre, action, s);
      auto priv = env.private_observation(observer, pre, action, s);
      pubstate.update(pub);
      istate.update(pub, priv);
   };
   step(sheriff::Action{sheriff::Load{script.cargo}});
   for(size_t r : std::views::iota(size_t{0}, script.config.rounds)) {
      step(sheriff::Action{sheriff::Offer{script.bribes.at(r)}});
      const bool last = r + 1 == script.config.rounds;
      if(last and stop_before_final_response) {
         break;
      }
      step(sheriff::Action{sheriff::Respond{script.accept.at(r)}});
   }
   return {std::move(istate), std::move(pubstate)};
}

}  // namespace

TEST_F(SheriffInfo, load_event_hides_size_until_an_inspection_publishes_it)
{
   using namespace sheriff;
   Environment env{sh_test::baseline_config()};

   State s{sh_test::baseline_config()};
   auto pre = s;
   env.transition(s, Action{Load{7}});
   auto pub = env.public_observation(pre, Action{Load{7}}, s);
   EXPECT_EQ(pub.kind, Observation::Kind::load);
   EXPECT_FALSE(pub.own_cargo.has_value());  // the value never reaches the public channel

   auto priv_sheriff = env.private_observation(nor::Player::bob, pre, Action{Load{7}}, s);
   EXPECT_EQ(priv_sheriff, sh::Observation{});
   auto priv_smuggler = env.private_observation(nor::Player::alex, pre, Action{Load{7}}, s);
   ASSERT_TRUE(priv_smuggler.own_cargo.has_value());
   EXPECT_EQ(*priv_smuggler.own_cargo, 7u);
}

TEST_F(SheriffInfo, sheriff_view_identical_under_divergent_hidden_cargos)
{
   // worlds A/B/C differ ONLY in the smuggler's secret cargo; the sheriff's view must be unable
   // to tell them apart before any inspection -- including after a bribe was ACCEPTED (a settled
   // bribe keeps the trunk shut forever)
   SheriffScript base{sh_test::baseline_config(), /*cargo=*/1, {1, 2}, {false, true}};
   SheriffScript other_a = base;
   other_a.cargo = 0;
   SheriffScript other_b = base;
   other_b.cargo = 10;

   // mid-game: right before the final response the sheriff cannot distinguish the worlds ...
   auto [mid_a, pub_mid_a] = observed_run(other_a, nor::Player::bob, /*stop=*/true);
   auto [mid_b, pub_mid_b] = observed_run(other_b, nor::Player::bob, /*stop=*/true);
   EXPECT_EQ(mid_a, mid_b);
   EXPECT_EQ(mid_a.hash(), mid_b.hash());
   std::unordered_set< sh::Infostate > merge;
   merge.emplace(mid_a);
   merge.emplace(mid_b);
   EXPECT_EQ(merge.size(), 1u);  // both worlds collapse into ONE of the sheriff's infosets

   // ... and even at the bribe-accepted TERMINAL the cargo stays hidden
   auto [end_a, pub_end_a] = observed_run(other_a, nor::Player::bob);
   auto [end_b, pub_end_b] = observed_run(other_b, nor::Player::bob);
   EXPECT_EQ(end_a, end_b);
   EXPECT_EQ(end_a.hash(), end_b.hash());
   EXPECT_EQ(pub_end_a, pub_end_b);

   // the smuggler himself distinguishes the worlds immediately through his own cargo echo
   auto alex_a = observed_run(base, nor::Player::alex).first;
   other_a.accept = {false, false};  // irrelevant; only the echo matters
   auto alex_b = observed_run(other_a, nor::Player::alex).first;
   EXPECT_NE(alex_a.history().front().second, alex_b.history().front().second);
}

TEST_F(SheriffInfo, inspection_reveals_the_cargo_publicly)
{
   using namespace sheriff;
   const Config config{5., 1., 1., 10, 2, 1};

   // transition manually so both observation channels can be inspected along the way
   State s{config};
   Environment env{config};
   env.transition(s, Action{Load{4}});
   env.transition(s, Action{Offer{0}});
   auto pre_inspect = s;
   env.transition(s, Action{Respond{false}});
   ASSERT_TRUE(s.terminal());

   auto pub = env.public_observation(pre_inspect, Action{Respond{false}}, s);
   EXPECT_EQ(pub.bribe_accepted, false);
   ASSERT_TRUE(pub.revealed_cargo.has_value());
   EXPECT_EQ(*pub.revealed_cargo, 4u);  // the opened trunk publishes the cargo count
   ASSERT_TRUE(pub.terminal_cause.has_value());
   EXPECT_EQ(*pub.terminal_cause, TerminalCause::inspection_goods);

   // the clean-inspection variant reveals the empty cargo as well
   State clean{config};
   env.transition(clean, Action{Load{0}});
   env.transition(clean, Action{Offer{2}});
   auto clean_pre = clean;
   env.transition(clean, Action{Respond{false}});
   auto clean_pub = env.public_observation(clean_pre, Action{Respond{false}}, clean);
   ASSERT_TRUE(clean_pub.revealed_cargo.has_value());
   EXPECT_EQ(*clean_pub.revealed_cargo, 0u);
   EXPECT_EQ(*clean_pub.terminal_cause, TerminalCause::inspection_clean);
}

TEST_F(SheriffInfo, environment_histories_respect_information_sets)
{
   using namespace sheriff;
   const Config config{5., 1., 1., 10, 2, 2};
   Environment env{config};
   auto final_state = play_script({config, /*cargo=*/6, {1, 2}, {false, true}});
   ASSERT_TRUE(final_state.terminal());

   // open history reveals everything: 1 load + 2*(offer+response)
   auto open = env.open_history(final_state);
   ASSERT_EQ(open.size(), 5u);
   const auto* open_load = std::get_if< Load >(std::get_if< Action >(&open[0].value()));
   EXPECT_NE(open_load, nullptr);
   EXPECT_EQ(open_load->items, 6u);

   // the public history hides the cargo payload but keeps every bargaining action visible
   auto public_hist = env.public_history(final_state);
   ASSERT_EQ(public_hist.size(), 5u);
   EXPECT_FALSE(public_hist[0].value().has_value());
   const auto* offer_one = std::get_if< Offer >(std::get_if< Action >(&*public_hist[1].value()));
   EXPECT_NE(offer_one, nullptr);
   EXPECT_EQ(offer_one->bribe, 1u);

   // the smuggler privately recalls his own cargo ...
   auto smuggler_hist = env.private_history(nor::Player::alex, final_state);
   ASSERT_EQ(smuggler_hist.size(), 5u);
   const auto* own_load = std::get_if< Load >(std::get_if< Action >(&*smuggler_hist[0].value()));
   EXPECT_NE(own_load, nullptr);
   EXPECT_EQ(own_load->items, 6u);
   // ... while the sheriff's private history masks it exactly like the public channel does
   auto sheriff_hist = env.private_history(nor::Player::bob, final_state);
   ASSERT_EQ(sheriff_hist.size(), 5u);
   EXPECT_FALSE(sheriff_hist[0].value().has_value());

   // pending-offer states reconstruct with the unanswered offer included (the load plus the
   // open first-round bribe; no round is resolved yet)
   State midway{config};
   midway.apply_action(Action{Load{6}});
   midway.apply_action(Action{Offer{1}});
   ASSERT_TRUE(midway.phase() == Phase::respond);
   auto open_midway = env.open_history(midway);
   ASSERT_EQ(open_midway.size(), 2u);
   EXPECT_EQ(open_midway.back().player(), nor::Player::alex);
}

// #####################################################################################################################
// deterministic-env trait checks
// #####################################################################################################################

static_assert(nor::concepts::deterministic_fosg< sh::Environment >);

TEST_F(SheriffTraits, deterministic_fosg_concepts)
{
   using Env = sh::Environment;
   static_assert(std::same_as< typename Env::chance_outcome_type, std::monostate >);
   static_assert(std::same_as<
                 typename Env::action_variant_type,
                 std::variant< sheriff::Action, std::monostate > >);
   EXPECT_TRUE((nor::concepts::fosg< Env >) );
   EXPECT_TRUE((nor::concepts::deterministic_env< Env >) );
   EXPECT_FALSE((nor::concepts::stochastic_env< Env >) );
   EXPECT_EQ(Env::stochasticity(), nor::Stochasticity::deterministic);
   EXPECT_FALSE(nor::concepts::has::method::chance_actions< Env >);
   EXPECT_FALSE(nor::concepts::has::method::chance_probability< Env >);
}

// #####################################################################################################################
// CFR convergence smoke: vanilla alternating CFR on the tiny instance
//
// METRIC NOTE (general-sum handling). The Sheriff benchmark is GENERAL-SUM, so exploitability()'s
// zero-sum normalization is NOT meaningful here. We assert on nash_conv(..., constant_sum=false),
// which per nor/exploitability.hpp is exactly the sum of per-player best-response improvements
// u_i(BR_i, pi_-i) - u_i(pi), and additionally REPORT each player's gap individually via
// per_player_br_gaps().
// #####################################################################################################################

namespace {

struct SheriffCFRConvergenceReport {
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

TEST_F(SheriffCFR, vanilla_alternating_nash_conv_decreases_below_modest_threshold)
{
   using namespace nor;
   using Env = games::sheriff::Environment;

   auto config = sh_test::tiny_config();  // v=5,p=1,s=1,n_max=2,b_max=1,r=1
   Env env{config};
   auto root_state = std::make_unique< games::sheriff::State >(config);

   auto avg_policy = factory::make_tabular_policy(
      std::unordered_map< games::sheriff::Infostate, HashmapActionPolicy< sheriff::Action > >{}
   );
   auto curr_policy = factory::make_tabular_policy(
      std::unordered_map< games::sheriff::Infostate, HashmapActionPolicy< sheriff::Action > >{}
   );

   auto
      solver = factory::make_cfr< rm::CFRConfig{.update_mode = rm::UpdateMode::alternating}, true >(
         std::move(env), std::move(root_state), curr_policy, avg_policy
      );
   Env expl_env{config};

   constexpr size_t kIterations = 500;
   constexpr size_t kFirstCheckpoint = 2;
   constexpr size_t kCheckpoint = 100;

   SheriffCFRConvergenceReport report{};
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
      double nc = nash_conv(expl_env, games::sheriff::State{config}, normalized_profile, false);
      const auto gaps = per_player_br_gaps(
         expl_env, games::sheriff::State{config}, normalized_profile
      );
      fmt::print(
         "[sheriff-cfr-baseline] iter={} nash_conv={:.6e} gap_alex={:.6e} gap_bob={:.6e}\n",
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
      "[sheriff-cfr-baseline] summary iterations={} nash_conv_at_iter_{}={:.6e} "
      "nash_conv_final={:.6e} gaps_final=({:.6e},{:.6e})\n",
      report.iterations,
      kFirstCheckpoint,
      report.nash_conv_first_checkpoint,
      report.nash_conv_final,
      report.gaps_final.at(0),
      report.gaps_final.at(1)
   );

   // (a) the profile becomes strictly less exploitable than early on ...
   EXPECT_LT(report.nash_conv_final, report.nash_conv_first_checkpoint);
   // ... and (b) it converges below a modest absolute threshold on the tiny instance
   EXPECT_LT(report.nash_conv_final, 0.25);
}
