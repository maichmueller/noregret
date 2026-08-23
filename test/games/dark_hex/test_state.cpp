
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dark_hex/dark_hex.hpp"
#include "fixtures.hpp"
#include "nor/concepts.hpp"
#include "nor/exploitability.hpp"
#include "nor/nor.hpp"

using namespace dark_hex;
using namespace nor::games::dark_hex;

namespace {

/// flattened cell index of (row, col) on an n x n board
constexpr uint8_t rc(size_t n, size_t row, size_t col)
{
   return uint8_t(row * n + col);
}

}  // namespace

// #####################################################################################################################
// connectivity / neighborhood semantics
// #####################################################################################################################

TEST_F(DarkHexState, connectivity_orients_rows_and_columns)
{
   state = State{Config{3}};
   const uint32_t col_zero = (1u << 0) | (1u << 3) | (1u << 6);  // cells (0,0),(1,0),(2,0)
   // player one bridges top <-> bottom (a fixed column) ...
   EXPECT_TRUE(state.connected(col_zero, /*top_bottom=*/true));
   // ... while player two needs a full row (left <-> right)
   EXPECT_FALSE(state.connected(col_zero, /*top_bottom=*/false));
   const uint32_t row_zero = (1u << 0) | (1u << 1) | (1u << 2);  // cells (0,*)
   EXPECT_TRUE(state.connected(row_zero, false));
   EXPECT_FALSE(state.connected(row_zero, true));
}

TEST_F(DarkHexState, connectivity_follows_hex_adjacency_incl_bridges)
{
   state = State{Config{3}};
   // zigzag chain (0,1)-(1,0)-(2,0): each consecutive pair is hex-adjacent ((+1,-1) and (+1,0))
   const uint32_t zigzag = (1u << rc(3, 0, 1)) | (1u << rc(3, 1, 0)) | (1u << rc(3, 2, 0));
   EXPECT_TRUE(state.connected(zigzag, true));

   // the diagonal neighbours (r-1,c+1)/(r+1,c-1) alone do NOT connect orthogonal directions:
   // (1,1) touches (2,0) only diagonally -- a top-bottom chain must route around
   const uint32_t diagonal_pair = (1u << rc(3, 1, 1)) | (1u << rc(3, 2, 0));
   EXPECT_FALSE(state.connected(diagonal_pair, false));  // cols 1..0 but col 2 missing anyway
   EXPECT_FALSE(state.connected(diagonal_pair | (1u << rc(3, 0, 2)), true));

   // bridge template: (1,1) and (3,0) on a 5-board are two common-neighbour steps apart; either
   // of the shared neighbours (2,0)/(2,1) completes the chain but nothing else does
   State big{Config{5}};
   const uint32_t a = 1u << rc(5, 1, 1);
   const uint32_t b = 1u << rc(5, 3, 0);
   EXPECT_FALSE(big.connected(a | b, true));
   EXPECT_TRUE(big.connected(a | b | (1u << rc(5, 2, 0)), true));
   EXPECT_TRUE(big.connected(a | b | (1u << rc(5, 2, 1)), true));
   // an unrelated stone in between the two bridge endpoints' other side stays disconnected
   EXPECT_FALSE(big.connected(a | b | (1u << rc(5, 0, 4)), true));

   // scattered stones never connect
   const uint32_t scattered =
      (1u << rc(3, 0, 0)) | (1u << rc(3, 1, 2)) | (1u << rc(3, 2, 1)) | (1u << rc(3, 0, 2));
   EXPECT_FALSE(state.connected(scattered, true));
   EXPECT_FALSE(state.connected(scattered, false));
}

TEST_F(DarkHexState, has_won_detects_winning_lines_of_both_players)
{
   state = State{Config{3}};
   // player one wins through the centre column
   state.apply_action(Move{rc(3, 0, 1)});
   state.apply_action(Move{rc(3, 0, 0)});
   state.apply_action(Move{rc(3, 1, 1)});
   state.apply_action(Move{rc(3, 2, 0)});
   state.apply_action(Move{rc(3, 2, 1)});
   EXPECT_TRUE(state.has_won(Player::one));
   EXPECT_FALSE(state.has_won(Player::two));
   EXPECT_TRUE(state.terminal());

   // player two wins through the middle row
   State s2{Config{3}};
   s2.apply_action(Move{rc(3, 0, 0)});
   s2.apply_action(Move{rc(3, 1, 0)});
   s2.apply_action(Move{rc(3, 0, 1)});
   s2.apply_action(Move{rc(3, 1, 1)});
   s2.apply_action(Move{rc(3, 2, 0)});
   s2.apply_action(Move{rc(3, 1, 2)});
   EXPECT_TRUE(s2.has_won(Player::two));
   EXPECT_FALSE(s2.has_won(Player::one));
}

// #####################################################################################################################
// rules-mode semantics: classical (turn retained) vs abrupt (turn consumed)
// #####################################################################################################################

TEST_F(DarkHexState, cdh_rejection_retains_turn_and_places_nothing)
{
   state = State{Config{3, RulesMode::cdh}};
   ASSERT_EQ(state.active_player(), Player::one);

   // a fresh placement flips the turn as usual
   state.apply_action(Move{rc(3, 0, 0)});
   EXPECT_EQ(state.active_player(), Player::two);
   EXPECT_EQ(state.move_count(), 1u);

   // player two attacks the occupied cell: told about the failure, keeps the turn, no stone
   state.apply_action(Move{rc(3, 0, 0)});
   EXPECT_TRUE(state.last_attempt_failed());
   EXPECT_EQ(state.active_player(), Player::two);  // turn RETAINED
   EXPECT_EQ(state.stones(Player::two), 0u);       // stone NOT placed
   EXPECT_EQ(state.attempts(Player::two), 1u);
   EXPECT_EQ(state.move_count(), 2u);
   EXPECT_FALSE(state.terminal());

   // the very same actor may immediately retry elsewhere and succeed
   state.apply_action(Move{rc(3, 2, 2)});
   EXPECT_FALSE(state.last_attempt_failed());
   EXPECT_EQ(state.stones(Player::two), 1u << rc(3, 2, 2));
   EXPECT_EQ(state.active_player(), Player::one);
}

TEST_F(DarkHexState, adh_rejection_consumes_the_turn)
{
   state = State{Config{3, RulesMode::adh}};
   state.apply_action(Move{rc(3, 0, 0)});
   ASSERT_EQ(state.active_player(), Player::two);

   // the failed attempt silently burns player two's turn
   state.apply_action(Move{rc(3, 0, 0)});
   EXPECT_TRUE(state.last_attempt_failed());
   EXPECT_EQ(state.active_player(), Player::one);  // turn PASSED
   EXPECT_EQ(state.stones(Player::two), 0u);
   EXPECT_EQ(state.move_count(), 2u);
}

TEST_F(DarkHexState, occupancy_blocks_success_on_own_and_enemy_stones)
{
   state = State{Config{3, RulesMode::cdh}};
   state.apply_action(Move{rc(3, 0, 0)});        // one owns (0,0)
   state.apply_action(Move{rc(3, 0, 1)});        // two owns (0,1)

   // player one may neither overwrite his own stone ...
   state.apply_action(Move{rc(3, 0, 0)});
   EXPECT_TRUE(state.last_attempt_failed());
   EXPECT_EQ(state.stones(Player::one), 1u << rc(3, 0, 0));

   // ... nor the enemy's
   state.apply_action(Move{rc(3, 0, 1)});
   EXPECT_TRUE(state.last_attempt_failed());
   EXPECT_EQ(state.stones(Player::one), 1u << rc(3, 0, 0));
   EXPECT_EQ(state.stones(Player::two), 1u << rc(3, 0, 1));

   // occupancy is the union of both fleets
   EXPECT_TRUE(state.is_occupied(rc(3, 0, 0)));
   EXPECT_TRUE(state.is_occupied(rc(3, 0, 1)));
   EXPECT_FALSE(state.is_occupied(rc(3, 1, 1)));

   // out-of-grid indices stay invalid while every in-grid attempt is legal upfront
   EXPECT_FALSE(state.is_valid(Move{uint8_t(9)}));
   EXPECT_TRUE(state.is_valid(Move{rc(3, 0, 0)}));  // occupied != illegal: it FAILS instead
   EXPECT_THROW(state.apply_action(Move{uint8_t(42)}), std::invalid_argument);
   EXPECT_THROW(State{Config{6}}.validate(), std::invalid_argument);
}

// #####################################################################################################################
// information correctness
// #####################################################################################################################

namespace {

/// replays `script` while accumulating the observation stream that `observer` receives into an
/// infostate + publicstate (mirroring exactly what the solvers' traversal does)
std::pair< Infostate, Publicstate > observed_states(
   const Environment& env,
   const DarkHexScript& script,
   nor::Player observer
)
{
   Infostate infostate{observer};
   Publicstate publicstate{};
   State state{script.config};
   for(const auto& [actor, cell] : script.moves) {
      if(state.terminal()) {
         break;
      }
      auto next = state;
      next.apply_action(Move{cell});
      publicstate.update(env.public_observation(state, Move{cell}, next));
      infostate.update(
         env.public_observation(state, Move{cell}, next),
         env.private_observation(observer, state, Move{cell}, next)
      );
      state = next;
   }
   return {infostate, publicstate};
}

}  // namespace

TEST(DarkHexInformation, opponent_board_invisible_given_identical_own_view_histories)
{
   using namespace nor;
   Environment env{Config{3, RulesMode::cdh}};

   // the two scripts differ ONLY in bob's hidden placements; alex never stumbles onto any of
   // bob's cells, so every observation alex receives coincides in both worlds
   DarkHexScript script_a{};
   script_a.config = Config{3, RulesMode::cdh};
   script_a.moves = {
      {Player::one, rc(3, 0, 0)},
      {Player::two, rc(3, 0, 1)},
      {Player::one, rc(3, 1, 1)},
      {Player::two, rc(3, 0, 2)},
      {Player::one, rc(3, 2, 0)},
      {Player::two, rc(3, 2, 1)}};
   DarkHexScript script_b = script_a;
   script_b.moves[1].second = rc(3, 2, 2);
   script_b.moves[3].second = rc(3, 1, 0);
   script_b.moves[5].second = rc(3, 1, 2);

   auto [infostate_a, publicstate_a] = observed_states(env, script_a, nor::Player::alex);
   auto [infostate_b, publicstate_b] = observed_states(env, script_b, nor::Player::alex);

   // identical own-view histories over differing opponent boards --> same infostate
   EXPECT_EQ(infostate_a, infostate_b);
   EXPECT_EQ(infostate_a.hash(), infostate_b.hash());
   EXPECT_EQ(publicstate_a, publicstate_b);
   EXPECT_EQ(publicstate_a.hash(), publicstate_b.hash());

   // bob himself distinguishes his placements through his private confirmations
   auto bob_pair_a = observed_states(env, script_a, nor::Player::bob);
   auto bob_pair_b = observed_states(env, script_b, nor::Player::bob);
   EXPECT_NE(bob_pair_a.first, bob_pair_b.first);
   EXPECT_NE(bob_pair_a.first.hash(), bob_pair_b.first.hash());
}

TEST(DarkHexInformation, visible_referee_feedback_splits_information_sets)
{
   using namespace nor;
   Environment env{Config{3, RulesMode::cdh}};

   // baseline: alex places freely
   DarkHexScript clean{};
   clean.config = Config{3, RulesMode::cdh};
   clean.moves = {{Player::one, rc(3, 0, 0)}, {Player::two, rc(3, 0, 1)},
                  {Player::one, rc(3, 1, 1)}, {Player::two, rc(3, 0, 2)},
                  {Player::one, rc(3, 1, 0)}};

   // collision: bob secretly took (1,0) instead of (0,2), so alex's last attempt is publicly
   // silent but privately REJECTED to him -- a visible outcome difference at equal history length
   DarkHexScript collision = clean;
   collision.moves[3].second = rc(3, 1, 0);
   collision.moves[4].second = rc(3, 1, 0);  // rejected, turn retained

   auto [clean_info, clean_pub] = observed_states(env, clean, nor::Player::alex);
   auto [collide_info, collide_pub] = observed_states(env, collision, nor::Player::alex);

   EXPECT_NE(clean_info, collide_info);
   EXPECT_NE(clean_info.hash(), collide_info.hash());
   // the public channel carries no payload in either world; only the private feedback splits
   for(const auto& obs : clean_pub.history()) {
      EXPECT_EQ(obs.kind, Observation::Kind::none);
   }
   for(const auto& obs : collide_pub.history()) {
      EXPECT_EQ(obs.kind, Observation::Kind::none);
   }
   EXPECT_EQ(clean_pub, collide_pub);
   EXPECT_EQ(clean_pub.hash(), collide_pub.hash());
   // sanity: the rejection really was delivered as a private observation to alex alone
   const auto& rejected_entry = collide_info.history()[4];
   EXPECT_EQ(rejected_entry.second.kind, Observation::Kind::attempt_rejected);
   EXPECT_EQ(rejected_entry.second.cell_index, rc(3, 1, 0));
}

TEST(DarkHexInformation, environment_observation_functions_deliver_privately)
{
   using namespace nor;
   Environment env{Config{3, RulesMode::cdh}};
   State s{env.config()};
   s.apply_action(Move{rc(3, 0, 0)});  // alex places
   auto next = s;
   next.apply_action(Move{rc(3, 0, 0)});  // bob collides with alex's stone

   // bob learns of his rejection; alex receives nothing
   auto priv_bob = env.private_observation(nor::Player::bob, s, Move{rc(3, 0, 0)}, next);
   EXPECT_EQ(priv_bob.kind, Observation::Kind::attempt_rejected);
   auto priv_alex = env.private_observation(nor::Player::alex, s, Move{rc(3, 0, 0)}, next);
   EXPECT_EQ(priv_alex.kind, Observation::Kind::none);

   // successful placements confirm the stone to its owner only
   State s2{Config{3, RulesMode::cdh}};
   State s2_after = s2;
   s2_after.apply_action(Move{rc(3, 1, 1)});
   auto priv_alex_ok =
      env.private_observation(nor::Player::alex, s2, Move{rc(3, 1, 1)}, s2_after);
   EXPECT_EQ(priv_alex_ok.kind, Observation::Kind::stone_placed);
   EXPECT_EQ(priv_alex_ok.cell_index, rc(3, 1, 1));
   auto priv_bob_other =
      env.private_observation(nor::Player::bob, s2, Move{rc(3, 1, 1)}, s2_after);
   EXPECT_EQ(priv_bob_other.kind, Observation::Kind::none);

   // the public channel never carries payload
   auto pub = env.public_observation(s, Move{rc(3, 0, 0)}, next);
   EXPECT_EQ(pub.kind, Observation::Kind::none);

   // histories: open reveals everything, public hides all, private shows own attempts only
   State s3{env.config()};
   s3.apply_action(Move{rc(3, 0, 0)});
   s3.apply_action(Move{rc(3, 2, 2)});
   s3.apply_action(Move{rc(3, 0, 0)});  // alex's rejected retry
   auto open = env.open_history(s3);
   ASSERT_EQ(open.size(), 3u);
   auto pub_hist = env.public_history(s3);
   ASSERT_EQ(pub_hist.size(), 3u);
   EXPECT_FALSE(pub_hist[0].value().has_value());
   auto alex_hist = env.private_history(nor::Player::alex, s3);
   ASSERT_EQ(alex_hist.size(), 3u);
   EXPECT_TRUE(alex_hist[0].value().has_value());
   EXPECT_FALSE(alex_hist[1].value().has_value());
   EXPECT_TRUE(alex_hist[2].value().has_value());
   auto bob_hist = env.private_history(nor::Player::bob, s3);
   ASSERT_EQ(bob_hist.size(), 3u);
   EXPECT_FALSE(bob_hist[0].value().has_value());
   EXPECT_TRUE(bob_hist[1].value().has_value());
}

// #####################################################################################################################
// payoff table + no-draw invariant
// #####################################################################################################################

TEST_P(DarkHexPayoffParamsF, payoff_table_plus_minus_one)
{
   const auto& [script, payoff_one, payoff_two] = GetParam();
   auto final_state = play_script(script);
   ASSERT_TRUE(final_state.terminal()) << common::to_string(final_state.move_count());
   EXPECT_DOUBLE_EQ(final_state.payoff(Player::one), payoff_one);
   EXPECT_DOUBLE_EQ(final_state.payoff(Player::two), payoff_two);
   EXPECT_DOUBLE_EQ(final_state.payoff(Player::one) + final_state.payoff(Player::two), 0.);
}

INSTANTIATE_TEST_SUITE_P(
   DarkHexPayoffs,
   DarkHexPayoffParamsF,
   ::testing::Values(
      // player one bridges top <-> bottom through the leftmost column
      std::make_tuple(
         DarkHexScript{
            Config{3, RulesMode::cdh},
            {{Player::one, rc(3, 0, 0)},
             {Player::two, rc(3, 0, 1)},
             {Player::one, rc(3, 1, 0)},
             {Player::two, rc(3, 0, 2)},
             {Player::one, rc(3, 2, 0)}}},
         1.,
         -1.
      ),
      // player two bridges left <-> right through the middle row
      std::make_tuple(
         DarkHexScript{
            Config{3, RulesMode::adh},
            {{Player::one, rc(3, 0, 0)},
             {Player::two, rc(3, 1, 0)},
             {Player::one, rc(3, 0, 1)},
             {Player::two, rc(3, 1, 1)},
             {Player::one, rc(3, 2, 0)},
             {Player::two, rc(3, 1, 2)}}},
         -1.,
         1.
      ),
      // cdh: player one wastes attempts on taken cells (retained turns) before winning
      std::make_tuple(
         DarkHexScript{
            Config{3, RulesMode::cdh},
            {{Player::one, rc(3, 1, 1)},
             {Player::two, rc(3, 0, 0)},
             {Player::one, rc(3, 1, 1)},  // rejected on own stone, turn retained
             {Player::one, rc(3, 1, 2)},
             {Player::two, rc(3, 1, 1)},  // rejected on enemy stone, turn retained
             {Player::two, rc(3, 0, 1)},
             {Player::one, rc(3, 0, 2)},
             {Player::two, rc(3, 0, 2)},  // rejected on enemy stone, turn retained
             {Player::two, rc(3, 1, 0)},
             {Player::one, rc(3, 2, 2)}}},
         1.,
         -1.
      )
   )
);

TEST(DarkHexRandomPlayouts, no_draws_and_consistent_payoffs)
{
   auto run_playouts = [&](RulesMode mode, unsigned seed_base) {
      for(unsigned seed = 0; seed < 100; ++seed) {
         State s{Config{3, mode}};
         std::mt19937 rng{seed_base + seed};
         size_t guard = 0;
         while(not s.terminal()) {
            auto legal = s.actions(s.active_player());
            ASSERT_FALSE(legal.empty());
            std::uniform_int_distribution< size_t > dist(0, legal.size() - 1);
            s.apply_action(legal[dist(rng)]);
            ASSERT_LT(++guard, 5000u) << "playout failed to terminate";
         }
         const bool one_won = s.has_won(Player::one);
         const bool two_won = s.has_won(Player::two);
         // draws are impossible: exactly one connector exists
         EXPECT_NE(one_won, two_won) << "seed=" << seed << " mode=" << common::to_string(mode);
         EXPECT_DOUBLE_EQ(s.payoff(Player::one), one_won ? 1. : -1.);
         EXPECT_DOUBLE_EQ(s.payoff(Player::two), two_won ? 1. : -1.);
      }
   };
   run_playouts(RulesMode::cdh, 1'000);
   run_playouts(RulesMode::adh, 2'000);
}

// #####################################################################################################################
// deterministic-env trait checks
// #####################################################################################################################

static_assert(nor::concepts::deterministic_fosg< nor::games::dark_hex::Environment >);

TEST(DarkHexTraits, deterministic_fosg_concepts)
{
   using Env = nor::games::dark_hex::Environment;
   static_assert(std::same_as< typename Env::chance_outcome_type, std::monostate >);
   static_assert(
      std::same_as< typename Env::action_variant_type, std::variant< Move, std::monostate > >
   );
   EXPECT_TRUE((nor::concepts::fosg< Env >) );
   EXPECT_TRUE((nor::concepts::deterministic_env< Env >) );
   EXPECT_FALSE((nor::concepts::stochastic_env< Env >) );
   EXPECT_EQ(Env::stochasticity(), nor::Stochasticity::deterministic);
   // deterministic envs expose no chance interface
   EXPECT_FALSE(nor::concepts::has::method::chance_actions< Env >);
   EXPECT_FALSE(nor::concepts::has::method::chance_probability< Env >);
}

// #####################################################################################################################
// CFR convergence smoke (baseline numbers)
// #####################################################################################################################

namespace {

struct DarkHexConvergenceReport {
   double exploitability_first_checkpoint = std::numeric_limits< double >::quiet_NaN();
   double exploitability_final = std::numeric_limits< double >::quiet_NaN();
   size_t iterations = 0;
};

inline DarkHexConvergenceReport
print_convergence_report(const DarkHexConvergenceReport& report)
{
   fmt::print(
      "[darkhex-cfr-baseline] iterations={} expl_at_iter_50={:.6e} expl_final={:.6e}\n",
      report.iterations,
      report.exploitability_first_checkpoint,
      report.exploitability_final
   );
   return report;
}

}  // namespace

TEST(DarkHexCFR, vanilla_alternating_2x2_cdh_converges)
{
   using namespace nor;
   using Env = games::dark_hex::Environment;
   auto config = dark_hex::Config{/*board_size=*/2, dark_hex::RulesMode::cdh};
   Env env{config};

   auto root_state = std::make_unique< dark_hex::State >(config);

   auto avg_policy = factory::make_tabular_policy(
      std::unordered_map< games::dark_hex::Infostate, HashmapActionPolicy< Move > >{}
   );
   auto curr_policy = factory::make_tabular_policy(
      std::unordered_map< games::dark_hex::Infostate, HashmapActionPolicy< Move > >{}
   );

   auto solver = factory::make_cfr< rm::CFRConfig{}, true >(
      std::move(env), std::move(root_state), curr_policy, avg_policy
   );
   Env expl_env{config};

   constexpr size_t kIterations = 300;
   constexpr size_t kCheckpoint = 50;

   DarkHexConvergenceReport report{};
   report.iterations = kIterations;
   for(size_t iter = 1; iter <= kIterations; ++iter) {
      solver.iterate(1);
      if(iter % kCheckpoint != 0) {
         continue;
      }
      const auto& avg_policies = solver.average_policy();
      using AvgTablePolicy = std::decay_t< decltype(avg_policies.at(nor::Player::alex)) >;
      double expl = exploitability(
         expl_env,
         games::dark_hex::State{config},
         player_hashmap< AvgTablePolicy >{
            std::pair{
               nor::Player::alex, normalize_state_policy(avg_policies.at(nor::Player::alex))},
            std::pair{nor::Player::bob, normalize_state_policy(avg_policies.at(nor::Player::bob))}}
      );
      fmt::print("[darkhex-cfr-baseline] iter={} exploitability={:.6e}\n", iter, expl);
      if(iter == kCheckpoint) {
         report.exploitability_first_checkpoint = expl;
      }
      report.exploitability_final = expl;
   }

   // vanilla CFR must make clear progress on the smallest dark hex instance
   EXPECT_LT(report.exploitability_final, report.exploitability_first_checkpoint);
   print_convergence_report(report);
}
