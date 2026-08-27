
#include <gtest/gtest.h>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "battleship_gs/battleship_gs.hpp"
#include "fixtures.hpp"
#include "nor/concepts.hpp"
#include "nor/exploitability.hpp"
#include "nor/nor.hpp"

using namespace battleship_gs;
using namespace nor::games::battleship_gs;

namespace {

constexpr Cell c(int row, int col)
{
   return Cell{int8_t(row), int8_t(col)};
}

/**
 * @brief Exhaustive uniform-profile enumeration over the whole game tree.
 *
 * Every placement is uniform over the grid, every shot uniform over the shooter's FRESH
 * (never-fired) targets -- this is exactly the SW-maximizing Nash equilibrium of the canonical
 * instance (Farina et al. 2019, Section 5.1), for which the paper reports the outcome
 * probabilities 5/9 / 1/3 / 1/9 and social welfare -8/9. These are asserted verbatim below;
 * note that they only arise for repeat-free shooting (the engine prunes already-fired cells).
 */
struct UniformProfileStats {
   double prob_one_sinks = 0.;  //< player one destroys the opponent's fleet
   double prob_two_sinks = 0.;
   double prob_timeout = 0.;  //< budgets exhausted without any sunk ship
   double expected_u1 = 0.;
   double expected_u2 = 0.;

   double social_welfare() const { return expected_u1 + expected_u2; }
};

void enumerate_uniform(const State& state, double prob, UniformProfileStats& stats)
{
   if(prob == 0.) {
      return;
   }
   if(state.terminal()) {
      const auto [u1, u2] = state.payoffs();
      stats.expected_u1 += prob * u1;
      stats.expected_u2 += prob * u2;
      if(u1 > 0.) {
         stats.prob_one_sinks += prob;
      } else if(u2 > 0.) {
         stats.prob_two_sinks += prob;
      } else {
         stats.prob_timeout += prob;
      }
      return;
   }
   const auto acts = state.actions(state.active_player());
   ASSERT_FALSE(acts.empty());
   const double p = prob / double(acts.size());
   for(const auto& action : acts) {
      auto next = state;
      next.apply_action(action);
      enumerate_uniform(next, p, stats);
   }
}

UniformProfileStats enumerate_uniform(const Config& config)
{
   UniformProfileStats stats{};
   enumerate_uniform(State{config}, 1., stats);
   return stats;
}

}  // namespace

// #####################################################################################################################
// config validation
// #####################################################################################################################

TEST(BattleshipGsConfig, defaults_transcribe_the_canonical_instance)
{
   // Farina et al. 2019, Section 5.1: board 3x1, one length-1 ship of value 1, r=2, gamma=2
   const Config config{};
   EXPECT_EQ(config.rows, 3u);
   EXPECT_EQ(config.cols, 1u);
   ASSERT_EQ(config.fleet.size(), 1u);
   EXPECT_EQ(config.fleet.front().length, 1u);
   EXPECT_DOUBLE_EQ(config.fleet.front().value, 1.);
   EXPECT_EQ(config.max_shots, 2u);
   EXPECT_DOUBLE_EQ(config.loss_multiplier, 2.);
   EXPECT_NO_THROW(config.validate());
}

TEST(BattleshipGsConfig, invalid_configurations_are_rejected)
{
   EXPECT_THROW((Config{0, 1, {{1u, 1.}}, 2, 2.}), std::invalid_argument);
   EXPECT_THROW((Config{3, 1, {}, 2, 2.}), std::invalid_argument);  // empty fleet
   EXPECT_THROW((Config{3, 1, {{0u, 1.}}, 2, 2.}), std::invalid_argument);  // zero-length ship
   EXPECT_THROW((Config{3, 1, {{7u, 1.}}, 2, 2.}), std::invalid_argument);  // overlong ship
   EXPECT_THROW((Config{3, 1, {{4u, 1.}}, 2, 2.}), std::invalid_argument);  // does not fit 3x1
   EXPECT_THROW((Config{3, 1, {{1u, 0.}}, 2, 2.}), std::invalid_argument);  // non-positive value
   EXPECT_THROW((Config{3, 1, {{1u, 1.}}, 0, 2.}), std::invalid_argument);  // no shots
   EXPECT_THROW((Config{3, 1, {{1u, 1.}}, 2, 0.5}), std::invalid_argument);  // gamma < 1
}

// #####################################################################################################################
// placement legality
// #####################################################################################################################

TEST_F(BattleshipGsState, placement_phase_starts_with_player_one)
{
   state = State{canonical_config()};
   EXPECT_EQ(state.phase(), Phase::one_placement);
   EXPECT_EQ(state.active_player(), Player::one);
   EXPECT_TRUE(state.placing());
   EXPECT_FALSE(state.firing());
   EXPECT_EQ(state.pending_ship_length(Player::one), 1u);
}

TEST_F(BattleshipGsState, placements_alternate_secretly_ship_by_ship)
{
   // ordered heterogeneous fleet: a length-1 ship of value 4 followed by a length-2 ship of
   // value 1 on a 3x2 field; both players place THE SAME spec in the SAME order
   Config config{3, 2, {{1u, 4.}, {2u, 1.}}, /*max_shots=*/3, /*loss_multiplier=*/2.};
   state = State{config};
   EXPECT_EQ(state.pending_ship_length(Player::one), 1u);
   state.apply_action(Action{Place{c(0, 0), int8_t{0}, int8_t{1}}});
   EXPECT_EQ(state.active_player(), Player::two);
   EXPECT_EQ(state.pending_ship_length(Player::two), 1u);
}

TEST_F(BattleshipGsState, placements_alternate_in_fleet_order)
{
   Config config{3, 2, {{1u, 4.}, {2u, 1.}}, 3, 2.};
   state = State{config};
   state.apply_action(Action{Place{c(0, 0), int8_t{0}, int8_t{1}}});  // P1 ship 0 (length 1)
   EXPECT_EQ(state.active_player(), Player::two);
   state.apply_action(Action{Place{c(2, 0), int8_t{0}, int8_t{1}}});  // P2 ship 0 (length 1)
   EXPECT_EQ(state.active_player(), Player::one);
   EXPECT_EQ(state.pending_ship_length(Player::one), 2u);  // now the length-2 ships are due
   state.apply_action(Action{Place{c(1, 0), int8_t{1}, int8_t{0}}});  // P1 ship 1
   state.apply_action(Action{Place{c(1, 1), int8_t{1}, int8_t{0}}});  // P2 ship 1
   EXPECT_EQ(state.phase(), Phase::one_fire);
   EXPECT_EQ(state.ships_placed(Player::one), 2u);
   EXPECT_EQ(state.ships_placed(Player::two), 2u);
   EXPECT_EQ(state.fleet_cells(Player::one).size(), 2u);
   EXPECT_EQ(state.fleet_cells(Player::one).at(1).size(), 2u);
}

TEST_F(BattleshipGsState, placement_rejects_off_grid_cells)
{
   state = State{canonical_config()};
   EXPECT_FALSE(state.is_valid(Action{Fire{c(0, 0)}}));
   EXPECT_THROW(state.apply_action(Action{Fire{c(0, 0)}}), std::invalid_argument);
   // off-grid starts are impossible here because actions() only enumerates in-grid starts, but
   // is_valid must still reject hand-crafted illegal lines
   auto wide = State{Config{2, 3, {{2u, 1.}}, 3, 2.}};
   EXPECT_FALSE(wide.is_valid(Action{Place{c(0, 2), int8_t{0}, int8_t{1}}}));  // runs off right
   EXPECT_FALSE(wide.is_valid(Action{Place{c(2, 0), int8_t{1}, int8_t{0}}}));  // runs off bottom
   EXPECT_THROW(
      wide.apply_action(Action{Place{c(0, 2), int8_t{0}, int8_t{1}}}), std::invalid_argument
   );
}

TEST_F(BattleshipGsState, placement_requires_canonical_straight_orientation)
{
   auto wide = State{Config{2, 3, {{2u, 1.}}, 3, 2.}};
   // only (+row)-stepping / (+col)-stepping canonical orientations are accepted
   EXPECT_FALSE(wide.is_valid(Action{Place{c(0, 1), int8_t{0}, int8_t{-1}}}));  // leftwards
   EXPECT_FALSE(wide.is_valid(Action{Place{c(1, 0), int8_t{-1}, int8_t{0}}}));  // upwards
   EXPECT_FALSE(wide.is_valid(Action{Place{c(0, 0), int8_t{1}, int8_t{1}}}));  // diagonal
   EXPECT_FALSE(wide.is_valid(Action{Place{c(0, 0), int8_t{2}, int8_t{0}}}));  // non-unit step
   EXPECT_TRUE(wide.is_valid(Action{Place{c(0, 0), int8_t{0}, int8_t{1}}}));
   EXPECT_TRUE(wide.is_valid(Action{Place{c(0, 0), int8_t{1}, int8_t{0}}}));
}

TEST_F(BattleshipGsState, placement_forbids_overlap_within_own_fleet)
{
   Config config{3, 2, {{2u, 1.}, {1u, 2.}}, 3, 2.};
   state = State{config};
   state.apply_action(Action{Place{c(0, 0), int8_t{0}, int8_t{1}}});
   state.apply_action(Action{Place{c(2, 0), int8_t{0}, int8_t{1}}});  // P2 mirrors far away
   EXPECT_EQ(state.active_player(), Player::one);
   // overlapping any own ship is illegal ...
   EXPECT_FALSE(state.is_valid(Action{Place{c(0, 1), int8_t{0}, int8_t{1}}}));
   EXPECT_FALSE(state.is_valid(Action{Place{c(0, 0), int8_t{1}, int8_t{0}}}));
   // ... while a fresh disjoint cell is accepted
   EXPECT_TRUE(state.is_valid(Action{Place{c(1, 1), int8_t{0}, int8_t{1}}}));
   // opposing fields are independent: both players may occupy identical cells of THEIR OWN field
   auto mirrored = State{config};
   mirrored.apply_action(Action{Place{c(0, 0), int8_t{0}, int8_t{1}}});
   mirrored.apply_action(Action{Place{c(0, 0), int8_t{0}, int8_t{1}}});
   EXPECT_EQ(mirrored.fleet_cells(Player::two), mirrored.fleet_cells(Player::one));
}

TEST_F(BattleshipGsState, placement_action_enumeration)
{
   // canonical instance: three length-1 placements on the 3x1 field, each generated exactly once
   state = State{canonical_config()};
   auto legal = state.actions(Player::one);
   ASSERT_EQ(legal.size(), 3u);
   for(size_t row : {size_t{0}, size_t{1}, size_t{2}}) {
      Action expected{Place{c(int8_t(row), 0), int8_t{0}, int8_t{1}}};
      EXPECT_NE(std::find(legal.begin(), legal.end(), expected), legal.end())
         << common::to_string(expected);
   }
   // no actions for the non-active player
   EXPECT_EQ(state.actions(Player::two), std::vector< Action >{});

   // mixed fleet on 3x2: six length-1 spots, then seven length-2 dominoes (3 horizontal rows,
   // 4 vertical) on a fresh field -- but only five after player one's scout blocks cell (0,0)
   Config config{3, 2, {{1u, 1.}, {2u, 1.}}, 3, 2.};
   auto mixed = State{config};
   EXPECT_EQ(mixed.actions(Player::one).size(), 6u);
   mixed.apply_action(Action{Place{c(0, 0), int8_t{0}, int8_t{1}}});
   mixed.apply_action(Action{Place{c(0, 0), int8_t{0}, int8_t{1}}});
   EXPECT_EQ(mixed.actions(Player::one).size(), 5u);

   // length-2 dominoes on a fresh 3x2 field
   auto dominoes = State{Config{3, 2, {{2u, 1.}}, 3, 2.}};
   EXPECT_EQ(dominoes.actions(Player::one).size(), 7u);
}

// #####################################################################################################################
// shot resolution truth table
// #####################################################################################################################

TEST(BattleshipGsShotResolution, miss_hit_sink_transitions_with_multi_cell_ships)
{
   // one length-2 ship worth 1 per player on a 1x3 strip, gamma = 2, r = 2
   Config config{1, 3, {{2u, 1.}}, 2, 2.};
   State state{config};
   // fleets: player one covers columns 0-1, player two columns 1-2
   state.apply_action(Action{Place{c(0, 0), int8_t{0}, int8_t{1}}});
   state.apply_action(Action{Place{c(0, 1), int8_t{0}, int8_t{1}}});
   ASSERT_EQ(state.phase(), Phase::one_fire);

   // player one misses (fires at his own stern cell)
   state.apply_action(Action{Fire{c(0, 0)}});
   EXPECT_EQ(state.shots_used(Player::one), 1u);
   EXPECT_TRUE(state.was_fired_at(Player::one, c(0, 0)));
   EXPECT_FALSE(state.was_hit(Player::one, c(0, 0)));
   EXPECT_EQ(state.phase(), Phase::two_fire);
   EXPECT_FALSE(state.terminal());

   // player two hits but does not sink yet (bow cell still intact)
   state.apply_action(Action{Fire{c(0, 1)}});
   EXPECT_EQ(state.phase(), Phase::one_fire);
   EXPECT_TRUE(state.was_hit(Player::two, c(0, 1)));
   EXPECT_FALSE(state.ship_sunk(Player::two, 0));  // player ONE's ship index 0 untouched
   EXPECT_FALSE(state.terminal());

   // player one hits back without sinking
   state.apply_action(Action{Fire{c(0, 2)}});
   EXPECT_TRUE(state.was_hit(Player::one, c(0, 2)));
   EXPECT_FALSE(state.ship_sunk(Player::two, 0));
   EXPECT_FALSE(state.terminal());
   EXPECT_FALSE(state.fleet_sunk(Player::two));

   // player two completes the destruction of player one's ship --> immediate game over
   state.apply_action(Action{Fire{c(0, 0)}});
   EXPECT_TRUE(state.ship_sunk(Player::one, 0));
   EXPECT_TRUE(state.fleet_sunk(Player::one));
   EXPECT_TRUE(state.terminal());
   // u(one) = -gamma * v = -2, u(two) = +v = +1 (general-sum!)
   EXPECT_DOUBLE_EQ(state.payoff(Player::one), -2.);
   EXPECT_DOUBLE_EQ(state.payoff(Player::two), 1.);
}

TEST(BattleshipGsShotResolution, sink_terminates_game_before_budgets_run_out)
{
   GameScript script{};
   script.config = canonical_config();
   script.fleet_one = {{c(0, 0)}};
   script.fleet_two = {{c(1, 0)}};
   // player one finds and sinks the opponent within his first two shots; player two's remaining
   // budget must never be used
   script.shots_one = {c(1, 0)};
   script.shots_two = {c(0, 0)};
   auto final_state = play_script(script);
   EXPECT_TRUE(final_state.terminal());
   EXPECT_EQ(final_state.shots_used(Player::one), 1u);
   EXPECT_EQ(final_state.shots_used(Player::two), 0u);
   EXPECT_DOUBLE_EQ(final_state.payoff(Player::one), 1.);
   EXPECT_DOUBLE_EQ(final_state.payoff(Player::two), -2.);
}

TEST(BattleshipGsShotResolution, budget_exhaustion_ends_the_game_without_payoffs)
{
   GameScript script{};
   script.config = canonical_config();
   script.fleet_one = {{c(0, 0)}};
   script.fleet_two = {{c(1, 0)}};
   // everybody wastes their shots on empty cells: mutual timeout with NO sink pays zero to
   // both players even though gamma = 2 (shots never repeat a previously fired cell --
   // cf. the no-repeat shooting rule of the state machine)
   script.shots_one = {c(0, 0), c(2, 0)};
   script.shots_two = {c(2, 0), c(1, 0)};
   auto final_state = play_script(script);
   EXPECT_TRUE(final_state.terminal());
   EXPECT_EQ(final_state.shots_used(Player::one), 2u);
   EXPECT_EQ(final_state.shots_used(Player::two), 2u);
   EXPECT_DOUBLE_EQ(final_state.payoff(Player::one), 0.);
   EXPECT_DOUBLE_EQ(final_state.payoff(Player::two), 0.);

   // mid-way states with unspent shot budgets are not terminal
   State midway{canonical_config()};
   midway.apply_action(Action{Place{c(0, 0), int8_t{0}, int8_t{1}}});
   midway.apply_action(Action{Place{c(1, 0), int8_t{0}, int8_t{1}}});
   midway.apply_action(Action{Fire{c(2, 0)}});
   EXPECT_FALSE(midway.terminal());
   EXPECT_EQ(midway.phase(), Phase::two_fire);
}

TEST(BattleshipGsShotResolution, shots_never_repeat_while_fresh_cells_remain)
{
   // canonical instance (r = 2 <= 3 cells): the engine prunes already-fired cells from the
   // action set and rejects them in is_valid -- wasting a shot on a known miss is strictly
   // dominated in every Nash equilibrium, and repeat-free play is exactly what reproduces the
   // published Section 5.1 statistics asserted by enumerate_uniform below
   auto decided = play_script(GameScript{
      canonical_config(), {{c(0, 0)}}, {{c(1, 0)}}, {c(2, 0), c(1, 0)}, {c(2, 0)}});
   // neither salvo scored before player one's second shot sank player two's scout
   ASSERT_TRUE(decided.terminal());
   EXPECT_EQ(decided.shots_used(Player::one), 2u);
   EXPECT_EQ(decided.shots_used(Player::two), 1u);
   EXPECT_FALSE(decided.was_fired_at(Player::one, c(0, 0)));
   EXPECT_FALSE(decided.was_fired_at(Player::two, c(0, 0)));
   EXPECT_DOUBLE_EQ(decided.payoff(Player::one), 1.);
   EXPECT_DOUBLE_EQ(decided.payoff(Player::two), -2.);

   // mid-game: the tried cell vanished from the fresh-cell action set ...
   auto midway = play_script(GameScript{
      canonical_config(), {{c(0, 0)}}, {{c(1, 0)}}, {c(2, 0)}, {c(2, 0)}});
   ASSERT_FALSE(midway.terminal());
   ASSERT_EQ(midway.phase(), Phase::one_fire);
   EXPECT_TRUE(midway.was_fired_at(Player::one, c(2, 0)));
   EXPECT_FALSE(midway.is_valid(Action{Fire{c(2, 0)}}));
   EXPECT_THROW(midway.apply_action(Action{Fire{c(2, 0)}}), std::invalid_argument);
   auto legal = midway.actions(Player::one);
   ASSERT_EQ(legal.size(), 2u);
   EXPECT_NE(std::find(legal.begin(), legal.end(), Action{Fire{c(0, 0)}}), legal.end());
   EXPECT_NE(std::find(legal.begin(), legal.end(), Action{Fire{c(1, 0)}}), legal.end());

   // ... while the opponent's fired cells stay irrelevant to this shooter's own freshness book-
   // keeping: both sides already grazed cell (2,0) on their respective boards
   EXPECT_EQ(midway.shots_used(Player::two), 1u);
}

// #####################################################################################################################
// payoff scenarios (hand-computed against Appendix E.1's payoff rule)
// #####################################################################################################################

TEST_P(BattleshipGsPayoffParamsF, payoff_table)
{
   const auto& [script, payoff_one, payoff_two] = GetParam();
   auto final_state = play_script(script);
   ASSERT_TRUE(final_state.terminal()) << common::to_string(final_state.phase());
   EXPECT_DOUBLE_EQ(final_state.payoff(Player::one), payoff_one);
   EXPECT_DOUBLE_EQ(final_state.payoff(Player::two), payoff_two);
   // reward() reads straight off the world state
   EXPECT_DOUBLE_EQ(Environment{script.config}.reward(nor::Player::alex, final_state), payoff_one);
   EXPECT_DOUBLE_EQ(Environment{script.config}.reward(nor::Player::bob, final_state), payoff_two);
}

INSTANTIATE_TEST_SUITE_P(
   BattleshipGsPayoffs,
   BattleshipGsPayoffParamsF,
   ::testing::Values(
      // canonical gamma=2 instance: sink win / get sunk loss --> (v, -gamma*v)
      std::make_tuple(
         GameScript{canonical_config(), {{c(1, 0)}}, {{c(2, 0)}}, {c(2, 0)}, {c(0, 0)}},
         1.,
         -2.
      ),
      // canonical gamma=2 instance: reversed outcome
      std::make_tuple(
         GameScript{canonical_config(), {{c(1, 0)}}, {{c(2, 0)}}, {c(0, 0), c(2, 0)}, {c(1, 0)}},
         -2.,
         1.
      ),
      // gamma=1 recovers the zero-sum family: payoffs always antisymmetric. The salvo needs a
      // filler second shot by player one (a fresh miss) so the alternation reaches player
      // two's game-ending second shot
      std::make_tuple(
         GameScript{
            Config{/*rows=*/1, /*cols=*/3, {{2u, 1.}}, 2, 1.},
            {{c(0, 0), c(0, 1)}},
            {{c(0, 1), c(0, 2)}},
            {c(0, 2), c(0, 0)},
            {c(0, 0), c(0, 1)}},
         -1.,
         1.
      ),
      // heterogeneous fleet {(1,4),(2,1)}, gamma=1.5: player one sinks the cheap enemy capital
      // (value 1) while losing his own expensive scout (value 4); the game runs to a mutual
      // budget timeout since neither fleet is fully destroyed:
      // u1 = 1 - 1.5*4 = -5, u2 = 4 - 1.5*1 = 2.5
      std::make_tuple(
         GameScript{
            Config{3, 2, {{1u, 4.}, {2u, 1.}}, 3, 1.5},
            {{c(0, 0)}, {c(1, 0), c(2, 0)}},
            {{c(0, 1)}, {c(1, 1), c(2, 1)}},
            {c(1, 1), c(2, 1), c(0, 0)},
            {c(0, 0), c(1, 1), c(2, 0)}},
         -5.,
         2.5
      ),
      // partial damage is worthless until a ship fully sinks: player one's single hit on the
      // enemy domino pays nothing, while the enemy sank one of ours
      std::make_tuple(
         GameScript{
            Config{2, 2, {{2u, 1.}}, 2, 2.},
            {{c(0, 0), c(0, 1)}},
            {{c(1, 0), c(1, 1)}},
            {c(1, 0), c(0, 0)},
            {c(0, 0), c(0, 1)}},
         -2.,
         1.
      ),
      // mutual timeout without any sunk ship: nobody pays anything despite gamma = 2
      std::make_tuple(
         GameScript{
            Config{2, 2, {{2u, 1.}}, 2, 2.},
            {{c(0, 0), c(0, 1)}},
            {{c(1, 0), c(1, 1)}},
            {c(0, 0), c(0, 1)},
            {c(1, 0), c(1, 1)}},
         0.,
         0.
      )
   )
);

// #####################################################################################################################
// the published NE numbers of the canonical instance (Farina et al. 2019, Section 5.1)
// #####################################################################################################################

TEST(BattleshipGsCanonicalInstance, uniform_ne_profile_reproduces_the_published_statistics)
{
   // Section 5.1: under the uniform NE, P(player one sinks) = 5/9, P(player two sinks) = 1/3,
   // P(peaceful) = 1/9, and the social welfare is exactly -8/9 (E[u1] = -1/9, E[u2] = -7/9).
   const auto stats = enumerate_uniform(canonical_config());
   EXPECT_NEAR(stats.prob_one_sinks, 5. / 9., 1e-12);
   EXPECT_NEAR(stats.prob_two_sinks, 1. / 3., 1e-12);
   EXPECT_NEAR(stats.prob_timeout, 1. / 9., 1e-12);
   EXPECT_DOUBLE_EQ(stats.prob_one_sinks + stats.prob_two_sinks + stats.prob_timeout, 1.);
   EXPECT_NEAR(stats.expected_u1, -1. / 9., 1e-12);
   EXPECT_NEAR(stats.expected_u2, -7. / 9., 1e-12);
   EXPECT_NEAR(stats.social_welfare(), -8. / 9., 1e-12);

   // sanity: the game is genuinely GENERAL SUM at the sink terminals ((1,-2): sum -1)
   State sink{canonical_config()};
   sink.apply_action(Action{Place{c(0, 0), int8_t{0}, int8_t{1}}});
   sink.apply_action(Action{Place{c(1, 0), int8_t{0}, int8_t{1}}});
   sink.apply_action(Action{Fire{c(1, 0)}});
   ASSERT_TRUE(sink.terminal());
   EXPECT_NE(sink.payoffs()[0] + sink.payoffs()[1], 0.);
}

// #####################################################################################################################
// information correctness
// #####################################################################################################################

namespace {

/// replays `script` and accumulates the observation stream that `observer` receives into an
/// infostate + publicstate (mirroring exactly what the solvers' traversal does)
std::pair< nor::games::battleship_gs::Infostate, nor::games::battleship_gs::Publicstate >
observed_states(
   const nor::games::battleship_gs::Environment& env,
   const GameScript& script,
   nor::Player observer
)
{
   using namespace battleship_gs;
   nor::games::battleship_gs::Infostate infostate{observer};
   nor::games::battleship_gs::Publicstate publicstate{};
   State state{script.config};
   auto record = [&](const Action& action) {
      auto next = state;
      next.apply_action(action);
      publicstate.update(env.public_observation(state, action, next));
      infostate.update(
         env.public_observation(state, action, next),
         env.private_observation(observer, state, action, next)
      );
      state = next;
   };
   const auto n_ships = std::max(script.fleet_one.size(), script.fleet_two.size());
   for(size_t ship = 0; ship < n_ships; ++ship) {
      if(ship < script.fleet_one.size()) {
         record(Action{bgs_test::place_of(script.fleet_one.at(ship))});
      }
      if(ship < script.fleet_two.size()) {
         record(Action{bgs_test::place_of(script.fleet_two.at(ship))});
      }
   }
   size_t next_shot_one = 0;
   size_t next_shot_two = 0;
   while(not state.terminal()) {
      if(state.phase() == Phase::one_fire) {
         if(next_shot_one >= script.shots_one.size()) {
            break;
         }
         record(Action{Fire{script.shots_one[next_shot_one++]}});
      } else if(state.phase() == Phase::two_fire) {
         if(next_shot_two >= script.shots_two.size()) {
            break;
         }
         record(Action{Fire{script.shots_two[next_shot_two++]}});
      } else {
         break;
      }
   }
   return {infostate, publicstate};
}

}  // namespace

TEST(BattleshipGsInformation, opponent_placement_not_inferable_from_observations)
{
   using namespace battleship_gs;
   using namespace nor;
   Environment env{Config{2, 3, {{2u, 1.}}, 3, 2.}};

   // two games identical from alex's point of view except for bob's hidden fleet layout.
   // alex's shot sequence only targets column 0 -- a miss against BOTH candidate layouts -- so
   // the referee verdicts (and hence all observations) coincide
   GameScript script_a{};
   script_a.config = Config{2, 3, {{2u, 1.}}, 3, 2.};
   script_a.fleet_one = {{c(0, 0), c(0, 1)}};
   script_a.fleet_two = {{c(1, 0), c(1, 1)}};
   script_a.shots_one = {c(0, 2)};
   script_a.shots_two = {c(0, 0), c(0, 1)};

   GameScript script_b = script_a;
   script_b.fleet_two = {{c(0, 1), c(1, 1)}};

   auto [infostate_a, publicstate_a] = observed_states(env, script_a, nor::Player::alex);
   auto [infostate_b, publicstate_b] = observed_states(env, script_b, nor::Player::alex);

   // identical public shot histories + invisible opponent placements --> same infostate
   EXPECT_EQ(infostate_a, infostate_b);
   EXPECT_EQ(infostate_a.hash(), infostate_b.hash());
   EXPECT_EQ(publicstate_a, publicstate_b);
   EXPECT_EQ(publicstate_a.hash(), publicstate_b.hash());

   // bob himself can tell his two placements apart through his private confirmations
   auto bob_pair_a = observed_states(env, script_a, nor::Player::bob);
   auto bob_pair_b = observed_states(env, script_b, nor::Player::bob);
   EXPECT_NE(bob_pair_a.first, bob_pair_b.first);
}

TEST(BattleshipGsInformation, placement_secrecy_of_observation_functions)
{
   using namespace battleship_gs;
   using namespace nor;
   // ordered fleet: one length-2 ship worth 1 followed by one length-1 ship worth 1
   const Config config{3, 2, {{2u, 1.}, {1u, 1.}}, 3, 2.};
   Environment env{config};
   State state{config};
   state.apply_action(Action{Place{c(0, 0), int8_t{0}, int8_t{1}}});  // alex's domino
   auto next = state;
   Action bobs_placement{Place{c(2, 0), int8_t{0}, int8_t{1}}};
   next.apply_action(bobs_placement);  // bob's domino

   // the public observation carries no positional payload
   auto pub = env.public_observation(state, bobs_placement, next);
   EXPECT_EQ(pub.kind, Observation::Kind::hidden_placement);
   EXPECT_FALSE(pub.target.has_value());
   EXPECT_FALSE(pub.placed_cells[0].has_value());

   // alex learns nothing privately ...
   auto priv_alex = env.private_observation(nor::Player::alex, state, bobs_placement, next);
   EXPECT_EQ(priv_alex.kind, Observation::Kind::none);

   // ... while bob receives his own domino's full confirmation
   auto priv_bob = env.private_observation(nor::Player::bob, state, bobs_placement, next);
   EXPECT_EQ(priv_bob.kind, Observation::Kind::hidden_placement);
   ASSERT_TRUE(priv_bob.placed_cells[0].has_value());
   ASSERT_TRUE(priv_bob.placed_cells[1].has_value());
   EXPECT_EQ(*priv_bob.placed_cells[0], c(2, 0));
   EXPECT_EQ(*priv_bob.placed_cells[1], c(2, 1));

   // multi-cell confirmations carry every covered cell -- here the single-cell second placement
   State two_ships{config};
   two_ships.apply_action(Action{Place{c(0, 0), int8_t{0}, int8_t{1}}});
   two_ships.apply_action(Action{Place{c(2, 0), int8_t{0}, int8_t{1}}});
   Action alexs_single{Place{c(1, 0), int8_t{1}, int8_t{0}}};
   auto after_single = two_ships;
   after_single.apply_action(alexs_single);
   auto priv_alex_single = env.private_observation(
      nor::Player::alex, two_ships, alexs_single, after_single
   );
   ASSERT_TRUE(priv_alex_single.placed_cells[0].has_value());
   EXPECT_EQ(*priv_alex_single.placed_cells[0], c(1, 0));
   EXPECT_FALSE(priv_alex_single.placed_cells[1].has_value());

   // shots are fully public including the referee verdict; the alternation (player one first)
   // is respected, so every snapshot below is taken at a properly sequenced decision point
   State shooting{Config{1, 3, {{2u, 1.}}, 2, 2.}};
   // fleets: player one covers columns 0-1, player two columns 1-2
   shooting.apply_action(Action{Place{c(0, 0), int8_t{0}, int8_t{1}}});
   shooting.apply_action(Action{Place{c(0, 1), int8_t{0}, int8_t{1}}});

   auto pre_hit = shooting;
   pre_hit.apply_action(Action{Fire{c(0, 2)}});  // alex grazes bob's stern: hit, not sink
   auto hit_pub = env.public_observation(shooting, Action{Fire{c(0, 2)}}, pre_hit);
   EXPECT_EQ(hit_pub.actor, nor::Player::alex);
   EXPECT_EQ(*hit_pub.target, c(0, 2));
   EXPECT_EQ(hit_pub.result, Result::hit);

   auto mid = pre_hit;
   mid.apply_action(Action{Fire{c(0, 0)}});  // bob answers with a bow hit on alex's domino

   auto pre_sink = mid;
   pre_sink.apply_action(Action{Fire{c(0, 1)}});  // alex completes the destruction: sink!
   ASSERT_TRUE(pre_sink.terminal());
   auto sink_pub = env.public_observation(mid, Action{Fire{c(0, 1)}}, pre_sink);
   EXPECT_EQ(sink_pub.actor, nor::Player::alex);
   EXPECT_EQ(sink_pub.result, Result::sink);
}

TEST(BattleshipGsInformation, environment_histories_respect_information_sets)
{
   using namespace battleship_gs;
   using namespace nor;
   Environment env{canonical_config()};

   GameScript script{};
   script.config = canonical_config();
   script.fleet_one = {{c(0, 0)}};
   script.fleet_two = {{c(1, 0)}};
   script.shots_one = {c(1, 0)};
   script.shots_two = {c(0, 0)};
   auto final_state = play_script(script);
   ASSERT_TRUE(final_state.terminal());

   // open history reveals everything: 2 placements + player one's game-ending shot (player two
   // never fires -- his fleet is already sunk)
   auto open = env.open_history(final_state);
   ASSERT_EQ(open.size(), 3u);

   // the public history hides the placement payloads but keeps every event slot
   auto public_hist = env.public_history(final_state);
   ASSERT_EQ(public_hist.size(), 3u);
   for(size_t i = 0; i < 2; ++i) {
      EXPECT_FALSE(public_hist[i].value().has_value());
      EXPECT_TRUE(
         public_hist[i].player() == nor::Player::alex or public_hist[i].player() == nor::Player::bob
      );
   }

   // alex's private history contains only his own placement ...
   auto alex_hist = env.private_history(nor::Player::alex, final_state);
   ASSERT_EQ(alex_hist.size(), 3u);
   const auto* alex_place = std::get_if< Place >(std::get_if< Action >(&*alex_hist[0].value()));
   EXPECT_NE(alex_place, nullptr);
   EXPECT_FALSE(alex_hist[1].value().has_value());
   // ... while bob's private history contains only his own
   auto bob_hist = env.private_history(nor::Player::bob, final_state);
   ASSERT_EQ(bob_hist.size(), 3u);
   EXPECT_FALSE(bob_hist[0].value().has_value());
   const auto* bob_place = std::get_if< Place >(std::get_if< Action >(&*bob_hist[1].value()));
   EXPECT_NE(bob_place, nullptr);
   EXPECT_FALSE(bob_hist[2].value().has_value());
   // the revealed placements match the scripted fleets
   EXPECT_EQ(*alex_place, (Place{c(0, 0), int8_t{0}, int8_t{1}}));
   EXPECT_EQ(*bob_place, (Place{c(1, 0), int8_t{0}, int8_t{1}}));

   // mid-placement reconstruction: only alex has placed so far
   State midway{canonical_config()};
   midway.apply_action(Action{Place{c(2, 0), int8_t{0}, int8_t{1}}});
   auto open_midway = env.open_history(midway);
   ASSERT_EQ(open_midway.size(), 1u);
   EXPECT_EQ(open_midway[0].player(), nor::Player::alex);
}

// #####################################################################################################################
// deterministic-env trait checks
// #####################################################################################################################

static_assert(nor::concepts::deterministic_fosg< nor::games::battleship_gs::Environment >);

TEST(BattleshipGsTraits, deterministic_fosg_concepts)
{
   using Env = nor::games::battleship_gs::Environment;
   static_assert(std::same_as< typename Env::chance_outcome_type, std::monostate >);
   static_assert(std::same_as<
                 typename Env::action_variant_type,
                 std::variant< Action, std::monostate > >);
   EXPECT_TRUE((nor::concepts::fosg< Env >) );
   EXPECT_TRUE((nor::concepts::deterministic_env< Env >) );
   EXPECT_FALSE((nor::concepts::stochastic_env< Env >) );
   EXPECT_EQ(Env::stochasticity(), nor::Stochasticity::deterministic);
   // deterministic envs expose no chance interface
   EXPECT_FALSE(nor::concepts::has::method::chance_actions< Env >);
   EXPECT_FALSE(nor::concepts::has::method::chance_probability< Env >);
}

// #####################################################################################################################
// CFR convergence smoke (vanilla CFR on the canonical instance; general-sum-safe metric)
// #####################################################################################################################

namespace {

struct ConvergenceReport {
   double nash_conv_first_checkpoint = std::numeric_limits< double >::quiet_NaN();
   double nash_conv_final = std::numeric_limits< double >::quiet_NaN();
   size_t iterations = 0;
};

inline ConvergenceReport print_convergence_report(const ConvergenceReport& report)
{
   fmt::print(
      "[battleship_gs-cfr-baseline] iterations={} nash_conv_at_iter_50={:.6e} nash_conv_final={:"
      ".6e}\n",
      report.iterations,
      report.nash_conv_first_checkpoint,
      report.nash_conv_final
   );
   return report;
}

}  // namespace

TEST(BattleshipGsCFR, vanilla_alternating_canonical_instance_nash_conv_decreases)
{
   using namespace nor;
   using Env = games::battleship_gs::Environment;

   // METRIC NOTE (general-sum handling): nash_conv(..., constant_sum=false) is the sum of
   // per-player best-response improvements u_i(BR_i, pi_-i) - u_i(pi) and therefore well-defined
   // for general-sum games (cf. the shapley test); exploitability()'s zero-sum normalization is
   // deliberately NOT used here.
   //
   // CONVERGENCE NOTE. The uniform profile IS already an equilibrium of the canonical instance,
   // which is also regret-matching's initialization -- exactly like in Shapley's game, nash_conv
   // collapses to numerical zero within the first checkpoint(s), so a strict monotone-decrease
   // assertion is vacuous/impossible here. We assert (a) no increase beyond float noise from the
   // first checkpoint to the final one and (b) actual convergence to zero, and REPORT the full
   // checkpoint trace as the baseline for future EFCE-solver comparisons on this benchmark.
   auto config = canonical_config();
   Env env{config};
   auto root_state = std::make_unique< battleship_gs::State >(config);

   auto avg_policy = factory::make_tabular_policy(
      std::unordered_map< games::battleship_gs::Infostate, HashmapActionPolicy< Action > >{}
   );
   auto curr_policy = factory::make_tabular_policy(
      std::unordered_map< games::battleship_gs::Infostate, HashmapActionPolicy< Action > >{}
   );

   auto solver = factory::make_cfr< rm::CFRConfig{}, true >(
      std::move(env), std::move(root_state), curr_policy, avg_policy
   );
   Env expl_env{config};

   constexpr size_t kIterations = 300;
   constexpr size_t kCheckpoint = 50;

   ConvergenceReport report{};
   report.iterations = kIterations;
   for(size_t iter = 1; iter <= kIterations; ++iter) {
      solver.iterate(1);
      if(iter % kCheckpoint != 0) {
         continue;
      }
      const auto& avg_policies = solver.average_policy();
      using AvgTablePolicy = std::decay_t< decltype(avg_policies.at(nor::Player::alex)) >;
      double nc = nash_conv(
         expl_env,
         battleship_gs::State{config},
         player_hashmap< AvgTablePolicy >{
            std::pair{
               nor::Player::alex, normalize_state_policy(avg_policies.at(nor::Player::alex))},
            std::pair{nor::Player::bob, normalize_state_policy(avg_policies.at(nor::Player::bob))}},
         /*constant_sum=*/false
      );
      fmt::print("[battleship_gs-cfr-baseline] iter={} nash_conv={:.6e}\n", iter, nc);
      if(iter == kCheckpoint) {
         report.nash_conv_first_checkpoint = nc;
      }
      report.nash_conv_final = nc;
   }

   // (a) the profile never becomes MORE exploitable than at the first checkpoint ...
   EXPECT_LE(report.nash_conv_final, report.nash_conv_first_checkpoint + 1e-12);
   // ... and (b) it converges to the uniform equilibrium up to floating-point noise
   EXPECT_LT(report.nash_conv_final, 1e-9);
   print_convergence_report(report);
}
