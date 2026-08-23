
#include <gtest/gtest.h>

#include <algorithm>
#include <unordered_map>

#include "battleship/battleship.hpp"
#include "fixtures.hpp"
#include "nor/concepts.hpp"
#include "nor/exploitability.hpp"
#include "nor/nor.hpp"

using namespace battleship;
using namespace nor::games::battleship;

namespace {

constexpr Cell c(int row, int col)
{
   return Cell{int8_t(row), int8_t(col)};
}

/// the four legal placements of a size-2 ship on the 2x2 grid
constexpr std::array< Place, 4 > placements_2x2 = {
   Place{c(0, 0), c(0, 1)},
   Place{c(1, 0), c(1, 1)},
   Place{c(0, 0), c(1, 0)},
   Place{c(0, 1), c(1, 1)}};

}  // namespace

// #####################################################################################################################
// placement legality
// #####################################################################################################################

TEST_F(BattleshipState, placement_phase_starts_with_player_one)
{
   state = State{light_config()};
   EXPECT_EQ(state.phase(), Phase::one_placement);
   EXPECT_EQ(state.active_player(), Player::one);
   EXPECT_TRUE(state.placing());
   EXPECT_FALSE(state.firing());
}

TEST_F(BattleshipState, placements_alternate_secretly_ship_by_ship)
{
   state = State{classic_config(3, 2, /*max_shots=*/3, /*ships_per_fleet=*/2)};
   EXPECT_EQ(state.active_player(), Player::one);
   state.apply_action(placements_2x2[0]);
   EXPECT_EQ(state.active_player(), Player::two);
   state.apply_action(Place{c(1, 0), c(1, 1)});
   // the second ship of player one follows his first one
   EXPECT_EQ(state.active_player(), Player::one);
   state.apply_action(Place{c(2, 0), c(2, 1)});
   EXPECT_EQ(state.active_player(), Player::two);
   state.apply_action(Place{c(2, 0), c(2, 1)});
   EXPECT_EQ(state.phase(), Phase::one_fire);
   EXPECT_EQ(state.ships_placed(Player::one), 2u);
   EXPECT_EQ(state.ships_placed(Player::two), 2u);
}

TEST_F(BattleshipState, placement_rejects_off_grid_cells)
{
   state = State{light_config()};
   EXPECT_FALSE(state.is_valid(Action{Place{c(-1, 0), c(0, 0)}}));
   EXPECT_FALSE(state.is_valid(Action{Place{c(0, 0), c(0, -1)}}));
   EXPECT_FALSE(state.is_valid(Action{Place{c(0, 0), c(0, 2)}}));
   EXPECT_FALSE(state.is_valid(Action{Place{c(0, 0), c(2, 0)}}));
   EXPECT_THROW(state.apply_action(Action{Place{c(0, 0), c(0, 2)}}), std::invalid_argument);
}

TEST_F(BattleshipState, placement_requires_adjacent_orientation)
{
   state = State{light_config()};
   // diagonal cells are not adjacent
   EXPECT_FALSE(state.is_valid(Action{Place{c(0, 0), c(1, 1)}}));
   // a gap of one cell is not adjacent either
   EXPECT_FALSE(state.is_valid(Action{Place{c(0, 0), c(0, 2)}}));
   // horizontal and vertical adjacency is fine
   EXPECT_TRUE(state.is_valid(Action{Place{c(0, 0), c(0, 1)}}));
   EXPECT_TRUE(state.is_valid(Action{Place{c(0, 0), c(1, 0)}}));
}

TEST_F(BattleshipState, placement_forbids_overlap_within_own_fleet)
{
   state = State{classic_config(3, 2, 3, 2)};
   state.apply_action(Place{c(0, 0), c(0, 1)});
   state.apply_action(Place{c(1, 0), c(1, 1)});
   // overlapping any own ship is illegal ...
   EXPECT_FALSE(state.is_valid(Action{Place{c(0, 1), c(1, 1)}}));
   EXPECT_FALSE(state.is_valid(Action{Place{c(0, 0), c(0, 1)}}));
   // ... while a fresh disjoint pair of cells is accepted
   EXPECT_TRUE(state.is_valid(Action{Place{c(2, 0), c(2, 1)}}));
   // opposing fleets may overlap: both players legally place on the very same cells
   auto mirrored = State{classic_config(3, 2, 3, 2)};
   mirrored.apply_action(Place{c(0, 0), c(0, 1)});
   mirrored.apply_action(Place{c(0, 0), c(0, 1)});
   EXPECT_EQ(mirrored.fleet_cells(Player::two), mirrored.fleet_cells(Player::one));
}

TEST_F(BattleshipState, placement_action_enumeration)
{
   state = State{light_config()};
   auto legal = state.actions(Player::one);
   ASSERT_EQ(legal.size(), 4u);  // 2 horizontal + 2 vertical dominoes on the 2x2 grid
   for(const auto& action : legal) {
      EXPECT_TRUE(state.is_valid(action)) << common::to_string(action);
   }
   // every canonical placement appears exactly once
   for(const auto& expected : placements_2x2) {
      EXPECT_NE(std::find(legal.begin(), legal.end(), Action{expected}), legal.end())
         << common::to_string(expected);
   }
   // the 3x2 grid offers 7 placements: 3 horizontal rows and 4 vertical dominoes
   auto classic = State{classic_config(3, 2, 3, 2)};
   EXPECT_EQ(classic.actions(Player::one).size(), 7u);
   // no actions for the non-active player
   EXPECT_EQ(state.actions(Player::two), std::vector< Action >{});
}

TEST_F(BattleshipState, wrong_action_type_in_phase_is_rejected)
{
   state = State{light_config()};
   EXPECT_FALSE(state.is_valid(Action{Fire{c(0, 0)}}));
   EXPECT_THROW(state.apply_action(Action{Fire{c(0, 0)}}), std::invalid_argument);
   state.apply_action(placements_2x2[0]);
   state.apply_action(placements_2x2[1]);
   ASSERT_TRUE(state.firing());
   // placements are rejected once the firing phase has begun
   EXPECT_FALSE(state.is_valid(Action{Place{c(0, 0), c(0, 1)}}));
   EXPECT_THROW(state.apply_action(Action{Place{c(0, 0), c(0, 1)}}), std::invalid_argument);
}

// #####################################################################################################################
// shot resolution truth table
// #####################################################################################################################

TEST(BattleshipShotResolution, miss_hit_sink_transitions)
{
   using namespace battleship;
   auto config = light_config();
   State state{config};
   // fleets: player one on row zero, player two on row one
   state.apply_action(Place{c(0, 0), c(0, 1)});
   state.apply_action(Place{c(1, 0), c(1, 1)});
   ASSERT_EQ(state.phase(), Phase::one_fire);

   // player one misses (fires at his own ship's cell)
   state.apply_action(Fire{c(0, 0)});
   EXPECT_EQ(state.shots_used(Player::one), 1u);
   EXPECT_TRUE(state.was_fired_at(Player::one, c(0, 0)));
   EXPECT_FALSE(state.was_hit(Player::one, c(0, 0)));
   EXPECT_EQ(state.phase(), Phase::two_fire);
   EXPECT_FALSE(state.terminal());

   // player two hits the ship of player one
   state.apply_action(Fire{c(0, 1)});
   EXPECT_EQ(state.phase(), Phase::one_fire);
   EXPECT_TRUE(state.was_hit(Player::two, c(0, 1)));
   EXPECT_FALSE(state.ship_sunk(Player::one, 0));

   // player one hits back but does not sink yet
   state.apply_action(Fire{c(1, 0)});
   EXPECT_TRUE(state.was_hit(Player::one, c(1, 0)));
   EXPECT_FALSE(state.ship_sunk(Player::two, 0));
   EXPECT_FALSE(state.terminal());

   // player two sinks the ship of player one --> immediate game over
   state.apply_action(Fire{c(0, 0)});
   EXPECT_TRUE(state.ship_sunk(Player::one, 0));
   EXPECT_TRUE(state.fleet_sunk(Player::one));
   EXPECT_TRUE(state.terminal());
   EXPECT_DOUBLE_EQ(state.payoff(Player::one), -2.);
   EXPECT_DOUBLE_EQ(state.payoff(Player::two), 2.);
}

TEST(BattleshipShotResolution, sink_terminates_light_game_immediately)
{
   GameScript script{};
   script.config = light_config();
   script.fleet_one = {{c(0, 0), c(0, 1)}};
   script.fleet_two = {{c(1, 0), c(1, 1)}};
   // player one finds and sinks the opponent within his first two shots; player two's
   // remaining budget must never be used
   script.shots_one = {c(1, 0), c(1, 1), c(0, 0)};
   script.shots_two = {c(0, 0), c(0, 1)};
   auto final_state = play_script(script);
   EXPECT_TRUE(final_state.terminal());
   EXPECT_EQ(final_state.shots_used(Player::one), 2u);
   EXPECT_EQ(final_state.shots_used(Player::two), 1u);
   EXPECT_DOUBLE_EQ(final_state.payoff(Player::one), 2.);
   EXPECT_DOUBLE_EQ(final_state.payoff(Player::two), -2.);
}

TEST(BattleshipShotResolution, budget_exhaustion_ends_the_game)
{
   auto config = classic_config(3, 2, /*max_shots=*/2, /*ships_per_fleet=*/2);
   GameScript script{};
   script.config = config;
   script.fleet_one = {{c(0, 0), c(0, 1)}, {c(2, 0), c(2, 1)}};
   script.fleet_two = {{c(1, 0), c(1, 1)}, {c(0, 0), c(0, 1)}};
   // player one wastes his second shot on his own cell while player two sinks ship B
   script.shots_one = {c(1, 0), c(0, 0)};
   script.shots_two = {c(2, 0), c(2, 1)};
   auto final_state = play_script(script);
   // consuming the last shot of both budgets without total destruction ends the duel
   EXPECT_TRUE(final_state.terminal());
   EXPECT_EQ(final_state.shots_used(Player::one), 2u);
   EXPECT_EQ(final_state.shots_used(Player::two), 2u);
   EXPECT_DOUBLE_EQ(final_state.payoff(Player::one), -4.);
   EXPECT_DOUBLE_EQ(final_state.payoff(Player::two), 4.);

   // mid-way states with unspent shot budgets are not terminal
   State midway{config};
   midway.apply_action(Place{c(0, 0), c(0, 1)});
   midway.apply_action(Place{c(1, 0), c(1, 1)});
   midway.apply_action(Place{c(2, 0), c(2, 1)});
   midway.apply_action(Place{c(0, 0), c(0, 1)});
   midway.apply_action(Fire{c(1, 0)});
   EXPECT_FALSE(midway.terminal());
   EXPECT_EQ(midway.phase(), Phase::two_fire);
}

TEST(BattleshipShotResolution, repeated_shots_are_permitted_and_idempotent)
{
   GameScript script{};
   script.config = light_config();
   script.fleet_one = {{c(0, 0), c(0, 1)}};
   script.fleet_two = {{c(1, 0), c(1, 1)}};
   // firing at the same cell twice wastes a shot but stays legal; all shots here target the
   // shooter's own cells and therefore miss
   script.shots_one = {c(0, 0), c(0, 0), c(0, 1)};
   script.shots_two = {c(1, 0), c(1, 0), c(1, 1)};
   auto state = play_script(script);
   EXPECT_TRUE(state.terminal());
   EXPECT_EQ(state.shots_used(Player::one), 3u);
   EXPECT_EQ(state.shots_used(Player::two), 3u);
   // nobody ever hit anything
   EXPECT_DOUBLE_EQ(state.payoff(Player::one), 0.);
}

// #####################################################################################################################
// payoff scenarios
// #####################################################################################################################

TEST_P(BattleshipPayoffParamsF, payoff_table)
{
   const auto& [script, payoff_one, payoff_two] = GetParam();
   auto final_state = play_script(script);
   ASSERT_TRUE(final_state.terminal()) << common::to_string(final_state.phase());
   EXPECT_DOUBLE_EQ(final_state.payoff(Player::one), payoff_one);
   EXPECT_DOUBLE_EQ(final_state.payoff(Player::two), payoff_two);
}

INSTANTIATE_TEST_SUITE_P(
   BattleshipPayoffs,
   BattleshipPayoffParamsF,
   ::testing::Values(
      // light variant: sink win / get sunk loss
      std::make_tuple(
         GameScript{
            light_config(),
            {{c(0, 0), c(0, 1)}},
            {{c(1, 0), c(1, 1)}},
            {c(1, 0), c(1, 1)},
            {c(0, 0)}},
         2.,
         -2.
      ),
      std::make_tuple(
         GameScript{
            light_config(),
            {{c(0, 0), c(1, 0)}},
            {{c(0, 1), c(1, 1)}},
            {c(0, 0), c(1, 0)},
            {c(1, 0), c(0, 0)}},
         -2.,
         2.
      ),
      // light variant: mutual timeout with no sink --> draw
      std::make_tuple(
         GameScript{
            light_config(),
            {{c(0, 0), c(0, 1)}},
            {{c(1, 0), c(1, 1)}},
            {c(0, 0), c(0, 1), c(0, 0)},
            {c(1, 0), c(1, 1), c(1, 0)}},
         0.,
         0.
      ),
      // classic-lite: symmetric trade (each side loses one ship worth 4) --> draw
      std::make_tuple(
         GameScript{
            classic_config(3, 2, 2),
            {{c(0, 0), c(0, 1)}, {c(2, 0), c(2, 1)}},
            {{c(1, 0), c(1, 1)}, {c(0, 0), c(0, 1)}},
            {c(1, 0), c(1, 1)},
            {c(2, 0), c(2, 1)}},
         0.,
         0.
      ),
      // classic-lite: partial score -- player one sinks an enemy ship without losing one
      std::make_tuple(
         GameScript{
            classic_config(3, 2, 2),
            {{c(0, 0), c(0, 1)}, {c(2, 0), c(2, 1)}},
            {{c(1, 0), c(1, 1)}, {c(0, 0), c(0, 1)}},
            {c(1, 0), c(1, 1)},
            {c(1, 0), c(1, 0)}},
         4.,
         -4.
      ),
      // classic-lite: total fleet destruction ends the game before the budgets run out
      std::make_tuple(
         GameScript{
            classic_config(3, 2, 4),
            {{c(0, 0), c(0, 1)}, {c(2, 0), c(2, 1)}},
            {{c(1, 0), c(1, 1)}, {c(0, 0), c(0, 1)}},
            {c(1, 0), c(1, 1), c(0, 0), c(0, 1)},
            {c(1, 0), c(1, 0), c(1, 0)}},
         8.,
         -8.
      )
   )
);

TEST(BattleshipClassicInstance, four_by_three_grid_smoke)
{
   // the largest published instance family: Battleship(R) on a 4x3 grid; player one sinks one
   // enemy ship before both shot budgets run out
   auto config = classic_config(4, 3, /*max_shots=*/2, /*ships_per_fleet=*/2);
   GameScript script{};
   script.config = config;
   script.fleet_one = {{c(0, 0), c(0, 1)}, {c(3, 1), c(3, 2)}};
   script.fleet_two = {{c(1, 1), c(2, 1)}, {c(0, 2), c(1, 2)}};
   script.shots_one = {c(1, 1), c(2, 1)};
   script.shots_two = {c(0, 0), c(3, 1)};
   auto final_state = play_script(script);
   EXPECT_TRUE(final_state.terminal());
   EXPECT_EQ(final_state.shots_used(Player::one), 2u);
   EXPECT_EQ(final_state.shots_used(Player::two), 2u);
   EXPECT_DOUBLE_EQ(final_state.payoff(Player::one), 4.);
   EXPECT_DOUBLE_EQ(final_state.payoff(Player::two), -4.);
}

// #####################################################################################################################
// information correctness
// #####################################################################################################################

namespace {

/// replays `script` and accumulates the observation stream that `observer` receives into an
/// infostate + publicstate (mirroring exactly what the solvers' traversal does)
std::pair< nor::games::battleship::Infostate, nor::games::battleship::Publicstate > observed_states(
   const nor::games::battleship::Environment& env,
   const GameScript& script,
   nor::Player observer
)
{
   using namespace battleship;
   nor::games::battleship::Infostate infostate{observer};
   nor::games::battleship::Publicstate publicstate{};
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
         record(Place{script.fleet_one[ship][0], script.fleet_one[ship][1]});
      }
      if(ship < script.fleet_two.size()) {
         record(Place{script.fleet_two[ship][0], script.fleet_two[ship][1]});
      }
   }
   size_t next_shot_one = 0;
   size_t next_shot_two = 0;
   while(not state.terminal()) {
      if(state.phase() == Phase::one_fire) {
         if(next_shot_one >= script.shots_one.size()) {
            break;
         }
         record(Fire{script.shots_one[next_shot_one++]});
      } else if(state.phase() == Phase::two_fire) {
         if(next_shot_two >= script.shots_two.size()) {
            break;
         }
         record(Fire{script.shots_two[next_shot_two++]});
      } else {
         break;
      }
   }
   return {infostate, publicstate};
}

}  // namespace

TEST(BattleshipInformation, opponent_placement_not_inferable_from_observations)
{
   using namespace battleship;
   using namespace nor;
   Environment env{light_config()};

   // two games identical from alex's point of view except for bob's hidden fleet layout.
   // alex's shot sequence only targets cell (0,0) -- a miss against BOTH candidate layouts --
   // so the referee verdicts (and hence all observations) coincide; bob's own shots hit the
   // identical fleet of alex in both games
   GameScript script_a{};
   script_a.config = light_config();
   script_a.fleet_one = {{c(0, 0), c(0, 1)}};
   script_a.fleet_two = {{c(1, 0), c(1, 1)}};
   script_a.shots_one = {c(0, 0), c(0, 0), c(0, 0)};
   script_a.shots_two = {c(1, 0), c(1, 1), c(1, 0)};

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

TEST(BattleshipInformation, placement_secrecy_of_observation_functions)
{
   using namespace battleship;
   using namespace nor;
   Environment env{light_config()};
   State state{light_config()};
   state.apply_action(Place{c(0, 0), c(0, 1)});
   auto next = state;
   next.apply_action(Place{c(1, 0), c(1, 1)});
   Action bobs_placement = Place{c(1, 0), c(1, 1)};

   // the public observation carries no positional payload
   auto pub = env.public_observation(state, bobs_placement, next);
   EXPECT_EQ(pub.kind, Observation::Kind::hidden_placement);
   EXPECT_FALSE(pub.target.has_value());
   EXPECT_FALSE(pub.placed_cells[0].has_value());

   // alex learns nothing privately ...
   auto priv_alex = env.private_observation(nor::Player::alex, state, bobs_placement, next);
   EXPECT_EQ(priv_alex.kind, Observation::Kind::none);

   // ... while bob receives his own ship's confirmation
   auto priv_bob = env.private_observation(nor::Player::bob, state, bobs_placement, next);
   EXPECT_EQ(priv_bob.kind, Observation::Kind::hidden_placement);
   EXPECT_TRUE(priv_bob.placed_cells[0].has_value());
   EXPECT_EQ(*priv_bob.placed_cells[0], c(1, 0));

   // shots are fully public including the referee verdict
   State after_fire = next;
   after_fire.apply_action(Fire{c(1, 0)});
   auto fire_pub = env.public_observation(next, Action{Fire{c(1, 0)}}, after_fire);
   EXPECT_EQ(fire_pub.kind, Observation::Kind::shot);
   EXPECT_EQ(fire_pub.target, Cell(c(1, 0)));
   EXPECT_EQ(fire_pub.result, Result::hit);
}

TEST(BattleshipInformation, environment_histories_respect_information_sets)
{
   using namespace battleship;
   using namespace nor;
   Environment env{light_config()};

   GameScript script{};
   script.config = light_config();
   script.fleet_one = {{c(0, 0), c(0, 1)}};
   script.fleet_two = {{c(1, 0), c(1, 1)}};
   script.shots_one = {c(1, 0), c(1, 1)};
   script.shots_two = {c(0, 0)};
   auto final_state = play_script(script);

   // open history reveals everything: 2 placements + 3 shots
   auto open = env.open_history(final_state);
   ASSERT_EQ(open.size(), 5u);

   // the public history hides the placement payloads but keeps their count and all shots
   auto public_hist = env.public_history(final_state);
   ASSERT_EQ(public_hist.size(), 5u);
   for(size_t i = 0; i < 2; ++i) {
      EXPECT_FALSE(public_hist[i].value().has_value());
      EXPECT_TRUE(
         public_hist[i].player() == nor::Player::alex or public_hist[i].player() == nor::Player::bob
      );
   }

   // alex's private history contains only his own placement ...
   auto alex_hist = env.private_history(nor::Player::alex, final_state);
   ASSERT_EQ(alex_hist.size(), 5u);
   const auto* alex_place = std::get_if< Place >(std::get_if< Action >(&*alex_hist[0].value()));
   EXPECT_NE(alex_place, nullptr);
   EXPECT_FALSE(alex_hist[1].value().has_value());
   // ... while bob's private history contains only his own
   auto bob_hist = env.private_history(nor::Player::bob, final_state);
   ASSERT_EQ(bob_hist.size(), 5u);
   EXPECT_FALSE(bob_hist[0].value().has_value());
   const auto* bob_place = std::get_if< Place >(std::get_if< Action >(&*bob_hist[1].value()));
   EXPECT_NE(bob_place, nullptr);
   // the revealed placements match the scripted fleets
   EXPECT_EQ(*alex_place, (Place{c(0, 0), c(0, 1)}));
   EXPECT_EQ(*bob_place, (Place{c(1, 0), c(1, 1)}));
}

// #####################################################################################################################
// deterministic-env trait checks
// #####################################################################################################################

static_assert(nor::concepts::deterministic_fosg< nor::games::battleship::Environment >);

TEST(BattleshipTraits, deterministic_fosg_concepts)
{
   using Env = nor::games::battleship::Environment;
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
// CFR convergence smoke (baseline numbers for future PCFR+ comparison)
// #####################################################################################################################

namespace {

struct ConvergenceReport {
   double exploitability_first_checkpoint = std::numeric_limits< double >::quiet_NaN();
   double exploitability_final = std::numeric_limits< double >::quiet_NaN();
   size_t iterations = 0;
};

inline ConvergenceReport print_convergence_report(const ConvergenceReport& report)
{
   fmt::print(
      "[battleship-cfr-baseline] iterations={} expl_at_iter_50={:.6e} expl_final={:.6e}\n",
      report.iterations,
      report.exploitability_first_checkpoint,
      report.exploitability_final
   );
   return report;
}

}  // namespace

TEST(BattleshipCFR, vanilla_alternating_light_2x2_converges)
{
   using namespace nor;
   using Env = games::battleship::Environment;
   auto config = battleship::light_config(/*cols=*/2, /*max_shots=*/3);
   Env env{config};

   auto root_state = std::make_unique< battleship::State >(config);

   auto avg_policy = factory::make_tabular_policy(
      std::unordered_map< games::battleship::Infostate, HashmapActionPolicy< Action > >{}
   );
   auto curr_policy = factory::make_tabular_policy(
      std::unordered_map< games::battleship::Infostate, HashmapActionPolicy< Action > >{}
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
      double expl = exploitability(
         expl_env,
         games::battleship::State{config},
         player_hashmap< AvgTablePolicy >{
            std::pair{
               nor::Player::alex, normalize_state_policy(avg_policies.at(nor::Player::alex))},
            std::pair{nor::Player::bob, normalize_state_policy(avg_policies.at(nor::Player::bob))}}
      );
      fmt::print("[battleship-cfr-baseline] iter={} exploitability={:.6e}\n", iter, expl);
      if(iter == kCheckpoint) {
         report.exploitability_first_checkpoint = expl;
      }
      report.exploitability_final = expl;
   }

   // substantial convergence for VANILLA CFR: the exploitability strictly decreases at every
   // checkpoint and shrinks to a small fraction of its early-iteration value. (Baseline for
   // future PCFR+ comparison: vanilla CFR is known to converge slowly on this game -- PCFR+'s
   // predictions exploit exactly this structure, see Farina et al., AAAI 2021, App. G.)
   EXPECT_LT(report.exploitability_final, report.exploitability_first_checkpoint);
   EXPECT_LT(report.exploitability_final, 0.4 * report.exploitability_first_checkpoint);
   print_convergence_report(report);
}
