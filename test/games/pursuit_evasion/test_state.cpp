
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <queue>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

#include "fixtures.hpp"
#include "nor/concepts.hpp"
#include "nor/exploitability.hpp"
#include "nor/nor.hpp"

// NOTE: 'Player' stays unambiguously bound to ::pursuit_evasion::Player; nor's player enum is
// always spelled 'nor::Player' in this file.

// #####################################################################################################################
// graph transcription sanity (PCFR+ App. G, Fig. 3)
// #####################################################################################################################

TEST_F(PEGraph, attacker_adjacency_matches_the_figure)
{
   // the 14 black directed arrows of the tikz source, incl. the one-way middle-column merge
   EXPECT_TRUE(has_attacker_edge(node_S, node_B));
   EXPECT_TRUE(has_attacker_edge(node_S, node_C));
   EXPECT_TRUE(has_attacker_edge(node_S, node_D));
   EXPECT_TRUE(has_attacker_edge(node_B, node_E));
   EXPECT_TRUE(has_attacker_edge(node_C, node_F));
   EXPECT_TRUE(has_attacker_edge(node_D, node_G));
   EXPECT_TRUE(has_attacker_edge(node_E, node_F));
   EXPECT_TRUE(has_attacker_edge(node_G, node_F));
   EXPECT_TRUE(has_attacker_edge(node_E, node_H));
   EXPECT_TRUE(has_attacker_edge(node_F, node_I));
   EXPECT_TRUE(has_attacker_edge(node_G, node_J));
   EXPECT_TRUE(has_attacker_edge(node_H, node_K));
   EXPECT_TRUE(has_attacker_edge(node_I, node_L));
   EXPECT_TRUE(has_attacker_edge(node_J, node_M));

   // directedness: no back edges out of the deeper columns ...
   EXPECT_FALSE(has_attacker_edge(node_B, node_S));
   EXPECT_FALSE(has_attacker_edge(node_H, node_E));
   // ... the merge into F is strictly one-way ...
   EXPECT_FALSE(has_attacker_edge(node_F, node_E));
   EXPECT_FALSE(has_attacker_edge(node_F, node_G));
   // ... and lateral links within a column are patrol-only (grey dashed), never attacker edges
   EXPECT_FALSE(has_attacker_edge(node_B, node_C));
   EXPECT_FALSE(has_attacker_edge(node_C, node_D));
   EXPECT_EQ(k_attacker_edges.size(), 14u);
}

TEST_F(PEGraph, patrol_edges_symmetric_and_confined_to_their_areas)
{
   for(uint8_t a = 0; a < node_count; ++a) {
      for(uint8_t b = 0; b < node_count; ++b) {
         EXPECT_EQ(has_patrol_edge(a, b), has_patrol_edge(b, a))
            << "asymmetric patrol adjacency " << unsigned(a) << "-" << unsigned(b);
      }
   }
   // every grey dashed edge stays inside ONE patrol area
   for(const auto& edge : k_patrol_edges) {
      EXPECT_TRUE(
         in_patrol1(edge.from) && in_patrol1(edge.to)
         || in_patrol2(edge.from) && in_patrol2(edge.to)
      ) << "patrol edge crossing areas: "
        << unsigned(edge.from) << "-" << unsigned(edge.to);
   }
   EXPECT_EQ(k_patrol_edges.size(), 4u);
   // the areas are disjoint and exclude exits + the middle column
   for(uint8_t n = 0; n < node_count; ++n) {
      EXPECT_FALSE(in_patrol1(n) && in_patrol2(n));
      if(is_exit(n)) {
         EXPECT_FALSE(in_patrol1(n));
         EXPECT_FALSE(in_patrol2(n));
      }
   }
}

namespace {

/// shortest attacker-path length over the black directed edges (size_t max if unreachable)
size_t attacker_bfs_distance(uint8_t from, uint8_t to)
{
   std::array< size_t, node_count > dist{};
   dist.fill(std::numeric_limits< size_t >::max());
   std::queue< uint8_t > frontier;
   dist[from] = 0;
   frontier.push(from);
   while(not frontier.empty()) {
      const auto cur = frontier.front();
      frontier.pop();
      for(uint8_t nxt = 0; nxt < node_count; ++nxt) {
         if(has_attacker_edge(cur, nxt) && dist[nxt] == std::numeric_limits< size_t >::max()) {
            dist[nxt] = dist[cur] + 1;
            frontier.push(nxt);
         }
      }
   }
   return dist[to];
}

}  // namespace

TEST_F(PEGraph, exits_reachable_in_four_moves_and_terminal)
{
   // S -> col1 -> col2 -> col3 -> exit is exactly 4 steps for every exit
   for(const auto& exit : k_exit_nodes) {
      ASSERT_EQ(attacker_bfs_distance(node_S, exit), 4u)
         << "exit " << unsigned(exit) << " not reachable in exactly 4 moves";
   }
   // exits end the game: no outgoing attacker edges anywhere on the right rim
   for(const auto& exit : k_exit_nodes) {
      for(uint8_t to = 0; to < node_count; ++to) {
         EXPECT_FALSE(has_attacker_edge(exit, to)) << "exit has outgoing edge";
      }
   }
   // every interior node keeps at least one outgoing move (so waiting is optional, never forced)
   for(uint8_t n = 0; n < node_count; ++n) {
      if(is_exit(n)) {
         continue;
      }
      bool has_out = false;
      for(uint8_t to = 0; to < node_count; ++to) {
         has_out = has_out || has_attacker_edge(n, to);
      }
      EXPECT_TRUE(has_out) << "interior node without outgoing edge: " << unsigned(n);
   }
}

// #####################################################################################################################
// world state basics
// #####################################################################################################################

TEST_F(PEState, initial_layout_matches_the_transcription_assumptions)
{
   EXPECT_EQ(state.attacker_node(), node_S);
   EXPECT_EQ(state.patrol_nodes()[0], node_C);
   EXPECT_EQ(state.patrol_nodes()[1], node_I);
   EXPECT_EQ(state.trace_mask(), 0u);
   EXPECT_EQ(state.round(), 0u);
   EXPECT_EQ(state.phase(), Phase::commit_attacker);
   EXPECT_EQ(state.active_player(), Player::one);
   EXPECT_FALSE(state.terminal());
   EXPECT_EQ(state.terminal_cause(), TerminalCause::none);
   EXPECT_DOUBLE_EQ(state.payoff(Player::one), 0.);
   EXPECT_DOUBLE_EQ(state.payoff(Player::two), 0.);

   auto copy = state;
   EXPECT_TRUE(copy == state);
   EXPECT_FALSE(copy != state);

   // config validation bounds
   EXPECT_THROW(Config{0}, std::invalid_argument);
   EXPECT_THROW(Config{Config::max_rounds + 1}, std::invalid_argument);
   EXPECT_NO_THROW(Config{6});
}

TEST_F(PEState, action_enumeration_counts_are_figure_faithful)
{
   // attacker at S: three outgoing edges plus wait
   auto att_actions = state.actions(Player::one);
   EXPECT_EQ(att_actions.size(), 4u);
   EXPECT_NE(
      std::find(att_actions.begin(), att_actions.end(), pe::Action{att_wait()}), att_actions.end()
   );

   // defender compound enumeration: 3 stay-or-step options per patrol = 9 combos
   state.apply_action(att_edge(node_S, node_B));
   ASSERT_EQ(state.phase(), Phase::commit_defender);
   auto def_actions = state.actions(Player::two);
   EXPECT_EQ(def_actions.size(), 9u);
   for(const auto& action : def_actions) {
      EXPECT_TRUE(state.is_valid(action));
   }
   EXPECT_EQ(state.actions(Player::one).size(), 0u);  // not his phase anymore
   EXPECT_EQ(State{Config{6}}.actions(Player::two).size(), 0u);  // not her phase yet either
}

// #####################################################################################################################
// simultaneity encoding: commit-commit-resolve with hidden commitments
// #####################################################################################################################

TEST_F(PESimultaneity, commit_phases_enforce_hidden_commitments)
{
   pe::Environment env{};
   ASSERT_EQ(state.phase(), Phase::commit_attacker);

   // the wrong side can neither query as legal nor commit during CommitA
   EXPECT_FALSE(state.is_valid(pe::Action{def_stay()}));
   EXPECT_THROW(state.apply_action(def_stay()), std::invalid_argument);

   // CommitA stores the attacker's move WITHOUT touching any position
   auto pre = state;
   state.apply_action(att_edge(node_S, node_B));
   EXPECT_EQ(state.attacker_node(), node_S);  // unresolved yet
   EXPECT_EQ(state.patrol_nodes()[0], node_C);
   EXPECT_EQ(state.patrol_nodes()[1], node_I);
   EXPECT_EQ(state.phase(), Phase::commit_defender);
   EXPECT_EQ(state.active_player(), Player::two);
   ASSERT_TRUE(state.committed_attacker_move().has_value());
   EXPECT_FALSE(state.committed_defender_move().has_value());

   // observation stream of the CommitA transition: public commitment event without payload,
   // private echo only for alex, nothing leaks to bob
   auto next = state;
   auto pub = env.public_observation(pre, pe::Action{att_edge(node_S, node_B)}, next);
   EXPECT_EQ(pub.committed_by, nor::Player::alex);
   EXPECT_FALSE(pub.own_att_move.has_value());
   EXPECT_FALSE(pub.terminal_cause.has_value());
   auto priv_bob = env.private_observation(
      nor::Player::bob, pre, pe::Action{att_edge(node_S, node_B)}, next
   );
   EXPECT_EQ(priv_bob, pe::Observation{});
   auto priv_alex = env.private_observation(
      nor::Player::alex, pre, pe::Action{att_edge(node_S, node_B)}, next
   );
   ASSERT_TRUE(priv_alex.own_att_move.has_value());
   EXPECT_EQ(*priv_alex.own_att_move, att_edge(node_S, node_B));

   // during CommitD the attacker cannot commit again
   EXPECT_THROW(state.apply_action(att_edge(node_S, node_C)), std::invalid_argument);
   EXPECT_THROW(env.transition(next, att_edge(node_S, node_C)), std::invalid_argument);

   // defender actions are enumerated from the UNCHANGED pre-resolve positions
   EXPECT_EQ(state.actions(Player::two).size(), 9u);

   // CommitD resolves both commitments simultaneously
   state.apply_action(def_to(node_D, node_I));
   EXPECT_EQ(state.attacker_node(), node_B);  // attacker commitment landed now
   EXPECT_EQ(state.patrol_nodes()[0], node_D);
   EXPECT_EQ(state.patrol_nodes()[1], node_I);
   EXPECT_EQ(state.round(), 1u);
   EXPECT_EQ(state.phase(), Phase::commit_attacker);
   EXPECT_EQ(state.active_player(), Player::one);
   EXPECT_FALSE(state.committed_attacker_move().has_value());
   EXPECT_FALSE(state.committed_defender_move().has_value());
}

TEST_F(PESimultaneity, resolve_applies_joint_outcome_without_shadowing)
{
   // crossing pattern: the attacker slips onto the node the patrol just vacated -- post-move
   // co-location decides, so nobody shadows anybody and nobody is captured
   state.apply_action(att_edge(node_S, node_C));
   state.apply_action(def_to(node_D, node_I));  // p1 leaves C as the attacker arrives there
   EXPECT_FALSE(state.terminal());
   EXPECT_EQ(state.attacker_node(), node_C);
   EXPECT_EQ(state.patrol_nodes()[0], node_D);

   // head-on collision: attacker walks onto a staying patrol -> immediate capture terminal
   State head_on{Config{4}};
   head_on.apply_action(att_edge(node_S, node_C));
   head_on.apply_action(def_stay());  // p1 holds C
   EXPECT_TRUE(head_on.terminal());
   EXPECT_EQ(head_on.terminal_cause(), TerminalCause::capture);
}

// #####################################################################################################################
// payoff truth table: capture -1/+1 | escapes {5,10,3} | timeout 0/0
// #####################################################################################################################

TEST_P(PEPayoffParamsF, scripted_payoffs_follow_the_truth_table)
{
   const auto& [script, expected_att, expected_def] = GetParam();
   auto final_state = play_script(script);
   ASSERT_TRUE(final_state.terminal());
   EXPECT_DOUBLE_EQ(final_state.payoff(Player::one), expected_att);
   EXPECT_DOUBLE_EQ(final_state.payoff(Player::two), expected_def);
   EXPECT_DOUBLE_EQ(final_state.payoff(Player::one) + final_state.payoff(Player::two), 0.);
}

INSTANTIATE_TEST_SUITE_P(
   PETruthTable,
   PEPayoffParamsF,
   ::testing::Values(
      // attacker runs into the standing centre patrol of P1 on round one
      std::make_tuple(PEScript{Config{4}, {{att_edge(node_S, node_C), def_stay()}}}, -1., 1.),
      // clean top lane: S-B-E-H-K escapes through exit K for 5 while patrols idle off-lane
      std::make_tuple(
         PEScript{
            Config{4},
            {{att_edge(node_S, node_B), def_stay()},
             {att_edge(node_B, node_E), def_stay()},
             {att_edge(node_E, node_H), def_stay()},
             {att_edge(node_H, node_K), def_stay()}}},
         5.,
         -5.
      ),
      // centre lane for the 10 exit: both patrols must vacate their centre start nodes in time
      std::make_tuple(
         PEScript{
            Config{4},
            {{att_edge(node_S, node_C), def_to(node_B, node_I)},
             {att_edge(node_C, node_F), def_to(node_B, node_I)},
             {att_edge(node_F, node_I), def_to(node_B, node_H)},
             {att_edge(node_I, node_L), def_to(node_B, node_H)}}},
         10.,
         -10.
      ),
      // bottom lane for the 3 exit needs no patrol cooperation at all
      std::make_tuple(
         PEScript{
            Config{4},
            {{att_edge(node_S, node_D), def_stay()},
             {att_edge(node_D, node_G), def_stay()},
             {att_edge(node_G, node_J), def_stay()},
             {att_edge(node_J, node_M), def_stay()}}},
         3.,
         -3.
      ),
      // everyone idles: m=2 rounds elapse -> timeout draw
      std::make_tuple(
         PEScript{Config{2}, {{att_wait(), def_stay()}, {att_wait(), def_stay()}}},
         0.,
         0.
      )
   )
);

// #####################################################################################################################
// trace lifecycle: deposit unless wait, sighting exactly on entering a traced node, clear on wait
// #####################################################################################################################

namespace {

/// replays `script` through the environment while accumulating observer's infostate/publicstate
inline std::pair< pe::Infostate, pe::Publicstate >
observed_run(const PEScript& script, nor::Player observer)
{
   pe::Environment env{script.config};
   pe::Infostate istate{observer};
   pe::Publicstate pubstate{};
   State s{script.config};
   for(const auto& [att, def] : script.rounds) {
      if(s.terminal()) {
         break;
      }
      auto step = [&](const auto& action) {
         auto pre = s;
         env.transition(s, action);
         auto pub = env.public_observation(pre, pe::Action{action}, s);
         auto priv = env.private_observation(observer, pre, pe::Action{action}, s);
         pubstate.update(pub);
         istate.update(pub, priv);
      };
      step(att);
      step(def);
   }
   return {std::move(istate), std::move(pubstate)};
}

}  // namespace

TEST_F(PETraces, deposit_sighting_and_clear_lifecycle)
{
   // r1: attacker enters B (trace@B) | r2: attacker advances B->E, patrol-1 ENTERS traced B ->
   // sighting | r3: attacker waits (clears ALL traces), patrol re-enters C -> no signal |
   // r4: patrol re-enters the cleaned B -> STILL no signal
   const PEScript script{
      Config{6},
      {{att_edge(node_S, node_B), def_stay()},
       {att_edge(node_B, node_E), def_to(node_B, node_I)},
       {att_wait(), def_to(node_C, node_I)},
       {att_wait(), def_to(node_B, node_I)}}};

   State s{script.config};
   const uint16_t bit_B = trace_bit(node_B);
   const uint16_t bit_E = trace_bit(node_E);

   s.apply_action(script.rounds[0].first);
   s.apply_action(script.rounds[0].second);
   EXPECT_EQ(s.trace_mask(), bit_B);  // arrival deposits; the spawn on S left no trace
   EXPECT_FALSE(s.terminal());

   s.apply_action(script.rounds[1].first);
   s.apply_action(script.rounds[1].second);
   EXPECT_EQ(s.trace_mask(), bit_B | bit_E);  // old traces persist until cleaned

   s.apply_action(script.rounds[2].first);
   s.apply_action(script.rounds[2].second);
   EXPECT_EQ(s.trace_mask(), 0u);  // waiting cleans every trace including the waiter's own node

   s.apply_action(script.rounds[3].first);
   s.apply_action(script.rounds[3].second);
   EXPECT_EQ(s.trace_mask(), 0u);
   EXPECT_FALSE(s.terminal());

   // observation-level truth: the defender sees EXACTLY one sighting, at node B, in round two;
   // the second visit after the cleaning raises nothing
   auto [bob_info, bob_pub] = observed_run(script, nor::Player::bob);
   size_t sight1_count = 0;
   uint8_t sighted_node = 255;
   for(const auto& [pub, priv] : bob_info.history()) {
      (void) pub;
      if(priv.sighting_p1.has_value()) {
         ++sight1_count;
         sighted_node = *priv.sighting_p1;
      }
      EXPECT_FALSE(priv.sighting_p2.has_value());  // patrol-2 never moved onto anything traced
   }
   EXPECT_EQ(sight1_count, 1u);
   EXPECT_EQ(sighted_node, node_B);

   // the attacker's stream carries his positions/wait flags but never patrol or sighting data
   auto [alex_info, alex_pub] = observed_run(script, nor::Player::alex);
   static constexpr std::array< uint8_t, 4 > k_expected_positions{node_B, node_E, node_E, node_E};
   static constexpr std::array< bool, 4 > k_expected_waited{false, false, true, true};
   size_t pos_idx = 0;
   for(const auto& [pub, priv] : alex_info.history()) {
      (void) pub;
      EXPECT_FALSE(priv.sighting_p1.has_value());
      EXPECT_FALSE(priv.sighting_p2.has_value());
      EXPECT_FALSE(priv.patrol_positions.has_value());
      if(priv.own_position.has_value()) {
         ASSERT_LT(pos_idx, 4u);
         EXPECT_EQ(*priv.own_position, k_expected_positions[pos_idx]);
         EXPECT_EQ(priv.waited, k_expected_waited[pos_idx]);
         ++pos_idx;
      }
   }
   EXPECT_EQ(pos_idx, 4u);
   // and the public channel stayed silent throughout (no terminal in this script)
   for(const auto& obs : bob_pub.history()) {
      EXPECT_FALSE(obs.terminal_cause.has_value());
   }
}

// #####################################################################################################################
// information correctness: hidden attacker paths invisible to the defender
// #####################################################################################################################

TEST(PEInformation, defender_view_identical_under_divergent_hidden_paths)
{
   // world A dashes along the top lane and escapes via K(5); world B along the bottom lane via
   // M(3). The defender plays the same idle compounds in both worlds and never moves, hence never
   // sights -- her view must be unable to tell the worlds apart even though the payoffs differ.
   const PEScript top_world{
      Config{4},
      {{att_edge(node_S, node_B), def_stay()},
       {att_edge(node_B, node_E), def_stay()},
       {att_edge(node_E, node_H), def_stay()},
       {att_edge(node_H, node_K), def_stay()}}};
   const PEScript bottom_world{
      Config{4},
      {{att_edge(node_S, node_D), def_stay()},
       {att_edge(node_D, node_G), def_stay()},
       {att_edge(node_G, node_J), def_stay()},
       {att_edge(node_J, node_M), def_stay()}}};

   auto [bob_top, pub_top] = observed_run(top_world, nor::Player::bob);
   auto [bob_bottom, pub_bottom] = observed_run(bottom_world, nor::Player::bob);
   EXPECT_EQ(bob_top, bob_bottom);
   EXPECT_EQ(bob_top.hash(), bob_bottom.hash());
   std::unordered_set< pe::Infostate > bob_set;
   bob_set.emplace(bob_top);
   bob_set.emplace(bob_bottom);
   EXPECT_EQ(bob_set.size(), 1u);

   EXPECT_EQ(pub_top, pub_bottom);
   EXPECT_EQ(pub_top.hash(), pub_bottom.hash());

   // the attacker himself distinguishes the worlds through his own positions
   auto [alex_top, alex_pub_top] = observed_run(top_world, nor::Player::alex);
   auto [alex_bottom, alex_pub_bottom] = observed_run(bottom_world, nor::Player::alex);
   EXPECT_NE(alex_top, alex_bottom);
   EXPECT_NE(alex_top.hash(), alex_bottom.hash());
   std::unordered_set< pe::Infostate > alex_set;
   alex_set.emplace(alex_top);
   alex_set.emplace(alex_bottom);
   EXPECT_EQ(alex_set.size(), 2u);
}

// #####################################################################################################################
// zero-sum invariant over random playouts
// #####################################################################################################################

TEST_F(PERandomPlayouts, always_terminal_zero_sum_with_consistent_causes)
{
   for(size_t m : {size_t(4), size_t(5), size_t(6)}) {
      SCOPED_TRACE(::testing::Message() << "m=" << m);
      pe::Environment env{Config{m}};
      for(unsigned seed = 0; seed < 60; ++seed) {
         State s{Config{m}};
         std::mt19937 rng{9000 + 7 * seed + m};
         const auto cause = random_pe_playout(s, rng);
         ASSERT_TRUE(s.terminal());
         const auto [u_a, u_d] = s.payoffs();
         EXPECT_NEAR(u_a + u_d, 0., 1e-12);
         switch(cause) {
            case TerminalCause::capture:
               EXPECT_DOUBLE_EQ(u_a, -1.);
               EXPECT_DOUBLE_EQ(u_d, 1.);
               break;
            case TerminalCause::escape: {
               EXPECT_TRUE(is_exit(s.attacker_node()));
               const double value = k_exit_payoffs[exit_index(s.attacker_node())];
               EXPECT_DOUBLE_EQ(u_a, value);
               EXPECT_DOUBLE_EQ(u_d, -value);
               break;
            }
            case TerminalCause::timeout:
               EXPECT_EQ(s.round(), m);
               EXPECT_DOUBLE_EQ(u_a, 0.);
               EXPECT_DOUBLE_EQ(u_d, 0.);
               break;
            case TerminalCause::none: ADD_FAILURE() << "non-terminal cause after playout";
         }
         EXPECT_LE(s.round(), m);
         // the FOSG adapter's rewards agree with the world-state payoffs
         EXPECT_DOUBLE_EQ(env.reward(nor::Player::alex, s), u_a);
         EXPECT_DOUBLE_EQ(env.reward(nor::Player::bob, s), u_d);
      }
   }
}

// #####################################################################################################################
// deterministic-env trait checks
// #####################################################################################################################

static_assert(nor::concepts::deterministic_fosg< pe::Environment >);

TEST_F(PETraits, deterministic_fosg_concepts)
{
   using Env = pe::Environment;
   static_assert(std::same_as< typename Env::chance_outcome_type, std::monostate >);
   // NOTE: the generator wraps (not flattens) the compound action variant
   static_assert(std::same_as<
                 typename Env::action_variant_type,
                 std::variant< std::variant< AttMove, DefMove >, std::monostate > >);
   EXPECT_TRUE((nor::concepts::fosg< Env >) );
   EXPECT_TRUE((nor::concepts::deterministic_env< Env >) );
   EXPECT_FALSE((nor::concepts::stochastic_env< Env >) );
   EXPECT_EQ(Env::stochasticity(), nor::Stochasticity::deterministic);
   EXPECT_FALSE(nor::concepts::has::method::chance_actions< Env >);
   EXPECT_FALSE(nor::concepts::has::method::chance_probability< Env >);
}

// #####################################################################################################################
// CFR convergence smoke: vanilla alternating CFR baseline numbers (PCFR+ comparison comes later)
// #####################################################################################################################

namespace {

struct PECFRConvergenceReport {
   double exploitability_first_checkpoint = std::numeric_limits< double >::quiet_NaN();
   double exploitability_final = std::numeric_limits< double >::quiet_NaN();
   size_t iterations = 0;
};

}  // namespace

TEST_F(PECFR, vanilla_alternating_m4_exploitability_decreases)
{
   using namespace nor;
   using Env = games::pursuit_evasion::Environment;

   auto config = Config{4};  // paper's smallest instance (search game with 4 turns)
   Env env{config};
   auto root_state = std::make_unique< games::pursuit_evasion::State >(config);

   auto avg_policy = factory::make_tabular_policy(
      std::unordered_map< games::pursuit_evasion::Infostate, HashmapActionPolicy< pe::Action > >{}
   );
   auto curr_policy = factory::make_tabular_policy(
      std::unordered_map< games::pursuit_evasion::Infostate, HashmapActionPolicy< pe::Action > >{}
   );

   auto
      solver = factory::make_cfr< rm::CFRConfig{.update_mode = rm::UpdateMode::alternating}, true >(
         std::move(env), std::move(root_state), curr_policy, avg_policy
      );
   Env expl_env{config};

   constexpr size_t kIterations = 300;
   constexpr size_t kCheckpoint = 50;

   PECFRConvergenceReport report{};
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
         games::pursuit_evasion::State{config},
         player_hashmap< AvgTablePolicy >{
            std::pair{
               nor::Player::alex, normalize_state_policy(avg_policies.at(nor::Player::alex))},
            std::pair{nor::Player::bob, normalize_state_policy(avg_policies.at(nor::Player::bob))}},
         /*constant_sum=*/true
      );
      fmt::print("[pe-cfr-baseline] iter={} exploitability={:.6e}\n", iter, expl);
      if(iter == kCheckpoint) {
         report.exploitability_first_checkpoint = expl;
      }
      report.exploitability_final = expl;
   }

   fmt::print(
      "[pe-cfr-baseline] summary iterations={} expl_at_iter_{}={:.6e} expl_final={:.6e}\n",
      report.iterations,
      kCheckpoint,
      report.exploitability_first_checkpoint,
      report.exploitability_final
   );

   EXPECT_LT(report.exploitability_final, report.exploitability_first_checkpoint);
}
