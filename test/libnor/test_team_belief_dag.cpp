#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <ranges>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "nor/nor.hpp"
#include "nor/rm/team/dag_cfr.hpp"
#include "nor/rm/team/team_belief_dag.hpp"
#include "nor/utils/utils.hpp"
#include "three_player_goofspiel/environment.hpp"

// NOTE: this suite covers the Team-Belief-DAG subsystem (Zhang, Farina & Sandholm, ICML 2022):
//   1. three-player-goofspiel environment mechanics,
//   2. TB-DAG construction correctness -- hand-built structural expectations PLUS an
//      independently coded, definition-faithful oracle over the enumerated world tree,
//   3. DAG size accounting,
//   4. DAG-CFR flow and evaluation invariants,
//   5. decentralization invariants of the extracted policies / coordinator program.

using namespace nor;
namespace tpg = games::three_player_goofspiel;

using EnvT = tpg::Environment;

/// game-domain seats mirroring the nor seats used throughout this suite
constexpr tpg::Player seat_alex = static_cast< tpg::Player >(Player::alex);
constexpr tpg::Player seat_bob = static_cast< tpg::Player >(Player::bob);
constexpr tpg::Player seat_cedric = static_cast< tpg::Player >(Player::cedric);

/////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// environment mechanics //////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(TeamGoofspielEnv, transitions_payoffs_and_round_outcomes)
{
   const tpg::GoofspielConfig cfg{.deck_size = 3};
   auto s = EnvT(cfg).initial_world_state();

   ASSERT_EQ(s.active_player(), tpg::Player::chance);
   const auto prizes = s.chance_actions();
   ASSERT_EQ(prizes.size(), size_t{3});
   for(const auto& p : prizes) {
      EXPECT_DOUBLE_EQ(s.chance_probability(p), 1. / 3.);
      EXPECT_TRUE(s.is_valid(p));
   }

   // round 1: highest unique bidder wins
   s.apply_action(tpg::ChanceOutcome{.kind = tpg::ChanceOutcome::Kind::prize, .value = 3});
   ASSERT_EQ(s.active_player(), seat_alex);
   s.apply_action(tpg::Bid{.card = 2});
   ASSERT_EQ(s.active_player(), seat_bob);
   s.apply_action(tpg::Bid{.card = 1});
   ASSERT_EQ(s.active_player(), seat_cedric);
   s.apply_action(tpg::Bid{.card = 1});
   ASSERT_TRUE(s.is_valid(tpg::ChanceOutcome{.kind = tpg::ChanceOutcome::Kind::confirm}));
   s.apply_action(tpg::ChanceOutcome{.kind = tpg::ChanceOutcome::Kind::confirm});
   EXPECT_EQ(s.round(), size_t{1});
   EXPECT_EQ(s.rounds()[0].outcome, tpg::RoundWinner::alex);
   EXPECT_EQ(s.score(seat_alex), 3);

   // round 2: bob uniquely takes prize 2 (cedric already spent his 1-card)
   s.apply_action(tpg::ChanceOutcome{.kind = tpg::ChanceOutcome::Kind::prize, .value = 2});
   s.apply_action(tpg::Bid{.card = 1});
   s.apply_action(tpg::Bid{.card = 3});
   s.apply_action(tpg::Bid{.card = 2});
   s.apply_action(tpg::ChanceOutcome{.kind = tpg::ChanceOutcome::Kind::confirm});
   EXPECT_EQ(s.rounds()[1].outcome, tpg::RoundWinner::bob);

   // round 3: alex and cedric tie at the max -> prize discarded
   s.apply_action(tpg::ChanceOutcome{.kind = tpg::ChanceOutcome::Kind::prize, .value = 1});
   s.apply_action(tpg::Bid{.card = 3});
   s.apply_action(tpg::Bid{.card = 2});
   s.apply_action(tpg::Bid{.card = 3});
   s.apply_action(tpg::ChanceOutcome{.kind = tpg::ChanceOutcome::Kind::confirm});
   ASSERT_TRUE(s.is_terminal());
   EXPECT_EQ(s.rounds()[2].outcome, tpg::RoundWinner::tie);

   // zero-sum accounting with the equal-split payoff rule
   const double team_score = double(s.score(seat_alex) + s.score(seat_bob));
   const double opp_score = double(s.score(seat_cedric));
   double sum_rewards = 0.;
   for(const auto seat : {tpg::Player::alex, tpg::Player::bob, tpg::Player::cedric}) {
      sum_rewards += s.payoff(seat);
   }
   EXPECT_NEAR(sum_rewards, 0., 1e-12);
   const double team = team_score;
   const double opp = opp_score;
   EXPECT_DOUBLE_EQ(s.payoff(seat_alex), 0.5 * (team - opp));
   EXPECT_DOUBLE_EQ(s.payoff(seat_bob), 0.5 * (team - opp));
   EXPECT_DOUBLE_EQ(s.payoff(seat_cedric), opp - team);
}

TEST(TeamGoofspielEnv, split_half_deal_enumerates_all_half_masks)
{
   const tpg::GoofspielConfig cfg{.deck_size = 4, .split_half_deal = true};
   auto s = EnvT(cfg).initial_world_state();
   ASSERT_EQ(s.phase(), tpg::Phase::deal);

   const auto deals = s.chance_actions();
   // all C(4,2)=6 halves that alex might receive are legal deals
   ASSERT_EQ(deals.size(), size_t{6});
   std::set< uint16_t > masks;
   for(const auto& d : deals) {
      EXPECT_EQ(d.kind, tpg::ChanceOutcome::Kind::deal);
      masks.insert(d.mask_a);
      EXPECT_DOUBLE_EQ(s.chance_probability(d), 1. / 6.);
      // every dealt half is disjoint from bob's complemented half inside one full deck
      uint16_t full = 0;
      for(size_t v = 1; v <= cfg.deck_size; ++v) {
         full |= tpg::card_bit(uint8_t(v));
      }
      EXPECT_TRUE((d.mask_a & ~full) == 0);
   }
   EXPECT_EQ(masks.size(), size_t{6});

   // the first deal shapes private hands immediately
   s.apply_action(deals.front());
   const auto [half_a, half_b] = s.dealt_halves();
   uint16_t full = 0;
   for(size_t v = 1; v <= cfg.deck_size; ++v) {
      full |= tpg::card_bit(uint8_t(v));
   }
   EXPECT_EQ(half_a | half_b, full);
   EXPECT_EQ(half_a & half_b, uint16_t{0});
   EXPECT_EQ(s.hand_mask(seat_alex), half_a);
   EXPECT_EQ(s.hand_mask(seat_bob), half_b);
   EXPECT_EQ(s.hand_mask(seat_cedric), full);
   EXPECT_EQ(s.phase(), tpg::Phase::prize_reveal);
}

TEST(TeamGoofspielEnv, identical_mode_has_no_deal_phase)
{
   const tpg::GoofspielConfig cfg{.deck_size = 3};
   auto s = EnvT(cfg).initial_world_state();
   EXPECT_EQ(s.phase(), tpg::Phase::prize_reveal);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////// independent definition-faithful oracle of the TB-DAG ///////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

using ObsPair = std::pair< tpg::Observation, tpg::Observation >;

/**
 * An INDEPENDENT re-derivation of the team decision problem semantics, deliberately coded
 * without touching the library's builder internals:
 *  - world-tree traversal on fresh state clones (utils::child_state-style discipline),
 *  - per-member visible streams reconstructed straight from the FOSG primitive
 *    observation calls along each edge,
 *  - connectivity classes via the literal Definition-4.1 predicate over subtree-shared
 *    team infosets, computed bottom-up.
 */
struct OracleWorld {
   size_t depth = 0;
   bool terminal = false;
   /// the deciding team member's global infoset id iff this is a team-decision node
   size_t infoset_id = std::numeric_limits< size_t >::max();
   /// sorted ascending global infoset ids reachable in this node's subtree
   std::vector< size_t > anc;
   std::vector< size_t > children;
};

struct Oracle {
   const EnvT* env = nullptr;
   std::vector< Player > members{};
   std::vector< OracleWorld > worlds{};

   /// per-member stream logs and their ordinal registries
   std::vector< std::vector< ObsPair > > streams{};
   std::vector< std::unordered_map< std::string, size_t > > stream_registry{};

   std::vector< size_t > infoset_owner_counter{};

   void enumerate(const EnvT& environment)
   {
      env = &environment;
      members = {Player::alex, Player::bob};
      streams.assign(members.size(), {});
      stream_registry.assign(members.size(), {});
      auto root = env->initial_world_state();
      visit(root, /*depth=*/0, /*parent=*/std::numeric_limits< size_t >::max());
      // bottom-up anc reduction
      for(size_t id = worlds.size(); id-- > 0;) {
         for(auto child : worlds[id].children) {
            worlds[id].anc.insert(
               worlds[id].anc.end(), worlds[child].anc.begin(), worlds[child].anc.end()
            );
         }
         if(worlds[id].infoset_id != npos_v) {
            worlds[id].anc.emplace_back(worlds[id].infoset_id);
         }
         std::ranges::sort(worlds[id].anc);
         worlds[id].anc.erase(std::ranges::unique(worlds[id].anc).begin(), worlds[id].anc.end());
      }
   }

   static constexpr size_t npos_v = std::numeric_limits< size_t >::max();

   size_t visit(const EnvT::world_state_type& state, size_t depth, size_t parent)
   {
      const size_t my_id = worlds.size();
      OracleWorld w{};
      w.depth = depth;
      worlds.emplace_back(std::move(w));
      if(parent != npos_v) {
         worlds[parent].children.emplace_back(my_id);
      }

      if(env->is_terminal(state)) {
         worlds[my_id].terminal = true;
         return my_id;
      }

      const Player acting = env->active_player(state);
      const bool alex_acts = acting == members[0];
      const bool bob_acts = acting == members[1];

      if(alex_acts or bob_acts) {
         const size_t slot = alex_acts ? 0 : 1;
         const auto& stream = streams[slot];
         std::string key;
         for(const auto& [pub, priv] : stream) {
            key += "(" + common::to_string(pub) + "|" + common::to_string(priv) + ");";
         }
         const auto found = stream_registry[slot].find(key);
         size_t ordinal = npos_v;
         if(found != stream_registry[slot].end()) {
            ordinal = found->second;
         } else {
            ordinal = stream_registry[slot].size();
            stream_registry[slot].emplace(key, ordinal);
         }
         worlds[my_id].infoset_id = next_global_infoset(slot, ordinal);

         const auto actions = env->actions(acting, state);
         for(const auto& action : actions) {
            advance_and_descend(state, action, depth, my_id);
         }
      } else if(acting == Player::chance) {
         for(const auto& outcome : env->chance_actions(state)) {
            advance_and_descend(state, outcome, depth, my_id);
         }
      } else {
         for(const auto& action : env->actions(acting, state)) {
            advance_and_descend(state, action, depth, my_id);
         }
      }
      return my_id;
   }

   template < typename EdgeT >
   void advance_and_descend(
      const EnvT::world_state_type& state,
      const EdgeT& edge,
      size_t depth,
      size_t my_id
   )
   {
      auto next_holder = child_state(*env, state, edge);
      const auto& next = *next_holder;
      const auto pub_obs = env->public_observation(state, edge, next);
      // fold this edge's visibility into every member's log
      for(auto [slot, member] : std::views::enumerate(members)) {
         const auto priv = env->private_observation(member, state, edge, next);
         streams[slot].emplace_back(pub_obs, priv);
      }
      const size_t child_id = visit(next, depth + 1, my_id);
      (void) child_id;
      for(auto [slot, member] : std::views::enumerate(members)) {
         static_cast< void >(member);
         streams[slot].pop_back();
      }
   }

   size_t next_global_infoset(size_t slot, size_t local_ordinal)
   {
      return slot * 1'000'000ull + local_ordinal;
   }
};

}  // namespace

/////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// TB-DAG construction checks ////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

/// structurally validates alternating layers, single inactive parents, component disjointness
/// and the terminal bijection directly on the public builder API
template < typename DagT >
auto expected_successors(
   const DagT& dag,
   typename DagT::BeliefId belief_id,
   const typename DagT::Prescription& prescription
) -> std::vector< typename DagT::NodeId >
{
   const auto& belief = dag.belief(belief_id);
   std::vector< typename DagT::NodeId > result;
   for(const auto world : belief.worlds) {
      const auto& node = dag.world(world);
      if(not node.team_decision) {
         result.insert(result.end(), node.children.begin(), node.children.end());
         continue;
      }
      size_t slot_pos = DagT::npos;
      for(auto [position, slot] : std::views::enumerate(belief.slots)) {
         if(slot.infoset_global == node.infoset_global) {
            slot_pos = static_cast< size_t >(position);
            break;
         }
      }
      if(slot_pos == DagT::npos) {
         continue;
      }
      result.emplace_back(node.children.at(prescription.slot_action_idx.at(slot_pos)));
   }
   std::ranges::sort(result);
   result.erase(std::ranges::unique(result).begin(), result.end());
   return result;
}

template < typename DagT >
void validate_dag_structure(const DagT& dag)
{
   EXPECT_EQ(dag.stats().tree_nodes, dag.tree_node_count());
   EXPECT_EQ(dag.stats().beliefs, dag.belief_count());
   EXPECT_EQ(dag.stats().inactives, dag.inactive_count());
   EXPECT_EQ(dag.stats().dag_edges, dag.edge_count());
   EXPECT_EQ(dag.stats().prescription_edges, dag.prescription_edge_count());
   EXPECT_EQ(dag.stats().observation_edges, dag.observation_edge_count());
   ASSERT_LT(dag.root_belief(), dag.belief_count());
   EXPECT_TRUE(dag.belief(dag.root_belief()).parents.empty());

   std::set< typename DagT::NodeId > terminal_worlds;
   EXPECT_EQ(dag.terminal_count(), dag.stats().tree_leaves);
   for(const auto leaf : std::views::iota(size_t{0}, dag.terminal_count())) {
      const typename DagT::BeliefId b = dag.terminal_belief_id(leaf);
      ASSERT_NE(b, dag.npos);
      const auto& rec = dag.belief(b);
      EXPECT_TRUE(rec.terminal);
      EXPECT_EQ(rec.worlds.size(), size_t{1});
      ASSERT_FALSE(rec.worlds.empty());
      const auto world = rec.worlds.front();
      EXPECT_EQ(dag.terminal_world(leaf), world);
      EXPECT_TRUE(dag.world(world).terminal);
      EXPECT_EQ(rec.leaf_index, leaf);
      terminal_worlds.insert(world);
   }
   EXPECT_EQ(terminal_worlds.size(), dag.terminal_count());

   for(const auto world : std::views::iota(size_t{0}, dag.tree_node_count())) {
      const auto& node = dag.world(world);
      EXPECT_TRUE(std::ranges::is_sorted(node.anc_infosets));
      EXPECT_EQ(
         std::adjacent_find(node.anc_infosets.begin(), node.anc_infosets.end()),
         node.anc_infosets.end()
      );
      if(node.terminal) {
         EXPECT_TRUE(node.children.empty());
         EXPECT_FALSE(node.team_decision);
         continue;
      }
      EXPECT_FALSE(node.children.empty());
      const bool is_team_node = dag.slot_of(node.owner) != dag.npos;
      EXPECT_EQ(node.team_decision, is_team_node);
      if(is_team_node) {
         ASSERT_NE(node.infoset_global, dag.npos);
         EXPECT_EQ(node.children.size(), dag.infoset_actions(node.infoset_global).size());
         EXPECT_FALSE(dag.infoset_actions(node.infoset_global).empty());
      } else {
         EXPECT_EQ(node.infoset_global, dag.npos);
      }
   }

   for(const auto o : std::views::iota(size_t{0}, dag.inactive_count())) {
      const auto& rec = dag.inactive(o);
      // single-parent invariant
      ASSERT_FALSE(rec.worlds.empty());
      ASSERT_TRUE(std::ranges::is_sorted(rec.worlds));
      // component labels are disjoint world sets; each connected component is one tree layer
      std::vector< typename DagT::NodeId > covered;
      EXPECT_GT(rec.components.size(), size_t{0});
      for(const typename DagT::BeliefId child : rec.components) {
         ASSERT_LT(child, dag.belief_count());
         const auto& child_belief = dag.belief(child);
         ASSERT_FALSE(child_belief.worlds.empty());
         const size_t depth = dag.world(child_belief.worlds.front()).depth;
         for(const auto w : child_belief.worlds) {
            covered.emplace_back(w);
            EXPECT_EQ(dag.world(w).depth, depth);
         }
         EXPECT_TRUE(std::ranges::contains(child_belief.parents, o));
      }
      std::ranges::sort(covered);
      EXPECT_EQ(std::ranges::unique(covered).begin(), covered.end());
      EXPECT_EQ(covered, rec.worlds);
   }

   for(const auto b : std::views::iota(size_t{0}, dag.belief_count())) {
      const auto& rec = dag.belief(b);
      ASSERT_FALSE(rec.worlds.empty());
      ASSERT_TRUE(std::ranges::is_sorted(rec.worlds));
      for(const auto world : rec.worlds) {
         ASSERT_LT(world, dag.tree_node_count());
      }
      if(rec.terminal) {
         EXPECT_EQ(rec.worlds.size(), size_t{1});
         continue;
      }
      // pass-through layers (observation-only, no team infoset intersects) carry exactly the
      // single empty prescription; decision layers always expose one slot per infoset
      EXPECT_EQ(rec.prescriptions.size(), rec.prescription_children.size());
      if(rec.slots.empty()) {
         ASSERT_EQ(rec.prescriptions.size(), size_t{1});
         ASSERT_EQ(rec.prescription_children.size(), size_t{1});
         EXPECT_TRUE(rec.prescriptions.front().slot_action_idx.empty());
      } else {
         EXPECT_GT(rec.slots.size(), size_t{0});
         size_t expected_fanout = 1;
         for(const auto& slot : rec.slots) {
            EXPECT_EQ(slot.actions, dag.infoset_actions(slot.infoset_global));
            EXPECT_FALSE(slot.actions.empty());
            expected_fanout *= slot.actions.size();
         }
         EXPECT_EQ(rec.prescriptions.size(), expected_fanout);
      }
      for(auto [prescription_idx, prescription] : std::views::enumerate(rec.prescriptions)) {
         const auto child = rec.prescription_children.at(static_cast< size_t >(prescription_idx));
         ASSERT_LT(child, dag.inactive_count());
         EXPECT_EQ(dag.inactive(child).parent_belief, b);
         EXPECT_EQ(dag.inactive(child).worlds, expected_successors(dag, b, prescription));
      }
      for(const auto& presc : rec.prescriptions) {
         ASSERT_EQ(presc.slot_action_idx.size(), rec.slots.size());
         for(auto [pos, idx] : std::views::enumerate(presc.slot_action_idx)) {
            EXPECT_LT(idx, rec.slots[static_cast< size_t >(pos)].actions.size());
         }
      }
   }

   // Every belief parent edge is reciprocal and has no duplicate parent.
   for(const auto b : std::views::iota(size_t{0}, dag.belief_count())) {
      const auto& parents = dag.belief(b).parents;
      EXPECT_TRUE(std::ranges::is_sorted(parents));
      EXPECT_EQ(std::adjacent_find(parents.begin(), parents.end()), parents.end());
      for(const auto parent : parents) {
         ASSERT_LT(parent, dag.inactive_count());
         EXPECT_TRUE(std::ranges::contains(dag.inactive(parent).components, b));
      }
   }
}

/// build a three-player-goofspiel TB-DAG over the team {alex, bob}
template < typename DagT = rm::team::TeamBeliefDAG< EnvT > >
DagT make_team_dag(
   const tpg::GoofspielConfig& cfg,
   size_t max_dag_nodes = DagT::k_default_max_dag_nodes
)
{
   typename DagT::Config config{};
   config.members = {Player::alex, Player::bob};
   config.max_dag_nodes = max_dag_nodes;
   return DagT(EnvT(cfg), std::move(config));
}

bool oracle_connected(const Oracle& oracle, size_t a, size_t b)
{
   if(a == b) {
      return true;
   }
   const auto& wa = oracle.worlds[a];
   const auto& wb = oracle.worlds[b];
   if(wa.depth != wb.depth) {
      return false;
   }
   size_t p = 0, q = 0;
   while(p < wa.anc.size() and q < wb.anc.size()) {
      if(wa.anc[p] < wb.anc[q]) {
         ++p;
      } else if(wb.anc[q] < wa.anc[p]) {
         ++q;
      } else {
         return true;
      }
   }
   return false;
}

}  // namespace

/**
 * Hand-built expectations on the smallest interesting instance: deck_size=2, identical
 * decks with limited-information observations. The root is a chance-only pass-through, while
 * hidden committed bid values create non-singleton beliefs at later public-observation boundaries.
 */
TEST(TeamBeliefDagConstruction, hand_built_deck2_identical_expectations)
{
   const tpg::GoofspielConfig cfg{.deck_size = 2, .imp_info = true};
   const auto dag = make_team_dag(cfg);
   validate_dag_structure(dag);

   EXPECT_GT(dag.tree_node_count(), size_t{0});
   EXPECT_GT(dag.stats().tree_leaves, size_t{0});

   // NOTE the game root is a CHANCE node (prize reveal), i.e. an observation-only origin --
   // paper section 4 explicitly notes this creates a trivial leading layer in the TB-DAG: the
   // root belief carries NO slots and the single empty prescription passes through.
   const auto& root = dag.belief(dag.root_belief());
   EXPECT_EQ(root.worlds.size(), size_t{1});
   EXPECT_EQ(root.worlds.front(), size_t{0});
   ASSERT_EQ(root.slots.size(), size_t{0});
   ASSERT_EQ(root.prescriptions.size(), size_t{1});

   // Its unique child decomposes the possible prizes into public observations.
   const auto& root_child = dag.inactive(root.prescription_children.front());
   ASSERT_EQ(root_child.components.size(), size_t{2});
   const auto& prize_belief = dag.belief(root_child.components.front());
   ASSERT_EQ(prize_belief.slots.size(), size_t{1});  // only alex acts after the first reveal
   ASSERT_EQ(prize_belief.prescriptions.size(), size_t{2});

   EXPECT_GE(dag.stats().max_belief_size, size_t{2});
   // A TB-DAG has alternating active and inactive layers, so its total node count is not a
   // compression metric. The belief layer itself must nevertheless merge at least one pair of
   // world nodes in this limited-information instance.
   EXPECT_LT(dag.belief_count(), dag.tree_node_count());
   EXPECT_EQ(dag.belief_count() + dag.inactive_count(), dag.node_count());
}

/**
 * The construction must match an INDEPENDENT enumeration driven by raw FOSG primitive calls:
 * identical world-tree shapes, team-node classification, and the exact connected components
 * of every public-observation inactive label.
 */
void compare_with_oracle(const tpg::GoofspielConfig& cfg)
{
   const auto dag = make_team_dag(cfg);
   validate_dag_structure(dag);

   Oracle oracle{};
   oracle.enumerate(EnvT(cfg));

   ASSERT_EQ(oracle.worlds.size(), dag.tree_node_count())
      << "independent enumeration must visit the same tree";

   for(auto w : std::views::iota(size_t{0}, oracle.worlds.size())) {
      const auto& oracle_world = oracle.worlds[w];
      const auto& dag_world = dag.world(w);
      EXPECT_EQ(dag_world.depth, oracle_world.depth);
      EXPECT_EQ(dag_world.terminal, oracle_world.terminal);
      EXPECT_EQ(dag_world.children, oracle_world.children);
      EXPECT_EQ(dag_world.team_decision, oracle_world.infoset_id != Oracle::npos_v);
   }

   // Compare each inactive label against the oracle's literal Definition-4.1 connectivity
   // predicate. A global partition comparison is invalid here: the same world can occur in
   // different contextual inactive labels and labels are not global equivalence tokens.
   for(auto inactive_id : std::views::iota(size_t{0}, dag.inactive_count())) {
      const auto& inactive = dag.inactive(inactive_id);
      std::unordered_map< size_t, size_t, common::value_hasher< size_t > > component_of;
      for(auto [component, belief_id] : std::views::enumerate(inactive.components)) {
         for(const auto world : dag.belief(belief_id).worlds) {
            ASSERT_TRUE(component_of.emplace(world, static_cast< size_t >(component)).second)
               << "inactive components overlap at world " << world;
         }
      }
      ASSERT_EQ(component_of.size(), inactive.worlds.size());
      for(auto [i, world_a] : std::views::enumerate(inactive.worlds)) {
         static_cast< void >(i);
         ASSERT_TRUE(component_of.contains(world_a));
         for(const auto world_b : inactive.worlds) {
            const bool actual_same = component_of.at(world_a) == component_of.at(world_b);
            EXPECT_EQ(actual_same, oracle_connected(oracle, world_a, world_b))
               << "connectivity mismatch in inactive " << inactive_id << " for worlds " << world_a
               << "," << world_b;
         }
      }
   }
}

TEST(TeamBeliefDagConstruction, partition_matches_independent_oracle_deck3_identical)
{
   compare_with_oracle(tpg::GoofspielConfig{.deck_size = 3, .imp_info = true});
}

TEST(TeamBeliefDagConstruction, partition_matches_independent_oracle_deck2_limited_information)
{
   compare_with_oracle(tpg::GoofspielConfig{.deck_size = 2, .imp_info = true});
}

TEST(TeamBeliefDagConstruction, exact_node_bound_rejects_nonterminal_expansion)
{
   EXPECT_THROW(
      make_team_dag(tpg::GoofspielConfig{.deck_size = 2}, /*max_dag_nodes=*/1), std::length_error
   );
}

/**
 * Size accounting keeps the construction bounded without asserting an accidental compression
 * ratio. Limited-information observations additionally check that hidden bids merge at least
 * one team belief.
 */
TEST(TeamBeliefDagSize, accounting_and_bounded_construction)
{
   const std::vector< tpg::GoofspielConfig > configs{
      {.deck_size = 2}, {.deck_size = 3}, {.deck_size = 2, .imp_info = true}};

   for(const auto& cfg : configs) {
      const auto dag = make_team_dag(cfg);
      validate_dag_structure(dag);
      fmt::print(
         "[team-dag-size] deck={} split={}: worlds={} leaves={} beliefs={} inactives={} "
         "edges={} (presc {} | obs {}) max_belief={}\n",
         cfg.deck_size,
         cfg.split_half_deal,
         dag.tree_node_count(),
         dag.stats().tree_leaves,
         dag.belief_count(),
         dag.inactive_count(),
         dag.edge_count(),
         dag.prescription_edge_count(),
         dag.observation_edge_count(),
         dag.stats().max_belief_size
      );
      EXPECT_LE(dag.node_count(), dag.k_default_max_dag_nodes);
      EXPECT_GE(dag.stats().max_belief_size, size_t{1});
   }

   const auto limited = make_team_dag(tpg::GoofspielConfig{.deck_size = 2, .imp_info = true});
   EXPECT_GT(limited.stats().max_belief_size, size_t{1});
   EXPECT_LT(limited.belief_count(), limited.tree_node_count());
}

/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// DAG-CFR convergence /////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

template < typename SolverT >
void validate_solver_plane(const SolverT& solver, size_t side)
{
   const auto& dag = solver.dag(side);
   const auto coordinator = solver.coordinator_plan(side);
   ASSERT_EQ(coordinator.size(), dag.belief_count());
   for(auto b : std::views::iota(size_t{0}, dag.belief_count())) {
      const auto& belief = dag.belief(b);
      const auto& row = coordinator[b];
      if(belief.terminal) {
         EXPECT_TRUE(row.empty());
         continue;
      }
      ASSERT_EQ(row.size(), belief.prescriptions.size());
      double total = 0.;
      for(const double probability : row) {
         EXPECT_TRUE(std::isfinite(probability));
         EXPECT_GE(probability, -1e-12);
         total += probability;
      }
      EXPECT_NEAR(total, 1., 1e-10);
   }

   const auto realizations = solver.average_realizations(side);
   ASSERT_EQ(realizations.size(), dag.terminal_count());
   double total_realization_mass = 0.;
   for(const double realization : realizations) {
      EXPECT_TRUE(std::isfinite(realization));
      EXPECT_GE(realization, -1e-12);
      total_realization_mass += realization;
   }
   // This is realization-form mass, not a probability distribution over leaves: inactive
   // chance/opponent branches preserve flow while their probabilities live in the leaf utility
   // column. A nontrivial game must carry positive terminal mass.
   EXPECT_GT(total_realization_mass, 0.);
}

}  // namespace

TEST(DagCfr, iteration_preserves_flow_and_evaluation_invariants)
{
   using Solver = rm::team::AdversarialTeamDagCfr< EnvT >;
   const tpg::GoofspielConfig cfg{.deck_size = 2, .imp_info = true};
   Solver::Config config{};
   config.team_members = {Player::alex, Player::bob};
   Solver solver(EnvT(cfg), config);

   EXPECT_EQ(solver.iteration(), size_t{0});
   solver.iterate(16);
   EXPECT_EQ(solver.iteration(), size_t{16});
   validate_solver_plane(solver, Solver::k_team_plane);
   validate_solver_plane(solver, Solver::k_adversary_plane);

   const auto evaluation = solver.evaluate();
   EXPECT_EQ(evaluation.iterations, size_t{16});
   for(const double value : evaluation.average_pair_values) {
      EXPECT_TRUE(std::isfinite(value));
   }
   EXPECT_NEAR(evaluation.average_pair_values[0] + evaluation.average_pair_values[1], 0., 1e-10);
   for(const double certificate : evaluation.regret_certificates) {
      EXPECT_TRUE(std::isfinite(certificate));
      EXPECT_GE(certificate, -1e-12);
   }
   EXPECT_TRUE(std::isfinite(evaluation.saddle_gap_proxy));
   EXPECT_GE(evaluation.saddle_gap_proxy, -1e-12);
   for(const double value : evaluation.br_surrogate_values) {
      EXPECT_TRUE(std::isfinite(value));
   }

   // The empirical BR helper is explicitly non-destructive.
   const auto before_br_iteration = solver.iteration();
   const double br = solver.br_surrogate_value(Solver::k_team_plane, /*replay_rounds=*/4);
   EXPECT_TRUE(std::isfinite(br));
   EXPECT_EQ(solver.iteration(), before_br_iteration);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////// decentralization invariants ////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(DagCfrDecentralization, extracted_policies_are_normalized_and_keep_joint_plan)
{
   using Solver = rm::team::AdversarialTeamDagCfr< EnvT >;
   const tpg::GoofspielConfig cfg{.deck_size = 2, .imp_info = true};
   Solver::Config config{};
   config.team_members = {Player::alex, Player::bob};
   Solver solver(EnvT(cfg), config);
   solver.iterate(8);

   const auto team_plane = Solver::k_team_plane;
   const auto& dag = solver.dag(team_plane);
   const auto policies = solver.decentralized_policies(team_plane);
   const auto coordinator = solver.coordinator_plan(team_plane);

   // The coordinator rows retain the correlated prescription object. The marginal policy view
   // below is deliberately checked separately because independent execution cannot preserve
   // this joint distribution in general. This sequential fixture has one active team slot per
   // belief; the generic shape check still verifies that every row is indexed by complete
   // prescriptions rather than by independently selected member actions.
   bool saw_nontrivial_prescription_row = false;
   for(auto b : std::views::iota(size_t{0}, dag.belief_count())) {
      const auto& belief = dag.belief(b);
      if(not belief.terminal and belief.prescriptions.size() > 1) {
         saw_nontrivial_prescription_row = true;
      }
      if(not belief.terminal) {
         ASSERT_EQ(coordinator[b].size(), belief.prescriptions.size());
         for(const auto& prescription : belief.prescriptions) {
            EXPECT_EQ(prescription.slot_action_idx.size(), belief.slots.size());
         }
      }
   }
   EXPECT_TRUE(saw_nontrivial_prescription_row);

   for(const auto member : dag.members()) {
      ASSERT_TRUE(policies.contains(member));
      EXPECT_GT(policies.at(member).size(), size_t{0});
      for(const auto& [istate, policy] : policies.at(member)) {
         EXPECT_EQ(istate.player(), member);
         EXPECT_GT(policy.size(), size_t{0});
         double total = 0.;
         for(const auto& [action, prob] : policy) {
            static_cast< void >(action);
            EXPECT_TRUE(std::isfinite(prob));
            EXPECT_GE(prob, -1e-12);
            total += prob;
         }
         EXPECT_NEAR(total, 1., 1e-8) << "marginal distributions must be normalized";
      }
   }
}

TEST(DagCfr, custom_root_is_shared_by_both_planes)
{
   using Solver = rm::team::AdversarialTeamDagCfr< EnvT >;
   const tpg::GoofspielConfig cfg{.deck_size = 2};
   EnvT env(cfg);
   auto root = std::make_unique< EnvT::world_state_type >(env.initial_world_state());
   root->apply_action(root->chance_actions().front());

   Solver::Config config{};
   config.team_members = {Player::alex, Player::bob};
   Solver solver(env, std::move(root), config);
   const auto& team_dag = solver.dag(Solver::k_team_plane);
   const auto& adversary_dag = solver.dag(Solver::k_adversary_plane);
   EXPECT_EQ(team_dag.root_state(), adversary_dag.root_state());
   EXPECT_EQ(team_dag.tree_node_count(), adversary_dag.tree_node_count());
   EXPECT_EQ(team_dag.terminal_count(), adversary_dag.terminal_count());
   for(auto leaf : std::views::iota(size_t{0}, team_dag.terminal_count())) {
      EXPECT_DOUBLE_EQ(team_dag.terminal_weight(leaf), adversary_dag.terminal_weight(leaf));
   }
   solver.iterate(1);
   EXPECT_EQ(solver.iteration(), size_t{1});
}
