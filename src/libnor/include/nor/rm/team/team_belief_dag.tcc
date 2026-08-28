
#pragma once

// included from "team_belief_dag.hpp" -- do not include directly

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nor::rm::team {

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// enumeration of H ///////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

template < typename Env >
void TeamBeliefDAG< Env >::_build()
{
   // -- resolve the roster and the member slots -----------------------------------------------
   for(auto player : m_env.players(*m_root_state) | utils::is_actual_player_filter) {
      m_roster_index.emplace(player, m_roster.size());
      m_roster.emplace_back(player);
   }
   for(auto [member_index, member] : std::views::enumerate(m_config.members)) {
      if(std::ranges::find(
            m_config.members.begin(), m_config.members.begin() + member_index, member
         )
         != m_config.members.begin() + member_index) {
         throw std::invalid_argument("TeamBeliefDAG: 'members' must not contain duplicates");
      }
      if(not m_roster_index.contains(member)) {
         throw std::invalid_argument(
            "TeamBeliefDAG: configured team member is not part of the root roster"
         );
      }
      MemberRegistry reg{};
      reg.player = member;
      m_member_infostates.emplace_back(std::move(reg));
      m_active_infostates.emplace(member, std::make_shared< info_state_type >(member));
      m_obs_buffers.emplace(member, std::vector< std::pair< observation_type, observation_type > >{});
   }

   // -- full-tree DFS of the team decision problem --------------------------------------------
   while(m_arena.size() < 1) {
      m_arena.emplace_back();
   }
   world_state_type& root = m_arena[0].construct_from(*m_root_state);

   m_tree.emplace_back();  // root node id 0 (depth/chance-weight default to identity)
   _enumerate_visit(root, /*node_id=*/0, /*depth=*/0);
   _finalize_enumeration();

   // -- TB-DAG recursion (paper Algorithm 1) --------------------------------------------------
   std::vector< NodeId > root_label{NodeId{0}};
   m_root_belief = _make_active(std::move(root_label));

   // -- statistics -----------------------------------------------------------------------------
   m_stats.tree_nodes = m_tree.size();
   for(const auto& node : m_tree) {
      if(node.terminal) {
         ++m_stats.tree_leaves;
      }
      if(node.team_decision) {
         ++m_stats.tree_decision_nodes;
      }
   }
   m_stats.beliefs = m_beliefs.size();
   m_stats.inactives = m_inactives.size();
   m_stats.dag_edges = edge_count();
   m_stats.prescription_edges = prescription_edge_count();
   m_stats.observation_edges = observation_edge_count();
   for(const auto& b : m_beliefs) {
      m_stats.max_prescription_width = std::max(m_stats.max_prescription_width, b.slots.size());
      m_stats.max_prescription_fanout =
         std::max(m_stats.max_prescription_fanout, b.prescriptions.size());
      m_stats.max_belief_size = std::max(m_stats.max_belief_size, b.worlds.size());
   }

   // leaf row -> terminal-belief cross-reference (every leaf has exactly one singleton belief)
   m_leaf_beliefs.assign(m_terminal_weights.size(), npos);
   for(auto [id, rec] : std::views::enumerate(m_beliefs)) {
      if(rec.terminal) {
         m_leaf_beliefs.at(rec.leaf_index) = static_cast< BeliefId >(id);
      }
   }
   if(std::ranges::find(m_leaf_beliefs, npos) != m_leaf_beliefs.end()) [[unlikely]] {
      throw std::logic_error(
         "TeamBeliefDAG: a terminal world did not materialize as a singleton belief"
      );
   }

   // global infoset id -> owner slot / local ordinal cross-references
   m_infoset_member.assign(m_infoset_owner.size(), npos);
   m_infoset_slot.assign(m_infoset_owner.size(), npos);
   m_infoset_local.assign(m_infoset_owner.size(), npos);
   for(auto [slot, registry] : std::views::enumerate(m_member_infostates)) {
      for(auto [local, global_id] : std::views::enumerate(registry.local_ids)) {
         m_infoset_member.at(global_id) = slot;
         m_infoset_slot.at(global_id) = slot;
         m_infoset_local.at(global_id) = local;
      }
   }
}

/**
 * Depth-first enumeration worker. The node 'node_id' has been pushed into 'm_tree' by the
 * caller with its depth and chance weight preset; this call fills in classification, assigns
 * (first-discovery) infoset ids for team decisions, validates perfect-recall action-list
 * consistency on re-discoveries, captures terminal payoffs, and recurses over all successors.
 *
 * NOTE: no references into 'm_tree' are held across the recursive descent -- deeper frames
 * append nodes and would invalidate them; every mutated field is written back through indices.
 */
template < typename Env >
void TeamBeliefDAG< Env >::_enumerate_visit(world_state_type& state, NodeId node_id, size_t depth)
{
   const bool terminal = m_env.is_terminal(state);
   m_tree[node_id].terminal = terminal;

   if(terminal) {
      auto& node = m_tree[node_id];
      node.payoffs.reserve(m_roster.size());
      for(auto player : m_roster) {
         node.payoffs.emplace_back(m_env.reward(player, state));
      }
      return;
   }

   Player active_player = m_env.active_player(state);
   m_tree[node_id].owner = active_player;

   const auto deciding_slot = slot_of(active_player);
   if(deciding_slot != npos) {
      // ---- a TEAM MEMBER decides here -------------------------------------------------------
      m_tree[node_id].team_decision = true;
      auto& registry = m_member_infostates[deciding_slot];
      const sptr< info_state_type >& istate = m_active_infostates.at(active_player);
      auto [id_it, inserted] =
         registry.ids.try_emplace(*istate, registry.local_ids.size());
      if(inserted) {
         registry.representatives.emplace_back(istate);
         const size_t global_id = m_infoset_global_count++;
         registry.local_ids.emplace_back(global_id);
         m_infoset_owner.emplace_back(active_player);
         m_tree[node_id].infoset_local = id_it->second;
         m_tree[node_id].infoset_global = global_id;
         m_infoset_actions.emplace_back(m_env.actions(active_player, state));
      } else {
         m_tree[node_id].infoset_local = id_it->second;
         m_tree[node_id].infoset_global = registry.local_ids.at(id_it->second);
      }

      // canonical action-list consistency across one infoset (perfect recall)
      const auto actions = m_env.actions(active_player, state);
      const auto& canonical = m_infoset_actions.at(m_tree[node_id].infoset_global);
      if(canonical != actions) [[unlikely]] {
         throw std::logic_error(
            "TeamBeliefDAG: inconsistent legal-action sets across one team infoset "
            "(perfect-recall violation or buggy environment)"
         );
      }

      std::vector< NodeId > children;
      children.reserve(actions.size());
      for(const auto& action : actions) {
         EdgeUndo undo;
         world_state_type& next = _advance_edge(state, depth, action, undo);
         NodeId child = m_tree.size();
         m_tree.emplace_back();
         m_tree.back().depth = depth + 1;
         m_tree.back().chance_weight = m_tree[node_id].chance_weight;
         _enumerate_visit(next, child, depth + 1);
         children.emplace_back(child);
         _undo_edge(undo);
      }
      m_tree[node_id].children = std::move(children);
   } else {
      // ---- inactive world node: expand chance outcomes or adversary actions ------------------
      std::vector< NodeId > children;
      const auto descend = [&](const auto& edge, double edge_weight) {
         EdgeUndo undo;
         world_state_type& next = _advance_edge(state, depth, edge, undo);
         NodeId child = m_tree.size();
         m_tree.emplace_back();
         m_tree.back().depth = depth + 1;
         m_tree.back().chance_weight = m_tree[node_id].chance_weight * edge_weight;
         _enumerate_visit(next, child, depth + 1);
         children.emplace_back(child);
         _undo_edge(undo);
      };

      if constexpr(concepts::stochastic_env< Env >) {
         if(active_player == Player::chance) {
            for(const auto& outcome : m_env.chance_actions(state)) {
               const double probability = m_env.chance_probability(state, outcome);
               if(not std::isfinite(probability) or probability < 0.) [[unlikely]] {
                  throw std::logic_error(
                     "TeamBeliefDAG: chance_probability must be finite and non-negative"
                  );
               }
               descend(outcome, probability);
            }
         } else {
            for(const auto& action : m_env.actions(active_player, state)) {
               descend(action, 1.);
            }
         }
      } else {
         // Deterministic FOSGs do not provide chance_actions; all non-team turns are ordinary
         // inactive actions in the induced team decision problem.
         for(const auto& action : m_env.actions(active_player, state)) {
            descend(action, 1.);
         }
      }
      m_tree[node_id].children = std::move(children);
   }
}

/**
 * Bottom-up finalization of the enumerated tree:
 * 1. ancestor-set reduction A(h) = union(A(children)) ∪ {own infoset} -- exactly the set of
 *    team infosets intersecting h's subtree, i.e. {I : h ⪯ I}, the connectivity encoding;
 * 2. flat terminal tables (leaf ids in discovery order, chance weights, payoff rows).
 */
template < typename Env >
void TeamBeliefDAG< Env >::_finalize_enumeration()
{
   // preorder enumeration guarantees every descendant carries a strictly larger id than its
   // ancestor, so a plain reverse-index sweep sees fully reduced children before their parent
   for(size_t id = m_tree.size(); id-- > 0;) {
      auto& node = m_tree[id];
      for(auto child : node.children) {
         node.anc_infosets.insert(
            node.anc_infosets.end(),
            m_tree[child].anc_infosets.begin(),
            m_tree[child].anc_infosets.end()
         );
      }
      if(node.team_decision) {
         node.anc_infosets.emplace_back(node.infoset_global);
      }
      std::ranges::sort(node.anc_infosets);
      node.anc_infosets.erase(
         std::ranges::unique(node.anc_infosets).begin(), node.anc_infosets.end()
      );
   }

   for(auto [id, node] : std::views::enumerate(m_tree)) {
      if(not node.terminal) {
         continue;
      }
      m_leaf_by_world.emplace(static_cast< NodeId >(id), m_terminal_weights.size());
      m_terminal_worlds.emplace_back(static_cast< NodeId >(id));
      m_terminal_weights.emplace_back(node.chance_weight);
      for(auto payoff : node.payoffs) {
         m_terminal_payoffs.emplace_back(payoff);
      }
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////// shared traversal edge mechanics /////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

template < typename Env >
template < typename ActionOrOutcome >
auto TeamBeliefDAG< Env >::_advance_edge(
   const world_state_type& state,
   size_t depth,
   const ActionOrOutcome& edge,
   EdgeUndo& undo
) -> world_state_type&
{
   while(m_arena.size() <= depth + 1) {
      m_arena.emplace_back();
   }
   world_state_type& next_wstate = m_arena[depth + 1].construct_from(state);
   m_env.transition(next_wstate, edge);

   const auto public_obs = m_env.public_observation(state, edge, next_wstate);
   const Player next_active_player = m_env.active_player(next_wstate);

   undo.flushes = false;
   undo.flush_target = Player::unknown;

   if(slot_of(next_active_player) != npos) {
      undo.flushes = true;
      undo.flush_target = next_active_player;
      undo.saved_infostate = m_active_infostates.at(next_active_player);
      undo.saved_flush_buffer = m_obs_buffers.at(next_active_player);
   }

   for(auto player : m_config.members) {
      if(undo.flushes and player == undo.flush_target) {
         continue;  // restored wholesale via saved_flush_buffer
      }
      undo.saved_sizes.emplace_back(player, m_obs_buffers.at(player).size());
      m_obs_buffers.at(player).emplace_back(
         public_obs, m_env.private_observation(player, state, edge, next_wstate)
      );
   }

   if(undo.flushes) {
      // flush the buffered observations into a clone of the member's current infostate, then
      // append this edge's own pair (mirrors next_infostate_and_obs_buffers_inplace)
      auto child_infostate = std::make_shared< info_state_type >(*undo.saved_infostate);
      auto& obs_history = m_obs_buffers.at(undo.flush_target);
      for(auto& obs : obs_history) {
         ::nor::detail::update_infostate(
            child_infostate, std::move(obs.first), std::move(obs.second)
         );
      }
      obs_history.clear();
      ::nor::detail::update_infostate(
         child_infostate,
         public_obs,
         m_env.private_observation(undo.flush_target, state, edge, next_wstate)
      );
      m_active_infostates.at(undo.flush_target) = std::move(child_infostate);
   }

   return next_wstate;
}

template < typename Env >
void TeamBeliefDAG< Env >::_undo_edge(const EdgeUndo& undo)
{
   if(undo.flushes) {
      m_active_infostates.at(undo.flush_target) = undo.saved_infostate;
      m_obs_buffers.at(undo.flush_target) = std::move(*undo.saved_flush_buffer);
   }
   for(const auto& [player, size] : undo.saved_sizes) {
      m_obs_buffers.at(player).resize(size);
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// TB-DAG construction (Alg. 1) ////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Paper Algorithm 1 MakeActiveNode: creates (or returns the memoized) belief labeled with the
 * world-set B. Terminal singletons become DAG terminals bound to their precomputed leaf rows.
 * Otherwise every intersecting team infoset contributes one prescription slot whose cartesian
 * product spawns the inactive children.
 *
 * NOTE: index-only mutations after recursive calls ('_make_inactive' grows 'm_inactives').
 */
template < typename Env >
auto TeamBeliefDAG< Env >::_make_active(std::vector< NodeId > b) -> BeliefId
{
   std::ranges::sort(b);
   b.erase(std::ranges::unique(b).begin(), b.end());

   if(auto found = m_belief_ids.find(b); found != m_belief_ids.end()) {
      return found->second;
   }
   const size_t used_nodes = node_count();
   if(used_nodes >= m_config.max_dag_nodes) {
      throw std::length_error(
         "TeamBeliefDAG exceeded max_dag_nodes=" + std::to_string(m_config.max_dag_nodes)
         + "; the game appears too large for an exact TB-DAG construction"
      );
   }

   const bool is_terminal = b.size() == 1 and m_tree[b.front()].terminal;
   if(not is_terminal and m_config.max_dag_nodes - used_nodes < 2) {
      throw std::length_error(
         "TeamBeliefDAG exceeded max_dag_nodes=" + std::to_string(m_config.max_dag_nodes)
         + "; the game appears too large for an exact TB-DAG construction"
      );
   }

   const BeliefId my_id = m_beliefs.size();
   Belief rec{};
   rec.worlds = b;
   // slots: distinct team infosets intersecting B, ascending-world discovery order
   for(auto world : b) {
      const auto& wn = m_tree[world];
      if(not wn.team_decision) {
         continue;
      }
      const auto g = wn.infoset_global;
      const bool already =
         std::ranges::any_of(rec.slots, [&](const BeliefSlot& s) { return s.infoset_global == g; });
      if(already) {
         continue;
      }
      BeliefSlot slot{};
      // The global-id cross-reference arrays are finalized only after the recursive DAG build;
      // resolve the owning member directly from the registry created during enumeration here.
      slot.member_idx = slot_of(m_infoset_owner.at(g));
      slot.infoset_global = g;
      slot.actions = m_infoset_actions.at(g);
      rec.slots.emplace_back(std::move(slot));
   }

   m_belief_ids.emplace(b, my_id);
   m_beliefs.emplace_back(std::move(rec));

   if(is_terminal) {
      m_beliefs[my_id].terminal = true;
      m_beliefs[my_id].leaf_index = m_leaf_by_world.at(b.front());
      return my_id;
   }

   // prescriptions: cartesian product over slot action lists (odometer order)
   const size_t n_slots = m_beliefs[my_id].slots.size();
   size_t total = 1;
   bool overflow_guard = false;
   for(const auto& slot : m_beliefs[my_id].slots) {
      const auto k = slot.actions.size();
      if(k == 0) [[unlikely]] {
         throw std::logic_error("TeamBeliefDAG: an intersecting infoset has no legal actions");
      }
      if(total > m_config.max_dag_nodes / k) {
         overflow_guard = true;
         break;
      }
      total *= k;
   }
   if(overflow_guard or total > m_config.max_dag_nodes) {
      throw std::length_error(
         "TeamBeliefDAG: prescription fan-out beyond max_dag_nodes="
         + std::to_string(m_config.max_dag_nodes) + " at a single belief"
      );
   }
   m_beliefs[my_id].prescriptions.reserve(total);
   m_beliefs[my_id].prescription_children.resize(total);
   {
      std::vector< size_t > odometer(n_slots, 0);
      for([[maybe_unused]] auto _ : std::views::iota(size_t{0}, total)) {
         Prescription presc{};
         presc.slot_action_idx = odometer;
         m_beliefs[my_id].prescriptions.emplace_back(std::move(presc));

         for(auto j : std::views::iota(size_t{0}, n_slots)) {
            ++odometer[j];
            if(odometer[j] < m_beliefs[my_id].slots[j].actions.size()) {
               break;
            }
            odometer[j] = 0;
         }
      }
   }

   // child world-set per prescription (children created in env-action order, so the child
   // position of a team-decision world equals its prescribed slot-action index)
   for(auto presc_idx : std::views::iota(size_t{0}, total)) {
      std::vector< NodeId > child_set;
      child_set.reserve(2 * b.size());
      for(auto world : b) {
         const auto& wn = m_tree[world];
         if(not wn.team_decision) {
            // inert world: ALL successors join the child set
            child_set.insert(child_set.end(), wn.children.begin(), wn.children.end());
            continue;
         }
         size_t slot_pos = npos;
         for(auto [j, slot_ref] : std::views::enumerate(m_beliefs[my_id].slots)) {
            if(slot_ref.infoset_global == wn.infoset_global) {
               slot_pos = j;
               break;
            }
         }
         if(slot_pos == npos) [[unlikely]] {
            throw std::logic_error(
               "TeamBeliefDAG: team decision world has no matching belief slot"
            );
         }
         const auto action_idx = m_beliefs[my_id].prescriptions[presc_idx].slot_action_idx.at(
            slot_pos
         );
         child_set.emplace_back(wn.children.at(action_idx));
      }
      std::ranges::sort(child_set);
      child_set.erase(std::ranges::unique(child_set).begin(), child_set.end());
      const InactiveId child = _make_inactive(std::move(child_set), my_id);
      m_beliefs[my_id].prescription_children[presc_idx] = child;
   }

   return my_id;
}

/**
 * Paper Algorithm 1 MakeInactiveNode: creates (or returns the memoized) inactive node labeled
 * with O and decomposes O into the connected components of the connectivity graph G[O]
 * (Definition 4.1): worlds o,o' link iff they share a tree layer AND their subtree-reachable
 * team-infoset sets intersect.
 */
template < typename Env >
auto TeamBeliefDAG< Env >::_make_inactive(std::vector< NodeId > o, BeliefId parent_belief)
   -> InactiveId
{
   std::ranges::sort(o);
   o.erase(std::ranges::unique(o).begin(), o.end());
   if(o.empty()) [[unlikely]] {
      throw std::logic_error("TeamBeliefDAG: encountered an empty inactive label");
   }

   if(auto found = m_inactive_ids.find(o); found != m_inactive_ids.end()) {
      const InactiveId existing = found->second;
      if(m_inactives[existing].parent_belief != parent_belief) [[unlikely]] {
         throw std::logic_error(
            "TeamBeliefDAG: inactive label reached from two beliefs -- violates the "
            "alternating/single-parent TB-DAG invariant"
         );
      }
      return existing;
   }
   if(node_count() >= m_config.max_dag_nodes) {
      throw std::length_error(
         "TeamBeliefDAG exceeded max_dag_nodes=" + std::to_string(m_config.max_dag_nodes)
         + "; the game appears too large for an exact TB-DAG construction"
      );
   }

   const InactiveId my_id = m_inactives.size();
   InactiveNode rec{};
   rec.worlds = o;
   rec.parent_belief = parent_belief;
   m_inactive_ids.emplace(o, my_id);
   m_inactives.emplace_back(std::move(rec));

   // ---- connected components of G[O] (union-find over pair connectivity) -------------------
   const size_t n = o.size();
   std::vector< size_t > uf(n);
   std::iota(uf.begin(), uf.end(), size_t{0});
   const auto find_root = [&](size_t x) {
      while(uf[x] != x) {
         uf[x] = uf[uf[x]];
         x = uf[x];
      }
      return x;
   };
   for(auto i : std::views::iota(size_t{0}, n)) {
      for(auto j : std::views::iota(i + 1, n)) {
         const auto& wi = m_tree[o[i]];
         const auto& wj = m_tree[o[j]];
         if(wi.depth != wj.depth) {
            continue;
         }
         // both ancestor lists are sorted ascending -- linear intersection test
         bool shares_info = false;
         size_t p = 0, q = 0;
         while(p < wi.anc_infosets.size() and q < wj.anc_infosets.size()) {
            if(wi.anc_infosets[p] < wj.anc_infosets[q]) {
               ++p;
            } else if(wj.anc_infosets[q] < wi.anc_infosets[p]) {
               ++q;
            } else {
               shares_info = true;
               break;
            }
         }
         if(shares_info) {
            uf[find_root(j)] = find_root(i);
         }
      }
   }

   std::vector< BeliefId > component_ids;
   {
      std::unordered_map< size_t, std::vector< NodeId > > buckets;
      for(auto [position, world] : std::views::enumerate(m_inactives[my_id].worlds)) {
         buckets[find_root(static_cast< size_t >(position))].emplace_back(world);
      }
      std::vector< std::vector< NodeId > > comps;
      comps.reserve(buckets.size());
      for(auto& [root_key, bucket] : buckets) {
         static_cast< void >(root_key);
         std::ranges::sort(bucket);
         comps.emplace_back(std::move(bucket));
      }
      std::ranges::sort(comps, [](const auto& x, const auto& y) { return x.front() < y.front(); });
      for(auto& comp : comps) {
         component_ids.emplace_back(_make_active(std::move(comp)));
      }
   }
   for(auto child : component_ids) {
      m_beliefs[child].parents.emplace_back(my_id);
   }
   m_inactives[my_id].components = std::move(component_ids);

   return my_id;
}

}  // namespace nor::rm::team
