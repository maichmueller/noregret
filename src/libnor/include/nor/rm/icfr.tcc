#pragma once

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ranges>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "icfr.hpp"

namespace nor::rm {

///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// construction / setup ////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

template < typename Env, typename InternalRM, typename ExternalRM >
void ICFR< Env, InternalRM, ExternalRM >::_init_roster()
{
   for(auto player : m_env.players(*m_root_state) | utils::is_actual_player_filter) {
      m_player_index.emplace(player, m_roster.size());
      m_roster.emplace_back(player);
   }
   m_structures.resize(m_roster.size());
   m_own_match.assign(m_roster.size(), 1.);
   m_decision_stacks.resize(m_roster.size());
}

template < typename Env, typename InternalRM, typename ExternalRM >
void ICFR< Env, InternalRM, ExternalRM >::_ensure_initialized()
{
   if(m_initialized) {
      return;
   }
   // fresh per-player traversal bookkeeping
   for(auto player : m_roster) {
      m_obs_buffers.emplace(
         player, std::vector< std::pair< observation_type, observation_type > >{}
      );
      m_infostates.emplace(player, std::make_shared< info_state_type >(player));
   }
   while(m_arena.size() < 1) {
      m_arena.emplace_back();
   }
   world_state_type& root = m_arena[0].construct_from(*m_root_state);
   _enumerate_visit(root, /*depth=*/0);
   _finalize_enumeration();
   m_initialized = true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// enumeration pass //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * One full-tree DFS discovering every infoset of every player together with its
 * legal actions, its own ancestor chain (root-first) and its immediate child
 * relation C(I,a). Infoset ids are assigned in first-discovery order, which
 * under perfect recall is a topological order of the chain precedence (all
 * chain ancestors of an infoset are discovered strictly earlier).
 *
 * The C(I,a) relation accumulates a UNION over ALL discovering paths: distinct
 * histories of one infoset may lead to different descendant infosets (e.g.
 * multiplayer poker response infosets span deals), so first-visit-only linking
 * would under-approximate the children sets.
 */
template < typename Env, typename InternalRM, typename ExternalRM >
void ICFR< Env, InternalRM, ExternalRM >::_enumerate_visit(world_state_type& state, size_t depth)
{
   if(_env().is_terminal(state)) {
      return;
   }
   Player active_player = _env().active_player(state);

   if constexpr(concepts::stochastic_env< env_type >) {
      if(active_player == Player::chance) {
         for(const auto& outcome : _env().chance_actions(state)) {
            EdgeUndo undo;
            world_state_type& next = _advance_edge(state, depth, outcome, undo);
            _enumerate_visit(next, depth + 1);
            _undo_edge(undo);
         }
         return;
      }
   }

   const size_t p_idx = m_player_index.at(active_player);
   auto& st = m_structures[p_idx];
   const sptr< info_state_type >& istate = m_infostates.at(active_player);

   auto [id_it, inserted] = st.ids.try_emplace(*istate, st.infosets.size());
   const size_t id = id_it->second;
   if(inserted) {
      auto& rec = st.infosets.emplace_back();
      rec.istate = istate;
      for(const auto& action : _env().actions(active_player, state)) {
         rec.actions.emplace_back(action);
         rec.internal_rm.register_action(action);
      }
      rec.children.resize(rec.actions.size());
      // inherit the discovering path's chain -- well-defined under perfect
      // recall since the ancestor chain of an infoset is unique
      if(not m_decision_stacks[p_idx].empty()) {
         const auto [parent_id, parent_action_idx] = m_decision_stacks[p_idx].back();
         rec.chain = st.infosets[parent_id].chain;
         rec.chain.emplace_back(parent_id, parent_action_idx);
      }
   }
   // link against the current path's previous decision of this player (on every
   // visit: later paths may contribute children the first visit could not see)
   if(not m_decision_stacks[p_idx].empty()) {
      const auto [parent_id, parent_action_idx] = m_decision_stacks[p_idx].back();
      auto& siblings = st.infosets[parent_id].children[parent_action_idx];
      if(std::ranges::find(siblings, id) == siblings.end()) {
         siblings.emplace_back(id);
      }
   }

   // COPY, not reference: deeper `_enumerate_visit` recursion below emplace_backs
   // infoset records into 'st.infosets', so a bound reference could dangle on vector
   // reallocation mid-descent.
   const std::vector< action_type > actions = st.infosets[id].actions;
   for(auto [action_idx, action] : std::views::enumerate(actions)) {
      m_decision_stacks[p_idx].emplace_back(id, action_idx);
      EdgeUndo undo;
      world_state_type& next = _advance_edge(state, depth, action, undo);
      _enumerate_visit(next, depth + 1);
      _undo_edge(undo);
      m_decision_stacks[p_idx].pop_back();
   }
}

template < typename Env, typename InternalRM, typename ExternalRM >
void ICFR< Env, InternalRM, ExternalRM >::_finalize_enumeration()
{
   for(auto& st : m_structures) {
      const auto n_infosets = st.infosets.size();
      // dense (infoset, action) slot offsets
      st.action_offsets.assign(n_infosets + 1, size_t{0});
      for(auto idx : std::views::iota(size_t{0}, n_infosets)) {
         st.action_offsets[idx + 1] = st.action_offsets[idx]
                                      + static_cast< size_t >(st.infosets[idx].actions.size());
      }
      const size_t total_slots = st.action_offsets.back();

      // external units: one per co-trigger sequence sigma in Sigma^c(I), i.e.
      // one per (ancestor chain position k, action b != required-at-chain[k])
      st.ext_unit_offsets.assign(n_infosets + 1, size_t{0});
      st.raw_external_offsets.assign(n_infosets + 1, size_t{0});
      for(auto idx : std::views::iota(size_t{0}, n_infosets)) {
         auto& rec = st.infosets[idx];
         for(auto [k, ancestor] : std::views::enumerate(rec.chain)) {
            const auto n_ancestor_actions = static_cast< size_t >(
               st.infosets[ancestor.first].actions.size()
            );
            for(auto b : std::views::iota(size_t{0}, n_ancestor_actions)) {
               if(b == ancestor.second) {
                  continue;
               }
               auto& unit = rec.ext_units.emplace_back();
               unit.chain_pos = k;
               unit.action_idx = b;
               for(const auto& action : rec.actions) {
                  unit.rm.register_action(action);
               }
            }
         }
         st.ext_unit_offsets[idx + 1] = st.ext_unit_offsets[idx] + rec.ext_units.size();
         st.raw_external_offsets[idx + 1] = st.raw_external_offsets[idx]
                                            + rec.ext_units.size() * rec.actions.size();
      }

      // raw trigger-sequence co-statistics: internal triggers sigma=(I,a*) get
      // an |A(I)| x |A(I)| block plus an |A(I)| obedient-scalar row, external
      // units an |A(I)| block each, and the on-path ancestor triggers one
      // |A(I)| block per chain position k of I (see the PlayerStructure note)
      st.raw_internal_offsets.assign(n_infosets + 1, size_t{0});
      st.raw_onpath_offsets.assign(n_infosets + 1, size_t{0});
      for(auto idx : std::views::iota(size_t{0}, n_infosets)) {
         const auto n_actions = static_cast< size_t >(st.infosets[idx].actions.size());
         st.raw_internal_offsets[idx + 1] = st.raw_internal_offsets[idx] + n_actions * n_actions;
         st.raw_onpath_offsets[idx + 1] = st.raw_onpath_offsets[idx]
                                          + st.infosets[idx].chain.size() * n_actions;
      }
      st.raw_internal.assign(st.raw_internal_offsets.back(), 0.);
      st.raw_external.assign(st.raw_external_offsets.back(), 0.);
      st.raw_onpath.assign(st.raw_onpath_offsets.back(), 0.);
      // obedient baselines + stop-value twins mirror the raw block shapes verbatim
      st.stop_internal.assign(st.raw_internal_offsets.back(), 0.);
      st.stop_external.assign(st.raw_external_offsets.back(), 0.);
      st.stop_onpath.assign(st.raw_onpath_offsets.back(), 0.);
      st.obed_internal.assign(st.raw_internal_offsets.back(), 0.);
      st.obed_external.assign(st.raw_external_offsets.back(), 0.);
      st.obed_onpath.assign(st.raw_onpath_offsets.back(), 0.);

      // per-iteration scratch
      st.choice.assign(n_infosets, size_t{0});
      st.reached.assign(n_infosets, char{0});
      st.prefix_match.assign(n_infosets, size_t{0});
      st.v_memo.assign(n_infosets, 0.);
      st.u_stop.assign(total_slots, 0.);
      st.x_sums.assign(total_slots, 0.);
      st.s_sums.assign(n_infosets, 0.);
      st.ext_active.assign(st.ext_unit_offsets.back(), char{0});
      st.last_recommendation.assign(n_infosets, {});
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////// shared traversal edge mechanics /////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template < typename Env, typename InternalRM, typename ExternalRM >
template < typename ActionOrOutcome >
auto ICFR< Env, InternalRM, ExternalRM >::_advance_edge(
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
   _env().transition(next_wstate, edge);

   const auto public_obs = _env().public_observation(state, edge, next_wstate);
   const Player next_active_player = _env().active_player(next_wstate);
   const auto infostate_entry_it = m_infostates.find(next_active_player);
   undo.flushes = next_active_player != Player::chance and infostate_entry_it != m_infostates.end();
   undo.flush_target = next_active_player;

   if(undo.flushes) {
      undo.saved_infostate = infostate_entry_it->second;
      undo.saved_flush_buffer = m_obs_buffers.at(next_active_player);
   }
   for(const auto& [player, buffer] : m_obs_buffers) {
      if(not undo.flushes or player != next_active_player) {
         undo.saved_sizes.emplace_back(player, buffer.size());
      }
   }
   for(auto player : _env().players(next_wstate)) {
      if(player == Player::chance) {
         continue;
      }
      if(undo.flushes and player == next_active_player) {
         // flush the buffered observations into a clone of the infostate, then
         // append this edge's own pair (mirrors next_infostate_and_obs_buffers_inplace)
         auto child_infostate = std::make_shared< info_state_type >(*undo.saved_infostate);
         auto& obs_history = m_obs_buffers[player];
         for(auto& obs : obs_history) {
            ::nor::detail::update_infostate(
               child_infostate, std::move(obs.first), std::move(obs.second)
            );
         }
         obs_history.clear();
         ::nor::detail::update_infostate(
            child_infostate,
            public_obs,
            _env().private_observation(player, state, edge, next_wstate)
         );
         infostate_entry_it->second = std::move(child_infostate);
      } else {
         m_obs_buffers[player].emplace_back(
            public_obs, _env().private_observation(player, state, edge, next_wstate)
         );
      }
   }
   return next_wstate;
}

template < typename Env, typename InternalRM, typename ExternalRM >
void ICFR< Env, InternalRM, ExternalRM >::_undo_edge(const EdgeUndo& undo)
{
   if(undo.flushes) {
      auto it = m_infostates.find(undo.flush_target);
      it->second = undo.saved_infostate;
      m_obs_buffers.at(undo.flush_target) = std::move(*undo.saved_flush_buffer);
   }
   for(const auto& [player, size] : undo.saved_sizes) {
      m_obs_buffers.at(player).resize(size);
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// iteration driver //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

template < typename Env, typename InternalRM, typename ExternalRM >
void ICFR< Env, InternalRM, ExternalRM >::iterate(size_t n_iters)
{
   _ensure_initialized();
   for([[maybe_unused]] auto _ : std::views::iota(size_t{0}, n_iters)) {
      // phase 1: sample a normal-form plan for every player (Algorithm 1 SampleInternal)
      for(auto p_idx : std::views::iota(size_t{0}, m_roster.size())) {
         _sample_plan(p_idx);
      }
      // phase 2: one full-tree pass computing every player's counterfactual
      // stop-values u^t[I,a] and accumulating mu_bar's terminal counts
      _traverse_values();
      // phase 3: feed the active regret-minimizer units (Algorithm 1 UpdateInternal)
      for(auto p_idx : std::views::iota(size_t{0}, m_roster.size())) {
         _update_player(p_idx);
      }
      ++m_iteration;
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// Algorithm 1: plan sampling ///////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template < typename Env, typename InternalRM, typename ExternalRM >
void ICFR< Env, InternalRM, ExternalRM >::_sample_plan(size_t p_idx)
{
   auto& st = m_structures[p_idx];
   std::ranges::fill(st.ext_active, char{0});

   constexpr size_t npos = std::numeric_limits< size_t >::max();
   for(auto id : std::views::iota(size_t{0}, st.infosets.size())) {
      auto& rec = st.infosets[id];

      // walk the own ancestor chain: the infoset is plan-reachable iff every
      // chain requirement is met; otherwise the FIRST mismatching position k
      // identifies the unique co-trigger sequence sigma^t_I =
      // (chain[k].J, pi^t(chain[k].J)) whose external unit is consulted here
      // (Algorithm 1 line 14; uniqueness holds because the plan agrees with the
      // chain everywhere before its first deviation by construction). The
      // number of matched leading positions is kept for the on-path laminar
      // accumulators of the update phase.
      size_t dev_pos = npos;
      size_t prefix_matched = 0;
      for(auto [k, ancestor] : std::views::enumerate(rec.chain)) {
         if(st.choice[ancestor.first] != ancestor.second) {
            dev_pos = k;
            break;
         }
         ++prefix_matched;
      }
      st.prefix_match[id] = prefix_matched;

      if(dev_pos == npos) {
         st.reached[id] = 1;
         // fresh carrier per consultation: registries differ across infosets and
         // a reused map would leak stale (unregistered-here) actions into sampling
         policy_out_type policy{};
         InternalRM::recommend(rec.internal_rm, policy, m_iteration);
         st.choice[id] = _sample_action(policy, rec.actions);
         auto& instr = st.last_recommendation[id];
         instr.assign(rec.actions.size(), 0.);
         for(auto [a_idx, action] : std::views::enumerate(rec.actions)) {
            instr[a_idx] = policy.at(action);
         }
      } else {
         st.reached[id] = 0;
         const auto deviating_action_idx = st.choice[rec.chain[dev_pos].first];
         bool found = false;
         for(auto [unit_idx, unit] : std::views::enumerate(rec.ext_units)) {
            if(unit.chain_pos == dev_pos and unit.action_idx == deviating_action_idx) {
               policy_out_type policy{};
               ExternalRM::recommend(unit.rm, policy, m_iteration);
               st.choice[id] = _sample_action(policy, rec.actions);
               st.ext_active[st.ext_unit_offsets[id] + unit_idx] = 1;
               found = true;
               break;
            }
         }
         if(not found) {
            throw std::logic_error(
               "ICFR: no external minimizer unit for the deviating sequence at this "
               "infoset -- the enumeration is inconsistent with the sampled plan"
            );
         }
      }
   }
}

template < typename Env, typename InternalRM, typename ExternalRM >
size_t ICFR< Env, InternalRM, ExternalRM >::_sample_action(
   const policy_out_type& dist,
   const std::vector< action_type >& actions
)
{
   std::uniform_real_distribution< double > uniform(0., 1.);
   const double r = uniform(m_rng);
   double acc = 0.;
   std::optional< action_type > chosen;
   for(const auto& [action, prob] : dist) {
      acc += prob;
      if(r < acc) {
         chosen = action;
         break;
      }
   }
   if(not chosen.has_value()) {
      // floating-point drift fallback: take the highest-probability entry
      double best = -1.;
      for(const auto& [action, prob] : dist) {
         if(prob > best) {
            best = prob;
            chosen = action;
         }
      }
   }
   for(auto [a_idx, action] : std::views::enumerate(actions)) {
      if(action == *chosen) {
         return a_idx;
      }
   }
   throw std::logic_error("ICFR: recommendation contains an unregistered action");
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////// counterfactual value traversal //////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template < typename Env, typename InternalRM, typename ExternalRM >
void ICFR< Env, InternalRM, ExternalRM >::_traverse_values()
{
   for(auto& st : m_structures) {
      std::ranges::fill(st.x_sums, 0.);
      std::ranges::fill(st.s_sums, 0.);
   }
   std::ranges::fill(m_own_match, 1.);
   m_chance_product = 1.;
   for(auto& stack : m_decision_stacks) {
      stack.clear();
   }
   for(auto player : m_roster) {
      m_obs_buffers.at(player).clear();
      m_infostates.at(player) = std::make_shared< info_state_type >(player);
   }
   m_path.clear();
   while(m_arena.size() < 1) {
      m_arena.emplace_back();
   }
   world_state_type& root = m_arena[0].construct_from(*m_root_state);
   _values_visit(root, /*depth=*/0);
}

template < typename Env, typename InternalRM, typename ExternalRM >
auto ICFR< Env, InternalRM, ExternalRM >::_values_visit(world_state_type& state, size_t depth)
   -> std::vector< double >
{
   if(_env().is_terminal(state)) {
      // realized trajectory of iteration t <=> every player's sampled plan
      // matched the path so far (chance has no plan; its probabilities are
      // folded into the values below)
      bool realized = true;
      for(double match : m_own_match) {
         if(match <= 0.) {
            realized = false;
            break;
         }
      }
      if(realized) {
         m_terminal_counts[m_path] += 1.;
      }
      const auto rewards = collect_rewards(_env(), state, _env().players(*m_root_state));
      std::vector< double > values(m_roster.size(), 0.);
      for(auto p_idx : std::views::iota(size_t{0}, m_roster.size())) {
         // M_i(z): gated by all OPPONENTS' plans only, weighted by the full
         // chance product p_c(z) (paper Definition of u^t_i[I,a])
         double gate = m_chance_product;
         for(auto q_idx : std::views::iota(size_t{0}, m_roster.size())) {
            if(q_idx != p_idx) {
               gate *= m_own_match[q_idx];
            }
         }
         values[p_idx] = gate * rewards.at(m_roster[p_idx]);
      }
      return values;
   }

   Player active_player = _env().active_player(state);

   if constexpr(concepts::stochastic_env< env_type >) {
      if(active_player == Player::chance) {
         std::vector< double > values(m_roster.size(), 0.);
         for(const auto& outcome : _env().chance_actions(state)) {
            const double outcome_prob = _env().chance_probability(state, outcome);
            // the FULL chance product stays carried in the descent scalar so
            // that every infoset's u^t aggregates share the paper's absolute
            // p_c(z) normalization (folding the prefix into returned values
            // would strip the above-the-infoset chance mass instead)
            const double saved_product = m_chance_product;
            m_chance_product = saved_product * outcome_prob;
            m_path.emplace_back(action_variant_type(outcome));
            EdgeUndo undo;
            world_state_type& next = _advance_edge(state, depth, outcome, undo);
            auto child_values = _values_visit(next, depth + 1);
            _undo_edge(undo);
            m_chance_product = saved_product;
            m_path.pop_back();
            for(auto p_idx : std::views::iota(size_t{0}, m_roster.size())) {
               values[p_idx] += child_values[p_idx];
            }
         }
         return values;
      }
   }

   const size_t p_idx = m_player_index.at(active_player);
   auto& st = m_structures[p_idx];
   const size_t id = st.ids.at(*m_infostates.at(active_player));
   // SAFE as a reference (unlike _enumerate_visit): infoset records are FROZEN by now --
   // '_values_visit' runs only after enumeration+freeze and never inserts into
   // 'st.infosets', so no reallocation can invalidate this binding.
   const auto& actions = st.infosets[id].actions;

   // per-action match indicators of the owner's sampled plan (the owner's OWN
   // play never gates M_j -- counterfactual weighting excludes the updating
   // player entirely; it gates everyone ELSE'S aggregates)
   std::vector< double > matches(actions.size(), 0.);
   matches[st.choice[id]] = 1.;

   std::vector< std::vector< double > > child_values{};
   child_values.reserve(actions.size());
   for(auto [action_idx, action] : std::views::enumerate(actions)) {
      const double saved_match = m_own_match[p_idx];
      m_own_match[p_idx] = saved_match * matches[action_idx];
      m_path.emplace_back(action_variant_type(action));
      EdgeUndo undo;
      world_state_type& next = _advance_edge(state, depth, action, undo);
      child_values.emplace_back(_values_visit(next, depth + 1));
      _undo_edge(undo);
      m_path.pop_back();
      m_own_match[p_idx] = saved_match;
   }

   // X(j,I,a) += M_j(h*a) over every visited history h of the infoset, and
   // S(j,I) += sum_a M_j(h*a) (subtree sums below each edge)
   const size_t offset = st.action_offsets[id];
   double s_accumulator = 0.;
   for(auto action_idx : std::views::iota(size_t{0}, actions.size())) {
      const double m_owner = child_values[action_idx][p_idx];
      st.x_sums[offset + action_idx] += m_owner;
      s_accumulator += m_owner;
   }
   st.s_sums[id] += s_accumulator;

   // aggregate the node's M-values for every player: the owner's own entries
   // ungated, everybody else's gated by the owner's plan indicator
   std::vector< double > values(m_roster.size(), 0.);
   for(auto action_idx : std::views::iota(size_t{0}, actions.size())) {
      for(auto q_idx : std::views::iota(size_t{0}, m_roster.size())) {
         values[q_idx] += child_values[action_idx][q_idx]
                          * (q_idx == p_idx ? 1. : matches[action_idx]);
      }
   }
   return values;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////// Algorithm 1: regret updates ////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template < typename Env, typename InternalRM, typename ExternalRM >
void ICFR< Env, InternalRM, ExternalRM >::_update_player(size_t p_idx)
{
   auto& st = m_structures[p_idx];
   const auto n_infosets = st.infosets.size();

   // ---- stop-values on the infoset graph:
   // u[I,a] = X[I,a] - sum_{J in C(I,a)} S[J]
   // (paper's u^t_i[I,a]: terminals immediately below (I,a))
   for(auto id : std::views::iota(size_t{0}, n_infosets)) {
      const auto offset = st.action_offsets[id];
      for(auto a_idx : std::views::iota(size_t{0}, st.infosets[id].actions.size())) {
         double u = st.x_sums[offset + a_idx];
         for(size_t child : st.infosets[id].children[a_idx]) {
            u -= st.s_sums[child];
         }
         st.u_stop[offset + a_idx] = u;
      }
   }

   // ---- continuation values V^t_J(pi^t) (paper Eq. 8); reverse topological
   // order guarantees all C-children are finished before their parents
   for(auto id : std::views::iota(size_t{0}, n_infosets) | std::views::reverse) {
      const auto offset = st.action_offsets[id];
      const size_t played = st.choice[id];
      double v = st.u_stop[offset + played];
      for(size_t child : st.infosets[id].children[played]) {
         v += st.v_memo[child];
      }
      st.v_memo[id] = v;
   }

   // ---- parameterized utilities u_hat_I(a) = u[I,a] + sum_{J in C(I,a)} V_J
   // (Eq. 9), fed to exactly the units whose recommendation was followed
   for(auto id : std::views::iota(size_t{0}, n_infosets)) {
      auto& rec = st.infosets[id];
      const auto offset = st.action_offsets[id];
      const auto n_actions = static_cast< size_t >(rec.actions.size());

      std::vector< double > u_hat(n_actions, 0.);
      for(auto a_idx : std::views::iota(size_t{0}, n_actions)) {
         double u = st.u_stop[offset + a_idx];
         for(size_t child : rec.children[a_idx]) {
            u += st.v_memo[child];
         }
         u_hat[a_idx] = u;
      }
      const double played_utility = u_hat[st.choice[id]];
      // internal unit observes iff the plan reaches I (indicator 1[pi^t in Pi_i(I)])
      if(st.reached[id] != 0) {
         InternalRM::observe_utilities(rec.internal_rm, u_hat);
         // raw AT-I trigger statistics sigma=(I,a*): on rounds where the plan
         // plays a* at I, accumulate u_hat(b) for every column b and the
         // obedient baseline u_hat(pi^t(I))
         const size_t base = st.raw_internal_offsets[id];
         for(auto a_star : std::views::iota(size_t{0}, n_actions)) {
            if(st.choice[id] == a_star) {
               for(auto b_idx : std::views::iota(size_t{0}, n_actions)) {
                  st.raw_internal[base + a_star * n_actions + b_idx] += u_hat[b_idx];
                  st.stop_internal[base + a_star * n_actions + b_idx] += st.u_stop[offset + b_idx];
               }
               st.obed_internal[base + a_star * n_actions + st.choice[id]] += played_utility;
            }
         }
      }

      // external units observe iff the plan plays their co-trigger sequence
      // (indicator 1[pi^t in Pi_i(sigma)]; resolved during sampling)
      const size_t ext_base = st.ext_unit_offsets[id];
      for(auto [unit_idx, unit] : std::views::enumerate(rec.ext_units)) {
         if(st.ext_active[ext_base + unit_idx] != 0) {
            for(auto a_idx : std::views::iota(size_t{0}, n_actions)) {
               ExternalRM::observe(unit.rm, rec.actions[a_idx], u_hat[a_idx]);
            }
            const size_t raw_base = st.raw_external_offsets[id] + unit_idx * n_actions;
            for(auto b_idx : std::views::iota(size_t{0}, n_actions)) {
               st.raw_external[raw_base + b_idx] += u_hat[b_idx];
               st.stop_external[raw_base + b_idx] += st.u_stop[offset + b_idx];
            }
            st.obed_external[raw_base + st.choice[id]] += played_utility;
         }
      }

      // on-path ancestor triggers sigma=(chain[k].J, required-at-J): raw slices
      // with indicator 1[pi^t agrees with chain positions 0..k]; these have no
      // minimizer unit of their own (see the PlayerStructure note) but enter
      // both the gap evaluation and the Lemma-1 assembly of trigger regrets
      const size_t onpath_base = st.raw_onpath_offsets[id];
      for(auto k : std::views::iota(size_t{0}, rec.chain.size())) {
         if(st.prefix_match[id] >= k + 1) {
            const size_t raw_base = onpath_base + k * n_actions;
            for(auto b_idx : std::views::iota(size_t{0}, n_actions)) {
               st.raw_onpath[raw_base + b_idx] += u_hat[b_idx];
               st.stop_onpath[raw_base + b_idx] += st.u_stop[offset + b_idx];
            }
            st.obed_onpath[raw_base + st.choice[id]] += played_utility;
         }
      }
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////// trigger regrets (Definition 3 / Lemma 1) /////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template < typename Env, typename InternalRM, typename ExternalRM >
double ICFR< Env, InternalRM, ExternalRM >::_trigger_regret_dp(
   const PlayerStructure& st,
   size_t trigger_infoset,
   size_t trigger_action_idx,
   size_t subtree_infoset,
   std::vector< double >& memo,
   std::vector< char >& computed
) const
{
   if(computed[subtree_infoset] != 0) {
      return memo[subtree_infoset];
   }
   const auto& rec = st.infosets[subtree_infoset];
   const auto n_actions = static_cast< size_t >(rec.actions.size());

   // Lemma-1 recursion over the ACTION COLUMN a of the deviation plan:
   //    R_{sigma,I'} = max_a { R_hat_{sigma,I',a} + sum_{I'' in C(I',a)} R_{sigma,I''} }
   // where R_hat_{sigma,I',a} lives in one of three raw accumulator slices
   // (see below for which).
   double best = -std::numeric_limits< double >::infinity();
   for(auto action_column : std::views::iota(size_t{0}, n_actions)) {
      // locate this subtree infoset's local slice for sigma:
      //   - at the trigger infoset itself: the INTERNAL block row a* (indicator
      //     1[pi^t reaches J and plays a*])
      //   - strictly below with a* deviating from the chain requirement at J:
      //     the EXTERNAL unit keyed by (position of J in the chain, a*)
      //     (indicator 1[pi^t in Pi_i(sigma)], accumulated on active rounds)
      //   - strictly below with a* being the chain-required action (sigma lies
      //     ON the path to this subtree): the ON-PATH block row of J's chain
      //     position (indicator 1[pi^t agrees with chain positions 0..k]; no
      //     minimizer unit owns this slice, see the PlayerStructure note)
      // R_hat_{sigma,I',a} = raw(sigma,I',a) - obed(sigma,I',a): the exact
      // cumulative laminar regret sum_t 1[x](u_hat(a) - u_hat(pi^t(I')))
      double local_value = 0.;
      if(subtree_infoset == trigger_infoset) {
         const size_t base = st.raw_internal_offsets[trigger_infoset]
                             + trigger_action_idx * n_actions;
         local_value = st.raw_internal[base + action_column]
                       - st.obed_internal[base + action_column];
      } else {
         const auto& chain = rec.chain;
         auto pos_it = std::ranges::find_if(chain, [&](const auto& entry) {
            return entry.first == trigger_infoset;
         });
         if(pos_it == chain.end()) {
            throw std::logic_error("ICFR: trigger infoset is not an ancestor of the subtree");
         }
         const auto chain_pos = static_cast< size_t >(pos_it - chain.begin());
         if(trigger_action_idx == chain[chain_pos].second) {
            const size_t base = st.raw_onpath_offsets[subtree_infoset] + chain_pos * n_actions;
            local_value = st.raw_onpath[base + action_column]
                          - st.obed_onpath[base + action_column];
         } else {
            bool found = false;
            for(auto [unit_idx, unit] : std::views::enumerate(rec.ext_units)) {
               if(unit.chain_pos == chain_pos and unit.action_idx == trigger_action_idx) {
                  const size_t base = st.raw_external_offsets[subtree_infoset]
                                      + unit_idx * n_actions;
                  local_value = st.raw_external[base + action_column]
                                - st.obed_external[base + action_column];
                  found = true;
                  break;
               }
            }
            if(not found) {
               throw std::logic_error("ICFR: missing external accumulator for trigger sequence");
            }
         }
      }

      double value = local_value;
      for(size_t child : rec.children[action_column]) {
         value += _trigger_regret_dp(
            st, trigger_infoset, trigger_action_idx, child, memo, computed
         );
      }
      best = std::max(best, value);
   }
   computed[subtree_infoset] = 1;
   memo[subtree_infoset] = best;
   return best;
}

template < typename Env, typename InternalRM, typename ExternalRM >
auto ICFR< Env, InternalRM, ExternalRM >::trigger_regrets() const -> std::vector< TriggerGapEntry >
{
   if(not m_initialized) {
      return {};
   }
   std::vector< TriggerGapEntry > entries;
   for(auto p_idx : std::views::iota(size_t{0}, m_structures.size())) {
      const auto& st = m_structures[p_idx];
      std::vector< double > memo(st.infosets.size(), 0.);
      std::vector< char > computed(st.infosets.size(), char{0});
      for(auto id : std::views::iota(size_t{0}, st.infosets.size())) {
         const auto& rec = st.infosets[id];
         for(auto a_idx : std::views::iota(size_t{0}, rec.actions.size())) {
            std::ranges::fill(computed, char{0});
            TriggerGapEntry entry{};
            entry.player = m_roster[p_idx];
            entry.infoset_id = id;
            entry.action = rec.actions[a_idx];
            entry.trigger_regret = _trigger_regret_dp(st, id, a_idx, id, memo, computed);
            entries.emplace_back(std::move(entry));
         }
      }
   }
   return entries;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////// EFCE gap evaluation against mu_bar^T (Eq. 7) ////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template < typename Env, typename InternalRM, typename ExternalRM >
auto ICFR< Env, InternalRM, ExternalRM >::evaluate_efce_gap() -> GapReport
{
   if(m_iteration == 0) {
      throw std::logic_error("ICFR: evaluate_efce_gap requires at least one iteration");
   }
   _ensure_initialized();

   GapReport report{};
   report.efce_gap = 0.;
   report.trigger_regret_bound = 0.;
   bool any_sequence = false;
   const double t_inv = 1. / double(m_iteration);

   for(auto p_idx : std::views::iota(size_t{0}, m_structures.size())) {
      const auto& st = m_structures[p_idx];
      std::vector< double > gap_memo(st.infosets.size(), 0.);
      std::vector< char > gap_computed(st.infosets.size(), char{0});
      std::vector< double > regret_memo(st.infosets.size(), 0.);
      std::vector< char > regret_computed(st.infosets.size(), char{0});
      for(auto id : std::views::iota(size_t{0}, st.infosets.size())) {
         const auto& rec = st.infosets[id];
         for(auto a_idx : std::views::iota(size_t{0}, rec.actions.size())) {
            TriggerGapEntry entry{};
            entry.player = m_roster[p_idx];
            entry.infoset_id = id;
            entry.action = rec.actions[a_idx];

            // empirical side of Eq. (7): the max over deviation plans of the
            // triggered agent's value (unnormalized raw sum), minus the
            // obedient baseline accumulated along the trigger condition.
            // obed(sigma) lives in the internal block's row a*: each round
            // writes u_hat(pi^t(I)) into exactly the played column, so the row
            // sum recovers sum_t 1[x] u_hat(played)
            std::ranges::fill(gap_computed, char{0});
            entry.triggered_value = _gap_triggered_dp(st, id, a_idx, id, gap_memo, gap_computed)
                                    * t_inv;
            const size_t obed_base = st.raw_internal_offsets[id] + a_idx * rec.actions.size();
            double obedient_sum = 0.;
            for(auto b_idx : std::views::iota(size_t{0}, rec.actions.size())) {
               obedient_sum += st.obed_internal[obed_base + b_idx];
            }
            entry.obedient_value = obedient_sum * t_inv;
            entry.gap = entry.triggered_value - entry.obedient_value;
            report.efce_gap = std::max(report.efce_gap, entry.gap);

            // certificate side: raw cumulative trigger regret R^T_sigma
            std::ranges::fill(regret_computed, char{0});
            entry.trigger_regret = _trigger_regret_dp(
               st, id, a_idx, id, regret_memo, regret_computed
            );
            report.trigger_regret_bound = std::max(
               report.trigger_regret_bound, entry.trigger_regret * t_inv
            );

            report.sequences.emplace_back(std::move(entry));
            any_sequence = true;
         }
      }
   }
   if(not any_sequence) {
      report.efce_gap = 0.;
      report.trigger_regret_bound = 0.;
   }
   return report;
}

template < typename Env, typename InternalRM, typename ExternalRM >
double ICFR< Env, InternalRM, ExternalRM >::_gap_triggered_dp(
   const PlayerStructure& st,
   size_t trigger_infoset,
   size_t trigger_action_idx,
   size_t subtree_infoset,
   std::vector< double >& memo,
   std::vector< char >& computed
) const
{
   if(computed[subtree_infoset] != 0) {
      return memo[subtree_infoset];
   }
   const auto& rec = st.infosets[subtree_infoset];
   const auto n_actions = static_cast< size_t >(rec.actions.size());

   double best = -std::numeric_limits< double >::infinity();
   for(auto action_column : std::views::iota(size_t{0}, n_actions)) {
      // raw(sigma, I', b) blocks: sum_t 1[x^t(sigma)] u_hat^t_{I'}(b)
      double local_value = 0.;
      if(subtree_infoset == trigger_infoset) {
         const size_t base = st.raw_internal_offsets[trigger_infoset]
                             + trigger_action_idx * n_actions;
         // STOP-value twin: the deviation DP composes immediate terminal values
         // along the deviation plan's own path (composing u_hat would double-count
         // the continuations embedded in it)
         local_value = st.stop_internal[base + action_column];
      } else {
         const auto& chain = rec.chain;
         auto pos_it = std::ranges::find_if(chain, [&](const auto& entry) {
            return entry.first == trigger_infoset;
         });
         if(pos_it == chain.end()) {
            throw std::logic_error("ICFR: trigger infoset is not an ancestor of the subtree");
         }
         const auto chain_pos = static_cast< size_t >(pos_it - chain.begin());
         if(trigger_action_idx == chain[chain_pos].second) {
            const size_t base = st.raw_onpath_offsets[subtree_infoset] + chain_pos * n_actions;
            local_value = st.stop_onpath[base + action_column];
         } else {
            bool found = false;
            for(auto [unit_idx, unit] : std::views::enumerate(rec.ext_units)) {
               if(unit.chain_pos == chain_pos and unit.action_idx == trigger_action_idx) {
                  const size_t base = st.raw_external_offsets[subtree_infoset]
                                      + unit_idx * n_actions;
                  local_value = st.stop_external[base + action_column];
                  found = true;
                  break;
               }
            }
            if(not found) {
               throw std::logic_error("ICFR: missing external accumulator for trigger sequence");
            }
         }
      }

      double value = local_value;
      for(size_t child : rec.children[action_column]) {
         value += _gap_triggered_dp(st, trigger_infoset, trigger_action_idx, child, memo, computed);
      }
      best = std::max(best, value);
   }
   computed[subtree_infoset] = 1;
   memo[subtree_infoset] = best;
   return best;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// diagnostic accessors ////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template < typename Env, typename InternalRM, typename ExternalRM >
size_t ICFR< Env, InternalRM, ExternalRM >::num_infosets(Player player) const
{
   if(not m_initialized) {
      return 0;
   }
   return m_structures.at(m_player_index.at(player)).infosets.size();
}

template < typename Env, typename InternalRM, typename ExternalRM >
size_t ICFR< Env, InternalRM, ExternalRM >::infoset_action_count(Player player, size_t infoset_id)
   const
{
   if(not m_initialized) {
      return 0;
   }
   return m_structures.at(m_player_index.at(player)).infosets.at(infoset_id).actions.size();
}

template < typename Env, typename InternalRM, typename ExternalRM >
size_t ICFR< Env, InternalRM, ExternalRM >::external_unit_count(Player player, size_t infoset_id)
   const
{
   if(not m_initialized) {
      return 0;
   }
   return m_structures.at(m_player_index.at(player)).infosets.at(infoset_id).ext_units.size();
}

template < typename Env, typename InternalRM, typename ExternalRM >
std::vector< std::pair< size_t, icfr_action_type_of< Env > > >
ICFR< Env, InternalRM, ExternalRM >::infoset_chain(Player player, size_t infoset_id) const
{
   using ret_entry = std::pair< size_t, icfr_action_type_of< Env > >;
   if(not m_initialized) {
      return {};
   }
   const auto& st = m_structures.at(m_player_index.at(player));
   const auto& chain = st.infosets.at(infoset_id).chain;
   return chain | std::views::transform([&](const auto& entry) {
             return ret_entry{entry.first, st.infosets[entry.first].actions[entry.second]};
          })
          | std::ranges::to< std::vector >();
}

template < typename Env, typename InternalRM, typename ExternalRM >
std::pair< size_t, icfr_action_type_of< Env > >
ICFR< Env, InternalRM, ExternalRM >::external_unit_descriptor(
   Player player,
   size_t infoset_id,
   size_t unit_idx
) const
{
   const auto& st = m_structures.at(m_player_index.at(player));
   const auto& rec = st.infosets.at(infoset_id);
   const auto& unit = rec.ext_units.at(unit_idx);
   const auto& ancestor = rec.chain.at(unit.chain_pos);
   return {unit.chain_pos, st.infosets[ancestor.first].actions[unit.action_idx]};
}

template < typename Env, typename InternalRM, typename ExternalRM >
const std::vector< double >& ICFR< Env, InternalRM, ExternalRM >::last_recommendation_distribution(
   Player player,
   size_t infoset_id
) const
{
   const static std::vector< double > empty{};
   if(not m_initialized) {
      return empty;
   }
   return m_structures.at(m_player_index.at(player)).last_recommendation.at(infoset_id);
}

}  // namespace nor::rm
