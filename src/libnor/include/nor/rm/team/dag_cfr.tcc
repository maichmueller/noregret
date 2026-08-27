
#pragma once

// included from "dag_cfr.hpp" -- do not include directly

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nor::rm::team {

///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// learner core (Alg. 2) ///////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

template < typename Env >
void AdversarialTeamDagCfr< Env >::iterate(size_t n_iters)
{
   for([[maybe_unused]] auto _ : std::views::iota(size_t{0}, n_iters)) {
      // phase 1: current plans of both planes (Algorithm 2 NextStrategy, top-down); the pass
      // also folds this iteration's reach-weighted mass into the average-plan accumulators
      _next_strategy(k_team_plane);
      _next_strategy(k_adversary_plane);

      // terminal-flow snapshots r_side(z) of this iteration's plans; leaf rows are shared
      // between both planes (identical underlying tree and enumeration order)
      const std::vector< double > flows_team = _terminal_flows(k_team_plane, m_rt[k_team_plane].inflow);
      const std::vector< double > flows_adv =
         _terminal_flows(k_adversary_plane, m_rt[k_adversary_plane].inflow);

      const auto n_leaves = m_leaf_utilities[k_team_plane].size();
      if(m_planes[k_team_plane].terminal_count() != n_leaves
         or m_planes[k_adversary_plane].terminal_count() != n_leaves) [[unlikely]] {
         throw std::logic_error("AdversarialTeamDagCfr: plane leaf-table misalignment");
      }

      // phase 2: joint terminal weights g_side(z) = p_c(z) * u_side(z) * r_opposing(z)
      std::vector< double > g0(n_leaves), g1(n_leaves);
      for(auto leaf : std::views::iota(size_t{0}, n_leaves)) {
         g0[leaf] = m_leaf_utilities[k_team_plane][leaf] * flows_adv[leaf];
         g1[leaf] = m_leaf_utilities[k_adversary_plane][leaf] * flows_team[leaf];
      }

      // phase 3: counterfactual backpropagation + RM+ regret folds (per plane)
      _observe_utility(k_team_plane, g0);
      _observe_utility(k_adversary_plane, g1);

      ++m_iteration;
   }
}

/**
 * Top-down pass of Algorithm 2 NextStrategy over one plane:
 *   x[root]=1;  x[s] = sum over parents;
 *   RM+ recommendation x'[s, a] = R+[s,a]/sum_a R+[s,a]  (uniform when the sum vanishes);
 *   x[s,a] = x'[s,a] * x[s].
 * The average-plan accumulators fold this iteration's contribution linearly weighted.
 */
template < typename Env >
void AdversarialTeamDagCfr< Env >::_next_strategy(size_t side)
{
   const auto& dag = m_planes[side];
   auto& rt = m_rt[side];

   std::ranges::fill(rt.inflow, 0.);
   rt.inflow[dag.root_belief()] = 1.;

   const double avg_weight = std::pow(
      static_cast< double >(m_iteration + 1), m_config.linear_weight_power
   );

   // beliefs are in creation order == topological order (parents constructed first)
   for(auto b : std::views::iota(size_t{0}, dag.belief_count())) {
      const auto& rec = dag.belief(b);
      if(rec.terminal) {
         continue;
      }
      const double inflow_b = rt.inflow[b];
      if(inflow_b <= 0.) {
         // unreachable under the current plan; keep edgep at its last recommendation but
         // zero the fold so stale weights cannot leak into the average plan
         continue;
      }
      const size_t k_count = rec.prescriptions.size();
      double pos_sum = 0.;
      for(double r : rt.regret[b]) {
         pos_sum += std::max(0., r);
      }
      auto& edgep = rt.current_edgep[b];
      if(pos_sum > 0.) {
         for(auto k : std::views::iota(size_t{0}, k_count)) {
            edgep[k] = std::max(0., rt.regret[b][k]) / pos_sum;
         }
      } else {
         const double uniform_prob = 1. / static_cast< double >(k_count);
         for(auto k : std::views::iota(size_t{0}, k_count)) {
            edgep[k] = uniform_prob;
         }
      }
      for(auto k : std::views::iota(size_t{0}, k_count)) {
         rt.strategy_sum[b][k] += avg_weight * inflow_b * edgep[k];
      }
      // route the flow through the inactive children (each child is exactly one prescription)
      for(auto k : std::views::iota(size_t{0}, k_count)) {
         // account lazily: store per-prescription routed masses as belief-relative values on
         // the inactive layer below
         rt.inactive_inflow[rec.prescription_children[k]] += inflow_b * edgep[k];
      }
   }

   // second sweep over inactive nodes distributing to their component child beliefs
   for(auto o : std::views::iota(size_t{0}, dag.inactive_count())) {
      const double inflow_o = rt.inactive_inflow[o];
      if(inflow_o <= 0.) {
         continue;
      }
      for(const BeliefId child : dag.inactive(o).components) {
         rt.inflow[child] += inflow_o;
      }
      rt.inactive_inflow[o] = 0.;  // reset scratch for the next iteration
   }
}

/**
 * Bottom-up pass of Algorithm 2 ObserveUtility over one plane given the joint terminal
 * weights 'g':
 *   u[z] = g[z]                                (singleton terminal beliefs)
 *   u[s] = sum_k u[child_k] * x'[s,k]
 *   R[s,k] += u[child_k] - u[s]
 *   u[parent(s)] += u[s]
 */
template < typename Env >
void AdversarialTeamDagCfr< Env >::_observe_utility(size_t side, const std::vector< double >& g)
{
   const auto& dag = m_planes[side];
   auto& rt = m_rt[side];

   auto& ubuf_belief = rt.ubuf_belief;
   auto& ubuf_inactive = rt.ubuf_inactive;
   std::ranges::fill(ubuf_belief, 0.);
   std::ranges::fill(ubuf_inactive, 0.);

   for(auto leaf : std::views::iota(size_t{0}, dag.terminal_count())) {
      ubuf_belief[dag.terminal_belief_id(leaf)] = g[leaf];
   }

   for(size_t b = dag.belief_count(); b-- > 0;) {
      const auto& rec = dag.belief(b);
      if(rec.terminal) {
         // singleton terminal: its preloaded g value flows straight into the inactive parent
         for(const InactiveId parent : rec.parents) {
            ubuf_inactive[parent] += ubuf_belief[b];
         }
         continue;
      }
      double val_mix = 0.;
      for(auto k : std::views::iota(size_t{0}, rec.prescriptions.size())) {
         val_mix += ubuf_inactive[rec.prescription_children[k]] * rt.current_edgep[b][k];
      }
      for(auto k : std::views::iota(size_t{0}, rec.prescriptions.size())) {
         // house RM+ arithmetic (== RegretMatchingPlus<>::observe)
         rt.regret[b][k] += ubuf_inactive[rec.prescription_children[k]] - val_mix;
      }
      ubuf_belief[b] = val_mix;
      for(const InactiveId parent : rec.parents) {
         ubuf_inactive[parent] += val_mix;
      }
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// evaluation ////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Normalized average plan of one plane: RMS-free plain strategy-sum normalization with
 * uniform fallbacks, followed by an average-flow recurrence mirroring _next_strategy.
 */
template < typename Env >
auto AdversarialTeamDagCfr< Env >::_average_plan(size_t side) const
   -> std::pair< std::vector< std::vector< double > >, std::vector< double > >
{
   const auto& dag = m_planes[side];
   const auto& sums = m_rt[side].strategy_sum;

   std::vector< std::vector< double > > edgep(dag.belief_count());
   for(auto b : std::views::iota(size_t{0}, dag.belief_count())) {
      const auto& rec = dag.belief(b);
      if(rec.terminal) {
         continue;
      }
      const auto k_count = rec.prescriptions.size();
      const double total = std::accumulate(sums[b].begin(), sums[b].end(), 0.);
      auto& probs = edgep[b];
      probs.resize(k_count);
      if(total > 0.) {
         for(auto k : std::views::iota(size_t{0}, k_count)) {
            probs[k] = sums[b][k] / total;
         }
      } else {
         const double uniform_prob = 1. / static_cast< double >(k_count);
         for(auto k : std::views::iota(size_t{0}, k_count)) {
            probs[k] = uniform_prob;
         }
      }
   }

   std::vector< double > inflow(dag.belief_count(), 0.);
   inflow[dag.root_belief()] = 1.;
   return {std::move(edgep), _terminal_flows(side, inflow, /*edgep_override=*/&edgep)};
}

template < typename Env >
auto AdversarialTeamDagCfr< Env >::average_realizations(size_t side) const
   -> std::vector< double >
{
   return _average_plan(side).second;
}

template < typename Env >
auto AdversarialTeamDagCfr< Env >::coordinator_plan(size_t side) const
   -> std::vector< std::vector< double > >
{
   std::vector< std::vector< double > > out;
   out.reserve(m_rt[side].strategy_sum.size());
   for(const auto& row : m_rt[side].strategy_sum) {
      const double total = std::accumulate(row.begin(), row.end(), 0.);
      if(total > 0.) {
         auto norm = row;
         for(double& entry : norm) {
            entry /= total;
         }
         out.emplace_back(std::move(norm));
      } else {
         out.emplace_back(row.size(), 1. / static_cast< double >(std::max(row.size(), size_t{1})));
      }
   }
   return out;
}

template < typename Env >
auto AdversarialTeamDagCfr< Env >::evaluate() -> Evaluation
{
   Evaluation eval{};
   eval.iterations = m_iteration;

   const auto [avg_edges0, avg_flows0] = _average_plan(k_team_plane);
   const auto [unused_edges1, avg_flows1] = _average_plan(k_adversary_plane);
   static_cast< void >(unused_edges1);

   const auto n_leaves = m_planes[k_team_plane].terminal_count();
   double pair_value = 0.;
   for(auto leaf : std::views::iota(size_t{0}, n_leaves)) {
      pair_value += m_leaf_utilities[k_team_plane][leaf] * avg_flows0[leaf] * avg_flows1[leaf];
   }
   eval.average_pair_values[k_team_plane] = pair_value;
   eval.average_pair_values[k_adversary_plane] =
      std::invoke([&] {
         double adv_value = 0.;
         const auto u1 = _coalition_leaf_utility(
            std::vector< Player >(m_planes[k_adversary_plane].members())
         );
         for(auto leaf : std::views::iota(size_t{0}, n_leaves)) {
            adv_value += u1[leaf] * avg_flows0[leaf] * avg_flows1[leaf];
         }
         return adv_value;
      });

   for(const auto side : {k_team_plane, k_adversary_plane}) {
      double positive_mass = 0.;
      for(const auto& row : m_rt[side].regret) {
         for(double r : row) {
            positive_mass += std::max(0., r);
         }
      }
      eval.regret_certificates[side] =
         positive_mass / static_cast< double >(std::max(eval.iterations, size_t{1}));
   }
   eval.saddle_gap_proxy =
      eval.regret_certificates[k_team_plane] + eval.regret_certificates[k_adversary_plane];

   eval.br_surrogate_values[k_team_plane] =
      br_surrogate_value(k_adversary_plane);  // adversary BR vs frozen team plan
   eval.br_surrogate_values[k_adversary_plane] =
      br_surrogate_value(k_team_plane);  // team BR proxy vs frozen adversary plan

   return eval;
}

/**
 * Non-destructive empirical best-response bracket: snapshots the queried plane's learner,
 * retrains it alone against the FROZEN opposing average realization vector, tracks the best
 * attained pairing value, then rolls every mutated buffer back.
 */
template < typename Env >
double AdversarialTeamDagCfr< Env >::br_surrogate_value(size_t side, size_t replay_rounds)
{
   const size_t other = side == k_team_plane ? k_adversary_plane : k_team_plane;

   PlaneRuntime snapshot = m_rt[side];
   const size_t saved_iteration = m_iteration;
   const auto frozen_opponent_flows = average_realizations(other);
   const auto& own_utilities = m_leaf_utilities[side];

   const auto attainable_value = [&]() {
      const auto mine = _terminal_flows(side, m_rt[side].inflow);
      double v = 0.;
      for(auto leaf : std::views::iota(size_t{0}, own_utilities.size())) {
         v += own_utilities[leaf] * frozen_opponent_flows[leaf] * mine[leaf];
      }
      return v;
   };

   double best = attainable_value();
   for([[maybe_unused]] auto _ : std::views::iota(size_t{0}, replay_rounds)) {
      _next_strategy(side);
      std::vector< double > g(own_utilities.size());
      for(auto leaf : std::views::iota(size_t{0}, own_utilities.size())) {
         g[leaf] = own_utilities[leaf] * frozen_opponent_flows[leaf];
      }
      _observe_utility(side, g);
      best = std::max(best, attainable_value());
   }

   m_rt[side] = std::move(snapshot);
   m_iteration = saved_iteration;
   return best;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// plan extraction //////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

template < typename Env >
auto AdversarialTeamDagCfr< Env >::decentralized_policies(size_t side) const
   -> player_hashmap<
      dense_hashmap< info_state_type, HashmapActionPolicy< action_type > > >
{
   using policy_table =
      dense_hashmap< info_state_type, HashmapActionPolicy< action_type > >;
   player_hashmap< policy_table > result;

   const auto& dag = m_planes[side];
   const auto [avg_edges, avg_flows] = _average_plan(side);

   // numerator/denominator accumulators per global team infoset of this plane
   std::unordered_map<
      size_t,
      HashmapActionPolicy< action_type >,
      common::value_hasher< size_t > >
      numerators;
   std::unordered_map< size_t, double, common::value_hasher< size_t > > engagement;

   for(auto b : std::views::iota(size_t{0}, dag.belief_count())) {
      const auto& rec = dag.belief(b);
      if(rec.terminal or avg_flows[b] <= 0.) {
         continue;
      }
      const double inflow_b = avg_flows[b];
      const auto k_count = rec.prescriptions.size();

      for(const auto& slot : rec.slots) {
         engagement[slot.infoset_global] += inflow_b;
      }
      for(auto presc_idx : std::views::iota(size_t{0}, k_count)) {
         const double mass = inflow_b * avg_edges[b][presc_idx];
         if(mass <= 0.) {
            continue;
         }
         for(auto [slot_pos, slot_ref] : std::views::enumerate(rec.slots)) {
            const size_t action_idx = rec.prescriptions[presc_idx].slot_action_idx[slot_pos];
            const action_type& action = slot_ref.actions[action_idx];
            // HashmapActionPolicy default-inserts 0 entries, enabling += semantics via []
            numerators[slot_ref.infoset_global][action] += mass;
         }
      }
   }

   for(const auto& [global_id, numerator_policy] : numerators) {
      const double denom = engagement.at(global_id);
      if(denom <= 0.) [[unlikely]] {
         continue;
      }
      const size_t member_idx = dag.slot_of_member_infoset(global_id);
      const Player member = dag.members().at(member_idx);
      HashmapActionPolicy< action_type > normalized{};
      for(const auto& [action, mass] : numerator_policy) {
         normalized.emplace(action, mass / denom);
      }
      const auto& canonical_istate =
         dag.member_infostate(member_idx, dag.local_of_member_infoset(global_id));
      result[member][canonical_istate] = std::move(normalized);
   }
   return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// internal helpers /////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

template < typename Env >
auto AdversarialTeamDagCfr< Env >::_coalition_leaf_utility(const std::vector< Player >& coalition)
   const -> std::vector< double >
{
   const auto& dag = m_planes[k_team_plane];  // shared payoff rows across planes
   std::vector< size_t > roster_slots;
   roster_slots.reserve(coalition.size());
   for(auto player : coalition) {
      roster_slots.emplace_back(dag.roster_index(player));
   }
   std::vector< double > column(dag.terminal_count(), 0.);
   for(auto leaf : std::views::iota(size_t{0}, dag.terminal_count())) {
      double acc = 0.;
      for(auto seat : roster_slots) {
         acc += dag.terminal_payoff(leaf, seat);
      }
      column[leaf] = dag.terminal_weight(leaf) * acc;
   }
   return column;
}

template < typename Env >
auto AdversarialTeamDagCfr< Env >::_terminal_flows(
   size_t side, const std::vector< double >& inflow,
   const std::vector< std::vector< double > >* edgep_override
) const -> std::vector< double >
{
   const auto& dag = m_planes[side];
   // forward-propagate the supplied/overridden belief inflow through the DAG layers and read
   // off the singleton-terminal belief masses
   std::vector< double > inactive_flow(dag.inactive_count(), 0.);
   std::vector< double > belief_flow = inflow;

   for(auto b : std::views::iota(size_t{0}, dag.belief_count())) {
      const auto& rec = dag.belief(b);
      if(rec.terminal or belief_flow[b] <= 0.) {
         continue;
      }
      for(auto k : std::views::iota(size_t{0}, rec.prescriptions.size())) {
         const double prob = edgep_override ? (*edgep_override)[b][k]
                                            : m_rt[side].current_edgep[b][k];
         inactive_flow[rec.prescription_children[k]] += belief_flow[b] * prob;
      }
   }
   for(auto o : std::views::iota(size_t{0}, dag.inactive_count())) {
      for(const BeliefId child : dag.inactive(o).components) {
         belief_flow[child] += inactive_flow[o];
      }
   }

   std::vector< double> out(dag.terminal_count());
   double mass_total = 0.;
   for(auto leaf : std::views::iota(size_t{0}, dag.terminal_count())) {
      out[leaf] = belief_flow[dag.terminal_belief_id(leaf)];
      mass_total += out[leaf];
   }
   if(mass_total > 0.) {
      for(double& f : out) {
         f /= mass_total;
      }
   }
   return out;
}

}  // namespace nor::rm::team
