#pragma once

#include <spdlog/spdlog.h>

#include <algorithm>
#include <limits>

#include "cfr.hpp"

namespace nor::rm {

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
auto VanillaCFR< config, Env, Policy, AveragePolicy >::iterate(size_t n_iters)
{
   std::vector< StateValueMap > root_values_per_iteration;
   root_values_per_iteration.reserve(n_iters);
   for([[maybe_unused]] auto _ : std::views::iota(size_t(0), n_iters)) {
      SPDLOG_DEBUG("Iteration number: {}", _iteration());
      StateValueMap value = std::invoke([&] {
         if constexpr(config.update_mode == UpdateMode::alternating) {
            auto player_to_update = _cycle_player_to_update();
            if(_iteration() < _env().players(*_root_state_uptr()).size() - 1) [[unlikely]] {
               return _iterate< true >(player_to_update);
            } else [[likely]] {
               return _iterate< false >(player_to_update);
            }
         } else {
            if(_iteration() == 0) [[unlikely]] {
               return _iterate< true >(std::nullopt);
            } else [[likely]] {
               return _iterate< false >(std::nullopt);
            }
         }
      });
      root_values_per_iteration.emplace_back(std::move(value));
      _iteration()++;
   }
   return root_values_per_iteration;
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
auto VanillaCFR< config, Env, Policy, AveragePolicy >::iterate(
   std::optional< Player > player_to_update
)
   requires(config.update_mode == UpdateMode::alternating)
{
   SPDLOG_DEBUG("Iteration number: {}", _iteration());
   // run the iteration
   StateValueMap values = [&] {
      if(_iteration() < _env().players(*_root_state_uptr()).size() - 1)
         return _iterate< true >(_cycle_player_to_update(player_to_update));
      else
         return _iterate< false >(_cycle_player_to_update(player_to_update));
   }();
   // and increment our iteration counter
   _iteration()++;
   return std::vector< StateValueMap >{std::move(values)};
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
template < bool initializing_run, bool use_current_policy >
auto VanillaCFR< config, Env, Policy, AveragePolicy >::_iterate(
   std::optional< Player > player_to_update
)
{
   auto root_players = _env().players(root_state());

   // start a fresh touched-infoset recording cycle (D3): the list is cleared
   // every iteration so its memory stays bounded by one traversal's worth of
   // infosets
   ++m_sweep_clock;
   m_touched_infonodes.clear();

   // the traversal containers are created once per iteration here (cheap) and
   // then mutated in place down the recursion with explicit save/restore at
   // every recursion boundary; no per-edge container copies happen anymore.
   auto rp_map = std::invoke([&] {
      ReachProbabilityMap rp{};
      for(auto player : root_players) {
         rp.get().emplace(player, 1.);
      }
      return rp;
   });
   auto obs_map = std::invoke([&] {
      ObservationbufferMap obs{};
      for(auto player : root_players | utils::is_actual_player_filter) {
         obs.emplace(player);
      }
      return obs;
   });
   auto infostates = std::invoke([&] {
      InfostateSptrMap istates{};
      for(auto player : root_players | utils::is_actual_player_filter) {
         istates.emplace(player, std::make_shared< info_state_type >(player));
      }
      return istates;
   });
   // per-depth reusable slots of the insertion-ordered action -> child-values
   // accumulator consumed by update_regret_and_policy: grown on demand by the
   // traversal and cleared between successive visitors of a depth -- the same
   // reuse discipline as the obs buffers above.
   ActionValueArena< action_variant_type > action_value_arena{};

   // copy-assign the root state into arena slot 0 (reused across iterations)
   world_state_type& root_arena_state = _arena_state(0, *_root_state_uptr());

   auto root_game_value = _traverse< initializing_run, use_current_policy >(
      player_to_update,
      root_arena_state,
      /*depth=*/0,
      rp_map,
      obs_map,
      infostates,
      action_value_arena
   );

   if(not m_infonode_presized) {
      m_infonode_presized = true;
      // D4: the first full traversal discovered every reachable infostate;
      // pre-size so later iterations never trigger a rehash (partial-pruning
      // configs may still grow beyond this, but never shrink below it)
      m_infonode.reserve(m_infonode.size());
   }

   if constexpr(use_current_policy) {
      _initiate_regret_minimization(player_to_update);
   }
   return root_game_value;
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
auto VanillaCFR< config, Env, Policy, AveragePolicy >::_arena_state(
   size_t depth,
   const world_state_type& source
) -> world_state_type&
{
   if(m_traversal_state_arena.size() <= depth) {
      m_traversal_state_arena.resize(depth + 1);
   }
   // reconstruct the slot from 'source' in place: no allocation after the
   // first visit to this depth and no requirement on copy assignment
   return m_traversal_state_arena[depth].construct_from(source);
}

namespace detail {

/// exact argmin over w >= 0 of
///    f(w) = sum_j max(0, R_j + w r_j)^2 / (w_sum + w)^2
/// -- the greedy-weights objective (Zhang, Lerer & Brown, AAAI 2022,
/// arXiv:2204.04826, Algorithm 1: w <- argmin_w phi((R + w r)/(wsum + w)) with
/// phi(x) = sum_j max(0, x_j)^2; Appendix F: for CFR the sum ranges over all
/// (infostate, action) pairs, which is exactly what 'regret_pairs' carries).
///
/// Each normalized component q_j(w) = (R_j + w r_j)/(w_sum + w) is monotone in w
/// (its derivative has the constant sign of r_j * w_sum - R_j) and crosses zero
/// at most once, at w = -R_j/r_j. f is therefore smooth on the finitely many
/// intervals between those breakpoints and attains its minimum on each interval
/// either at a boundary or at the unique stationary point of that interval:
///    0 = d/dw sum_{j in S} q_j(w)^2   =>   w_S = (A0 - A1 w_sum) / (A2 w_sum - A1)
/// with A0/A1/A2 = sums over the interval's active set S of R^2 / R*r / r^2.
/// Evaluating all breakpoints plus one clamped stationary point per interval
/// finds the global minimum exactly -- the O(|P||A|)-points procedure mentioned
/// in the paper ("computed exactly by checking O(|P||A|) points"); weights above
/// 1 are later rescaled by the solver anyway, so tail-region precision is moot.
/// Returns 0 when no component carries signal (all r_j == 0); callers apply the
/// weight floor in that case.
[[nodiscard]] inline double greedy_optimal_weight(
   const std::vector< std::pair< double, double > >& regret_pairs,
   double w_sum,
   std::vector< double >& scratch_breakpoints
)
{
   scratch_breakpoints.clear();
   scratch_breakpoints.emplace_back(0.);
   bool has_signal = false;
   for(const auto& [cumul_regret, instant_regret] : regret_pairs) {
      if(instant_regret != 0.) {
         has_signal = true;
         const double crossing = -cumul_regret / instant_regret;
         if(crossing > 0. and std::isfinite(crossing)) {
            scratch_breakpoints.emplace_back(crossing);
         }
      }
   }
   if(not has_signal) {
      return 0.;
   }
   std::ranges::sort(scratch_breakpoints);
   scratch_breakpoints.erase(
      std::ranges::unique(scratch_breakpoints).begin(), scratch_breakpoints.end()
   );

   const auto objective = [&](double w) {
      const double denom = w_sum + w;
      double acc = 0.;
      for(const auto& [cumul_regret, instant_regret] : regret_pairs) {
         const double value = cumul_regret + w * instant_regret;
         if(value > 0.) {
            acc += value * value;
         }
      }
      return acc / (denom * denom);
   };

   // stationary point of one constant-sign region [lo, hi]; the active set is probed at
   // the region midpoint since signs only flip at the breakpoints. hi < 0 encodes the
   // unbounded tail region [lo, infinity)
   const auto region_stationary = [&](double lo, double hi) -> double {
      const double probe = hi < 0. ? lo * 2. + 1. : 0.5 * (lo + hi);
      double sum_rr = 0., sum_rcumr = 0., sum_cumul_sq = 0.;
      bool any_active = false;
      for(const auto& [cumul_regret, instant_regret] : regret_pairs) {
         if(cumul_regret + probe * instant_regret > 0.) {
            any_active = true;
            sum_cumul_sq += cumul_regret * cumul_regret;
            sum_rcumr += cumul_regret * instant_regret;
            sum_rr += instant_regret * instant_regret;
         }
      }
      if(not any_active or sum_rr * w_sum == sum_rcumr) {
         return std::numeric_limits< double >::quiet_NaN();
      }
      double w_stat = (sum_cumul_sq - sum_rcumr * w_sum) / (sum_rr * w_sum - sum_rcumr);
      if(hi < 0.) {
         // tail: clamp into [lo, infinity)
         w_stat = std::max(w_stat, lo);
      } else {
         w_stat = std::clamp(w_stat, lo, hi);
      }
      return w_stat;
   };

   double best_w = 0.;
   double best_value = objective(best_w);
   const auto consider = [&](double candidate) {
      if(std::isfinite(candidate) and candidate >= 0.) {
         const double value = objective(candidate);
         if(value < best_value) {
            best_value = value;
            best_w = candidate;
         }
      }
   };
   for(auto idx : std::views::iota(size_t{1}, scratch_breakpoints.size())) {
      const double lo = scratch_breakpoints[idx - 1];
      const double hi = scratch_breakpoints[idx];
      consider(hi);
      consider(region_stationary(lo, hi));
   }
   // unbounded tail beyond the largest breakpoint
   consider(region_stationary(scratch_breakpoints.back(), -1.));

   return best_w;
}

}  // namespace detail

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
void VanillaCFR< config, Env, Policy, AveragePolicy >::_initiate_regret_minimization(
   const std::optional< Player >& player_to_update
)
{
   // here we now invoke the actual regret minimization procedure for each infostate individually.
   // The sweep is intentionally serial: per-infostate workloads are tiny and a parallel
   // schedule would render float summation orders (and hence table contents)
   // non-deterministic. See the determinism note in rm_utils.hpp.
   if constexpr(config.pruning_mode == CFRPruningMode::none) {
      // D3: sweep exactly the infosets updated during THIS traversal, in
      // pre-order. Without pruning the full-tree traversal visits every
      // infoset of the updating player(s) every iteration, so this set equals
      // the former whole-map filter scan exactly; per-infostate updates are
      // self-contained (recommend/scaling touch only their own node record),
      // hence the visit order is result-invariant and the tables stay bitwise
      // identical.
      if constexpr(config.weighting_mode == CFRWeightingMode::greedy) {
         // greedy weights needs the instantaneous regrets of the COMPLETED traversal
         // aggregated across ALL swept infostates before any table may fold them -- a
         // single common weight per iteration is what retains the convergence guarantee
         // of Theorem 1 in Zhang, Lerer & Brown (AAAI 2022), cf. their Appendix F.
         // (Greedy weights is restricted to simultaneous updates by
         // 'sanity_check_cfr_config', so this sweep always covers every player.)
         _finalize_greedy_iteration(m_touched_infonodes);
         return;
      }
      for(auto& [infostate_ptr, node_ptr] : m_touched_infonodes) {
         _invoke_regret_minimizer(*infostate_ptr, *node_ptr);
      }
   } else {
      // under regret-based/dynamic-thresholding pruning subtrees are skipped,
      // so unvisited-but-alive nodes would drop out of a touched list; keep
      // the exhaustive filter to preserve the exact legacy update set.
      auto node_view = std::invoke([&] {
         if constexpr(config.update_mode == UpdateMode::alternating) {
            return _infonodes()
                   | std::views::filter(
                      [update_player = *player_to_update](const auto& infostate_ptr_data) {
                         return std::get< 0 >(infostate_ptr_data)->player() == update_player;
                      }
                   );
         } else {
            return std::views::all(_infonodes());
         }
      });

      if constexpr(config.weighting_mode == CFRWeightingMode::greedy) {
         // greedy weights: aggregate first, fold second (see the pruning==none branch)
         _finalize_greedy_iteration(node_view);
         return;
      }

      std::for_each(node_view.begin(), node_view.end(), [&](auto& infostate_ptr_data) {
         auto& [infostate_ptr, data] = infostate_ptr_data;
         _invoke_regret_minimizer(*infostate_ptr, data);
      });
   }
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
void VanillaCFR< config, Env, Policy, AveragePolicy >::_publish_recommendation(
   infostate_data_type& istate_data
)
{
   auto& cache = istate_data.current_strategy();
   for(auto [idx, entry] : std::views::enumerate(m_recommend_scratch.entries)) {
      cache[idx] = entry.second;
   }
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
void VanillaCFR< config, Env, Policy, AveragePolicy >::_touch(
   const info_state_type& infostate,
   infostate_data_type& node
)
{
   if(node.sweep_stamp != m_sweep_clock) {
      node.sweep_stamp = m_sweep_clock;
      m_touched_infonodes.emplace_back(&infostate, &node);
   }
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
void VanillaCFR< config, Env, Policy, AveragePolicy >::_materialize_current_policy() const
{
   m_curr_policy_view.clear();
   // every root player keeps an (initially empty) table entry, exactly like the
   // constructor-initialized maps did
   for(auto player : env().players(root_state()) | utils::is_actual_player_filter) {
      m_curr_policy_view.emplace(player, Policy());
   }
   for(auto& [infostate_ptr, node] : m_infonode) {
      const auto& actions = node.actions();
      auto& entry = m_curr_policy_view[infostate_ptr->player()](
         *infostate_ptr, actions, typename base::uniform_policy_type{}
      );
      for(auto [idx, action] : std::views::enumerate(actions)) {
         entry[action] = node.current_prob(idx);
      }
   }
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
void VanillaCFR< config, Env, Policy, AveragePolicy >::_materialize_average_policy() const
{
   m_avg_policy_view.clear();
   for(auto player : env().players(root_state()) | utils::is_actual_player_filter) {
      m_avg_policy_view.emplace(player, AveragePolicy());
   }
   for(auto& [infostate_ptr, node] : m_infonode) {
      if(not node.average_active()) {
         continue;
      }
      const auto& actions = node.actions();
      auto& entry = m_avg_policy_view[infostate_ptr->player()](
         *infostate_ptr, actions, typename base::zero_policy_type{}
      );
      const auto& sums = node.strategy_sum();
      for(auto [idx, action] : std::views::enumerate(actions)) {
         entry[action] = sums[idx];
      }
   }
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
void VanillaCFR< config, Env, Policy, AveragePolicy >::_seed_node_from_user_tables(
   Player active_player,
   const info_state_type& infostate,
   infostate_data_type& node
)
{
   // honor user-provided starting strategies: entries present in the
   // constructor-time tables overlay the uniform defaults for the FIRST
   // traversal of this infoset (after which the minimizer refreshes the
   // recommendation as usual). Absent actions read as zero in the former
   // table-backed representation and are replicated as such.
   auto& curr_table = _policy()[active_player];
   if constexpr(requires { curr_table.find(infostate); }) {
      if(auto found_it = curr_table.find(infostate); found_it != curr_table.end()) {
         auto& cache = node.current_strategy();
         for(auto [idx, action] : std::views::enumerate(node.actions())) {
            if constexpr(requires { found_it->second.find(action); }) {
               if(auto act_it = found_it->second.find(action); act_it != found_it->second.end()) {
                  cache[idx] = act_it->second;
               } else {
                  cache[idx] = 0.;
               }
            } else {
               cache[idx] = found_it->second.at(action);
            }
         }
      }
   }
   auto& avg_table = _average_policy()[active_player];
   if constexpr(requires { avg_table.find(infostate); }) {
      if(auto found_it = avg_table.find(infostate); found_it != avg_table.end()) {
         auto& sums = node.strategy_sum();
         std::ranges::fill(sums, 0.);
         for(auto [idx, action] : std::views::enumerate(node.actions())) {
            if constexpr(requires { found_it->second.find(action); }) {
               if(auto act_it = found_it->second.find(action); act_it != found_it->second.end()) {
                  sums[idx] = act_it->second;
               }
            } else {
               sums[idx] = found_it->second.at(action);
            }
         }
         // seeded records count as active WITHOUT the uniform baseline: the
         // former table held exactly the seeded values and nothing else
         node.activate_average(/*with_uniform_baseline=*/false);
      }
   }
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
template < typename NodeView >
void VanillaCFR< config, Env, Policy, AveragePolicy >::_finalize_greedy_iteration(
   NodeView&& node_view
)
{
   // element-type adapter for the two sweep shapes this sweep is invoked with:
   // the touched-list holds (const info_state_type*, record*) pairs while the
   // exhaustive-filter views iterate the node MAP (sptr keys, record values).
   // Both reduce to a mutable node-record reference here.
   auto record_of = [](auto& entry) -> decltype(auto) {
      if constexpr(requires { std::get< 1 >(entry)->data(); }) {
         return *std::get< 1 >(entry);
      } else {
         return std::get< 1 >(entry);
      }
   };

   // ---- pass 1: aggregate the potential line-search inputs over all swept infostates ----
   auto& regret_pairs = m_greedy_scratch.regret_pairs;
   regret_pairs.clear();
   for(auto& entry : node_view) {
      auto& node_data = record_of(entry).data();
      for(auto idx : std::views::iota(size_t{0}, node_data.regret.size())) {
         regret_pairs.emplace_back(node_data.regret[idx], node_data.instant_regret[idx]);
      }
   }

   // ---- raw iteration weight: exact line search, floored per the paper's CFR recipe -----
   // (arXiv:2204.04826 Appendix F: "a minimum weight floor of 100% of the average weight
   // accrued thus far" worked best for extensive-form games; the fraction is exposed as
   // CFRConfig::greedy_weight_floor_fraction. The denominator counts the implicit weight-1
   // seed iterate, mirroring the reference implementation's loop index.)
   double raw_weight = std::invoke([&] {
      if(m_greedy_state.wsum <= 0.) {
         // first weighted update: no accrued mass to trade off against -- neutral
         // weight of 1 (the paper seeds its accumulator with wsum = 1 analogously)
         return 1.;
      }
      const double searched = detail::greedy_optimal_weight(
         regret_pairs, m_greedy_state.wsum, m_greedy_scratch.breakpoints
      );
      const double floor = config.greedy_weight_floor_fraction * m_greedy_state.wsum
                           / double(m_greedy_state.updates + 1);
      return std::max(searched, floor);
   });
   if(not(std::isfinite(raw_weight) and raw_weight > 0.)) {
      raw_weight = 1.;
   }

   // A near-infinite search result means the objective's infimum lies at w -> infinity:
   // this iteration INVALIDATES the accumulated history (its positive-regret potential is
   // smaller than the running average's). The reference implementation realizes that case
   // as a 1e-6 dilution of all previous mass plus a weight-1 update; we adopt it verbatim
   // (with a finite threshold guarding double overflow). Regular iterations are applied
   // directly with scale 1, exactly like the reference.
   const bool discards_history = not std::isfinite(raw_weight) or raw_weight > 1e12;
   const double effective_weight = discards_history ? 1. : raw_weight;
   const double history_scale = discards_history ? 1e-6 : 1.;
   if(discards_history) {
      ++m_greedy_state.stats.history_discards;
   }

   auto& stats = m_greedy_state.stats;
   ++stats.weight_draws;
   stats.total_weight += effective_weight;
   stats.min_weight = std::min(stats.min_weight, effective_weight);
   stats.max_weight = std::max(stats.max_weight, effective_weight);
   stats.last_weight = effective_weight;

   // ---- pass 2: weighted fold + recommendation refresh across all swept infostates ----
   // D1 adaptation: develop fetched PolicyLabel::current/average TABLES here; under the
   // node-record representation those tables are lazily-materialized VIEWS, so instead the
   // fold is routed straight into the owning records: the recommendation scratch carries
   // the pre-refresh current strategy sigma^t (seeded from the records below, published
   // back after the fold) and the average accumulator writes the strategy sums with the
   // former fetch-time uniform baseline creation. Numerics are unchanged: identical reads,
   // identical writes, identical baselines -- only the storage moved.
   std::for_each(node_view.begin(), node_view.end(), [&](auto& entry) {
      auto& istate_data = record_of(entry);
      const auto& actions = istate_data.actions();
      m_recommend_scratch.prepare(actions);
      for(auto [idx, scratch_entry] : std::views::enumerate(m_recommend_scratch.entries)) {
         scratch_entry.second = istate_data.current_prob(idx);
      }
      detail::average_strategy_accumulator< infostate_data_type > avg_accumulator{&istate_data};
      m_regret_minimizer.apply_weighted_fold(
         istate_data.data(),
         history_scale,
         effective_weight,
         m_recommend_scratch,
         avg_accumulator,
         _iteration()
      );
      _publish_recommendation(istate_data);
   });
   m_curr_view_dirty = true;
   m_avg_view_dirty = true;

   // bookkeeping advances on the same scale as the tables themselves
   m_greedy_state.wsum = history_scale * m_greedy_state.wsum + effective_weight;
   ++m_greedy_state.updates;
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
void VanillaCFR< config, Env, Policy, AveragePolicy >::_invoke_regret_minimizer(
   [[maybe_unused]] const info_state_type& infostate,
   infostate_data_type& istate_data
)
{
   auto& node_data = istate_data.data();
   const auto& actions = istate_data.actions();

   if constexpr(config.weighting_mode == CFRWeightingMode::exponential) {
      // exponential cfr defers all cumulative updates to the end of an
      // iteration: the L1-weighted regret increments are applied here, the
      // average policy numerator is accumulated into the node record (the
      // per-action denominator lives in the minimizer payload) and finally the
      // current policy is refreshed through regular regret matching.
      // NOTE: the scratch is seeded with the node's current strategy because
      // finalize_iteration READS the pre-refresh recommendations for its
      // numerator accumulations before recommend() overwrites them.
      m_recommend_scratch.prepare(actions);
      for(auto [idx, entry] : std::views::enumerate(m_recommend_scratch.entries)) {
         entry.second = istate_data.current_prob(idx);
      }
      detail::average_strategy_accumulator< infostate_data_type > avg_accumulator{&istate_data};
      m_regret_minimizer.finalize_iteration(
         node_data,
         m_recommend_scratch,
         avg_accumulator,
         _iteration(),
         [this](double instant_regret, size_t iteration) {
            return m_expcfr_params.beta(instant_regret, iteration);
         }
      );
      _publish_recommendation(istate_data);
   } else {
      // Lazy-CFR (Zhou et al., ICLR 2020): the recommendation is recomputed ONLY when the
      // latest traversal closed this infostate's open segment (opponent-reach budget
      // exhausted and its buffers folded); otherwise the strategy stays FROZEN and the whole
      // sweep pass collapses to a skipped-refresh counter increment. Under the node-record
      // representation "frozen" means the record's current-strategy cache is left untouched.
      bool refresh_strategy = true;
      if constexpr(config.lazy_update_mode != CFRLazyUpdateMode::off) {
         refresh_strategy = _lazy_consume_refresh(infostate);
      }
      if(refresh_strategy) {
         // derive the recommendation from the (possibly weighted) stored regret
         // into the scratch buffer and publish it as the node's current strategy
         m_recommend_scratch.prepare(actions);
         m_regret_minimizer.recommend(node_data, m_recommend_scratch, _iteration());
         _publish_recommendation(istate_data);

         // scale the accumulated average policy by this iteration's weight.
         // B3: with 'weight_by_cycle' the gamma exponentiation indexes by the
         // cycle number (iteration / num_players) instead of the raw iteration.
         if constexpr(config.weighting_mode == CFRWeightingMode::discounted) {
            const size_t weight_index = m_regret_minimizer.discounted_parameters().weight_by_cycle
                                            ? this->cycle()
                                            : _iteration();
            const double policy_weight = m_regret_minimizer.policy_weight(weight_index);
            for(double& prob : istate_data.strategy_sum()) {
               prob *= policy_weight;
            }
         }
      } else {
         ++m_lazy_stats.skipped_refreshes;
      }
   }
   if constexpr(config.pruning_mode == CFRPruningMode::regret_based or config.pruning_mode == CFRPruningMode::dynamic_thresholding) {
      // the regret tables are now in their post-iteration state (RM+RBP has folded its
      // instantaneous buffer via the replace-if-positive rule); scan for entries negative
      // enough to open a pruning window for the NEXT iterations
      _arm_pruning_windows(infostate, node_data);
   }
   m_curr_view_dirty = true;
   m_avg_view_dirty = true;
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
void VanillaCFR< config, Env, Policy, AveragePolicy >::_force_warm_start_policy(
   const info_state_type& infostate,
   const std::vector< action_type >& actions,
   infostate_data_type& node
)
{
   if constexpr(config.warm_start_iterations > 0) {
      auto& cache = node.current_strategy();
      if(m_warm_start_policy.distribution) {
         const auto fixed_distribution = m_warm_start_policy.distribution(infostate, actions);
         for(auto [idx, action] : std::views::enumerate(actions)) {
            // .at() on purpose: an incomplete fixed distribution is a caller bug we want to
            // fail loudly on instead of silently zeroing actions
            cache[idx] = fixed_distribution.at(action);
         }
      } else {
         const double uniform_prob = 1. / static_cast< double >(actions.size());
         std::ranges::fill(cache, uniform_prob);
      }
   }
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
template < bool initialize_infonodes, bool use_current_policy >
StateValueMap VanillaCFR< config, Env, Policy, AveragePolicy >::_traverse(
   std::optional< Player > player_to_update,
   world_state_type& state,
   size_t depth,
   ReachProbabilityMap& reach_probability,
   ObservationbufferMap& observation_buffer,
   InfostateSptrMap& infostates,
   ActionValueArena< action_variant_type >& action_value_arena
)
{
   if(_env().is_terminal(state)) {
      // NOTE: pass the ROOT participant set explicitly. The default of
      // collect_rewards derives the player set from env.players(terminal
      // state), which excludes players that have folded out of poker-like
      // games. Downstream, update_regret_and_policy looks up every actual
      // player in each action's value map and would throw out_of_range for
      // such subtrees. Environments must support reward() for all initial
      // participants (e.g. leduc pays folded players their sunk stakes).
      return StateValueMap{collect_rewards(_env(), state, _env().players(root_state()))};
   }

   if constexpr(config.pruning_mode == CFRPruningMode::partial) {
      if(_partial_pruning_condition(player_to_update, reach_probability)) {
         // if the entire subtree is pruned then the values that could be found are all 0. for
         // each player
         return StateValueMap{std::invoke([&] {
            StateValueMap::UnderlyingType map;
            for(auto player : _env().players(state) | utils::is_actual_player_pred) {
               map[player] = 0.;
            }
            return map;
         })};
      }
   }

   Player active_player = _env().active_player(state);
   // the state's value for each player. To be filled by the action traversal functions.
   StateValueMap state_value{};
   // each action's value for each player. To be filled by the action traversal functions.
   // REUSED STORAGE: this depth's arena slot replaces the former fresh
   // unordered_map per visit. DFS never interleaves same-depth frames, so
   // clearing on entry/exit keeps successive visitors of a depth isolated
   // without any content save/restore.
   if(action_value_arena.size() <= depth) {
      action_value_arena.resize(depth + 1);
   }
   auto& action_value = action_value_arena[depth];
   action_value.clear();
   // traverse all child states from this state. The constexpr check for determinism in the env
   // allows deterministic envs to not provide certain functions that are only needed in the
   // stochastic case.
   if constexpr(concepts::stochastic_env< env_type >) {
      if(active_player == Player::chance) {
         _traverse_chance_actions< initialize_infonodes, use_current_policy >(
            player_to_update,
            active_player,
            state,
            depth,
            reach_probability,
            observation_buffer,
            infostates,
            state_value,
            action_value,
            action_value_arena
         );
         // if this is a chance node then we don't need to update any regret or average policy
         // after the traversal; release the slot for the next visitor of this depth
         action_value.clear();
         return state_value;
      }
   }

   sptr< info_state_type > this_infostate = infostates.at(active_player);

   _traverse_player_actions< initialize_infonodes, use_current_policy >(
      player_to_update,
      active_player,
      state,
      depth,
      reach_probability,
      observation_buffer,
      infostates,
      state_value,
      action_value,
      action_value_arena
   );

   if constexpr(use_current_policy) {
      // we can only update our regrets and policies if we are traversing with the current
      // policy, since the average policy is not to be changed directly (but through averaging up
      // all current policies)
      if constexpr(config.update_mode == UpdateMode::alternating) {
         // in alternating updates, we only update the regret and strategy if the current
         // player is the chosen player to update.
         if(active_player == player_to_update.value()) {
            update_regret_and_policy(*this_infostate, reach_probability, state_value, action_value);
         }
      } else {
         // if we do simultaenous updates, then we always update the regret and strategy
         // values of the node's active player.
         update_regret_and_policy(*this_infostate, reach_probability, state_value, action_value);
      }
   }
   // release the slot for the next visitor of this depth (the child value maps
   // were consumed above; clearing keeps the slot's heap capacity for reuse)
   action_value.clear();
   return StateValueMap{std::move(state_value)};
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
template < bool initialize_infonodes, bool use_current_policy >
void VanillaCFR< config, Env, Policy, AveragePolicy >::_traverse_player_actions(
   std::optional< Player > player_to_update,
   Player active_player,
   world_state_type& state,
   size_t depth,
   ReachProbabilityMap& reach_probability,
   ObservationbufferMap& observation_buffer,
   InfostateSptrMap& infostates,
   StateValueMap& state_value,
   ActionValueTable< action_variant_type >& action_value,
   ActionValueArena< action_variant_type >& action_value_arena
)
{
   const auto& this_infostate = infostates.at(active_player);
   if constexpr(initialize_infonodes) {
      auto [node_iter, inserted] = _infonodes().try_emplace(
         this_infostate, _env().actions(active_player, state)
      );
      if(inserted) {
         // first registration: overlay user-seeded starting strategies from the
         // constructor-time policy tables so externally provided starting
         // points keep their historical meaning for the FIRST traversal
         _seed_node_from_user_tables(active_player, *this_infostate, node_iter->second);
      }
   }
   auto& this_node = _infonode(this_infostate);
   const auto& actions = this_node.actions();
   if constexpr(config.warm_start_iterations > 0 and use_current_policy) {
      // WARM START pre-play phase: while warm_start_active(_iteration()) holds, EVERY
      // player's played strategy is forced to the fixed warm-start policy by overwriting
      // the node record's cached current-strategy AT THE VISIT, i.e. at the exact point
      // where the edge probabilities are read (D1: current_prob() resolves the cache);
      // the counterfactual regret updates below therefore
      // accumulate best-response information about a fully STATIONARY opposition into
      // whoever is being updated this iteration. Roles alternate exactly like regular CFR.
      // Everything else -- regret increments and all minimizer bookkeeping (recommend(),
      // discounting, clamping) -- runs exactly as in plain CFR; only what is PLAYED
      // differs, and the average strategy is left untouched by the pre-play phase
      // altogether (see update_regret_and_policy). Overwriting here rather than
      // intercepting recommend() preserves every minimizer invariant bit-for-bit.
      //
      // PROVENANCE: this is the 'naive' fixed-opposition pre-play regime -- analyzed and
      // DISMISSED as a warm start in Brown & Sandholm, "Strategy-Based Warm Starting for
      // Regret Minimization in Games" (AAAI 2016), sec. 2, because it cannot substitute
      // for their regret-table substitution; it survives as an empirical early-descent
      // device (cf. the "warm-start form" of DeepStack, Moravčík et al., Science 2017,
      // and the warm-start references around DDCFR, Xu et al., ICLR 2024). No dedicated
      // convergence analysis exists for it; see rm::CFRConfig::warm_start_iterations.
      if(warm_start_active(_iteration())) {
         _force_warm_start_policy(*this_infostate, actions, this_node);
      }
   }
   double normalizing_factor = std::invoke([&] {
      if constexpr(not use_current_policy) {
         // we try to normalize only for the average policy, since iterations with the current
         // policy are for the express purpose of updating the average strategy. As such, we
         // should not intervene to change these values, as that may alter the values incorrectly
         // NOTE: activating here reproduces the former fetch_policy<average>
         // side effect of creating a uniform entry on first read.
         this_node.activate_average();
         m_avg_view_dirty = true;
         double normalization = std::ranges::fold_left(
            this_node.strategy_sum(), double(0.), std::plus{}
         );
         if(std::abs(normalization) < 1e-20) {
            throw std::invalid_argument(
               "Average policy likelihoods accumulate to 0. Such values cannot be normalized."
            );
         }
         return normalization;
      } else
         return 1.;
   });
   // deferred per-visit records of pruned edges traversed in THIS visit; their value
   // contributions are pushed once state_value below is fully aggregated (the buffer update
   // needs the final v(I), which includes all sibling actions)
   [[maybe_unused]] std::vector< std::pair< size_t, double > > pending_window_visits{};
   for(auto [action_idx, action] : std::views::enumerate(actions)) {
      if constexpr(config.pruning_mode == CFRPruningMode::regret_based or config.pruning_mode == CFRPruningMode::dynamic_thresholding) {
         if constexpr(use_current_policy) {
            if(_rbp_gate(
                  player_to_update,
                  active_player,
                  this_node.data(),
                  action_idx,
                  action,
                  infostates,
                  observation_buffer,
                  state,
                  depth
               )) {
               // the pruned action's current probability is exactly zero, so its contribution
               // to v(I) and its average-strategy increment would both vanish anyway; skipping
               // the recursion leaves state_value and action_value untouched for this action
               pending_window_visits.emplace_back(
                  action_idx, rm::cf_reach_probability(active_player, reach_probability.get())
               );
               continue;
            }
         }
      }
      double action_prob;
      if constexpr(use_current_policy) {
         // D1: the current policy comes straight from the node record's cached
         // recommendation (uniform before the first recommend) -- no policy-table hop
         action_prob = this_node.current_prob(action_idx);
      } else {
         action_prob = this_node.strategy_sum()[action_idx] / normalizing_factor;
      }

      // ---- save/restore bookkeeping for the recursion boundary --------------
      // reach probability: only the acting player's entry is scaled
      // NOTE: no reference is held across the recursion -- deeper frames may
      // insert new keys into the compacted table (vector storage shifts), so
      // the entry is re-resolved via operator[] for the restore.
      double& player_reach_entry = reach_probability.get()[active_player];
      const double saved_reach_prob = player_reach_entry;
      // scale in place instead of copying the whole reach map per edge
      player_reach_entry *= action_prob;

      // advance the arena slot of the next depth: copy-assign + transition
      // (no allocation after the first visit to this depth)
      world_state_type& next_wstate = _arena_state(depth + 1, state);
      _env().transition(next_wstate, action);

      // fold the transition's observations into the buffers / the advanced
      // infostate exactly like next_infostate_and_obs_buffers_inplace would --
      // but on the live seat-indexed containers, with restoration after the
      // recursion.
      // NOTE: a transition may leave the chance player in charge (multi-draw
      // deals); like the inplace helper we never buffer/flush FOR chance and
      // never touch the infostate table then.
      auto& obs_table = observation_buffer;
      const auto public_obs = _env().public_observation(state, action, next_wstate);
      const Player next_active_player = _env().active_player(next_wstate);
      // the infostate that advances across this edge belongs to the player who
      // is active in the CHILD state (exactly like the inplace helper): when
      // that player takes over, his buffered observations are flushed into a
      // clone of his infostate while every other player's sptr stays untouched.
      // A null pointer stands in for the former find()==end().
      const auto infostate_slot = next_active_player == Player::chance
                                     ? nullptr
                                     : infostates.find(next_active_player);
      const bool flushes = infostate_slot != nullptr;

      sptr< info_state_type > saved_infostate{};
      sptr< info_state_type > child_infostate{};
      // fixed-size STACK bookkeeping (bounded by max_player_seats): sizes of
      // exactly the buffers this edge appends to; restoring is an O(1)-each
      // resize afterwards. Replaces the former heap vector<pair<Player,size_t>>.
      std::array< size_t, max_player_seats > saved_buffer_sizes{};
      std::array< size_t, max_player_seats > saved_seats{};
      size_t saved_size_count = 0;
      if(flushes) {
         // remember the pre-edge infostate so it can be restored verbatim once
         // the recursion returns; the flush-target's buffer is SWAPPED into
         // this frame's reusable scratch slot instead of being copied
         saved_infostate = *infostate_slot;
         child_infostate = std::make_shared< info_state_type >(*saved_infostate);
         auto& obs_scratch = this->_obs_scratch_slot(depth);
         // drop dead residue a previous sibling edge left in the scratch slot
         // so that the flush target's live buffer is EMPTY during the recursion
         // below (the swap keeps capacities on both sides)
         obs_scratch.clear();
         obs_scratch.swap(obs_table[next_active_player]);
      }
      for(auto player : _env().players(next_wstate)) {
         if(player == Player::chance) {
            continue;
         }
         if(not (flushes and player == next_active_player)) {
            const auto seat_idx = static_cast< size_t >(player);
            saved_seats[saved_size_count] = seat_idx;
            saved_buffer_sizes[seat_idx] = obs_table[player].size();
            ++saved_size_count;
         }
      }
      for(auto player : _env().players(next_wstate)) {
         if(player == Player::chance) {
            continue;
         }
         if(flushes and player == next_active_player) {
            // after the swap the pre-edge history lives in the scratch slot:
            // drain it into the clone, clear the scratch for reuse; the live
            // buffer stays empty for the recursion below
            auto& scratch_history = this->_obs_scratch_slot(depth);
            // NOTE: the scratch slot now OWNS the pre-edge buffer contents
            // (acquired via the swap above); they are the restoration source
            // for sibling edges. Drain them by COPY -- moving out or clearing
            // here would destroy the saved state.
            for(auto& obs : scratch_history) {
               ::nor::detail::update_infostate(child_infostate, obs.first, obs.second);
            }
            ::nor::detail::update_infostate(
               child_infostate,
               public_obs,
               _env().private_observation(player, state, action, next_wstate)
            );
         } else {
            obs_table[player].emplace_back(
               public_obs, _env().private_observation(player, state, action, next_wstate)
            );
         }
      }
      if(flushes) {
         *infostate_slot = std::move(child_infostate);
      }

      StateValueMap child_rewards_map = _traverse< initialize_infonodes, use_current_policy >(
         player_to_update,
         next_wstate,
         depth + 1,
         reach_probability,
         observation_buffer,
         infostates,
         action_value_arena
      );

      // ---- restore everything the recursion mutated -------------------------
      if(flushes) {
         *infostate_slot = std::move(saved_infostate);
         // swap the pre-edge buffer back out of the scratch slot
         this->_obs_scratch_slot(depth).swap(obs_table[next_active_player]);
      }
      for(auto idx : std::views::iota(size_t{0}, saved_size_count)) {
         const auto seat_idx = saved_seats[idx];
         obs_table[static_cast< Player >(seat_idx)].resize(saved_buffer_sizes[seat_idx]);
      }
      reach_probability.get()[active_player] = saved_reach_prob;

      // add the child state's value to the respective player's value table, multiplied by the
      // policies likelihood of playing this action
      for(auto [player, child_value] : child_rewards_map.get()) {
         state_value.get()[player] += action_prob * child_value;
      }
      detail::emplace_action_value(action_value, action, std::move(child_rewards_map));
   }

   if constexpr(config.pruning_mode == CFRPruningMode::regret_based or config.pruning_mode == CFRPruningMode::dynamic_thresholding) {
      if(not pending_window_visits.empty()) {
         auto& node_data = this_node.data();
         const double v_I = state_value.get().at(active_player);
         for(auto [action_idx, cf_reach] : pending_window_visits) {
            _push_window_visit(
               node_data, active_player, *this_infostate, action_idx, cf_reach, v_I
            );
         }
      }
   }
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
template < bool initialize_infonodes, bool use_current_policy >
void VanillaCFR< config, Env, Policy, AveragePolicy >::_traverse_chance_actions(
   std::optional< Player > player_to_update,
   Player active_player,
   world_state_type& state,
   size_t depth,
   ReachProbabilityMap& reach_probability,
   ObservationbufferMap& observation_buffer,
   InfostateSptrMap& infostates,
   StateValueMap& state_value,
   ActionValueTable< action_variant_type >& action_value,
   ActionValueArena< action_variant_type >& action_value_arena
)
{
   for(auto&& outcome : _env().chance_actions(state)) {
      auto outcome_prob = _env().chance_probability(state, outcome);

      // NOTE: environments differ in whether 'Player::chance' is part of
      // env.players(root_state) (kuhn: yes, leduc: no). Default-initializing
      // the entry via operator[] here would multiply 0 by the outcome
      // probability and permanently zero out the counterfactual reach of every
      // descendant infostate, silently disabling all regret updates. Seed the
      // entry with the outcome probability when absent. Save/restore keeps the
      // caller's table intact across sibling edges without per-edge copies.
      auto& rp_table = reach_probability.get();
      const bool inserted = rp_table.try_emplace(active_player, outcome_prob).second;
      // NOTE: as above, no reference is held across the recursion (deeper
      // inserts may reallocate); every access re-resolves through operator[].
      double& chance_reach_entry = rp_table[active_player];
      const double saved_chance_reach = chance_reach_entry;
      if(not inserted) {
         chance_reach_entry *= outcome_prob;
      }

      world_state_type& next_wstate = _arena_state(depth + 1, state);
      _env().transition(next_wstate, outcome);

      // chance transitions fold into the buffers/infostates exactly like the
      // inplace helper: observations are buffered for every actual player and a
      // flush only happens once an ACTUAL player becomes active (multi-draw
      // deals chain several chance edges with buffering only).
      auto& obs_table = observation_buffer;
      const auto public_obs = _env().public_observation(state, outcome, next_wstate);
      const Player next_active_player = _env().active_player(next_wstate);
      // null pointer stands in for the former find()==end()
      const auto infostate_slot = next_active_player == Player::chance
                                     ? nullptr
                                     : infostates.find(next_active_player);
      const bool flushes = infostate_slot != nullptr;

      sptr< info_state_type > saved_infostate{};
      sptr< info_state_type > child_infostate{};
      // fixed-size STACK bookkeeping bounded by max_player_seats (see the
      // player-actions traversal for the full rationale)
      std::array< size_t, max_player_seats > saved_buffer_sizes{};
      std::array< size_t, max_player_seats > saved_seats{};
      size_t saved_size_count = 0;
      if(flushes) {
         saved_infostate = *infostate_slot;
         child_infostate = std::make_shared< info_state_type >(*saved_infostate);
         auto& obs_scratch = this->_obs_scratch_slot(depth);
         // drop dead residue a previous sibling edge left in the scratch slot
         // (see the player-actions traversal for the full rationale)
         obs_scratch.clear();
         obs_scratch.swap(obs_table[next_active_player]);
      }
      for(auto player : _env().players(next_wstate)) {
         if(player == Player::chance) {
            continue;
         }
         if(not (flushes and player == next_active_player)) {
            const auto seat_idx = static_cast< size_t >(player);
            saved_seats[saved_size_count] = seat_idx;
            saved_buffer_sizes[seat_idx] = obs_table[player].size();
            ++saved_size_count;
         }
      }
      for(auto player : _env().players(next_wstate)) {
         if(player == Player::chance) {
            continue;
         }
         if(flushes and player == next_active_player) {
            auto& scratch_history = this->_obs_scratch_slot(depth);
            // NOTE: the scratch slot now OWNS the pre-edge buffer contents
            // (acquired via the swap above); they are the restoration source
            // for sibling edges. Drain them by COPY -- moving out or clearing
            // here would destroy the saved state.
            for(auto& obs : scratch_history) {
               ::nor::detail::update_infostate(child_infostate, obs.first, obs.second);
            }
            ::nor::detail::update_infostate(
               child_infostate,
               public_obs,
               _env().private_observation(player, state, outcome, next_wstate)
            );
         } else {
            obs_table[player].emplace_back(
               public_obs, _env().private_observation(player, state, outcome, next_wstate)
            );
         }
      }
      if(flushes) {
         *infostate_slot = std::move(child_infostate);
      }

      StateValueMap child_rewards_map = _traverse< initialize_infonodes, use_current_policy >(
         player_to_update,
         next_wstate,
         depth + 1,
         reach_probability,
         observation_buffer,
         infostates,
         action_value_arena
      );

      if(flushes) {
         *infostate_slot = std::move(saved_infostate);
         this->_obs_scratch_slot(depth).swap(obs_table[next_active_player]);
      }
      for(auto idx : std::views::iota(size_t{0}, saved_size_count)) {
         const auto seat_idx = saved_seats[idx];
         obs_table[static_cast< Player >(seat_idx)].resize(saved_buffer_sizes[seat_idx]);
      }
      // replicate the former unordered_map insert/erase reset semantics:
      // a mid-traversal-inserted chance entry is removed again, an existing
      // entry is restored verbatim
      if(inserted) {
         rp_table.erase(active_player);
      } else {
         rp_table[active_player] = saved_chance_reach;
      }

      // add the child state's value to the respective player's value table, multiplied by the
      // policies likelihood of playing this action
      for(auto [player, child_value] : child_rewards_map.get()) {
         state_value.get()[player] += outcome_prob * child_value;
      }
      detail::emplace_action_value(action_value, std::move(outcome), std::move(child_rewards_map));
   }
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
void VanillaCFR< config, Env, Policy, AveragePolicy >::update_regret_and_policy(
   const info_state_type& infostate,
   const ReachProbabilityMap& reach_probability,
   const StateValueMap& state_value,
   const ActionValueTable< action_variant_type >& action_value_map
)
{
   auto& istate_data = _infonode(infostate);
   auto& node_data = istate_data.data();
   const auto& actions = istate_data.actions();
   auto player = infostate.player();
   double cf_reach_prob = rm::cf_reach_probability(player, reach_probability.get());
   double player_reach_prob = reach_probability.get().at(player);
   double player_state_value = state_value.get().at(player);

   // D3: remember this node for the end-of-iteration sweep (once per iteration)
   _touch(infostate, istate_data);

   if constexpr(config.lazy_update_mode != CFRLazyUpdateMode::off) {
      // ------------------- Lazy-CFR (Zhou et al., ICLR 2020) buffering path ------------------
      // The infostate's strategy is FROZEN inside an open segment: instead of the eager
      // observe()/average-policy increments below, this visit's contributions are buffered
      // and attributed when the segment closes. A close is triggered BEFORE attributing the
      // current visit once the accumulated opponent reach has exhausted the budget B; the
      // closing fold applies both buffers at once and flags the end-of-iteration sweep to
      // recompute the recommendation. Deferral of the average-strategy mass is exact because
      // sum_t pi_i^t * sigma collapses to (sum_t pi_i^t) * sigma for the frozen sigma.
      // NOTE (documented deviation from the paper's immediate refresh): the recommendation
      // is recomputed at the END of the closing iteration rather than inside the traversal,
      // preserving this codebase's one-sweep-per-iteration recommendation discipline. Visits
      // to this infostate LATER within the closing iteration therefore play the old strategy
      // but are attributed to the new segment -- an O(single-traversal) own-reach mass
      // ascribed one segment late, vanishing in the T -> infinity limit.
      auto& seg = _lazy_segment(infostate, actions);
      if(seg.pending_cf_reach >= config.lazy_update_threshold_b) {
         _lazy_fold_segment(infostate, seg);
      }
      for(const auto& [action_variant, action_value] : action_value_map) {
         // we only call this function with action values from a non-chance player, so we can
         // safely assume that the action is of action_type
         const auto& action = std::get< 0 >(action_variant);
         seg.regret_buffer[istate_data.index_of(action)] +=
            cf_reach_prob * (action_value.get().at(player) - player_state_value);
      }
      seg.pending_cf_reach += cf_reach_prob;
      seg.pending_player_reach += player_reach_prob;
      return;
   }

   for(const auto& [action_variant, action_value] : action_value_map) {
      // we only call this function with action values from a non-chance player, so we can safely
      // assume that the action is of action_type
      const auto& action = std::get< 0 >(action_variant);
      // update the cumulative regret according to the formula:
      // let I be the infostate, p be the player, r the cumulative regret
      //    r = \sum_a counterfactual_reach_prob_{p}(I) * (value_{p}(I-->a) - value_{p}(I))
      if constexpr(
         config.weighting_mode != CFRWeightingMode::exponential
         and config.weighting_mode != CFRWeightingMode::greedy
      ) {
         if(cf_reach_prob > 0) {
            // this if statement effectively introduces partial pruning. But this is such a
            // slight modification (and gain, if any) that it is to be included in all variants
            // of CFR
            //
            // all other cfr variants currently implemented need the average regret update at
            // history update time
            m_regret_minimizer.observe(
               node_data,
               action,
               cf_reach_prob * (action_value.get().at(player) - player_state_value)
            );
         }
      } else {
         // for the exponential cfr method we need to remember these regret increments of
         // iteration t, until the end of iteration t, and apply them once we have computed the
         // L1 weights (i.e. at infostate update time, not history update time). After iteration
         // t ends we have to delete them again, so that this is only a memory of the current
         // iteration! Each history h that passed through infostate I will increment here the
         // instantaneous regret values r(h,a), in order to accumulate r(I, a) = sum_h r(h, a)
         //
         // greedy weights defers for the same structural reason (arXiv:2204.04826): its
         // iteration weight is only known once the whole traversal's regrets are aggregated
         m_regret_minimizer.observe(
            node_data, action, cf_reach_prob * (action_value.get().at(player) - player_state_value)
         );
      }
      if constexpr(
         config.weighting_mode != CFRWeightingMode::exponential
         and config.weighting_mode != CFRWeightingMode::greedy
      ) {
         // WARM START: pre-play iterations are 'BEFORE play' -- they exist purely to seed
         // the regret tables with best-response information about the fixed profile, so
         // they contribute NOTHING to the average strategy (the degenerate weighting
         // schedule of the weighted-averaging family, cf. DCFR's downweighting of early
         // rounds). Without this exclusion the phase's uniform rounds would pollute the
         // average and negate the seeding benefit.
         //
         // GREEDY WEIGHTS (arXiv:2204.04826): deferred exactly like the exponential family --
         // the greedy iteration weight is only known once the whole traversal's regrets are
         // aggregated, so the weighted average-policy increment is applied once per iteration
         // by the end-of-iteration sweep ('_finalize_greedy_iteration').
         if(not warm_start_active(_iteration())) {
            // D1: accumulate the average strategy INTO the node record according
            // to the formula:
            // let 'I' be the infostate, 'p' the player, 'a' the chosen action and
            // 'sigma^t' the current policy:
            // -->  avg_sigma^{t+1}(a) += reach_prob_{p}(I) * sigma^t(I, a)
            // activating on first touch reproduces the former fetch_policy<average>
            // behavior of starting every entry from its uniform baseline
            const size_t action_idx = istate_data.index_of(action);
            istate_data.activate_average();
            auto& sums = istate_data.strategy_sum();
            sums[action_idx] += player_reach_prob * istate_data.current_prob(action_idx);
         }
         // For exponential CFR we update the average policy after the tree traversal
      }
   }
   if constexpr(config.weighting_mode == CFRWeightingMode::greedy) {
      // greedy weights accumulates the average-policy mass pi^t(h) of every visited history h
      // of this infostate; the weighted increment is applied once per iteration by the
      // end-of-iteration sweep ('_finalize_greedy_iteration'), which also resets the snapshot.
      // ACCUMULATION (not overwrite) is required because an infostate may be reached through
      // several histories within one traversal.
      istate_data.data().reach_prob_snapshot += player_reach_prob;
   } else if constexpr(config.weighting_mode == CFRWeightingMode::exponential) {
      // For exponential CFR we need to store the reach probability of the active player until
      // the end of the iteration
      node_data.reach_prob_snapshot = player_reach_prob;
   }
   m_avg_view_dirty = true;
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
lazy::SegmentState& VanillaCFR< config, Env, Policy, AveragePolicy >::_lazy_segment(
   const info_state_type& infostate,
   const std::vector< action_type >& actions
)
{
   auto& table = m_lazy_segments[infostate.player()];
   auto& seg = table.try_emplace(infostate).first->second;
   if(seg.regret_buffer.size() != actions.size()) {
      // first touch (or defensive resync): keep the buffer index-aligned with the registry
      seg.regret_buffer.assign(actions.size(), 0.);
   }
   return seg;
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
void VanillaCFR< config, Env, Policy, AveragePolicy >::_lazy_fold_segment(
   const info_state_type& infostate,
   lazy::SegmentState& seg
)
{
   static_assert(config.lazy_update_mode != CFRLazyUpdateMode::off);
   auto& istate_data = _infonode(infostate);
   const auto& actions = istate_data.actions();
   for(auto idx : std::views::iota(size_t{0}, seg.regret_buffer.size())) {
      if(seg.regret_buffer[idx] != 0.) {
         // fold the segment's buffered counterfactual regret increments into the minimizer's
         // own cumulative update rule ("R <- R + sum r~"; CFR+ clamps at its recommend step)
         m_regret_minimizer.observe(istate_data.data(), actions[idx], seg.regret_buffer[idx]);
      }
   }
   // one deferred average-strategy increment for the WHOLE closed segment: exact because the
   // frozen strategy makes sum_t pi_i^t * sigma equal (sum_t pi_i^t) * sigma. Must run BEFORE
   // the sweep recomputes the recommendation from the freshly folded regrets.
   // D1: written straight into the node record's strategy sums (the former
   // fetch_policy<current/average> tables ARE these caches under the record representation);
   // current_prob() reads the frozen recommendation exactly like the former table did.
   if(seg.pending_player_reach != 0.) {
      istate_data.activate_average();
      auto& sums = istate_data.strategy_sum();
      for(auto idx : std::views::iota(size_t{0}, seg.regret_buffer.size())) {
         sums[idx] += seg.pending_player_reach * istate_data.current_prob(idx);
      }
   }
   std::ranges::fill(seg.regret_buffer, 0.);
   seg.pending_cf_reach = 0.;
   seg.pending_player_reach = 0.;
   seg.refresh_pending = true;
   ++m_lazy_stats.segment_refreshes;
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
bool VanillaCFR< config, Env, Policy, AveragePolicy >::_lazy_consume_refresh(
   const info_state_type& infostate
)
{
   auto& table = m_lazy_segments[infostate.player()];
   const auto seg_it = table.find(infostate);
   if(seg_it == table.end()) {
      // never visited (or never buffered): nothing can have closed -> strategy stays frozen
      return false;
   }
   const bool pending = seg_it->second.refresh_pending;
   seg_it->second.refresh_pending = false;
   return pending;
}
namespace detail {

/// @brief a verification of the correctness of the chosen configuration
template < CFRConfig config >
consteval bool sanity_check_cfr_config()
{
   if constexpr(config.regret_minimizing_mode == RegretMinimizingMode::predictive_regret_matching_plus or config.regret_minimizing_mode == RegretMinimizingMode::sap_predictive_regret_matching_plus or config.regret_minimizing_mode == RegretMinimizingMode::ap_predictive_regret_matching_plus or config.regret_minimizing_mode == RegretMinimizingMode::p2p_predictive_regret_matching_plus or config.regret_minimizing_mode == RegretMinimizingMode::smooth_predictive_regret_matching_plus or config.regret_minimizing_mode == RegretMinimizingMode::stable_predictive_regret_matching_plus or config.regret_minimizing_mode == RegretMinimizingMode::discounted_regret_matching_plus or config.regret_minimizing_mode == RegretMinimizingMode::discounted_predictive_regret_matching_plus) {
      // the predictive regret minimizers (PCFR+/SAPCFR+/APCFR+/P2PCFR+ and the
      // Smooth/Stable-PRM+ robustifications) pair the strategy
      // snapshot of the previous recommendation with the instantaneous regrets
      // of exactly one full iteration. This pairing is only well-defined for
      // alternating updates over unpruned traversals (in simultaneous updates
      // and under pruning the rho/sigma_snap correspondence breaks). The DCFR+/PDCFR+
      // kernels (arXiv:2404.13891) share the same one-phase-per-recommend contract
      // (their deferred fold consumes exactly one fully observed instantaneous
      // regret vector per recommend) and the paper prescribes alternating updates.
      // The discounted weighting mode is REQUIRED as carrier of the gamma-side
      // average-policy accumulation; its own alpha/beta regret discounts are
      // compiled out by the minimizer selection for these modes
      if constexpr(config.update_mode != UpdateMode::alternating or config.pruning_mode != CFRPruningMode::none or config.weighting_mode != CFRWeightingMode::discounted) {
          return false;
      }
   }
   if constexpr(config.lazy_update_mode != CFRLazyUpdateMode::off) {
      // Lazy-CFR (Zhou et al., ICLR 2020) freezes recommendations across iterations, so any
      // kernel whose update rule pairs ONE recommendation with EXACTLY ONE fully observed
      // iteration's regret vector breaks: the predictive family's rho/sigma_snap
      // correspondence and the DCFR+/PDCFR+ deferred folds (arXiv:2404.13891, sec. 4)
      // both assume a refresh every iteration.
      if constexpr(
         config.regret_minimizing_mode == RegretMinimizingMode::predictive_regret_matching_plus
         or config.regret_minimizing_mode
            == RegretMinimizingMode::sap_predictive_regret_matching_plus
         or config.regret_minimizing_mode
            == RegretMinimizingMode::ap_predictive_regret_matching_plus
         or config.regret_minimizing_mode
            == RegretMinimizingMode::p2p_predictive_regret_matching_plus
         or config.regret_minimizing_mode
            == RegretMinimizingMode::smooth_predictive_regret_matching_plus
         or config.regret_minimizing_mode
            == RegretMinimizingMode::stable_predictive_regret_matching_plus
         or config.regret_minimizing_mode == RegretMinimizingMode::discounted_regret_matching_plus
         or config.regret_minimizing_mode
            == RegretMinimizingMode::discounted_predictive_regret_matching_plus
      ) {
         return false;
      }
      // the deferred average-strategy accumulation assumes UNIFORM weighting: the
      // discounted/linear gamma-side rescaling multiplies the accumulated policy EVERY
      // iteration, which would also hit mass still sitting in the lazy buffers; the
      // exponential family defers its own updates to finalize_iteration and was not analyzed
      if constexpr(config.weighting_mode != CFRWeightingMode::uniform) {
         return false;
      }
      // pruning windows are armed/consumed on a per-iteration cadence tied to the recommend
      // step; freezing recommendations across iterations breaks that contract. Kept out of
      // scope initially.
      if constexpr(config.pruning_mode != CFRPruningMode::none) {
         return false;
      }
      // a non-positive opponent-reach budget would close segments before any visit's
      // contributions are buffered
      if constexpr(not(config.lazy_update_threshold_b > 0.)) {
         return false;
      }
   }
   if constexpr(config.regret_minimizing_mode == RegretMinimizingMode::internal_regret_matching) {
      // the swap-basis phi-regret kernel buffers instantaneous regrets during
      // the traversal and folds them against its own recommendation snapshot
      // inside 'recommend'. The discounted weighting decorator would re-scale
      // the already-folded transformed-regret table and the exponential kernel
      // routes updates through a deferred L1 machinery this minimizer does not
      // implement, so both are statically rejected (the selection pins the mode
      // to InternalRegretMatching regardless of the weighting axis).
      // Regret-based pruning prescribes the RM+-style replace-if-positive fold
      // contract on a 'cumulative_instant_regret' layout this kernel does not
      // carry; dynamic thresholding stays composable since it only reshapes the
      // derived recommendation.
      if constexpr(
         config.weighting_mode != CFRWeightingMode::uniform
         or config.pruning_mode == CFRPruningMode::regret_based
      ) {
         return false;
      }
   }
   if constexpr(config.weighting_mode == CFRWeightingMode::greedy) {
      // Greedy weights (Zhang, Lerer & Brown, AAAI 2022) re-weighs WHOLE iterations
      // dynamically from their observed counterfactual regrets. Combinations that were
      // never analyzed or are known to break are statically rejected:
      //  * ALTERNATING UPDATES: the published formulation (Algorithm 1 and the CFR
      //    extension of their Appendix F) is a SIMULTANEOUS scheme -- one joint weight
      //    drawn from the potential summed over all players. Running one independent
      //    greedy instance per player under alternating updates demonstrably destroys
      //    convergence (the per-player objective prefers near-total replacement of the
      //    instance's history, turning the average strategy into a short-window moving
      //    average that oscillates instead of converging -- observed on 2p/3p kuhn).
      //  * dynamic thresholding reshapes recommendations mid-flight while the greedy fold
      //    assumes recommendations derived straight from the freshly folded cumulative
      //    regret table;
      //  * regret-based pruning is rejected below regardless (its windows buffer UNweighted
      //    increments and require the uniform mode);
      //  * the predictive / DCFR+-style kernels pair their strategy snapshots with exactly
      //    one fully observed instantaneous regret vector per recommend and REQUIRE the
      //    discounted carrier weighting (first block below), which already rules out
      //    combining them with greedy weights.
      // NOTE also that the paper's CFR extension explicitly assumes full-expansion
      // traversals: chance-sampled settings risk upweighting 'lucky' sampled outcomes
      // (their Appendix F), hence greedy weights is vanilla-engine only.
      if constexpr(
         config.update_mode != UpdateMode::simultaneous
         or config.pruning_mode == CFRPruningMode::dynamic_thresholding
      ) {
         return false;
      }
   }
   if constexpr(
      (config.weighting_mode == CFRWeightingMode::exponential)
      and (config.pruning_mode == CFRPruningMode::regret_based)
      and (config.regret_minimizing_mode == RegretMinimizingMode::regret_matching_plus)
   ) {
      // there is currently no theoretic work on combining these methods and the update rule
      // for the cumulative regret in both clash with different approaches (exp weighting
      // wants e^L1 weighted updates) while regret-based-pruning with CFR+ wants to replace
      // the cumulative regret with r(I,a) only if r(I,a) > 0 and R^T(I,a) < 0, otherwise do a
      // normal cumulative regret update (i.e. R^t+1(I,a) = R^t(I,a) + r(I,a))
      return false;
   }
   if constexpr(config.pruning_mode == CFRPruningMode::regret_based) {
      // Regret-based pruning is proven for alternating updates whose local recommendations are
      // regret-matching based with UNIFORM average-strategy accumulation (Brown & Sandholm,
      // NIPS 2015, secs. 3-4.2; their sec. 4.2 even flags linear averaging as noticeably noisier
      // under RBP). The discounted/linear/exponential families re-weight exactly the regret
      // increments a pruned window buffers and were not analyzed; simultaneous updates traverse
      // every player in one pass so skipped windows would have to be folded per-player.
      // CHOICE: statically reject instead of implementing an unanalyzed safe variant.
      if constexpr(config.regret_minimizing_mode != RegretMinimizingMode::regret_matching_plus or config.update_mode != UpdateMode::alternating or config.weighting_mode != CFRWeightingMode::uniform) {
         return false;
      }
   }
   // dynamic_thresholding composes with any base minimizer recommending via regret matching on
   // its cumulative table (plain RM, RM+ via DiscountedCFR carrier, ExponentialCFR) -- which is
   // what the selection in minimizers.hpp produces; it also keeps working under simultaneous
   // updates since thresholding only reshapes recommendations. The predictive/discounted-kernel
   // modes are already rejected above because they require pruning_mode == none altogether.
   if constexpr(config.warm_start_iterations > 0) {
      // the warm-start pre-play phase forces the PLAYED strategy during early traversals at
      // the policy-fetch point while regret/average updates run unmodified (see
      // _traverse_player_actions). Exponential CFR defers both its cumulative updates AND its
      // policy refresh into finalize_iteration with a separate L1-weighted
      // numerator/denominator machinery; forcing played policies mid-phase is unanalyzed
      // there, so the combination is statically rejected. All regret-matching-based modes
      // (RM / RM+, incl. their discounted/linear carriers) are supported.
      if constexpr(config.weighting_mode == CFRWeightingMode::exponential) {
         return false;
      }
   }
   return true;
}

}  // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////
////////////////////// regret-based pruning / dynamic thresholding engine /////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

/// resolves the registered action at 'action_idx' (gate/fold helpers speak in table indices)
template < typename NodeData >
decltype(auto) action_of_index(NodeData& node_data, size_t action_idx)
{
   return node_data.registry.actions[action_idx];
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
pruning::PayoffBound VanillaCFR< config, Env, Policy, AveragePolicy >::_edge_bound(
   const info_state_type& infostate,
   const action_type& action
)
{
   if constexpr(config.pruning_mode == CFRPruningMode::regret_based or config.pruning_mode == CFRPruningMode::dynamic_thresholding) {
      if(m_payoff_bounds.empty()) {
         auto root_players = _env().players(root_state());
         for(auto p : root_players) {
            if(p != Player::chance) {
               m_root_player_order.push_back(p);
            }
         }
         if constexpr(concepts::has::supports_payoff_bounds< env_type >) {
            // B4 trait contract: the environment reports per-player global bounds; honor them
            for(auto p : m_root_player_order) {
               auto [lo, hi] = _env().payoff_bounds(p);
               m_payoff_bounds.emplace(p, pruning::PayoffBound{.lower = lo, .upper = hi});
            }
         } else {
            // fallback: probe PER-(infostate,action) terminal-reward ranges once -- the tight,
            // faithful reading of the paper's U(I,a)/L(I). The full-tree walk costs one
            // traversal and runs outside any active recursion (arena is idle there).
            _probe_edge_bounds();
         }
      }
      if constexpr(not concepts::has::supports_payoff_bounds< env_type >) {
         // D2: resolve from the owning node record's registry-aligned bounds table
         const auto node_it = m_infonode.find(infostate);
         if(node_it != m_infonode.end()) {
            const auto& node = node_it->second;
            const auto& actions = node.actions();
            const auto found = std::ranges::find(actions, action);
            if(found != actions.end()) {
               const auto& bound = node.edge_bounds(
               )[static_cast< size_t >(found - actions.begin())];
               // {+inf,-inf} marks an action whose subtree was never probed;
               // treat it (and any infostate missing a node record) as a miss:
               // fall back to the global per-player interval -- strictly WIDER,
               // hence still sound.
               if(bound.upper != -std::numeric_limits< double >::infinity()) {
                  return bound;
               }
            }
         }
      }
      const auto& global = m_payoff_bounds.at(infostate.player());
      return global;
   } else {
      return pruning::PayoffBound{};
   }
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
void VanillaCFR< config, Env, Policy, AveragePolicy >::_probe_edge_bounds()
{
   player_hashmap< sptr< info_state_type > > infostates{};
   player_hashmap< std::vector< std::pair< observation_type, observation_type > > > buffers{};
   for(auto p : m_root_player_order) {
      infostates.emplace(p, std::make_shared< info_state_type >(p));
      buffers.emplace(p, std::vector< std::pair< observation_type, observation_type > >{});
   }
   world_state_type& root_ref = *_root_state_uptr();
   world_state_type& arena_root = _arena_state(0, root_ref);
   // the ROOT hull doubles as the conservative global per-player interval used whenever an
   // edge lookup misses
   const auto global_hull = _probe_dfs(arena_root, 0, infostates, buffers);
   for(auto idx : std::views::iota(size_t{0}, m_root_player_order.size())) {
      m_payoff_bounds.emplace(m_root_player_order[idx], global_hull[idx]);
   }

   // convert raw per-edge {min,max} of u_owner below h*a into the paper's pair
   // { lower = L(I), upper = U(I,a) } with L(I) = min over the node's probed edges.
   // D2: the ranges live in the node records now; min/max folds are exact
   // operations, so the fold order does not influence the resulting bits.
   for(auto& [infostate_ptr, node] : m_infonode) {
      auto& bounds = node.edge_bounds();
      double l_of_I = std::numeric_limits< double >::infinity();
      for(const auto& bound : bounds) {
         l_of_I = std::min(l_of_I, bound.lower);  // still raw min here
      }
      for(auto& bound : bounds) {
         // leave never-probed slots untouched: they must keep falling back to
         // the (wider) global per-player interval exactly like a former
         // edge-table miss did
         if(bound.upper == -std::numeric_limits< double >::infinity()) {
            continue;
         }
         bound.lower = l_of_I;
      }
   }
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
std::vector< pruning::PayoffBound > VanillaCFR< config, Env, Policy, AveragePolicy >::_probe_dfs(
   world_state_type& next_wstate,
   size_t depth,
   player_hashmap< sptr< info_state_type > >& infostates,
   player_hashmap< std::vector< std::pair< observation_type, observation_type > > >&
      observation_buffers
)
{
   // NOTE: this worker assumes the EDGE ALREADY APPLIED by the caller (next_wstate
   // transitioned and its observations flushed/buffered); it restores nothing itself -- the
   // caller snapshots/restores around each expansion. Returns the {lo,hi} interval of every
   // ROOT player's terminal reward below next_wstate (order = m_root_player_order).
   const size_t n = m_root_player_order.size();
   if(_env().is_terminal(next_wstate)) {
      auto rewards = collect_rewards(_env(), next_wstate, _env().players(root_state()));
      std::vector< pruning::PayoffBound > out(n);
      for(auto idx : std::views::iota(size_t{0}, n)) {
         const double r = rewards.at(m_root_player_order[idx]);
         out[idx] = pruning::PayoffBound{.lower = r, .upper = r};
      }
      return out;
   }

   auto hull = [&](
                  std::vector< pruning::PayoffBound >& acc,
                  const std::vector< pruning::PayoffBound >& sub
               ) {
      for(auto idx : std::views::iota(size_t{0}, n)) {
         acc[idx].lower = std::min(acc[idx].lower, sub[idx].lower);
         acc[idx].upper = std::max(acc[idx].upper, sub[idx].upper);
      }
   };

   std::vector< pruning::PayoffBound > ranges(
      n,
      pruning::PayoffBound{
         .lower = std::numeric_limits< double >::infinity(),
         .upper = -std::numeric_limits< double >::infinity()}
   );

   const Player active = _env().active_player(next_wstate);

   auto expand = [&](const auto& action_or_outcome) -> std::vector< pruning::PayoffBound > {
      world_state_type& child = _arena_state(depth + 1, next_wstate);
      _env().transition(child, action_or_outcome);
      const auto pub_obs = _env().public_observation(next_wstate, action_or_outcome, child);
      const Player na = _env().active_player(child);
      const bool flushes = na != Player::chance and infostates.contains(na);
      sptr< info_state_type > saved_entry{};
      std::vector< std::pair< observation_type, observation_type > > saved_flush_buffer;
      std::vector< std::pair< Player, size_t > > saved_sizes;
      if(flushes) {
         saved_entry = infostates.at(na);
         saved_flush_buffer = observation_buffers.at(na);
      }
      for(const auto& [player, buffer] : observation_buffers) {
         saved_sizes.emplace_back(player, buffer.size());
      }
      if(flushes) {
         auto child_ist = std::make_shared< info_state_type >(*saved_entry);
         auto& history = observation_buffers[na];
         for(auto& obs : history) {
            ::nor::detail::update_infostate(child_ist, std::move(obs.first), std::move(obs.second));
         }
         history.clear();
         ::nor::detail::update_infostate(
            child_ist,
            pub_obs,
            _env().private_observation(na, next_wstate, action_or_outcome, child)
         );
         infostates.at(na) = std::move(child_ist);
      }
      for(auto player : _env().players(child)) {
         if(player == Player::chance or player == na) {
            continue;
         }
         observation_buffers[player].emplace_back(
            pub_obs, _env().private_observation(player, next_wstate, action_or_outcome, child)
         );
      }
      auto sub = _probe_dfs(child, depth + 1, infostates, observation_buffers);
      if(flushes) {
         infostates.at(na) = std::move(saved_entry);
         observation_buffers.at(na) = std::move(saved_flush_buffer);
      }
      for(const auto& [player, size] : saved_sizes) {
         observation_buffers.at(player).resize(size);
      }
      return sub;
   };

   if constexpr(concepts::stochastic_env< env_type >) {
      if(active == Player::chance) {
         for(auto&& outcome : _env().chance_actions(next_wstate)) {
            hull(ranges, expand(outcome));
         }
         return ranges;
      }
   }

   for(const auto& action : _env().actions(active, next_wstate)) {
      // record BEFORE expanding: the edge's key is this node's CURRENT infostate and the
      // subtree range is returned by the expansion itself
      auto owner_idx = static_cast< size_t >(
         std::ranges::find(m_root_player_order, active) - m_root_player_order.begin()
      );
      auto sub = expand(action);
      // D2: record the probed range in the owning infoset's node record
      // (index-aligned with its action registry). A node-record or registry
      // miss cannot occur once the tree has been traversed once (the probe
      // only ever runs after a full registration pass); if it ever did, the
      // bound is simply absent and queries fall back to the wider global hull.
      const auto node_it = m_infonode.find(*infostates.at(active));
      if(node_it != m_infonode.end()) {
         auto& node = node_it->second;
         const auto& registered = node.actions();
         const auto found = std::ranges::find(registered, action);
         if(found != registered.end()) {
            auto& bound = node.edge_bounds()[static_cast< size_t >(found - registered.begin())];
            bound.lower = std::min(bound.lower, sub[owner_idx].lower);
            bound.upper = std::max(bound.upper, sub[owner_idx].upper);
         }
      }
      hull(ranges, sub);
   }
   return ranges;
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
template < typename NodeData >
void VanillaCFR< config, Env, Policy, AveragePolicy >::_arm_pruning_windows(
   const info_state_type& infostate,
   NodeData& node_data
)
{
   auto& rbp = node_data.rbp;
   const size_t t0 = _iteration() + 1;
   for(auto idx : std::views::iota(size_t{0}, node_data.regret.size())) {
      if(rbp.pruned_until[idx] != 0) {
         continue;
      }
      const auto& action = node_data.registry.actions[idx];
      // Theorem-1 window denominator: U(I,a) - L(I), resolved from the per-edge probed ranges
      // (or the env's global B4 bounds when provided)
      const double bound_range = _edge_bound(infostate, action).range();
      // conservative per-iteration regret growth bound r^t(I,a) <= increment_bound; exponential
      // weighting inflates by e^range because its L1 factors rescale increments (pruning.hpp)
      const double inc_bound = pruning::window_increment_bound(config.weighting_mode, bound_range);
      const double R = node_data.regret[idx];
      // Appendix-B minimum-skip filter (NIPS'15): only open a window when even the WORST-CASE
      // window length floor(|R| / inc-bound) clears the configured minimum
      if(R >= -config.rbp_min_skip_iterations * inc_bound) {
         continue;
      }
      const size_t m = pruning::theorem1_window(R, inc_bound);
      rbp.pruned_until[idx] = t0 + m;
      rbp.last_pruned_iteration[idx] = t0;
      rbp.pessimistic_regret[idx] = R;
      rbp.br_regret_buffer[idx] = 0.;
      rbp.cached_br_value[idx] = 0.;
      rbp.visits_since_refresh[idx] = 0;
      ++m_pruning_stats.windows_armed;
   }
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
template < typename NodeData >
bool VanillaCFR< config, Env, Policy, AveragePolicy >::_rbp_gate(
   std::optional< Player > player_to_update,
   Player active_player,
   NodeData& node_data,
   size_t action_idx,
   const action_type& /*action*/,
   InfostateSptrMap& infostates,
   ObservationbufferMap& observation_buffer,
   const world_state_type& state,
   size_t depth
)
{
   auto& rbp = node_data.rbp;
   if(rbp.pruned_until[action_idx] == 0) {
      return false;
   }
   if(not player_to_update.has_value() or active_player != player_to_update.value()) {
      // windows only gate the OWNING player's own update traversals: opponents passing through
      // (I,a) still need the subtree for their average-strategy updates (their counterfactual
      // regret updates below (I,a) are zero-reach anyway)
      return false;
   }
   const size_t current_iteration = _iteration();
   if(current_iteration < rbp.pruned_until[action_idx]) {
      auto& since_refresh = rbp.visits_since_refresh[action_idx];
      if(since_refresh % config.rbp_br_refresh_period == 0) {
         // periodic best-response traversal of D(h,a) against the opponents' AVERAGE strategies.
         // The BR walk still operates on classic player hashmaps (cold path), so the live
         // seat-indexed containers are materialized through named lvalue views; the walk
         // restores every mutation per edge, making the copies observationally equivalent
         auto raw_istates = raw_infostate_view(infostates);
         auto raw_buffers = raw_observation_buffer_view(observation_buffer);
         rbp.cached_br_value[action_idx] = _br_expectimax_from_edge(
            active_player,
            action_of_index(node_data, action_idx),
            state,
            depth,
            raw_istates,
            raw_buffers
         );
         ++m_pruning_stats.br_refreshes;
      }
      ++since_refresh;
      ++m_pruning_stats.skipped_edge_visits;
      return true;
   }
   // deadline expired: fold the buffered best-response regret and resume normal traversal
   _rbp_fold(node_data, action_idx, action_of_index(node_data, action_idx));
   return false;
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
template < typename NodeData >
void VanillaCFR< config, Env, Policy, AveragePolicy >::_push_window_visit(
   NodeData& node_data,
   Player active_player,
   const info_state_type& infostate,
   size_t action_idx,
   double cf_reach_prob,
   double state_value_for_player
)
{
   auto& rbp = node_data.rbp;
   // eq-(9) pessimistic tracker: assume the pruned action delivered its maximal payoff U(I,a);
   // this upper-bounds the true regret evolution because v(I->a) <= U(I,a) pointwise
   const double u_upper = _edge_bound(infostate, node_data.registry.actions[action_idx]).upper;
   rbp.pessimistic_regret[action_idx] += cf_reach_prob * (u_upper - state_value_for_player);
   // best-response buffer: the missing true increments pi_{-i}(v(I->a) - v(I)) are replaced by
   // pi_{-i}(v_BR - v(I)); folded into the regret table at unfold time
   rbp.br_regret_buffer[action_idx] += cf_reach_prob
                                       * (rbp.cached_br_value[action_idx] - state_value_for_player);
   if(pruning::pessimistic_unfold_required(rbp.pessimistic_regret[action_idx])) {
      _rbp_fold(node_data, action_idx, node_data.registry.actions[action_idx]);
   }
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
template < typename NodeData >
void VanillaCFR< config, Env, Policy, AveragePolicy >::_rbp_fold(
   NodeData& node_data,
   size_t action_idx,
   const action_type& action
)
{
   auto& rbp = node_data.rbp;
   if(rbp.br_regret_buffer[action_idx] != 0.) {
      // "update the regrets to match this": announce the window's best response as having been
      // played on every skipped iteration; observe routes into the minimizer's own update rule
      // (RM+RBP's replace-if-positive fold happens at its next recommend)
      m_regret_minimizer.observe(node_data, action, rbp.br_regret_buffer[action_idx]);
   }
   rbp.pruned_until[action_idx] = 0;
   rbp.br_regret_buffer[action_idx] = 0.;
   rbp.pessimistic_regret[action_idx] = 0.;
   rbp.cached_br_value[action_idx] = 0.;
   rbp.visits_since_refresh[action_idx] = 0;
   ++m_pruning_stats.window_folds;
}
template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
std::vector< double > VanillaCFR< config, Env, Policy, AveragePolicy >::_normalized_average_policy(
   const info_state_type& infostate,
   const std::vector< action_type >& actions
)
{
   std::vector< double > probs;
   probs.reserve(actions.size());
   double sum = 0.;
   // NOTE: 'actions' here comes from the environment and need not share the
   // infostate registry's registration order -- resolve and accumulate in the
   // environment's order to preserve the former floating-point summation order.
   if constexpr(config.weighting_mode == CFRWeightingMode::exponential) {
      // exponential weighting keeps an unnormalized numerator plus a separate denominator
      // table; normalize against the denominators exactly like average_policy() does.
      // Infostates never visited by the exponential update path have no node data yet --
      // fall back to uniform for them.
      const auto node_it = m_infonode.find(infostate);
      if(node_it == m_infonode.end()) {
         std::vector< double > uniform(actions.size(), 1. / static_cast< double >(actions.size()));
         return uniform;
      }
      auto& node = node_it->second;
      // reproduces the former fetch_policy<average> side effect of creating
      // the uniform-baseline entry on first read
      node.activate_average();
      const auto& node_data = node.data();
      const auto& sums = node.strategy_sum();
      const auto& denominators = node_data.avg_policy_denominator;
      for(const auto& action : actions) {
         // resolve each action's table slot explicitly
         const size_t idx = node.index_of(action);
         const double denominator = std::max(denominators[idx], 1e-20);
         const double p = sums[idx] / denominator;
         probs.push_back(p);
         sum += p;
      }
   } else {
      const auto node_it = m_infonode.find(infostate);
      if(node_it != m_infonode.end()) {
         auto& node = node_it->second;
         node.activate_average();
         const auto& sums = node.strategy_sum();
         for(const auto& action : actions) {
            sum += sums[node.index_of(action)];
         }
         for(const auto& action : actions) {
            probs.push_back(sums[node.index_of(action)]);
         }
      } else {
         // no node record yet: behave like the formerly fetch-created fresh
         // (uniform) table entry, including its additive summation order
         const double uniform_prob = 1. / static_cast< double >(actions.size());
         for([[maybe_unused]] const auto& action : actions) {
            sum += uniform_prob;
         }
         probs.assign(actions.size(), uniform_prob);
      }
   }
   if(sum < 1e-20) {
      // never-visited infostate: fall back to uniform
      std::ranges::fill(probs, 1. / static_cast< double >(actions.size()));
      return probs;
   }
   for(double& p : probs) {
      p /= sum;
   }
   return probs;
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
template < typename ActionOrOutcome >
auto VanillaCFR< config, Env, Policy, AveragePolicy >::_br_advance(
   const ActionOrOutcome& action_or_outcome,
   const world_state_type& state,
   size_t depth,
   player_hashmap< sptr< info_state_type > >& infostates,
   player_hashmap< std::vector< std::pair< observation_type, observation_type > > >&
      observation_buffers
) -> world_state_type&
{
   // NOTE on arena safety: this walk only ever touches arena slots strictly DEEPER than the
   // gating traversal frame's depth, and that frame holds no live references to those slots
   world_state_type& next_wstate = _arena_state(depth + 1, state);
   _env().transition(next_wstate, action_or_outcome);

   const auto public_obs = _env().public_observation(state, action_or_outcome, next_wstate);
   const Player next_active_player = _env().active_player(next_wstate);
   // NOTE: unlike a naive early-return design we must NOT skip buffering when the child is
   // still a chance state -- multi-draw deals chain chance edges and every observation pair
   // has to reach the eventual flush target (mirrors the main traversal exactly)
   if(next_active_player != Player::chance) {
      auto infostate_entry_it = infostates.find(next_active_player);
      if(infostate_entry_it != infostates.end()) {
         auto child_infostate = std::make_shared< info_state_type >(*infostate_entry_it->second);
         auto& obs_history = observation_buffers[next_active_player];
         for(auto& obs : obs_history) {
            ::nor::detail::update_infostate(
               child_infostate, std::move(obs.first), std::move(obs.second)
            );
         }
         obs_history.clear();
         ::nor::detail::update_infostate(
            child_infostate,
            public_obs,
            _env().private_observation(next_active_player, state, action_or_outcome, next_wstate)
         );
         infostate_entry_it->second = std::move(child_infostate);
      }
   }
   for(auto player : _env().players(next_wstate)) {
      if(player == Player::chance or player == next_active_player) {
         continue;
      }
      observation_buffers[player].emplace_back(
         public_obs, _env().private_observation(player, state, action_or_outcome, next_wstate)
      );
   }
   return next_wstate;
}

template < CFRConfig config, typename Env, typename Policy, typename AveragePolicy >
template < typename ActionOrOutcome >
double VanillaCFR< config, Env, Policy, AveragePolicy >::_br_expectimax_from_edge(
   Player br_player,
   const ActionOrOutcome& action_or_outcome,
   const world_state_type& state,
   size_t depth,
   player_hashmap< sptr< info_state_type > >& infostates,
   player_hashmap< std::vector< std::pair< observation_type, observation_type > > >&
      observation_buffers
)
{
   // ---- edge application (in place) -------------------------------------------------------
   world_state_type& next_wstate = _arena_state(depth + 1, state);
   _env().transition(next_wstate, action_or_outcome);
   const auto public_obs = _env().public_observation(state, action_or_outcome, next_wstate);
   const Player next_active_player = _env().active_player(next_wstate);

   // targeted snapshots for restoration after the recursion (cheap: one shared_ptr per
   // player plus buffer size bookkeeping -- no container copies)
   const bool flushes = next_active_player != Player::chance
                        and infostates.contains(next_active_player);
   sptr< info_state_type > saved_entry{};
   std::vector< std::pair< observation_type, observation_type > > saved_flush_buffer;
   std::vector< std::pair< Player, size_t > > saved_buffer_sizes;
   if(flushes) {
      saved_entry = infostates.at(next_active_player);
      saved_flush_buffer = observation_buffers.at(next_active_player);
   }
   for(const auto& [player, buffer] : observation_buffers) {
      saved_buffer_sizes.emplace_back(player, buffer.size());
   }

   if(flushes) {
      auto child_infostate = std::make_shared< info_state_type >(*saved_entry);
      auto& obs_history = observation_buffers[next_active_player];
      for(auto& obs : obs_history) {
         ::nor::detail::update_infostate(
            child_infostate, std::move(obs.first), std::move(obs.second)
         );
      }
      obs_history.clear();
      ::nor::detail::update_infostate(
         child_infostate,
         public_obs,
         _env().private_observation(next_active_player, state, action_or_outcome, next_wstate)
      );
      infostates.at(next_active_player) = child_infostate;
   } else if(next_active_player != Player::chance) {
      // player without an infostate entry yet: buffer only
   }
   for(auto player : _env().players(next_wstate)) {
      if(player == Player::chance or player == next_active_player) {
         continue;
      }
      observation_buffers[player].emplace_back(
         public_obs, _env().private_observation(player, state, action_or_outcome, next_wstate)
      );
   }

   // ---- expectimax step -------------------------------------------------------------------
   double value = 0.;
   if(_env().is_terminal(next_wstate)) {
      // pass the ROOT participant set explicitly (see _traverse's terminal note)
      value = collect_rewards(_env(), next_wstate, _env().players(root_state())).at(br_player);
   } else if constexpr(concepts::stochastic_env< env_type >) {
      const Player active_player = _env().active_player(next_wstate);
      if(active_player == Player::chance) {
         for(auto&& outcome : _env().chance_actions(next_wstate)) {
            const double prob = _env().chance_probability(next_wstate, outcome);
            if(prob <= 0.) {
               continue;
            }
            value += prob
                     * _br_expectimax_from_edge(
                        br_player, outcome, next_wstate, depth + 1, infostates, observation_buffers
                     );
         }
         // restore happens below (shared epilogue)
      } else {
         value = std::invoke([&] {
            const auto& actions = _env().actions(active_player, next_wstate);
            if(active_player == br_player) {
               // best response: greedy max over the subtree values
               double best = std::numeric_limits< double >::lowest();
               for(const auto& action : actions) {
                  best = std::max(
                     best,
                     _br_expectimax_from_edge(
                        br_player, action, next_wstate, depth + 1, infostates, observation_buffers
                     )
                  );
               }
               return best;
            }
            // opponent: expectation under their CURRENT AVERAGE strategy
            const auto& active_infostate = *infostates.at(active_player);
            const std::vector< double > avg_probs = _normalized_average_policy(
               active_infostate, actions
            );
            double expected = 0.;
            for(auto [idx, action] : std::views::enumerate(actions)) {
               if(avg_probs[idx] <= 0.) {
                  continue;
               }
               expected += avg_probs[idx]
                           * _br_expectimax_from_edge(
                              br_player,
                              action,
                              next_wstate,
                              depth + 1,
                              infostates,
                              observation_buffers
                           );
            }
            return expected;
         });
      }
   } else {
      const Player active_player = _env().active_player(next_wstate);
      const auto& actions = _env().actions(active_player, next_wstate);
      if(active_player == br_player) {
         // best response: greedy max over the subtree values
         value = std::numeric_limits< double >::lowest();
         for(const auto& action : actions) {
            value = std::max(
               value,
               _br_expectimax_from_edge(
                  br_player, action, next_wstate, depth + 1, infostates, observation_buffers
               )
            );
         }
      } else {
         // opponent: expectation under their CURRENT AVERAGE strategy (fetch_policy<average>)
         const auto& active_infostate = *infostates.at(active_player);
         const std::vector< double > avg_probs = _normalized_average_policy(
            active_infostate, actions
         );
         for(auto [idx, action] : std::views::enumerate(actions)) {
            if(avg_probs[idx] <= 0.) {
               continue;
            }
            value += avg_probs[idx]
                     * _br_expectimax_from_edge(
                        br_player, action, next_wstate, depth + 1, infostates, observation_buffers
                     );
         }
      }
   }

   // ---- restore ---------------------------------------------------------------------------
   if(flushes) {
      infostates.at(next_active_player) = std::move(saved_entry);
      observation_buffers.at(next_active_player) = std::move(saved_flush_buffer);
   }
   for(const auto& [player, size] : saved_buffer_sizes) {
      observation_buffers.at(player).resize(size);
   }
   return value;
}

}  // namespace nor::rm
