
#ifndef NOR_RM_MINIMIZERS_HPP
#define NOR_RM_MINIMIZERS_HPP

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/common.hpp"
#include "nor/concepts.hpp"
#include "nor/rm/cfr_tabular/cfr_config.hpp"
#include "nor/rm/pruning.hpp"
#include "nor/rm/rm_utils.hpp"

namespace nor::rm {

/// the canonical per-action double table used by all tabular regret minimizers.
/// Entry i of the table belongs to the i-th action registered at the owning
/// infostate node (see 'detail::action_registry'); tables are grown in lockstep
/// with the registry by 'register_action'.
template < concepts::action Action >
using per_action_table = std::vector< double >;

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////// minimizer node data protocol /////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Every regret minimizer type M exposes:
//    using node_data_type            -- the per-infostate tables it needs. It always contains a
//                                       'regret' member of type per_action_table<Action> whose
//                                       entries are aligned to the node's action registry, and
//    static void register_action(node_data_type&, const Action&)
//                                    -- append 'action' to the registry and zero-initialize all
//                                       per-action table slots for it
//    static void observe(node_data_type&, const Action&, double increment)
//                                    -- fold one counterfactually weighted instantaneous regret
//                                       increment into the tables
//    static void recommend(node_data_type&, PolicyOut&, size_t iteration)
//                                    -- derive the current policy from the stored regret. May
//                                       mutate the tables (e.g. CFR+ clamping or predictive
//                                       minimizers refreshing their strategy snapshot).
//    static double policy_weight(size_t iteration)
//                                    -- multiplier the solver applies to the accumulated average
//                                       policy after this iteration's traversal
//
// The concept below fixes this contract.

template < typename M, typename Action, typename PolicyOut >
concept regret_minimizer_for = concepts::action< Action > and std::is_default_constructible_v< M >
                               and requires(
                                  M minimizer,
                                  typename M::node_data_type& node_data,
                                  const Action& action,
                                  PolicyOut& policy_out,
                                  size_t iteration
                               ) {
                                      typename M::node_data_type;
                                      M::register_action(node_data, action);
                                      M::observe(node_data, action, 1.);
                                      minimizer.recommend(node_data, policy_out, iteration);
                                      {
                                         minimizer.policy_weight(iteration)
                                      } -> std::convertible_to< double >;
                                   };

namespace detail {

/// the registration-order list of an infostate node's legal actions. It is the
/// single source of truth for the action -> index mapping that all flat
/// per-action tables ('per_action_table') of a node are aligned to: entry i of
/// every table belongs to 'actions[i]'.
template < concepts::action Action >
struct action_registry {
   std::vector< Action > actions;

   void register_action(const Action& action) { actions.emplace_back(action); }

   /// resolves the table index of 'action' (linear scan; legal-action counts
   /// per infostate are tiny, so this beats any keyed structure)
   [[nodiscard]] size_t index_of(const Action& action) const
   {
      const auto found = std::ranges::find(actions, action);
      if(found == actions.end()) {
         throw std::out_of_range("action is not registered at this infostate node");
      }
      return static_cast< size_t >(found - actions.begin());
   }
};

/// writes `positive-part-of-regret / sum(positive parts)` (or uniform when the
/// sum vanishes) for every registered action into the policy output
template < typename Action, typename PolicyOut >
void _recommend_from_regret(
   const std::vector< Action >& actions,
   const std::vector< double >& regret,
   PolicyOut& policy_out
)
{
   double pos_regret_sum{0.};
   for(double cumul_regret : regret) {
      pos_regret_sum += std::max(0., cumul_regret);
   }
   if(pos_regret_sum > 0.) {
      for(auto&& [action, cumul_regret] : std::views::zip(actions, regret)) {
         policy_out[action] = std::max(0., cumul_regret) / pos_regret_sum;
      }
   } else {
      const double uniform_prob = 1. / static_cast< double >(regret.size());
      for(const auto& action : actions) {
         policy_out[action] = uniform_prob;
      }
   }
}

}  // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// concrete minimizers /////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

/// vanilla external-regret matching (CFR)
template < concepts::action Action >
struct RegretMatching {
   struct node_data_type {
      detail::action_registry< Action > registry;
      /// cumulative counterfactual regret z(I,a); entry i belongs to registry.actions[i]
      per_action_table< Action > regret;

      void register_action(const Action& action)
      {
         registry.register_action(action);
         regret.emplace_back(0.);
      }

      [[nodiscard]] size_t index_of(const Action& action) const
      {
         return registry.index_of(action);
      }
   };

   static void observe(node_data_type& data, const Action& action, double increment)
   {
      data.regret[data.registry.index_of(action)] += increment;
   }

   template < typename PolicyOut >
   static void recommend(const node_data_type& data, PolicyOut& policy_out, size_t /*iteration*/)
   {
      detail::_recommend_from_regret(data.registry.actions, data.regret, policy_out);
   }

   static constexpr double policy_weight(size_t /*iteration*/) { return 1.; }
};

/// optimistic external-regret matching (CFR+): the cumulative regret table is
/// clamped to non-negative values during each recommendation step
template < concepts::action Action >
struct RegretMatchingPlus {
   struct node_data_type {
      detail::action_registry< Action > registry;
      per_action_table< Action > regret;

      void register_action(const Action& action)
      {
         registry.register_action(action);
         regret.emplace_back(0.);
      }

      [[nodiscard]] size_t index_of(const Action& action) const
      {
         return registry.index_of(action);
      }
   };

   static void observe(node_data_type& data, const Action& action, double increment)
   {
      data.regret[data.registry.index_of(action)] += increment;
   }

   template < typename PolicyOut >
   static void recommend(node_data_type& data, PolicyOut& policy_out, size_t /*iteration*/)
   {
      // CFR+ clamps the cumulative regret in place before deriving the policy
      double pos_regret_sum{0.};
      for(double& cumul_regret : data.regret) {
         cumul_regret = std::max(0., cumul_regret);
         pos_regret_sum += cumul_regret;
      }
      if(pos_regret_sum > 0.) {
         for(auto&& [action, cumul_regret] : std::views::zip(data.registry.actions, data.regret)) {
            policy_out[action] = cumul_regret / pos_regret_sum;
         }
      } else {
         const double uniform_prob = 1. / static_cast< double >(data.regret.size());
         for(const auto& action : data.registry.actions) {
            policy_out[action] = uniform_prob;
         }
      }
   }

   static constexpr double policy_weight(size_t /*iteration*/) { return 1.; }
};

/// CFR+ with regret-based pruning (Brown & Sandholm, NIPS 2015, sec. 4.2). The cumulative
/// regret follows CFR+'s modified update rule
///    R^T(I,a) = r^T(I,a)   if r^T(I,a) > 0 and R^{T-1}(I,a) <= 0
///    R^T(I,a) = R^{T-1}(I,a) + r^T(I,a)   otherwise
/// which reproduces plain CFR+ whenever the result would be non-negative but lets the table
/// drop below zero so that pruning windows can be armed on deeply negative entries. The
/// iteration-aggregated instantaneous regret r^T is buffered by 'observe' across a traversal
/// and folded in -- with the replace-if-positive rule -- by 'recommend'.
///
/// The node data additionally carries the per-(infostate,action) RBP bookkeeping tables
/// (prune deadlines, best-response buffers) consulted by the solver's traversal gate.
template < concepts::action Action >
struct RegretMatchingPlusRBP {
   struct node_data_type {
      detail::action_registry< Action > registry;
      per_action_table< Action > regret;
      /// instantaneous regret increments r(I, a) of the current iteration
      per_action_table< Action > cumulative_instant_regret;
      /// regret-based pruning window bookkeeping (deadlines + best-response buffers)
      pruning::RBPTables rbp;

      void register_action(const Action& action)
      {
         registry.register_action(action);
         regret.emplace_back(0.);
         cumulative_instant_regret.emplace_back(0.);
         rbp.register_action();
      }

      [[nodiscard]] size_t index_of(const Action& action) const
      {
         return registry.index_of(action);
      }
   };

   static void observe(node_data_type& data, const Action& action, double increment)
   {
      // buffer the increment; the paper's rule aggregates over all histories h in I visited
      // within one iteration before deciding between replace and add
      data.cumulative_instant_regret[data.registry.index_of(action)] += increment;
   }

   template < typename PolicyOut >
   static void recommend(node_data_type& data, PolicyOut& policy_out, size_t /*iteration*/)
   {
      // fold the buffered instantaneous regret into the cumulative table via the modified
      // CFR+ rule (replace-if-positive) while deriving the new recommendation
      double pos_regret_sum{0.};
      for(auto idx : std::views::iota(size_t{0}, data.regret.size())) {
         double& cumul_regret = data.regret[idx];
         double& instant_regret = data.cumulative_instant_regret[idx];
         cumul_regret = instant_regret > 0. and cumul_regret <= 0. ? instant_regret
                                                                   : cumul_regret + instant_regret;
         instant_regret = 0.;
         pos_regret_sum += std::max(0., cumul_regret);
      }
      if(pos_regret_sum > 0.) {
         for(auto&& [action, cumul_regret] : std::views::zip(data.registry.actions, data.regret)) {
            policy_out[action] = std::max(0., cumul_regret) / pos_regret_sum;
         }
      } else {
         const double uniform_prob = 1. / static_cast< double >(data.regret.size());
         for(const auto& action : data.registry.actions) {
            policy_out[action] = uniform_prob;
         }
      }
   }

   static constexpr double policy_weight(size_t /*iteration*/) { return 1.; }
};

/// exponential CFR ("exponential-weighted CFR"). Owns three extra named tables
/// per infostate:
///    instant_regret          -- buffers r(I, a) = sum_h r(h, a) of iteration t
///    reach_prob_snapshot     -- pi^t(I) of iteration t
///    avg_policy_denominator  -- sum_t pi^t(I) * exp(L1^t(I, a))
/// Cumulative regret updates are deferred until the end of an iteration, where
/// they are weighted by the L1 factors exp(r(I,a) - mean_r(I)).
template < concepts::action Action >
struct ExponentialCFR {
   struct node_data_type {
      detail::action_registry< Action > registry;
      per_action_table< Action > regret;
      per_action_table< Action > instant_regret;
      double reach_prob_snapshot = 0.;
      per_action_table< Action > avg_policy_denominator;

      void register_action(const Action& action)
      {
         registry.register_action(action);
         regret.emplace_back(0.);
         instant_regret.emplace_back(0.);
         avg_policy_denominator.emplace_back(0.);
      }

      [[nodiscard]] size_t index_of(const Action& action) const
      {
         return registry.index_of(action);
      }
   };

   static void observe(node_data_type& data, const Action& action, double increment)
   {
      data.instant_regret[data.registry.index_of(action)] += increment;
   }

   static void snapshot_reach_probability(node_data_type& data, double reach_prob)
   {
      data.reach_prob_snapshot = reach_prob;
   }

   template < typename PolicyOut >
   static void recommend(const node_data_type& data, PolicyOut& policy_out, size_t /*iteration*/)
   {
      detail::_recommend_from_regret(data.registry.actions, data.regret, policy_out);
   }

   static constexpr double policy_weight(size_t /*iteration*/) { return 1.; }

   /// end-of-iteration bookkeeping: applies L1-weighted cumulative regret
   /// updates, updates the average policy numerator/denominator pair, consumes
   /// the instantaneous buffer and finally refreshes the current policy.
   template < typename CurrentPolicy, typename AveragePolicy, typename BetaFn >
   static void finalize_iteration(
      node_data_type& data,
      CurrentPolicy& curr_policy,
      AveragePolicy& avg_policy,
      size_t iteration,
      BetaFn&& beta
   )
   {
      const auto n_actions = data.regret.size();
      auto& instant_table = data.instant_regret;
      // L1(I, a) = exp(r(I, a) - mean_r(I)); computed before mutating anything.
      // The scratch buffer is HOISTED out of the per-infostate sweep: minimizers
      // are stateless shared objects, so the reuse storage lives here as a
      // thread-local static (the end-of-iteration sweep is intentionally serial
      // -- see the determinism note in rm_utils.hpp -- and finalize_iteration
      // is never reentered recursively). CONTRACT: 'beta' must NOT invoke CFR
      // machinery (traversals, iterate, finalize_iteration itself, ...) on the
      // SAME thread while this hoisted buffer holds unsaved scratch state --
      // reentrancy would silently clobber it mid-sweep. Reuse is result-neutral:
      // the buffer is cleared and refilled with exactly the same values in exactly
      // the same order on every call.
      static thread_local per_action_table< Action > l1_weights;
      l1_weights.clear();
      l1_weights.reserve(n_actions);
      double average_instant_regret = std::ranges::fold_left(instant_table, double(0.), std::plus{})
                                      / double(n_actions);
      for(double instant_regret : instant_table) {
         l1_weights.push_back(std::exp(instant_regret - average_instant_regret));
      }

      for(auto idx : std::views::iota(size_t{0}, n_actions)) {
         double& cumul_regret = data.regret[idx];
         double& instant_regret = instant_table[idx];
         if(instant_regret >= 0.) {
            cumul_regret += l1_weights[idx] * instant_regret;
         } else {
            cumul_regret += l1_weights[idx] * beta(instant_regret, iteration);
         }
         // reset the buffer so the next iteration starts fresh
         instant_regret = 0.;
      }

      for(auto idx : std::views::iota(size_t{0}, n_actions)) {
         const auto& action = data.registry.actions[idx];
         const double l1_weight = l1_weights[idx];
         // cumulative numerator update
         avg_policy[action] += l1_weight * data.reach_prob_snapshot * curr_policy[action];
         // cumulative denominator update
         data.avg_policy_denominator[idx] += l1_weight * data.reach_prob_snapshot;
      }

      recommend(data, curr_policy, iteration);
   }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////// internal (phi-) regret matching /////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Internal ("phi-") regret matching over the canonical swap basis
 * ("RM-X" style; Hart & Mas-Colell "Simple Adaptive Strategies", 2000).
 *
 * Where RegretMatching accumulates the plain counterfactual value offset, this
 * kernel maintains one accumulator per action of the CANONICAL SWAP BASIS: the
 * transformation phi_a that moves ALL strategy mass onto action a. The
 * cumulative transformed regret of iteration t is
 *
 *    R^t(I, phi_a) = sum_tau <r^tau(I), phi_a(sigma^tau(I)) - sigma^tau(I)>
 *                  = sum_tau [ r^tau(I, a) - <sigma^tau(I), r^tau(I)> ]
 *
 * i.e. the target-a column aggregate of the full pairwise (i -> a)
 * internal-regret matrix in its mixed-strategy form -- summing the pairwise
 * regrets over their SOURCE index collapses the |A|^2 matrix onto these |A|
 * basis accumulators. The recommendation is regret matching on the positive
 * parts of R(phi_a).
 *
 * MECHANICS. Counterfactual increments r(h, a) arrive through 'observe' (as for
 * every minimizer) and are buffered into an instantaneous table; 'recommend'
 * folds the whole buffer against a snapshot sigma^tau(I) of the recommendation
 * under which it was incurred (before the very first recommendation the played
 * strategy is the solver's uniform initialization), derives the new policy and
 * refreshes the snapshot so that the next fold pairs with the actually played
 * strategy.
 *
 * COMPLEXITY. O(|A|) time per observe and per recommend-fold (the fold is two
 * |A|-sized dot-product/addition passes), O(|A|) extra memory per infostate
 * beyond vanilla RM's table: three aligned |A| tables total (swap regret,
 * instantaneous buffer, policy snapshot). The full pairwise internal-regret
 * matrix is never materialized.
 */
template < concepts::action Action >
struct InternalRegretMatching {
   struct node_data_type {
      detail::action_registry< Action > registry;
      /// cumulative transformed regret R^t(I, phi_a); entry i belongs to
      /// registry.actions[i]. Named 'regret' as required by the
      /// InfostateNodeData protocol (and accurate: it IS the cumulative
      /// phi-regret table)
      per_action_table< Action > regret;
      /// instantaneous counterfactual regret buffer r^t(I, a) = sum_h pi_{-i}(h)(v(h,a) - v(h))
      /// of the current iteration; consumed (reset to zero) by the next fold
      per_action_table< Action > instant_regret;
      /// sigma^t(I): snapshot of the last recommendation, refreshed by every
      /// 'recommend' call; the pairing partner of the buffered increments
      per_action_table< Action > policy_snapshot;
      /// false until the first 'recommend' has populated 'policy_snapshot';
      /// beforehand the played strategy is the solver's uniform initialization
      bool snapshot_live = false;

      void register_action(const Action& action)
      {
         registry.register_action(action);
         regret.emplace_back(0.);
         instant_regret.emplace_back(0.);
         policy_snapshot.emplace_back(0.);
      }

      [[nodiscard]] size_t index_of(const Action& action) const
      {
         return registry.index_of(action);
      }
   };

   static void register_action(node_data_type& data, const auto& action)
   {
      data.register_action(action);
   }

   static void observe(node_data_type& data, const Action& action, double increment)
   {
      data.instant_regret[data.registry.index_of(action)] += increment;
   }

   template < typename PolicyOut >
   static void recommend(node_data_type& data, PolicyOut& policy_out, size_t /*iteration*/)
   {
      const auto n_actions = data.regret.size();

      // fold the buffered instantaneous regret against the strategy that generated it:
      //    v = <sigma^t, r^t>   and   R(phi_a) += <r^t, phi_a(sigma^t) - sigma^t> = r^t(a) - v
      // (linear in the per-history increments, so buffering + one fold reproduces
      // the exact per-iteration inner products)
      double expected_strategy_value{0.};
      if(data.snapshot_live) {
         for(auto&& [sigma_a, instant_regret] :
             std::views::zip(data.policy_snapshot, data.instant_regret)) {
            expected_strategy_value += sigma_a * instant_regret;
         }
      } else {
         const double uniform_prob = 1. / static_cast< double >(n_actions);
         for(double instant_regret : data.instant_regret) {
            expected_strategy_value += uniform_prob * instant_regret;
         }
      }
      for(auto idx : std::views::iota(size_t{0}, n_actions)) {
         data.regret[idx] += data.instant_regret[idx] - expected_strategy_value;
         data.instant_regret[idx] = 0.;
      }

      // regret-match on the positive parts of the transformed-regret table ...
      detail::_recommend_from_regret(data.registry.actions, data.regret, policy_out);

      // ... and refresh the play-snapshot so the next iteration's fold pairs
      // with the strategy actually recommended now ('policy_out' is keyed by
      // action, hence the registry-ordered read-back instead of a zip)
      for(auto&& [action, snapshot] :
          std::views::zip(data.registry.actions, data.policy_snapshot)) {
         snapshot = policy_out[action];
      }
      data.snapshot_live = true;
   }

   static constexpr double policy_weight(size_t /*iteration*/) { return 1.; }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// weighting decorators //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Discounted-CFR (DCFR) weighting decorator around an inner minimizer.
 *
 * At recommendation time the cumulative regret entries are scaled by
 * t^alpha / (t^alpha + 1) (positive regrets) resp. t^beta / (t^beta + 1)
 * (non-positive regrets); after each iteration the accumulated average policy
 * is scaled by (t / (t + 1))^gamma, where t is the logical (1-based) iteration
 * number. Linear CFR is the special case alpha = beta = gamma = 1.
 *
 * The 'discount_regrets' flag allows using the decorator purely for its
 * average-policy side: when set to false the alpha/beta regret scaling is
 * compiled out and only the gamma-side policy weight remains active. This is
 * how PCFR+ composes its required quadratic averaging (gamma = 2) without
 * perturbing the predictive regret updates.
 */
template < typename Inner, bool discount_regrets = true >
class DiscountedCFR {
  public:
   constexpr DiscountedCFR() = default;
   explicit constexpr DiscountedCFR(CFRDiscountedParameters params) : m_params(params) {}

   using node_data_type = typename Inner::node_data_type;

   static void register_action(node_data_type& data, const auto& action)
   {
      Inner::register_action(data, action);
   }

   static void observe(node_data_type& data, const auto& action, double increment)
   {
      Inner::observe(data, action, increment);
   }

   template < typename PolicyOut >
   void recommend(node_data_type& data, PolicyOut& policy_out, size_t iteration) const
   {
      // note: the regret scaling uses the raw iteration counter (no +1 offset)
      // while the policy weight below uses the logical iteration number. This
      // mirrors the historical behavior which empirically converged faster.
      // 'discount_factor' reproduces the historical arithmetic bit-for-bit for
      // the constant exponents and additionally defines the raw-index-0 /
      // negative-exponent corner (NaN guard, see its documentation).
      if constexpr(discount_regrets) {
         const double factor_positive = discount_factor(iteration, m_params.alpha_at(iteration));
         const double factor_negative = discount_factor(iteration, m_params.beta_at(iteration));
         for(double& cumul_regret : data.regret) {
            cumul_regret *= cumul_regret > 0. ? factor_positive : factor_negative;
         }
      }
      Inner::recommend(data, policy_out, iteration);
   }

   [[nodiscard]] double policy_weight(size_t iteration) const
   {
      // add +1 to obtain the logical iteration number t (iterations count from
      // 0 numerically but from 1 theoretically); the resulting fraction is then
      // exponentiated by gamma. When 'weight_by_cycle' is active the solver
      // passes the CYCLE index (= iteration / num_players) instead of the raw
      // iteration (see TabularCFRBase::cycle()).
      double t = double(iteration) + 1.;
      return std::pow(t / (t + 1.), m_params.gamma_at(iteration));
   }

   /// the stored parameters (exposes 'weight_by_cycle' to the solver)
   [[nodiscard]] const CFRDiscountedParameters& discounted_parameters() const { return m_params; }

  private:
   CFRDiscountedParameters m_params;
};

/**
 * @brief Stateless LINEAR weighting decorator around an inner minimizer --
 * "Linear CFR" (Brown & Sandholm, "Solving Imperfect-Information Games via
 * Discounted Regret Minimization", AAAI 2014), i.e. exactly the DCFR family
 * member DiscountedCFR with the exponents frozen at their unit values
 * alpha = beta = gamma = 1.
 *
 * The schedules need no stored parameters in that corner: the cumulative
 * regrets are scaled by t / (t + 1) at recommendation time (raw 0-based index,
 * replicating DiscountedCFR's arithmetic call-for-call -- including the
 * historical evaluation of BOTH discount branches even though they coincide
 * numerically) and the accumulated average policy is rescaled by the logical
 * iteration fraction pow(t / (t + 1), 1) after every iteration. Bit-for-bit
 * parity with historical linear-parameterized DiscountedCFR runs is the design
 * contract; see the CFRLinear alias and the accompanying regression test.
 *
 * Being parameter-free, ALL protocol members are static. That is precisely what
 * makes this carrier composable with the Thresholded<Inner> dynamic-thresholding
 * wrapper (whose forwards are qualified static-style calls), whereas the
 * parameter-carrying DiscountedCFR decorator is not. Runtime-SCHEDULE carriers
 * (HS-schedules, DDCFR agents) deliberately remain on the instance carrier and
 * are statically excluded from dynamic thresholding (see
 * sanity_check_cfr_config).
 */
template < typename Inner >
class LinearCFR {
  public:
   using node_data_type = typename Inner::node_data_type;

   static void register_action(node_data_type& data, const auto& action)
   {
      Inner::register_action(data, action);
   }

   static void observe(node_data_type& data, const auto& action, double increment)
   {
      Inner::observe(data, action, increment);
   }

   template < typename PolicyOut >
   static void recommend(node_data_type& data, PolicyOut& policy_out, size_t iteration)
   {
      // bit-for-bit replication of DiscountedCFR<Inner>::recommend at constant
      // unit exponents: same raw-index convention, same discount_factor calls
      // and the same scale-then-forward-to-inner operation order (note that
      // factor_positive == factor_negative here -- alpha == beta == 1 -- but
      // both are computed separately to preserve the historical FP sequence)
      const double factor_positive = discount_factor(iteration, 1.);
      const double factor_negative = discount_factor(iteration, 1.);
      for(double& cumul_regret : data.regret) {
         cumul_regret *= cumul_regret > 0. ? factor_positive : factor_negative;
      }
      Inner::recommend(data, policy_out, iteration);
   }

   /// the gamma-side average-policy weight: logical iteration fraction
   /// pow(t / (t + 1), gamma = 1), mirroring DiscountedCFR::policy_weight
   /// (which the solver indexes by the raw iteration count here, matching the
   /// historical linear spelling with weight_by_cycle == false)
   static double policy_weight(size_t iteration)
   {
      double t = double(iteration) + 1.;
      return std::pow(t / (t + 1.), 1.);
   }
};

/**
 * @brief Greedy-weights decorator (Zhang, Lerer & Brown, "Equilibrium Finding in
 * Normal-Form Games Via Greedy Regret Minimization", AAAI 2022,
 * arXiv:2204.04826) around an inner RM / RM+ minimizer.
 *
 * Unlike every other weighting mode, the greedy iteration weight depends on the
 * instantaneous regrets of a COMPLETED traversal and must be identical for all
 * infostates of the updating player(s) (arXiv:2204.04826, Appendix F), so the
 * fold cannot happen incrementally during the descent. This decorator therefore
 * buffers each iteration's counterfactual regret increments ('observe' writes
 * only into 'instant_regret', mirroring ExponentialCFR's deferral pattern) and
 * exposes 'apply_weighted_fold' for the solver's end-of-iteration greedy sweep:
 *
 *    1. the solver aggregates the (cumulative regret R_j, buffered regret r_j)
 *       pairs over all swept infostates and computes the scalar weight w that
 *       greedily minimizes phi((R + w r)/(w_sum + w)) with
 *       phi(x_+) = sum_j max(0, x_j)^2  (Algorithm 1; Appendix F: the objective
 *       is "the sum of all local potential functions at all infosets");
 *    2. per infostate this decorator then applies the update
 *          R <- scale * R + w * r,     avg <- scale * avg + w * pi_reach * sigma^t
 *       with BOTH accumulators discounted by the same 'history_scale' so that the
 *       weighted-mean invariant between regret tables and average policy is exact.
 *       Regular iterations use scale = 1 and weight = w directly (like the authors'
 *       reference implementation, github.com/hughbzhang/greedy-weights); only a
 *       near-infinite search result -- the objective's infimum at w -> infinity,
 *       i.e. an iteration that invalidates the accumulated history -- is realized
 *       as their degenerate-case handling: dilute all previous mass by 1e-6 and
 *       weigh this iteration by 1;
 *    3. finally the inner recommendation refreshes sigma^{t+1} from the updated
 *       cumulative regret (Algorithm 1 derives the next iterate from R + w r).
 *       For Inner == RegretMatchingPlus this also performs CFR+'s clamping at
 *       exactly its canonical position (recommendation time).
 *
 * The weighting used for all infosets of one iteration is equal by construction,
 * which is what retains the O(1/sqrt(T)) convergence guarantee (Theorem 1).
 */
template < concepts::action Action, typename Inner >
class GreedyWeights {
   using inner_node_data_type = typename Inner::node_data_type;

  public:
   /// inner tables plus the end-of-iteration bookkeeping owned by the greedy sweep
   struct node_data_type: public inner_node_data_type {
      /// buffered instantaneous regret increments r(I, a) of the current iteration
      /// (cleared by 'apply_weighted_fold' once they are folded into 'regret')
      per_action_table< Action > instant_regret;
      /// sum of the updating player's own reach probabilities pi^t(h) over every
      /// history h of this infostate visited during the current iteration; drives
      /// the deferred average-policy increment. Consumed and reset to zero by
      /// 'apply_weighted_fold'
      double reach_prob_snapshot = 0.;

      void register_action(const Action& action)
      {
         inner_node_data_type::register_action(action);
         instant_regret.emplace_back(0.);
      }
   };

   static void register_action(node_data_type& data, const auto& action)
   {
      data.register_action(action);
   }

   static void observe(node_data_type& data, const auto& action, double increment)
   {
      data.instant_regret[data.registry.index_of(action)] += increment;
   }

   template < typename PolicyOut >
   static void recommend(node_data_type& data, PolicyOut& policy_out, size_t iteration)
   {
      Inner::recommend(data, policy_out, iteration);
   }

   static constexpr double policy_weight(size_t /*iteration*/) { return 1.; }

   /// end-of-iteration fold driven by the solver's greedy sweep: discounts the previously
   /// accumulated mass of BOTH the regret table and the average-policy accumulator by
   /// 'history_scale' (1.0 on the regular path), folds the buffered instantaneous regrets
   /// scaled by 'weight', adds the weighted average-policy increment (numerator side only --
   /// the average policy stays in its unnormalized cumulative representation) and refreshes
   /// the current policy from the freshly updated regret table. The two-sided discount keeps
   /// regret tables and average-policy accumulator on a COMMON scale so that the
   /// weighted-mean invariant pi_bar ∝ sum_t w_t pi^t is preserved exactly.
   template < typename CurrentPolicy, typename AveragePolicy >
   static void apply_weighted_fold(
      node_data_type& data,
      double history_scale,
      double weight,
      CurrentPolicy& current_policy,
      AveragePolicy& average_policy,
      size_t iteration
   )
   {
      const auto n_actions = data.regret.size();
      for(auto idx : std::views::iota(size_t{0}, n_actions)) {
         // discount previous iterations' mass, then add this iteration's buffered
         // increment with the effective weight (Algorithm 1: R <- R + w r)
         double& cumul_regret = data.regret[idx];
         cumul_regret = history_scale * cumul_regret + weight * data.instant_regret[idx];
         data.instant_regret[idx] = 0.;
      }
      for(const auto& action : data.registry.actions) {
         // deferred average-policy increment (Algorithm 1: pi_bar <- (w_sum pi_bar + w pi)/(w_sum +
         // w), realized as scale-history-then-add on the cumulative representation)
         auto& entry = average_policy[action];
         entry = history_scale * entry + weight * data.reach_prob_snapshot * current_policy[action];
      }
      // consume the snapshot so that unvisited infostates contribute nothing on
      // their next fold and no stale mass survives into the following iteration
      data.reach_prob_snapshot = 0.;
      // derive sigma^{t+1} from the updated regret table (Inner::recommend also
      // performs RM+'s clamping at its canonical position)
      Inner::recommend(data, current_policy, iteration);
   }
};

/**
 * @brief Dynamic-thresholding decorator (Brown, Kroer, Sandholm, AAAI 2017,
 * DOI 10.1609/aaai.v31i1.10603) around an inner regret minimizer.
 *
 * At every recommendation, actions whose probability falls below the schedule
 *
 *    tau_t = (C^2 - 1) / (2 C |A(I)|^2 sqrt(t))     (their Theorem 2; RM-family recommenders,
 *                                                   which is what all our local minimizers are,
 *                                                   including ExponentialCFR whose
 *                                                   recommendation step is regret matching on
 *                                                   its cumulative table)
 *
 * are set to exactly zero probability and the remainder is renormalized to sum to one. t is the
 * logical (1-based) iteration and C >= 1 the aggressiveness constant. This makes low-probability
 * actions prunable even for inner minimizers that would otherwise assign positive mass to every
 * action, and their Theorem 2 guarantees the regret bound degrades only by the constant factor C.
 *
 * The decorator honors the full observe/recommend/policy_weight protocol of the minimizer
 * framework, forwards ExponentialCFR's finalize_iteration (re-applying thresholding after the
 * inner policy refresh), and carries the per-(infostate,action) RBP tables in its node data so
 * that the solver's traversal gate can be reused unchanged for pruning_mode ==
 * dynamic_thresholding.
 */
template < typename Inner, double ThresholdC >
class Thresholded {
   static_assert(ThresholdC >= 1., "dynamic thresholding requires C >= 1");

  public:
   constexpr Thresholded() = default;

   /// node data = inner node data + the RBP window bookkeeping consulted by the traversal gate
   struct node_data_type: public Inner::node_data_type {
      pruning::RBPTables rbp;

      void register_action(const auto& action)
      {
         Inner::node_data_type::register_action(action);
         rbp.register_action();
      }
   };

   static void register_action(node_data_type& data, const auto& action)
   {
      data.register_action(action);
   }

   static void observe(node_data_type& data, const auto& action, double increment)
   {
      Inner::observe(data, action, increment);
   }

   template < typename PolicyOut >
   static void recommend(node_data_type& data, PolicyOut& policy_out, size_t iteration)
   {
      Inner::recommend(data, policy_out, iteration);
      const double tau = pruning::rm_dynamic_threshold(
         data.registry.actions.size(), iteration + 1, ThresholdC
      );
      _apply_threshold(policy_out, tau);
   }

   static double policy_weight(size_t iteration) { return Inner::policy_weight(iteration); }

   /// forwards ExponentialCFR's deferred end-of-iteration update; the inner refresh ends with
   /// its own recommendation, after which thresholding is re-applied here
   template < typename CurrentPolicy, typename AveragePolicy, typename BetaFn >
      requires requires(
         node_data_type& data,
         CurrentPolicy& curr,
         AveragePolicy& avg,
         size_t it,
         BetaFn&& beta
      ) { Inner::finalize_iteration(data, curr, avg, it, std::forward< BetaFn >(beta)); }
   static void finalize_iteration(
      node_data_type& data,
      CurrentPolicy& curr,
      AveragePolicy& avg,
      size_t iteration,
      BetaFn&& beta
   )
   {
      Inner::finalize_iteration(data, curr, avg, iteration, std::forward< BetaFn >(beta));
      recommend(data, curr, iteration);
   }

  private:
   /// zeroes every entry below tau and renormalizes the survivors; if EVERY entry falls below
   /// tau the argmax entry is kept at probability 1 (a pure recommendation cannot be all-zero,
   /// so the renormalizer's denominator is rescued from vanishing)
   template < typename PolicyOut >
   static void _apply_threshold(PolicyOut& policy_out, double tau)
   {
      if(tau <= 0.) {
         return;
      }
      using entry_type = std::remove_reference_t< decltype(*std::begin(policy_out)) >;
      const entry_type* best_kept = nullptr;
      double kept_mass = 0.;
      for(auto& entry : policy_out) {
         if(entry.second >= tau) {
            kept_mass += entry.second;
            if(best_kept == nullptr or entry.second > best_kept->second) {
               best_kept = &entry;
            }
         } else {
            entry.second = 0.;
         }
      }
      if(kept_mass > 0.) {
         for(auto& entry : policy_out) {
            entry.second /= kept_mass;
         }
         return;
      }
      const entry_type* best_any = nullptr;
      for(auto& entry : policy_out) {
         if(best_any == nullptr or entry.second > best_any->second) {
            best_any = &entry;
         }
      }
      if(best_any != nullptr) {
         for(auto& entry : policy_out) {
            entry.second = (&entry == best_any) ? 1. : 0.;
         }
      }
   }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// predictive minimizers //////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// predictive minimizers //////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

// the predictive minimizer lives in its own header which opens 'nor::rm'
// itself, hence the temporary namespace closure here
}  // namespace nor::rm

#include "predictive.hpp"

// the DCFR+/PDCFR+ kernels + HS schedule factories follow the same pattern
#include "discounted_predictive.hpp"

// the behaviorally-constrained (perturbed) RM+ kernel follows it as well
#include "constrained.hpp"

// the optimistic multiplicative-weights (OMWU) kernel of the EC 2022
// predictive trigger-regret dynamics follows the same pattern
#include "omwu.hpp"

namespace nor::rm {

/////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// minimizer selection ////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <
   concepts::action Action,
   CFRWeightingMode weighting,
   CFRPruningMode pruning,
   RegretMinimizingMode rm_mode,
   double PerturbationFloor = 0. >
consteval auto select_vanilla_minimizer()
{
   if constexpr(rm_mode == RegretMinimizingMode::constrained_regret_matching_plus) {
      // behaviorally-constrained (perturbed) RM+ kernel (Farina, Kroer & Sandholm,
      // ICML 2017): RM+ over the linearly constrained simplex; the uniform floor
      // comes in as a compile-time value from CFRConfig::perturbation_floor.
      // Statically pinned to the uniform weighting mode by 'sanity_check_cfr_config'
      return std::type_identity< ConstrainedRMPlus< Action, PerturbationFloor > >{};
   } else if constexpr(rm_mode == RegretMinimizingMode::internal_regret_matching) {
      // swap-basis phi-regret kernel (Hart-Mas-Colell style internal-regret
      // matching); selected ahead of the weighting-mode branches because it is
      // statically pinned to the uniform weighting mode by the config check
      return std::type_identity< InternalRegretMatching< Action > >{};
   } else if constexpr(rm_mode == RegretMinimizingMode::predictive_regret_matching_plus) {
      // PCFR+ needs the quadratic average-policy accumulation of the discounted
      // weighting machinery (gamma = 2) but NOT the alpha/beta regret discounts
      return std::type_identity< DiscountedCFR<
         PredictiveRegretMatchingPlus< Action >,
         /*discount_regrets=*/false > >{};
   } else if constexpr(rm_mode == RegretMinimizingMode::sap_predictive_regret_matching_plus) {
      return std::type_identity< DiscountedCFR<
         SAPPredictiveRegretMatchingPlus< Action >,
         /*discount_regrets=*/false > >{};
   } else if constexpr(rm_mode == RegretMinimizingMode::ap_predictive_regret_matching_plus) {
      // APCFR+: adaptive per-infostate asymmetry of step sizes
      // (arXiv:2503.12770, Eq. (4) + adaptive rule Eq. (10))
      return std::type_identity< DiscountedCFR<
         APPredictiveRegretMatchingPlus< Action >,
         /*discount_regrets=*/false > >{};
   } else if constexpr(rm_mode == RegretMinimizingMode::p2p_predictive_regret_matching_plus) {
      // P2PCFR+ ("Pessimistic PCFR+", OpenReview njyZgDDeY4): fixed prediction
      // damping with alpha = 5
      return std::type_identity< DiscountedCFR<
         P2PPredictiveRegretMatchingPlus< Action >,
         /*discount_regrets=*/false > >{};
   } else if constexpr(rm_mode == RegretMinimizingMode::smooth_predictive_regret_matching_plus) {
      // Smooth PRM+: chop off the origin via the norm floor
      // (arXiv:2305.14709, Algorithm 2)
      return std::type_identity< DiscountedCFR<
         SmoothPredictiveRegretMatchingPlus< Action >,
         /*discount_regrets=*/false > >{};
   } else if constexpr(rm_mode == RegretMinimizingMode::stable_predictive_regret_matching_plus) {
      // Stable PRM+: componentwise restart schedule
      // (arXiv:2305.14709, Algorithm 1)
      return std::type_identity< DiscountedCFR<
         StablePredictiveRegretMatchingPlus< Action >,
         /*discount_regrets=*/false > >{};
   } else if constexpr(rm_mode == RegretMinimizingMode::discounted_regret_matching_plus) {
      // DCFR+ (arXiv:2404.13891): the kernel owns the alpha-side discounting
      // (fold-time, discount-before-add) AND the gamma-side averaging; it must be
      // selected directly -- a DiscountedCFR wrapper would double the gamma weight
      return std::type_identity< DiscountedRegretMatchingPlus< Action > >{};
   } else if constexpr(rm_mode == RegretMinimizingMode::discounted_predictive_regret_matching_plus) {
      // PDCFR+ (arXiv:2404.13891): DCFR+ with persistence-prediction recommendations
      return std::type_identity< DiscountedPredictiveRegretMatchingPlus< Action > >{};
   } else if constexpr(weighting == CFRWeightingMode::greedy) {
      // greedy weights wraps a plain RM / RM+ kernel; the predictive and
      // DCFR+-style kernels are statically rejected with this weighting mode by
      // 'sanity_check_cfr_config' (never analyzed together with dynamic weights)
      if constexpr(rm_mode == RegretMinimizingMode::regret_matching_plus) {
         return std::type_identity< GreedyWeights< Action, RegretMatchingPlus< Action > > >{};
      } else {
         return std::type_identity< GreedyWeights< Action, RegretMatching< Action > > >{};
      }
   } else if constexpr(weighting == CFRWeightingMode::exponential) {
      return std::type_identity< ExponentialCFR< Action > >{};
   } else if constexpr(weighting == CFRWeightingMode::linear) {
      // the STATELESS linear carrier (fixed unit exponents). Unlike the
      // parameter-carrying DiscountedCFR below, its fully static protocol
      // satisfies the qualified call pattern of Thresholded<Inner>, which is
      // what admits linear x dynamic_thresholding (see sanity_check_cfr_config)
      if constexpr(rm_mode == RegretMinimizingMode::regret_matching_plus) {
         return std::type_identity< LinearCFR< RegretMatchingPlus< Action > > >{};
      } else {
         return std::type_identity< LinearCFR< RegretMatching< Action > > >{};
      }
   } else if constexpr(weighting == CFRWeightingMode::discounted) {
      if constexpr(rm_mode == RegretMinimizingMode::regret_matching_plus) {
         return std::type_identity< DiscountedCFR< RegretMatchingPlus< Action > > >{};
      } else {
         return std::type_identity< DiscountedCFR< RegretMatching< Action > > >{};
      }
   } else if constexpr(pruning == CFRPruningMode::regret_based and rm_mode == RegretMinimizingMode::regret_matching_plus) {
      return std::type_identity< RegretMatchingPlusRBP< Action > >{};
   } else if constexpr(rm_mode == RegretMinimizingMode::regret_matching_plus) {
      return std::type_identity< RegretMatchingPlus< Action > >{};
   } else {
      return std::type_identity< RegretMatching< Action > >{};
   }
}

}  // namespace detail

/// maps a CFR configuration onto the concrete regret minimizer type acting on
/// actions of type 'Action' (before any pruning-mode decoration)
template < CFRConfig config, concepts::action Action >
using base_minimizer_for_t = typename decltype(detail::select_vanilla_minimizer<
                                               Action,
                                               config.weighting_mode,
                                               config.pruning_mode,
                                               config.regret_minimizing_mode,
                                               config.perturbation_floor >())::type;

/// final minimizer selection: dynamic thresholding wraps whatever base minimizer the rest of
/// the configuration selects (Brown, Kroer, Sandholm, AAAI 2017) -- composable with every
/// base whose protocol is fully static (RM / RM+, ExponentialCFR, LinearCFR); only the
/// parameter-carrying DiscountedCFR carrier is statically excluded from thresholded
/// configurations (see sanity_check_cfr_config). regret_based pruning needs no wrapper: it
/// selects RegretMatchingPlusRBP directly inside 'select_vanilla_minimizer'.
template < CFRConfig config, concepts::action Action >
using minimizer_for_t = std::conditional_t<
   config.pruning_mode == CFRPruningMode::dynamic_thresholding,
   Thresholded< base_minimizer_for_t< config, Action >, config.dynamic_threshold_c >,
   base_minimizer_for_t< config, Action > >;

///////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// MCCFR minimizers //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

struct no_mccfr_extras {};

/// lazy average-policy weighting keeps a per-action table of accumulated
/// average-policy mass of actions that were not sampled since the last visit
template < concepts::action Action >
struct lazy_weighting_extras {
   per_action_table< Action > pending_avg_accumulator;

   void register_action(const Action&) { pending_avg_accumulator.emplace_back(0.); }
};

/// pure CFR samples one 'pure strategy' action per infostate per iteration
template < concepts::action Action >
struct pure_sampling_extras {
   std::optional< Action > sampled_action;

   void register_action(const Action&) {}
};

struct optimistic_weighting_extras {
   /// the last iteration this infostate was visited at (for delay weighting)
   size_t last_visit_iteration = 0;

   void register_action(const auto&) {}
};

/// state-action baselines b̂(I, a) for variance-reduced outcome-sampling MCCFR
/// (Schmid et al., AAAI 2019): one scalar per legal action, regressed onto the
/// baseline-corrected value estimates of the sampled trajectories visiting I.
template < concepts::action Action >
struct vr_baseline_extras {
   per_action_table< Action > baseline;

   void register_action(const Action&) { baseline.emplace_back(0.); }
};

/// the set of regret-minimizing modes admissible inside MCCFR: plain RM for every
/// traversal scheme plus the whole PRM+/PCFR+ family of predictive kernels for
/// OUTCOME SAMPLING (see 'MCCFRMinimizer' for the composition-theory notes).
/// Deliberately queryable so tests can assert the selection behavior without
/// instantiating rejecting configurations (the engine-level static_asserts remain
/// hard errors by design).
template < RegretMinimizingMode rm_mode >
inline constexpr bool
   mccfr_admissible_rm_mode = rm_mode == RegretMinimizingMode::regret_matching
                              or rm_mode == RegretMinimizingMode::predictive_regret_matching_plus
                              or rm_mode
                                    == RegretMinimizingMode::sap_predictive_regret_matching_plus
                              or rm_mode == RegretMinimizingMode::ap_predictive_regret_matching_plus
                              or rm_mode
                                    == RegretMinimizingMode::p2p_predictive_regret_matching_plus
                              or rm_mode
                                    == RegretMinimizingMode::smooth_predictive_regret_matching_plus
                              or rm_mode
                                    == RegretMinimizingMode::stable_predictive_regret_matching_plus;

/// whether a (traversal scheme, regret-minimizing mode) pair forms an admissible
/// MCCFR configuration. The predictive kernels require outcome sampling: their
/// composition theory (and our integration contract) is built on the single-
/// trajectory importance-weighted increment stream of OS-MCCFR, while the other
/// traversal schemes fold regrets through different code paths whose interplay
/// with the prediction buffer is not established.
template < MCCFRAlgorithmMode algorithm, RegretMinimizingMode rm_mode >
inline constexpr bool mccfr_rm_mode_compatible =
   mccfr_admissible_rm_mode< rm_mode >
   and (rm_mode == RegretMinimizingMode::regret_matching or algorithm == MCCFRAlgorithmMode::outcome_sampling);

/// maps the admissible predictive modes onto their PRM+-family kernel
/// implementation ('void' for the plain-RM default whose storage stays inline in
/// 'MCCFRMinimizer' so that historical layouts are preserved bit-for-bit)
template < concepts::action Action, RegretMinimizingMode rm_mode >
struct mccfr_predictive_kernel_of {
   using type = void;
};

template < concepts::action Action >
struct mccfr_predictive_kernel_of< Action, RegretMinimizingMode::predictive_regret_matching_plus > {
   using type = PredictiveRegretMatchingPlus< Action >;
};

template < concepts::action Action >
struct mccfr_predictive_kernel_of<
   Action,
   RegretMinimizingMode::sap_predictive_regret_matching_plus > {
   using type = SAPPredictiveRegretMatchingPlus< Action >;
};

template < concepts::action Action >
struct mccfr_predictive_kernel_of<
   Action,
   RegretMinimizingMode::ap_predictive_regret_matching_plus > {
   using type = APPredictiveRegretMatchingPlus< Action >;
};

template < concepts::action Action >
struct mccfr_predictive_kernel_of<
   Action,
   RegretMinimizingMode::p2p_predictive_regret_matching_plus > {
   using type = P2PPredictiveRegretMatchingPlus< Action >;
};

template < concepts::action Action >
struct mccfr_predictive_kernel_of<
   Action,
   RegretMinimizingMode::smooth_predictive_regret_matching_plus > {
   using type = SmoothPredictiveRegretMatchingPlus< Action >;
};

template < concepts::action Action >
struct mccfr_predictive_kernel_of<
   Action,
   RegretMinimizingMode::stable_predictive_regret_matching_plus > {
   using type = StablePredictiveRegretMatchingPlus< Action >;
};

template < concepts::action Action, RegretMinimizingMode rm_mode >
using mccfr_predictive_kernel_t = typename mccfr_predictive_kernel_of< Action, rm_mode >::type;

/// legacy plain-RM per-infostate storage of 'MCCFRMinimizer' (registry +
/// cumulative table). Kept as the base of the node data exactly as before so
/// that existing configs see identical member layout and arithmetic.
template < concepts::action Action >
struct mccfr_plain_kernel_data {
   action_registry< Action > registry;
   /// cumulative counterfactual regret z(I,a); entry i belongs to registry.actions[i]
   per_action_table< Action > regret;

   void register_action(const Action& action)
   {
      registry.register_action(action);
      regret.emplace_back(0.);
   }

   [[nodiscard]] size_t index_of(const Action& action) const { return registry.index_of(action); }
};

/// resolves the per-infostate table payload of the selected kernel. Specialized
/// on 'void' (the plain-RM marker) so the fallback never instantiates
/// 'void::node_data_type' -- plain configs must keep their historical layout.
template < concepts::action Action, typename KernelOrVoid >
struct mccfr_kernel_data_of {
   using type = typename KernelOrVoid::node_data_type;
};

template < concepts::action Action >
struct mccfr_kernel_data_of< Action, void > {
   using type = mccfr_plain_kernel_data< Action >;
};

template < MCCFRAlgorithmMode algorithm, MCCFRWeightingMode weighting, concepts::action Action >
using mccfr_extras_t = std::conditional_t<
   algorithm == MCCFRAlgorithmMode::pure_cfr,
   pure_sampling_extras< Action >,
   std::conditional_t<
      weighting == MCCFRWeightingMode::lazy,
      lazy_weighting_extras< Action >,
      std::conditional_t<
         weighting == MCCFRWeightingMode::optimistic,
         optimistic_weighting_extras,
         no_mccfr_extras > > >;

}  // namespace detail

/**
 * @brief the MCCFR regret minimizer: plain external regret matching by default,
 * or a predictive RM+ kernel of the PRM+/PCFR+ family (Farina, Kroer, Sandholm,
 * AAAI 2021 and successors) when the config's 'regret_minimizing_mode' selects
 * one -- in both cases on top of whatever per-config extras the
 * sampling/weighting scheme requires.
 *
 * COMPOSITION THEORY (predictive kernels under outcome sampling)
 * --------------------------------------------------------------
 * The kernel consumes the OS-MCCFR increment stream UNCHANGED: 'observe' folds
 * exactly the importance-weighted sampled counterfactual increments r̂^t(I,a)
 * that vanilla OS-MCCFR accumulates (Lanctot et al., NIPS 2009). These are
 * conditionally unbiased, E[r̂^t | F_t] = r^t, where r^t is the true
 * counterfactual instantaneous regret under the current strategy profile and
 * F_t is the filtration generated by the sampling randomness up to t. The
 * recommendation at time t is derived from theta(a) = max(0, clip(z)(a) +
 * s*rho(a)) with rho = l̂^{t-1} = the last REALIZED sampled instantaneous
 * regret vector (persistence prediction on samples); rho is measurable w.r.t.
 * F_{t-1}, as is the recommended strategy sigma^t.
 *
 * What survives, precisely:
 *  1. Unbiasedness of the increment estimator is untouched -- prediction only
 *     shifts the recommendation source, never the increments.
 *  2. PRM+'s external-regret guarantee (Farina, Kroer, Sandholm, AAAI 2021,
 *     thm. 4) is PATHWISE over arbitrary loss sequences with arbitrary
 *     predictions bounded like the losses; it therefore applies to each single
 *     realization of the sampled-loss sequence {l̂^t} with predictions
 *     m^t = l̂^{t-1}. This is exactly the structural requirement of the
 *     stochastic-CFR framework of Farina, Kroer & Sandholm (ICML 2020), where
 *     any local minimizer with an expected-regret bound over the ESTIMATED
 *     loss sequence composes with any conditionally-unbiased gradient
 *     estimator to yield average-strategy convergence.
 *  3. Taking expectations over the sampling randomness then transfers the
 *     bound from estimated to true counterfactual regrets (sigma^t is
 *     F_{t-1}-measurable, so E[<sigma^t, l̂^t>] = E[<sigma^t, r^t>]).
 *
 * What degrades / is NOT claimed (honesty note):
 *  - The prediction-error term of the pathwise bound now contains SAMPLING
 *    NOISE: ||l̂^{t+1} - l̂^t|| >= ||r^{t+1} - r^t|| with inflation from the
 *    heavy-tailed 1/q(a*) factor of the sampled action's component. The
 *    benefit of prediction therefore shrinks as sampling variance grows;
 *    epsilon-on-policy exploration (the default) bounds q away from zero and
 *    plays the same heavy-tail-guarding role it plays for vanilla OS-MCCFR.
 *    Damped shifts (SAPCFR+/P2PCFR+) are the robustification knobs.
 *  - Full-information PCFR+'s headline O(1/T)-style behavior additionally
 *    relies on quadratic average-policy accumulation (gamma = 2). Under MCCFR
 *    the average strategy must keep its SAMPLING-correct weighting (lazy /
 *    stochastic / ... modes, orthogonal below), which does not reproduce the
 *    quadratic scheme; we claim convergence at OS-MCCFR-style rates modulo
 *    the inflated prediction term, not PCFR+'s constants.
 *  - Alternating updates visit an infostate several times per update cycle:
 *    the FIRST recommendation after a regret fold carries the prediction
 *    shift, subsequent re-recommendations see an already-consumed (empty)
 *    prediction buffer and degenerate to plain RM+. This is benign -- the
 *    strategy actually played immediately after the update (whose sampled
 *    values feed the next fold) includes the prediction -- but it means the
 *    prediction is not re-applied at every intermediate visit.
 *
 * Published precedent for this composition: VR-DeepPDCFR+ / VR-DeepDCFR+
 * (Xu et al., AAAI 2026, arXiv:2511.08174) drive DCFR+/PDCFR+-style
 * discount/clip/predict updates with outcome-sampled advantage estimates
 * (their thms. 1-2 establish the estimator expectations), and ESCHER (McAleer
 * et al., ICLR 2023) clips cumulative regrets under sampling. We found no
 * published pathwise analysis of PRM+ specifically inside tabular OS-MCCFR;
 * the guarantee statement above is our own composition of the two cited
 * frameworks.
 *
 * The 'variance_reduction' switch additionally attaches the VR-MCCFR
 * state-action baseline table to every node (Schmid et al., AAAI 2019) when
 * set to VarianceReductionMode::action_baseline. It is kept orthogonal to the
 * weighting extras so that both can be active at once; under
 * VarianceReductionMode::history_value (ESCHER-style history values,
 * McAleer et al., ICLR 2023) no per-infostate table is needed -- the
 * history-value store lives in the MCCFR engine keyed by world-state-edge
 * hashes -- so the extras collapse to the empty struct as well. 'none' is the
 * plain outcome-sampling layout, identical to the historical flag-off path.
 * Predictive kernels and VR baselines are currently mutually exclusive (see
 * the static checks below): the VR machinery re-evaluates recommendations
 * mid-update for its predictive-baseline rule, which would consume the
 * prediction buffer off its canonical schedule.
 */
template <
   concepts::action Action,
   MCCFRAlgorithmMode algorithm,
   MCCFRWeightingMode weighting,
   RegretMinimizingMode rm_mode = RegretMinimizingMode::regret_matching,
   VarianceReductionMode variance_reduction = VarianceReductionMode::none >
struct MCCFRMinimizer {
   static_assert(
      detail::mccfr_admissible_rm_mode< rm_mode >,
      "This regret-minimizing mode is not available inside MCCFR. Admissible: "
      "regret_matching plus the PRM+/PCFR+-family predictive kernels "
      "(predictive/sap/ap/p2p/smooth/stable_predictive_regret_matching_plus) under "
      "outcome sampling. Plain RM+ and internal-regret matching remain CFR-only."
   );
   static_assert(
      detail::mccfr_rm_mode_compatible< algorithm, rm_mode >,
      "Predictive (PRM+/PCFR+-style) regret kernels inside MCCFR currently support "
      "OUTCOME SAMPLING only: their composition theory rests on the "
      "importance-weighted single-trajectory increment stream of OS-MCCFR. Use "
      "MCCFRAlgorithmMode::outcome_sampling."
   );
   static_assert(
      variance_reduction == VarianceReductionMode::none
         or rm_mode == RegretMinimizingMode::regret_matching,
      "Predictive kernels and VR-MCCFR/ESCHER baselines are not combinable yet: the "
      "baseline maintenance rules re-evaluate recommendations mid-update (predictive "
      "baseline, Davis et al., ICML 2020), consuming the prediction buffer off its "
      "canonical schedule. Run them separately."
   );

   /// the selected predictive kernel ('void' for the plain-RM default)
   using kernel_type = detail::mccfr_predictive_kernel_t< Action, rm_mode >;
   static constexpr bool predictive_active = not std::is_void_v< kernel_type >;

   /// per-infostate tables of the active kernel (predictive node data carries the
   /// instantaneous-regret buffer + strategy snapshot + shift context)
   using kernel_data_type = typename detail::mccfr_kernel_data_of< Action, kernel_type >::type;

   using extras_type = detail::mccfr_extras_t< algorithm, weighting, Action >;
   using vr_extras_type = std::conditional_t<
      variance_reduction == VarianceReductionMode::action_baseline,
      detail::vr_baseline_extras< Action >,
      detail::no_mccfr_extras >;

   struct node_data_type: public kernel_data_type {
      [[no_unique_address]] extras_type extras;
      [[no_unique_address]] vr_extras_type vr_extras;

      void register_action(const Action& action)
      {
         kernel_data_type::register_action(action);
         if constexpr(not std::is_empty_v< extras_type >) {
            extras.register_action(action);
         }
         if constexpr(not std::is_empty_v< vr_extras_type >) {
            vr_extras.register_action(action);
         }
      }
   };

   static void observe(node_data_type& data, const Action& action, double increment)
   {
      if constexpr(predictive_active) {
         // PRM+ family: clip-at-fold into z AND buffer into rho (the persistence
         // prediction source of the next recommendation)
         kernel_type::observe(data, action, increment);
      } else {
         data.regret[data.registry.index_of(action)] += increment;
      }
   }

   template < typename PolicyOut >
   static void recommend(node_data_type& data, PolicyOut& policy_out, size_t iteration)
   {
      if constexpr(predictive_active) {
         // theta(a) = max(0, clip(z)(a) + s*rho(a)); consumes rho afterwards
         kernel_type::recommend(data, policy_out, iteration);
      } else {
         detail::_recommend_from_regret(data.registry.actions, data.regret, policy_out);
      }
   }

   static constexpr double policy_weight(size_t /*iteration*/) { return 1.; }
};

/// maps an MCCFR configuration onto the concrete regret minimizer type acting
/// on actions of type 'Action'
template < MCCFRConfig config, concepts::action Action >
using mccfr_minimizer_for_t = MCCFRMinimizer<
   Action,
   config.algorithm,
   config.weighting,
   config.regret_minimizing_mode,
   effective_variance_reduction(config) >;

}  // namespace nor::rm

#endif  // NOR_RM_MINIMIZERS_HPP
