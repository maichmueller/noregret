
#ifndef NOR_RM_MINIMIZERS_HPP
#define NOR_RM_MINIMIZERS_HPP

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <optional>
#include <ranges>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "common/common.hpp"
#include "nor/concepts.hpp"
#include "nor/rm/cfr_tabular/cfr_config.hpp"
#include "nor/rm/rm_utils.hpp"

namespace nor::rm {

/// the canonical per-action double table used by all tabular regret minimizers.
/// Keys are reference wrappers into the infostate node's owned action storage,
/// hence the custom hash/compare.
template < concepts::action Action >
using per_action_map = std::unordered_map<
   std::reference_wrapper< const Action >,
   double,
   common::ref_wrapper_hasher< const Action >,
   common::ref_wrapper_comparator< const Action > >;

///////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// minimizer node data protocol /////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Every regret minimizer type M exposes:
//    using node_data_type            -- the per-infostate tables it needs. It always contains a
//                                       'regret' member of type per_action_map<Action>.
//    static void register_action(node_data_type&, const Action&)
//                                    -- zero-initialize all per-action tables for 'action'
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

/// writes `positive-part-of-regret / sum(positive parts)` (or uniform when the
/// sum vanishes) for every registered action into the policy output
template < typename RegretTable, typename PolicyOut >
void _recommend_from_regret(const RegretTable& regret, PolicyOut& policy_out)
{
   double pos_regret_sum{0.};
   for(const auto& [action_ref, cumul_regret] : regret) {
      pos_regret_sum += std::max(0., cumul_regret);
   }
   if(pos_regret_sum > 0.) {
      for(const auto& [action_ref, cumul_regret] : regret) {
         policy_out[action_ref.get()] = std::max(0., cumul_regret) / pos_regret_sum;
      }
   } else {
      const double uniform_prob = 1. / static_cast< double >(regret.size());
      for(const auto& [action_ref, _] : regret) {
         policy_out[action_ref.get()] = uniform_prob;
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
      per_action_map< Action > regret;

      void register_action(const Action& action) { regret.emplace(std::cref(action), 0.); }
   };

   static void observe(node_data_type& data, const Action& action, double increment)
   {
      // all legal actions are pre-registered via register_action, so this never
      // inserts a key referencing storage owned outside this node
      data.regret[std::cref(action)] += increment;
   }

   template < typename PolicyOut >
   static void recommend(const node_data_type& data, PolicyOut& policy_out, size_t /*iteration*/)
   {
      detail::_recommend_from_regret(data.regret, policy_out);
   }

   static constexpr double policy_weight(size_t /*iteration*/) { return 1.; }
};

/// optimistic external-regret matching (CFR+): the cumulative regret table is
/// clamped to non-negative values during each recommendation step
template < concepts::action Action >
struct RegretMatchingPlus {
   struct node_data_type {
      per_action_map< Action > regret;

      void register_action(const Action& action) { regret.emplace(std::cref(action), 0.); }
   };

   static void observe(node_data_type& data, const Action& action, double increment)
   {
      data.regret[std::cref(action)] += increment;
   }

   template < typename PolicyOut >
   static void recommend(node_data_type& data, PolicyOut& policy_out, size_t /*iteration*/)
   {
      // CFR+ clamps the cumulative regret in place before deriving the policy
      double pos_regret_sum{0.};
      for(auto& [action_ref, cumul_regret] : data.regret) {
         cumul_regret = std::max(0., cumul_regret);
         pos_regret_sum += cumul_regret;
      }
      if(pos_regret_sum > 0.) {
         for(const auto& [action_ref, cumul_regret] : data.regret) {
            policy_out[action_ref.get()] = cumul_regret / pos_regret_sum;
         }
      } else {
         const double uniform_prob = 1. / static_cast< double >(data.regret.size());
         for(const auto& [action_ref, _] : data.regret) {
            policy_out[action_ref.get()] = uniform_prob;
         }
      }
   }

   static constexpr double policy_weight(size_t /*iteration*/) { return 1.; }
};

/// CFR+ with regret-based pruning: negative cumulative regrets may be REPLACED
/// (rather than merely incremented) by positive instantaneous regrets. The
/// instantaneous buffer is consumed (reset to 0) on each recommendation.
template < concepts::action Action >
struct RegretMatchingPlusRBP {
   struct node_data_type {
      per_action_map< Action > regret;
      /// instantaneous regret increments r(I, a) of the current iteration
      per_action_map< Action > cumulative_instant_regret;

      void register_action(const Action& action)
      {
         regret.emplace(std::cref(action), 0.);
         cumulative_instant_regret.emplace(std::cref(action), 0.);
      }
   };

   static void observe(node_data_type& data, const Action& action, double increment)
   {
      data.regret[std::cref(action)] += increment;
   }

   template < typename PolicyOut >
   static void recommend(node_data_type& data, PolicyOut& policy_out, size_t /*iteration*/)
   {
      // apply the replace-if-positive rule on the buffered instantaneous regret
      // and consume the buffer while deriving the new recommendation
      double pos_regret_sum{0.};
      for(auto& [action_ref, cumul_regret] : data.regret) {
         auto& instant_regret = data.cumulative_instant_regret[action_ref];
         cumul_regret = instant_regret > 0. and cumul_regret < 0. ? instant_regret
                                                                  : cumul_regret + instant_regret;
         instant_regret = 0.;
         pos_regret_sum += std::max(0., cumul_regret);
      }
      if(pos_regret_sum > 0.) {
         for(const auto& [action_ref, cumul_regret] : data.regret) {
            policy_out[action_ref.get()] = std::max(0., cumul_regret) / pos_regret_sum;
         }
      } else {
         const double uniform_prob = 1. / static_cast< double >(data.regret.size());
         for(const auto& [action_ref, _] : data.regret) {
            policy_out[action_ref.get()] = uniform_prob;
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
      per_action_map< Action > regret;
      per_action_map< Action > instant_regret;
      double reach_prob_snapshot = 0.;
      per_action_map< Action > avg_policy_denominator;

      void register_action(const Action& action)
      {
         regret.emplace(std::cref(action), 0.);
         instant_regret.emplace(std::cref(action), 0.);
         avg_policy_denominator.emplace(std::cref(action), 0.);
      }
   };

   static void observe(node_data_type& data, const Action& action, double increment)
   {
      // canonicalize the key through the cumulative regret table so that the
      // buffered entry references stable storage owned by this node
      auto [iter, _] = data.regret.try_emplace(action, 0.);
      data.instant_regret[std::cref(iter->first)] += increment;
   }

   static void snapshot_reach_probability(node_data_type& data, double reach_prob)
   {
      data.reach_prob_snapshot = reach_prob;
   }

   template < typename PolicyOut >
   static void recommend(const node_data_type& data, PolicyOut& policy_out, size_t /*iteration*/)
   {
      detail::_recommend_from_regret(data.regret, policy_out);
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
      auto& instant_table = data.instant_regret;
      // L1(I, a) = exp(r(I, a) - mean_r(I)); computed before mutating anything
      auto l1_weights = std::invoke([&] {
         per_action_map< Action > l1;
         double average_instant_regret = std::ranges::fold_left(
                                            instant_table | std::views::values,
                                            double(0.),
                                            std::plus{}
                                         )
                                         / double(instant_table.size());
         std::ranges::for_each(instant_table, [&](const auto& actionref_to_instant_regret) {
            const auto& [action_ref, instant_regret] = actionref_to_instant_regret;
            l1[action_ref] = std::exp(instant_regret - average_instant_regret);
         });
         return l1;
      });

      for(auto& [action_ref, cumul_regret] : data.regret) {
         auto& instant_regret = instant_table[action_ref];
         if(instant_regret >= 0.) {
            cumul_regret += l1_weights[action_ref] * instant_regret;
         } else {
            cumul_regret += l1_weights[action_ref] * beta(instant_regret, iteration);
         }
         // reset the buffer so the next iteration starts fresh
         instant_regret = 0.;
      }

      for(auto& [action_ref, denominator] : data.avg_policy_denominator) {
         const auto& action = action_ref.get();
         const double l1_weight = l1_weights[action_ref];
         // cumulative numerator update
         avg_policy[action] += l1_weight * data.reach_prob_snapshot * curr_policy[action];
         // cumulative denominator update
         denominator += l1_weight * data.reach_prob_snapshot;
      }

      recommend(data, curr_policy, iteration);
   }
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
      if constexpr(discount_regrets) {
         double t_alpha = std::pow(double(iteration), m_params.alpha);
         double t_beta = std::pow(double(iteration), m_params.beta);
         for(auto& [action_ref, cumul_regret] : data.regret) {
            (void) action_ref;
            cumul_regret *= cumul_regret > 0. ? t_alpha / (t_alpha + 1.) : t_beta / (t_beta + 1.);
         }
      }
      Inner::recommend(data, policy_out, iteration);
   }

   [[nodiscard]] double policy_weight(size_t iteration) const
   {
      // add +1 to obtain the logical iteration number t (iterations count from
      // 0 numerically but from 1 theoretically); the resulting fraction is then
      // exponentiated by gamma
      double t = double(iteration) + 1.;
      return std::pow(t / (t + 1.), m_params.gamma);
   }

  private:
   CFRDiscountedParameters m_params;
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

namespace nor::rm {

/////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// minimizer selection ////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <
   concepts::action Action,
   CFRWeightingMode weighting,
   CFRPruningMode pruning,
   RegretMinimizingMode rm_mode >
consteval auto select_vanilla_minimizer()
{
   if constexpr(rm_mode == RegretMinimizingMode::predictive_regret_matching_plus) {
      // PCFR+ needs the quadratic average-policy accumulation of the discounted
      // weighting machinery (gamma = 2) but NOT the alpha/beta regret discounts
      return std::type_identity< DiscountedCFR<
         PredictiveRegretMatchingPlus< Action >,
         /*discount_regrets=*/false > >{};
   } else if constexpr(rm_mode == RegretMinimizingMode::sap_predictive_regret_matching_plus) {
      return std::type_identity< DiscountedCFR<
         PredictiveRegretMatchingPlus< Action, true >,
         /*discount_regrets=*/false > >{};
   } else if constexpr(weighting == CFRWeightingMode::exponential) {
      return std::type_identity< ExponentialCFR< Action > >{};
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
/// actions of type 'Action'
template < CFRConfig config, concepts::action Action >
using minimizer_for_t = typename decltype(detail::select_vanilla_minimizer<
                                          Action,
                                          config.weighting_mode,
                                          config.pruning_mode,
                                          config.regret_minimizing_mode >())::type;

///////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// MCCFR minimizers //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

struct no_mccfr_extras {};

/// lazy average-policy weighting keeps a per-action table of accumulated
/// average-policy mass of actions that were not sampled since the last visit
template < concepts::action Action >
struct lazy_weighting_extras {
   per_action_map< Action > pending_avg_accumulator;

   void register_action(const Action& action)
   {
      pending_avg_accumulator.emplace(std::cref(action), 0.);
   }
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
   per_action_map< Action > baseline;

   void register_action(const Action& action) { baseline.emplace(std::cref(action), 0.); }
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
 * @brief the MCCFR regret minimizer: plain external regret matching on top of
 * whatever per-config extras the sampling/weighting scheme requires.
 *
 * The 'variance_reduced' switch additionally attaches the VR-MCCFR
 * state-action baseline table to every node (Schmid et al., AAAI 2019). It is
 * kept orthogonal to the weighting extras so that both can be active at once;
 * when false, 'vr_extras' collapses to the empty struct and costs nothing.
 */
template <
   concepts::action Action,
   MCCFRAlgorithmMode algorithm,
   MCCFRWeightingMode weighting,
   RegretMinimizingMode rm_mode = RegretMinimizingMode::regret_matching,
   bool variance_reduced = false >
struct MCCFRMinimizer {
   static_assert(
      rm_mode == RegretMinimizingMode::regret_matching,
      "MCCFR+ is not yet implemented."
   );

   using extras_type = detail::mccfr_extras_t< algorithm, weighting, Action >;
   using vr_extras_type = std::conditional_t<
      variance_reduced,
      detail::vr_baseline_extras< Action >,
      detail::no_mccfr_extras >;

   struct node_data_type {
      per_action_map< Action > regret;
      [[no_unique_address]] extras_type extras;
      [[no_unique_address]] vr_extras_type vr_extras;

      void register_action(const Action& action)
      {
         regret.emplace(std::cref(action), 0.);
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
      data.regret[std::cref(action)] += increment;
   }

   template < typename PolicyOut >
   static void recommend(const node_data_type& data, PolicyOut& policy_out, size_t /*iteration*/)
   {
      detail::_recommend_from_regret(data.regret, policy_out);
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
   config.variance_reduced_baselines >;

}  // namespace nor::rm

#endif  // NOR_RM_MINIMIZERS_HPP
