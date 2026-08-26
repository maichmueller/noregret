
#ifndef NOR_RM_MINIMIZERS_CONSTRAINED_HPP
#define NOR_RM_MINIMIZERS_CONSTRAINED_HPP

// NOTE: this header relies on 'per_action_table', 'detail::action_registry' and
// the minimizer node data protocol which are defined in
// nor/rm/minimizers/minimizers.hpp BEFORE it includes this file. Include the
// former instead of this file directly.

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace nor::rm {

/**
 * @brief behaviorally-constrained ("perturbed") RM+ regret minimizer.
 *
 * Implements the per-infoset regret matching step of CFR+ over LINEARLY
 * CONSTRAINED simplexes (Farina, Kroer & Sandholm, "Regret Minimization in
 * Behaviorally-Constrained Zero-Sum Games", ICML 2017, arXiv:1711.03441,
 * Prop. 6 + Algorithm 2). The strategy space of every infostate I is the
 * lower-bounded simplex (Selten behavioral perturbation; their Def. 9)
 *
 *    Q^I = { sigma in Delta^{|A(I)|} : sigma(I,a) >= p(I,a) for all a },
 *
 * which is a finitely-generated convex polytope with vertices b_i = p + tau*e_i,
 * tau = 1 - sum_a p(I,a) (their Prop. 6). Running vanilla RM+ over the vertex
 * coordinates and re-expressing the update against the UNPERTURBED per-action
 * instantaneous regrets phi_t(I,a) = pi_{-i}(v(I,a) - v(I)) yields their
 * closed-form Algorithm 2, which is what this kernel executes:
 *
 *    r_t^I <- [ r_{t-1}^I + tau_p(I) * phi_t^I + 1 * <p(I,.), phi_t^I> ]^+
 *    x_t^I = p(I,.) + tau_p(I) * [r_{t-1}^I]^+ / sum_a [r_{t-1}^I]_a^+
 *
 * (the Lambda == 0 case recommends the polytope CENTER p + tau/|A|, the natural
 * analogue of RM's uniform fallback). Every recommendation satisfies the floors
 * EXACTLY by construction -- x >= p componentwise because a non-negative mass is
 * ADDED to each floor entry -- so the solved profile approximates an extensive-
 * form perfect equilibrium refinement while Theorem 7 preserves CFR+'s
 * O(1/sqrt(T)) regret rate. The reach-weighted average accumulated by the solver
 * from these recommendations matches Algorithm 2's averaging prescription.
 *
 * INTEGRATION CONTRACT with rm::VanillaCFR: exactly one full iteration's
 * counterfactual increments arrive through 'observe()' between two 'recommend()'
 * calls, mirroring InternalRegretMatching/PredictiveRegretMatchingPlus. Because
 * the coupling term 1*<p,phi> ties all components together, the perturbed
 * specialization BUFFERS the iteration's increments ('instant_regret') and folds
 * them once at recommendation time; the buffer is consumed there.
 *
 * UNPERTURBED LIMIT: with UniformFloor == 0 (the default), tau = 1 and the
 * coupling term vanishes, so the kernel compiles the plain RegretMatchingPlus
 * arithmetic instead -- increments fold directly into the cumulative table at
 * 'observe()' time with the identical summation grouping, reproducing CFR+
 * trajectories BIT-FOR-BIT while sharing the constrained code path's interface.
 * This specialization cannot honor non-uniform floors; solvers therefore require
 * CFRConfig::perturbation_floor > 0 when the environment overrides floors through
 * the B8 trait 'action_probability_floors'.
 *
 * FLOOR SOURCE: 'probability_floors' is seeded per registered action with the
 * compile-time UniformFloor; the solver refreshes it from the environment's B8
 * trait ahead of every recommend when such a trait exists. Floors are validated
 * at every constrained fold: entries outside [0, 1] or a total sum >= 1 (no free
 * mass left) throw std::invalid_argument.
 */
template <
   concepts::action Action,
   double UniformFloor = 0.,
   bool BufferIncrements = (UniformFloor > 0.) >
struct ConstrainedRMPlus {
   static_assert(UniformFloor >= 0., "the uniform perturbation floor must be non-negative");

   /// true iff the kernel degenerates to bit-for-bit plain RM+
   static constexpr bool buffered_updates = BufferIncrements;
   /// the uniform floor seeded into every newly registered action
   static constexpr double uniform_floor = UniformFloor;

   struct node_data_type {
      detail::action_registry< Action > registry;
      /// cumulative transformed regret r^I; entry i belongs to registry.actions[i].
      /// Named 'regret' as required by the InfostateNodeData protocol
      per_action_table< Action > regret;
      /// instantaneous counterfactual regret buffer of the running iteration
      /// (only grown and consumed by the PERTURBED specialization)
      per_action_table< Action > instant_regret;
      /// per-action probability floors sigma(I,a) >= floor; seeded with
      /// 'uniform_floor' at registration and optionally refreshed per sweep by
      /// the solver from the environment's B8 trait
      per_action_table< Action > probability_floors;

      void register_action(const Action& action)
      {
         registry.register_action(action);
         regret.emplace_back(0.);
         probability_floors.emplace_back(uniform_floor);
         if constexpr(buffered_updates) {
            instant_regret.emplace_back(0.);
         }
      }

      [[nodiscard]] size_t index_of(const Action& action) const
      {
         return registry.index_of(action);
      }
   };

   /// protocol entry point: append 'action' to the registry and zero-initialize
   /// all per-action table slots for it
   static void register_action(node_data_type& data, const Action& action)
   {
      data.register_action(action);
   }

   static void observe(node_data_type& data, const Action& action, double increment)
   {
      const auto idx = data.registry.index_of(action);
      if constexpr(buffered_updates) {
         // buffered; folded into the cumulative table by the next recommend()
         data.instant_regret[idx] += increment;
      } else {
         // unperturbed limit: identical grouping to RegretMatchingPlus
         data.regret[idx] += increment;
      }
   }

   template < typename PolicyOut >
   static void recommend(node_data_type& data, PolicyOut& policy_out, size_t /*iteration*/)
   {
      const auto n_actions = data.regret.size();
      if constexpr(not buffered_updates) {
         // ---- unperturbed limit: verbatim RegretMatchingPlus arithmetic -------
         double pos_regret_sum{0.};
         for(double& cumul_regret : data.regret) {
            cumul_regret = std::max(0., cumul_regret);
            pos_regret_sum += cumul_regret;
         }
         if(pos_regret_sum > 0.) {
            for(auto&& [action, cumul_regret] :
                std::views::zip(data.registry.actions, data.regret)) {
               policy_out[action] = cumul_regret / pos_regret_sum;
            }
         } else {
            const double uniform_prob = 1. / static_cast< double >(n_actions);
            for(const auto& action : data.registry.actions) {
               policy_out[action] = uniform_prob;
            }
         }
         return;
      }

      // ---- constrained path: validate the perturbation -----------------------
      double floor_sum{0.};
      for(double floor_prob : data.probability_floors) {
         if(not (0. <= floor_prob and floor_prob <= 1.)) {
            throw std::invalid_argument("ConstrainedRMPlus: probability floors must lie in [0, 1]");
         }
         floor_sum += floor_prob;
      }
      if(not (floor_sum < 1.)) {
         throw std::invalid_argument(
            "ConstrainedRMPlus: infeasible perturbation, the floors already sum "
            "to >= 1 and leave no free probability mass"
         );
      }
      const double tau = 1. - floor_sum;

      // Algorithm 2 fold: r <- [r + tau * phi + 1 * <p, phi>]^+ ; the clamp IS the
      // RM+ forgetting step, applied once per iteration like the paper's line
      double floor_dot_phi{0.};
      for(auto&& [floor_prob, instant_regret] :
          std::views::zip(data.probability_floors, data.instant_regret)) {
         floor_dot_phi += floor_prob * instant_regret;
      }
      double lambda{0.};
      for(auto idx : std::views::iota(size_t{0}, n_actions)) {
         double& cumul_regret = data.regret[idx];
         cumul_regret = std::max(0., cumul_regret + tau * data.instant_regret[idx] + floor_dot_phi);
         lambda += cumul_regret;
      }
      // consume the buffer: the next accumulation phase starts from scratch
      std::ranges::fill(data.instant_regret, 0.);

      // recommendation x = p + tau * r / Lambda (polytope center at Lambda == 0):
      // x >= p holds EXACTLY since only non-negative mass is added to the floors
      if(lambda > 0.) {
         for(auto idx : std::views::iota(size_t{0}, n_actions)) {
            policy_out[data.registry.actions[idx]] = data.probability_floors[idx]
                                                     + tau * data.regret[idx] / lambda;
         }
      } else {
         for(auto idx : std::views::iota(size_t{0}, n_actions)) {
            policy_out[data.registry.actions[idx]] = data.probability_floors[idx]
                                                     + tau / static_cast< double >(n_actions);
         }
      }
   }

   /// uniform averaging over the constrained iterates: Algorithm 2 accumulates the
   /// reach-weighted current strategies x̄ <- x̄ + pi_1 x_t and normalizes at the end,
   /// which is exactly what the solver's uniform weighting mode performs
   static constexpr double policy_weight(size_t /*iteration*/) { return 1.; }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////// concept conformance checks ////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

/// minimal writable policy sink used solely to probe concept conformance below
template < concepts::action Action >
struct constrained_policy_out_stub {
   double& operator[](const Action&) { return value; }
   double value = 0.;
};

}  // namespace detail

static_assert(
   regret_minimizer_for<
      ConstrainedRMPlus< int >,
      int,
      detail::constrained_policy_out_stub< int > >,
   "the unperturbed ConstrainedRMPlus does not satisfy the regret minimizer protocol."
);
static_assert(
   regret_minimizer_for<
      ConstrainedRMPlus< int, 0.05 >,
      int,
      detail::constrained_policy_out_stub< int > >,
   "the perturbed ConstrainedRMPlus does not satisfy the regret minimizer protocol."
);

}  // namespace nor::rm

#endif  // NOR_RM_MINIMIZERS_CONSTRAINED_HPP
