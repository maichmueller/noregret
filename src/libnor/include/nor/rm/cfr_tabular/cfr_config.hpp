
#ifndef NOR_CFR_CONFIG_HPP
#define NOR_CFR_CONFIG_HPP

#include <functional>

namespace nor::rm {

enum class RegretMinimizingMode {
   regret_matching = 0,
   regret_matching_plus = 1,
   // PCFR+ (Farina, Kroer, Sandholm, AAAI 2021): CFR+ whose recommendation step
   // is conditioned on a persistence prediction of the next instantaneous
   // counterfactual regret. Requires alternating updates, no pruning and the
   // discounted weighting mode (used solely for its quadratic average-policy
   // accumulation with gamma = 2)
   predictive_regret_matching_plus = 2,
   // SAPCFR+ (arXiv:2503.12770): PCFR+ with the prediction shift term damped by
   // 1/(1 + alpha), alpha = 2. Same configuration constraints as PCFR+
   sap_predictive_regret_matching_plus = 3,
   // DCFR+ (Xu et al., "Minimizing Weighted Counterfactual Regret with Optimistic
   // Online Mirror Descent", IJCAI 2024, arXiv:2404.13891, sec. 4):
   //    R^t = [ R^{t-1} * (t-1)^alpha / ((t-1)^alpha + 1) + r^t ]^+
   // i.e. RM+-style folding WITH the positive-part alpha discount applied BEFORE
   // the instantaneous regret is added and the sum is clipped. Because the stored
   // table is clamped at every fold, the beta branch of plain DCFR is vacuous.
   // Requires alternating updates, no pruning, the discounted weighting mode
   // (used solely for its gamma-side average-policy accumulation)
   discounted_regret_matching_plus = 4,
   // PDCFR+ (arXiv:2404.13891, sec. 4): DCFR+ whose recommendation is computed
   // from the predicted next cumulative regret
   //    R~^{t+1} = [ R^t * t^alpha / (t^alpha + 1) + v^{t+1} ]^+
   // with the persistence prediction v^{t+1} = r^t. Paper defaults alpha = 2.3,
   // gamma = 5 (sec. 5.2). Same configuration constraints as DCFR+
   discounted_predictive_regret_matching_plus = 5
};

enum class UpdateMode { simultaneous = 0, alternating = 1 };

enum class CFRWeightingMode {
   // no particular weighting scheme applied to updates of regret or average policy.
   // Both are incremented by unweighted increments
   uniform = 0,
   // The average policy is being incremented by the weight 't' in iteration 't'
   linear = 1,
   // Both the regret and average policy are updated by the weights
   // t^alpha / (t^alpha +1), t^beta / (t^beta + 1), (t / t+1)^gamma
   discounted = 2,
   // The regret and average policy are weighted by an L1 factor: L1(I, a) = r(I,a) - E[v(I)]
   // where r(I,a) is the instantaneous regret and E[v(I)] is the expected value of the infostate
   exponential = 3
};

enum class CFRPruningMode {
   // No pruning
   none = 0,
   // Partial pruning drops the subtree if a player policy upstream hits 0
   partial = 1,
   // TODO: Regret-based pruning skips subtrees for all t > t_0 if an action's regret is < 0 at time
   // t_0 and updates upon take up t_1 with a best-response against the average strategy of the
   // opponents during this period.
   regret_based = 2,
   // TODO:
   dynamic_thresholding = 3
};

struct CFRConfig {
   UpdateMode update_mode = UpdateMode::alternating;
   RegretMinimizingMode regret_minimizing_mode = RegretMinimizingMode::regret_matching;
   CFRWeightingMode weighting_mode = CFRWeightingMode::uniform;
   CFRPruningMode pruning_mode = CFRPruningMode::none;
};

struct CFRPlusConfig {
   UpdateMode update_mode = UpdateMode::alternating;
};

struct CFRDiscountedConfig {
   UpdateMode update_mode = UpdateMode::alternating;
   RegretMinimizingMode regret_minimizing_mode = RegretMinimizingMode::regret_matching;
};

struct CFRLinearConfig {
   // should always be exact same as Discounted Config!
   UpdateMode update_mode = UpdateMode::alternating;
   RegretMinimizingMode regret_minimizing_mode = RegretMinimizingMode::regret_matching;
};

struct CFRExponentialConfig {
   UpdateMode update_mode = UpdateMode::alternating;
   RegretMinimizingMode regret_minimizing_mode = RegretMinimizingMode::regret_matching;
};

enum class MCCFRAlgorithmMode {
   // sample only the chance players action according to the chance distribution
   chance_sampling = 0,
   // sample only a single trajectory of the game tree and update the policies
   outcome_sampling = 1,
   // traverse each action of a traversing player, but sample only a single action of each opponent
   // and chance player
   external_sampling = 2,
   // traverse each action of a traversing player, but sample only a single action of each opponent
   // and chance player. Pure CFR is technically not an MCCFR family member, but it follows external
   // sampling's logic closely
   pure_cfr = 3
};

enum class MCCFRWeightingMode {
   none = 0,
   // the lazy update scheme is the correct average policy update scheme which maintains a table of
   // unsampled action policy values that are pushed alongside once such an action is sampled
   lazy = 1,
   // optimistic updates the average action policy by weighting the current increment with the delay
   // in number of iterations (t-c) this action has not been sampled and updated last
   optimistic = 2,
   // stochastic updates the average policy by weighting the current increment with the reciprocal
   // sampled action sample-probability
   stochastic = 3
};

enum class MCCFRExplorationMode {
   // explore the action space by drawing a legal action uniformly with probability epsilon and
   // otherwise sampling according to the current action policy
   epsilon_on_policy = 0,
   // sample via a custom sampling policy
   // TODO: implement this method
   custom_sampling_policy = 1
};

struct MCCFRConfig {
   UpdateMode update_mode = UpdateMode::alternating;
   MCCFRAlgorithmMode algorithm = MCCFRAlgorithmMode::outcome_sampling;
   MCCFRExplorationMode exploration = MCCFRExplorationMode::epsilon_on_policy;
   MCCFRWeightingMode weighting = MCCFRWeightingMode::lazy;
   RegretMinimizingMode regret_minimizing_mode = RegretMinimizingMode::regret_matching;
   CFRPruningMode pruning_mode = CFRPruningMode::none;
   /// Variance-reduced outcome sampling through state-action baselines
   /// (VR-MCCFR; Schmid et al., AAAI 2019, eqs (7)-(11)). At every updated
   /// infoset the sampled action's value is baseline-corrected
   /// b̂(I,a*) + (u(ha*|z) − b̂(I,a*))/ξ(h,a*), off-trajectory actions are
   /// valued by their baselines, regrets accumulate v̂ᵇ(I,a) − v̂ᵇ(I) per
   /// action, and the σ-weighted mixture propagates bootstrapped up the
   /// single sampled trajectory. Unbiased for any positive sampling rule,
   /// including ε-on-policy exploration. Requires alternating updates
   /// (enforced statically); under simultaneous updates the low-variance
   /// increments make both players' policies chase each other within one
   /// trajectory and average-strategy convergence stalls.
   /// Orthogonal to all other config axes; when false the generated code is
   /// identical to the plain outcome-sampling scheme.
   bool variance_reduced_baselines = false;
   /// baseline learning rate β of the update b̂(I,a) ← b̂(I,a) + β·(v̂ − b̂(I,a))
   /// (only meaningful together with 'variance_reduced_baselines').
   /// NOTE: the regression target carries the eq-(9) importance factor 1/ξ,
   /// so a full step β = 1 chases heavy-tailed spikes and destabilizes the
   /// off-trajectory baseline values; the paper likewise prescribes an
   /// exponentially-decaying average rather than a cumulative one.
   double baseline_update_rate = 0.1;
};

struct CFRDiscountedParameters {
   /// the parameter to exponentiate the weight of positive cumulative regrets with
   double alpha = 1.5;
   /// the parameter to exponentiate the weight of negative cumulative regrets with
   double beta = 0.;
   /// the parameter to exponentiate the weight of the cumulative policy with
   double gamma = 2.;

   /// OPTIONAL per-iteration schedules overriding the constants above.
   /// Each schedule is invoked with the raw (0-based) iteration index and must
   /// return the parameter value for that iteration. When null the constants
   /// are used (static fast path; identical arithmetic to previous versions).
   /// Consumers: HS-schedules / DDCFR agents.
   std::function< double(size_t) > alpha_schedule{};
   std::function< double(size_t) > beta_schedule{};
   std::function< double(size_t) > gamma_schedule{};

   [[nodiscard]] double alpha_at(size_t iteration) const
   {
      return alpha_schedule ? alpha_schedule(iteration) : alpha;
   }
   [[nodiscard]] double beta_at(size_t iteration) const
   {
      return beta_schedule ? beta_schedule(iteration) : beta;
   }
   [[nodiscard]] double gamma_at(size_t iteration) const
   {
      return gamma_schedule ? gamma_schedule(iteration) : gamma;
   }

   /// B3: when true, the gamma-side policy weighting indexes by the CYCLE
   /// number (= iteration / num_players) instead of the raw iteration index.
   /// Default false keeps the historical indexing bit-for-bit.
   bool weight_by_cycle = false;
};

namespace detail {

inline double _zero(double, size_t)
{
   return 0.;
}

}  // namespace detail

struct CFRExponentialParameters {
   /// the parameter function beta (can depend on the instantaneous regret of the action) to limit
   /// negative regrets to
   double (*beta)(double, size_t) = &detail::_zero;
};

}  // namespace nor::rm

#endif  // NOR_CFR_CONFIG_HPP
