
#ifndef NOR_CFR_CONFIG_HPP
#define NOR_CFR_CONFIG_HPP

#include <cmath>
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
   // APCFR+ (Meng et al., arXiv:2503.12770v2): PCFR+ with an ADAPTIVE per-infostate
   // asymmetry of step sizes between the implicit (prediction-carrying) and
   // explicit accumulated regret updates; alpha^t_I is learned from running sums
   // of squared L2 norms (their Eq. (10)), clamped by alpha_max = 5.
   // Same configuration constraints as PCFR+
   ap_predictive_regret_matching_plus = 4,
   // P2PCFR+ ("Pessimistic PCFR+", ICLR'25 submission, OpenReview njyZgDDeY4):
   // fixed prediction damping 1/(1 + alpha) with alpha = 5 (their experimental
   // choice; theory covers alpha in [0,1]). Same configuration constraints as
   // PCFR+; see rm::P2PPredictionShift for provenance caveats
   p2p_predictive_regret_matching_plus = 5,
   // Smooth PRM+ (Farina, Grand-Clément, Kroer, Lee, Luo, NeurIPS 2023,
   // arXiv:2305.14709, Algorithm 2): PCFR+ whose predicted-regret vector is kept
   // at 1-norm >= epsilon (= 1) before normalization, "chopping off" the origin.
   // Same configuration constraints as PCFR+
   smooth_predictive_regret_matching_plus = 6,
   // Stable PRM+ (arXiv:2305.14709, Algorithm 1): PCFR+ with componentwise
   // restart -- whenever every cumulative-regret entry is <= R0 = 1, the table
   // resets to R0 * 1 and the next prediction is suppressed.
   // Same configuration constraints as PCFR+
   stable_predictive_regret_matching_plus = 7,
   // DCFR+ (Xu et al., "Minimizing Weighted Counterfactual Regret with Optimistic
   // Online Mirror Descent", IJCAI 2024, arXiv:2404.13891, sec. 4):
   //    R^t = [ R^{t-1} * (t-1)^alpha / ((t-1)^alpha + 1) + r^t ]^+
   // i.e. RM+-style folding WITH the positive-part alpha discount applied BEFORE
   // the instantaneous regret is added and the sum is clipped. Because the stored
   // table is clamped at every fold, the beta branch of plain DCFR is vacuous.
   // Requires alternating updates, no pruning, the discounted weighting mode
   // (used solely for its gamma-side average-policy accumulation)
   discounted_regret_matching_plus = 8,
   // PDCFR+ (arXiv:2404.13891, sec. 4): DCFR+ whose recommendation is computed
   // from the predicted next cumulative regret
   //    R~^{t+1} = [ R^t * t^alpha / (t^alpha + 1) + v^{t+1} ]^+
   // with the persistence prediction v^{t+1} = r^t. Paper defaults alpha = 2.3,
   // gamma = 5 (sec. 5.2). Same configuration constraints as DCFR+
   discounted_predictive_regret_matching_plus = 9
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
   // Regret-based pruning (Brown & Sandholm, "Regret-Based Pruning in Extensive-Form Games",
   // NIPS 2015). When the cumulative counterfactual regret of an action a at infostate I is
   // sufficiently negative, the subtree D(I,a) is skipped for a sound number of iterations
   // (Theorem 1: m = floor(|R(I,a)| / (U(I,a) - L(I)))) while a best-response value against the
   // opponents' average strategies is buffered; at the deadline the buffer folds into the regret
   // table and normal traversal resumes. Requires alternating updates, RM+-style recommendations
   // and non-exponential weighting (statically enforced).
   regret_based = 2,
   // Dynamic thresholding (Brown, Kroer, Sandholm, "Dynamic Thresholding and Pruning for Regret
   // Minimization", AAAI 2017, DOI 10.1609/aaai.v31i1.10603). Every recommendation zeroes actions
   // below the schedule tau_t and renormalizes (their Theorems 1 and 2), which makes
   // low-probability
   // subtrees prunable even for local regret minimizers that would otherwise put positive mass on
   // every action. Implemented as the Thresholded<Inner> minimizer wrapper plus reuse of the
   // regret-based traversal gate.
   dynamic_thresholding = 3
};

enum class CFRLazyUpdateMode {
   // Eager CFR: every visited infostate's recommendation is refreshed by regret matching at
   // the end of EVERY iteration (classical behavior)
   off = 0,
   // Lazy-CFR / Lazy-CFR+ (Zhou et al., "Lazy-CFR: fast and near-optimal regret minimization
   // for extensive games", ICLR 2020, arXiv:1810.04433): time is segmented PER INFOSET. Inside
   // a segment the infoset's strategy stays FROZEN -- counterfactual regret increments and
   // own-reach-weighted average-strategy mass arriving during the segment are buffered instead
   // of being applied eagerly. A segment CLOSES once its accumulated opponent reach
   // pi_{-i}(I) exhausts the threshold B ('lazy_update_threshold_b'); the closing fold applies
   // both buffers at once (exact, because the frozen strategy makes sum_t pi_i^t sigma equal
   // (sum_t pi_i^t) sigma) and only THEN is the regret-matching recommendation recomputed.
   // Composes with the plain RM and RM+ kernels (=> Lazy-CFR and Lazy-CFR+); statically
   // incompatible with the predictive/discounted kernels, weighted averaging and pruning.
   reach_threshold = 1
};

struct CFRConfig {
   UpdateMode update_mode = UpdateMode::alternating;
   RegretMinimizingMode regret_minimizing_mode = RegretMinimizingMode::regret_matching;
   CFRWeightingMode weighting_mode = CFRWeightingMode::uniform;
   CFRPruningMode pruning_mode = CFRPruningMode::none;

   /// ---- lazy-update segmentation knobs (only meaningful when lazy_update_mode != off) -------
   /// see CFRLazyUpdateMode::reach_threshold for the mechanism (Zhou et al., ICLR 2020)
   CFRLazyUpdateMode lazy_update_mode = CFRLazyUpdateMode::off;
   /// opponent-reach budget B of the per-infoset segmentation: a segment closes once the
   /// counterfactual (opponent) reach accumulated since the last refresh reaches B. The paper
   /// prescribes a fixed budget; B == 1 degenerates to at-most-one frozen iteration between
   /// refreshes for fully-reachable infostates.
   double lazy_update_threshold_b = 1.;

   /// ---- regret-based pruning knobs (only meaningful when pruning_mode == regret_based or
   /// dynamic_thresholding) -------------------------------------------------------------

   /// Minimum worst-case window length -- computed via the Theorem-1 bound
   /// m = floor(|R(I,a)| / (U(I,a) - L(I))) -- before a pruning window is armed. The NIPS'15
   /// paper's Appendix B recommends such a minimum-skip filter (their experiments use 25 on
   /// Leduc-5); small bed games need smaller values so that windows engage at all.
   double rbp_min_skip_iterations = 1.;
   /// Period (in visits to a pruned edge) between recomputations of the buffered best-response
   /// value against the opponents' current average strategies ("periodic BR traversals").
   /// Higher periods trade surrogate freshness for less BR-walk overhead; the NIPS'15 appendix
   /// argues the scheme is insensitive to such cadence choices.
   size_t rbp_br_refresh_period = 16;

   /// Aggressiveness constant C >= 1 of the dynamic-thresholding schedules
   ///   RM-family (their Theorem 2): tau_t = (C^2 - 1) / (2 C |A(I)|^2 sqrt(t))
   ///   Hedge-family (their Theorem 1): tau_t = (C - 1) sqrt(ln |A(I)|) / (sqrt(2) |A(I)|^2
   ///   sqrt(t))
   /// with t the logical iteration number. Their experiments show insensitivity to C; C == 1
   /// disables thresholding entirely (tau_t collapses to 0).
   double dynamic_threshold_c = 3.;
};

struct CFRPlusConfig {
   UpdateMode update_mode = UpdateMode::alternating;
};

/// configuration carrier of the Lazy-CFR family (see CFRLazyUpdateMode::reach_threshold).
/// Consumed by the rm::LazyCFR / rm::LazyCFRPlus aliases and the factory's make_cfr_lazy /
/// make_cfr_lazy_plus methods; the same knobs are reachable directly through rm::CFRConfig.
struct CFRLazyConfig {
   UpdateMode update_mode = UpdateMode::alternating;
   /// regret_matching => Lazy-CFR, regret_matching_plus => Lazy-CFR+
   RegretMinimizingMode regret_minimizing_mode = RegretMinimizingMode::regret_matching;
   /// opponent-reach budget B per infoset segment
   double threshold_b = 1.;
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

/// Which variance-reduction machinery is attached to outcome-sampling updates.
/// Supersedes the boolean 'variance_reduced_baselines' (kept as a legacy shim;
/// see 'effective_variance_reduction').
enum class VarianceReductionMode {
   // plain outcome-sampling MCCFR: no baselines, importance-weighted terminal values
   none = 0,
   // VR-MCCFR (Schmid et al., AAAI 2019): per-infostate-and-action running-mean
   // baselines b(I,a) correcting the sampled continuation values
   action_baseline = 1,
   // ESCHER-style history-value function V(h) (McAleer, Farina, Lanctot,
   // Sandholm, ICLR 2023, arXiv:2206.04122): one scalar per visited world-state
   // edge (h -> h a), keyed by a rolling hash of the sampled trajectory prefix
   // plus the local action index; bootstrapped along the single sampled
   // trajectory exactly like the action baselines, but at HISTORY granularity
   // (no generalization across the histories of an infoset)
   history_value = 2
};

/// How the baseline of the sampled edge is maintained after the regret update.
enum class BaselineUpdateRule {
   // exponentially-decaying running mean toward the baseline-corrected value
   // estimate of the sampled branch (VR-MCCFR paper step (e))
   running_mean = 0,
   // predictive baseline (Davis, Schmid, Bowling, ICML 2020, eq (8)): regress
   // the sampled edge's baseline onto the trajectory value RE-EVALUATED under
   // the next strategy sigma^{t+1} (obtained by recommending from the freshly
   // updated regret table). Provably optimal (zero-variance under complete
   // sampling schemes); empirically lower variance under outcome sampling.
   predictive = 1
};

/// Distribution used to sample the UPDATING player's actions during an
/// outcome-sampling traversal.
enum class UpdaterSamplingMode {
   // epsilon-on-policy sampling w.r.t. the current strategy (classical OS-MCCFR
   // / VR-MCCFR). Time-varying sampling rule => importance corrections required.
   current_policy = 0,
   // ESCHER (sec. 3): sample the updating player's actions from a FIXED uniform
   // distribution over legal actions (opponents keep their current policy).
   // Because the sampling rule no longer changes across iterations, all
   // importance-sampling corrections are dropped: neither the eq-(9) 1/xi
   // deviation factor nor the eq-(11) pi_-i/q(h) reach weighting is applied.
   // The per-infoset visit frequency under the fixed policy plays the role of
   // the positive scaling w(s) of ESCHER's Theorem 1. Requires
   // VarianceReductionMode::history_value (enforced statically).
   fixed_uniform = 1
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
   ///////////////////////////////////////////////////////////////////////////
   /// tri-state variance-reduction surface (supersedes the boolean above) ///
   ///////////////////////////////////////////////////////////////////////////
   /// Selects among no reduction, VR-MCCFR action baselines b(I,a), and
   /// ESCHER-style history values V(h). The legacy boolean
   /// 'variance_reduced_baselines' remains fully functional as a shim: when it
   /// is true and this field is left at 'none', the effective mode resolves to
   /// 'action_baseline' (see 'effective_variance_reduction'). Explicitly
   /// setting both wins for the enum.
   VarianceReductionMode variance_reduction = VarianceReductionMode::none;
   /// baseline maintenance rule applied to the sampled edge's baseline after
   /// the regret update ('running_mean' = VR-MCCFR step (e); 'predictive' =
   /// Davis et al. eq (8), regressing onto the next-strategy re-evaluation).
   BaselineUpdateRule baseline_update_rule = BaselineUpdateRule::running_mean;
   /// sampling distribution of the updating player's actions; 'fixed_uniform'
   /// activates the ESCHER importance-sampling-free scheme (requires
   /// variance_reduction == history_value).
   UpdaterSamplingMode updater_sampling = UpdaterSamplingMode::current_policy;
};

/// Resolves the LEGACY boolean flag against the tri-state enum. Call sites
/// (minimizer selection, engine constexpr branches) must consult this instead
/// of reading either field directly so that old designated-initializer configs
/// keep their exact historical behavior:
///   .variance_reduced_baselines = true  => action_baseline
///   everything else                     => the enum value verbatim
[[nodiscard]] inline constexpr VarianceReductionMode effective_variance_reduction(MCCFRConfig config
)
{
   if(config.variance_reduction != VarianceReductionMode::none) {
      return config.variance_reduction;
   }
   return config.variance_reduced_baselines ? VarianceReductionMode::action_baseline
                                            : VarianceReductionMode::none;
}

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

/// discount factor d(t; e) = t^e / (t^e + 1) of the DCFR family, with the raw
/// (0-based) index made well-defined for NEGATIVE exponents: the papers index
/// discounts by COMPLETED iterations t >= 1, so raw index 0 has no counterpart
/// there. For e >= 0 the historical convention 0^e = 0 is preserved bit-for-bit
/// (factor 0 for e > 0, factor 1/2 for e == 0); for e < 0 -- where pow(0, e)
/// evaluates to +inf and the naive factor becomes NaN, silently poisoning every
/// non-positive regret entry (observed with scheduled HS_beta starting at -1) --
/// the neutral limit 1/2 is substituted.
[[nodiscard]] inline double discount_factor(size_t raw_index, double exponent)
{
   if(raw_index == 0 and exponent < 0.) {
      return 0.5;
   }
   const double base = static_cast< double >(raw_index);
   const double p = std::pow(base, exponent);
   return p / (p + 1.);
}

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
