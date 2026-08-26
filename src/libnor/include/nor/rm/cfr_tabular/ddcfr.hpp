
#ifndef NOR_RM_CFR_TABULAR_DDCFR_HPP
#define NOR_RM_CFR_TABULAR_DDCFR_HPP

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "nor/exploitability.hpp"
#include "nor/rm/cfr_tabular/cfr_config.hpp"

namespace nor::rm::ddcfr {

///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////// runtime-adaptive dynamic discounting ////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief DDCFR-style RUNTIME-ADAPTIVE discounting for the tabular DCFR carrier.
 *
 * Published background. Dynamic Discounted CFR (Xu, Li, Fu, Fu, Xing, Cheng,
 * "Dynamic Discounted Counterfactual Regret Minimization", ICLR 2024,
 * openreview 6PbvbLyqT6) replaces DCFR's fixed exponents with per-decision
 * outputs of a policy trained OFFLINE by evolutionary strategies across a
 * distribution of games:
 *
 *    a^t = [alpha_t, beta_t, gamma_t, tau_t] = pi_theta(s_t)
 *    alpha in [0,5], beta in [-5,0], gamma in [0,5]   (continuous, tanh-scaled)
 *    tau in {1, 2, 5, 10, 20}                         (discrete hold interval)
 *    s_t = ( t/T , conv_frac )                        (paper sec. 3.3, obs dim = 2)
 *
 * where conv_frac is the normalized remaining log-exploitability
 *    conv_frac = (log10(e_t) - log10(e_floor)) / (log10(e_0) - log10(e_floor))
 * anchored at e_floor = 1e-12, and tau_t HOLDS the weights constant for tau
 * consecutive CFR iterations (their CFREnv.step loops 'tau' times between
 * network queries). The held (alpha_t, beta_t, gamma_t) drive exactly the
 * DCFR update rules this engine implements through CFRWeightingMode::discounted
 * (see rm::DiscountedCFR / rm::CFRDiscountedParameters).
 *
 * WHAT THIS HEADER PROVIDES (tabular-faithful subset). The neural part of
 * DDCFR -- offline RL/ES training of pi_theta over a game distribution -- is
 * OUT OF SCOPE. What is provided is the runtime-adaptive discounting LAYER:
 *
 *  1. rm::ddcfr::DDCFRFeatures      -- the runtime observation consumed by a
 *     weight policy: normalized iteration fraction t/T, the paper's
 *     convergence proxy (normalized log-exploitability, computed by an
 *     optional periodic probe), plus tabular proxies that stand in for the
 *     paper's exploitability feature when no probe is wired: mean normalized
 *     average-strategy entropy and cumulative-regret statistics (mean L1 norm,
 *     positive-entry fraction).
 *  2. rm::ddcfr::DDCFRWeightPolicy  -- a std::function<DDCFRWeights(features,
 *     index)> INJECTION POINT (same spirit as the B1 *_schedule hooks it feeds)
 *     so users can plug their OWN trained network later.
 *  3. Two concrete deterministic policies:
 *       - PiecewiseDDCFRPolicy: a hand-designed piecewise-linear scheme
 *         mirroring the learned qualitative behavior (aggressive early
 *         history-downweighting -> late near-uniform weighting), documented as
 *         an APPROXIMATION -- not the published network's output;
 *       - ExploitabilityProxyPolicy: modulates the discounts from a cheap
 *         exploitability estimate recomputed every K iterations on a SMALL game
 *         (faster discounting while far from equilibrium).
 *  4. rm::ddcfr::DDCFRController    -- the glue that consumes the B1 schedule
 *     hooks (rm::CFRDiscountedParameters::alpha_/beta_/gamma_schedule): it
 *     computes features from the attached solver's runtime state once per
 *     query, sanitizes/clamps the policy output into the paper's parameter
 *     ranges, realizes tau's weight-HOLD semantics, and records the trajectory
 *     for inspection.
 *
 * HOW TO REACH PUBLISHED DDCFR FROM HERE. Train pi_theta offline (ES or PPO
 * against the reward = decrease in log-exploitability, cf. their sec. 4) on a
 * distribution of small games, export its inference as a stateless callable,
 * and inject it as the DDCFRWeightPolicy below -- ideally after replacing the
 * feature vector's tabular proxies with true exploitability via bind()'s probe
 * overload. Everything else (feature plumbing, tau-hold mechanics, schedule
 * integration, clamping) already matches the published loop.
 *
 * SAMPLING-POINT CONTRACT. The B1 hooks fire inside the end-of-iteration sweep
 * (_invoke_regret_minimizer) with the RAW 0-based iteration index; features are
 * therefore extracted from post-traversal/pre-recommendation tables of the
 * running iteration. Under alternating updates one query fires per PLAYER
 * update; the published simultaneous-update loop queries once per joint
 * iteration -- same granularity modulo the player cycle. The predictive
 * kernels (PDCFR+) additionally evaluate alpha at index+1 inside recommend();
 * the controller's tau-hold/memoization resolves both lookups deterministically.
 */

/// the four-parameter action a^t = [alpha_t, beta_t, gamma_t, tau_t]; the first
/// three feed the DCFR discount factors d(t; e) = t^e/(t^e + 1) (positive /
/// negative regrets / average-policy side), 'tau' is the number of consecutive
/// iterations the drawn weights stay HELD before the next policy query
struct DDCFRWeights {
   double alpha = 1.5;
   double beta = 0.;
   double gamma = 2.;
   size_t tau = 1;
};

/// the runtime observation s_t handed to a weight policy. Field selection
/// follows the paper where applicable and documents its deviations elsewhere:
///  - iteration_fraction : t/T (published feature 1)
///  - convergence_proxy  : the paper's conv_frac (normalized remaining
///                         log-exploitability in [0,1]; 1 = start, 0 = floor).
///                         quiet_NaN when NO exploitability probe is bound --
///                         consumers should fall back to a tabular proxy then.
///  - strategy_entropy   : TABULAR PROXY (our extension): mean over registered
///                         infostates of the AVERAGE strategy's entropy
///                         normalized by ln|A(I)| in [0,1]
///  - regret_l1_norm     : TABULAR PROXY (our extension): mean L1 norm of the
///                         cumulative counterfactual regret tables
///  - positive_regret_fraction : TABULAR PROXY (our extension): share of
///                         strictly positive entries across all regret tables
struct DDCFRFeatures {
   double iteration_fraction = 0.;
   double convergence_proxy = std::numeric_limits< double >::quiet_NaN();
   double strategy_entropy = 0.;
   double regret_l1_norm = 0.;
   double positive_regret_fraction = 0.;
};

/// injection point for arbitrary weight policies -- including a user's own
/// trained DDCFR network exported as a plain callable. Receives the current
/// features and the raw (0-based) index at which the weights will take effect;
/// must be deterministic given (features, index) to keep solver runs
/// reproducible. Outputs are sanitized+clamped by the controller (see
/// sanitize_weights) before they reach the schedules.
using DDCFRWeightPolicy = std::function< DDCFRWeights(const DDCFRFeatures&, size_t) >;

/// the paper's action-space bounds (CFREnv: alpha_range/beta_range/gamma_range
/// = [0,5]/[-5,0]/[0,5], tau_list = {1,2,5,10,20}); every controller-produced
/// weight is clamped into these intervals
struct ParameterRanges {
   static constexpr double alpha_min = 0., alpha_max = 5.;
   static constexpr double beta_min = -5., beta_max = 0.;
   static constexpr double gamma_min = 0., gamma_max = 5.;
   static constexpr size_t tau_min = 1, tau_max = 20;
};

inline DDCFRWeights sanitize_weights(DDCFRWeights w)
{
   if(not std::isfinite(w.alpha)) {
      // non-finite output: fall back to the neutral fixed-DCFR default instead
      // of propagating NaN into the tables (the discount_factor NaN guard in
      // cfr_config.hpp remains the second line of defense for negative
      // exponents at the undefined raw index 0)
      w.alpha = 1.5;
   }
   if(not std::isfinite(w.beta)) {
      w.beta = 0.;
   }
   if(not std::isfinite(w.gamma)) {
      w.gamma = 2.;
   }
   w.alpha = std::clamp(w.alpha, ParameterRanges::alpha_min, ParameterRanges::alpha_max);
   w.beta = std::clamp(w.beta, ParameterRanges::beta_min, ParameterRanges::beta_max);
   w.gamma = std::clamp(w.gamma, ParameterRanges::gamma_min, ParameterRanges::gamma_max);
   // NOTE: unlike the paper we accept any integral tau in [1, 20] rather than
   // snapping to their discrete set {1,2,5,10,20} -- a documented superset that
   // keeps custom policies' cadence choices intact
   if(w.tau < ParameterRanges::tau_min) {
      w.tau = ParameterRanges::tau_min;
   }
   if(w.tau > ParameterRanges::tau_max) {
      w.tau = ParameterRanges::tau_max;
   }
   return w;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// built-in weight policies ////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief hand-designed piecewise scheme mirroring DDCFR's learned qualitative
 * behavior: AGGRESSIVE early downweighting of accumulated history (low alpha =>
 * fast positive-regret turnover, strongly negative beta => fast negative-regret
 * clearing, high gamma => stale average-strategy mass decays quickly) easing
 * toward a NEAR-UNIFORM late regime (high alpha, beta -> 0, low gamma) where
 * late iterates are barely discounted.
 *
 * HONEST PROVENANCE: this is NOT the published network's output -- the learned
 * weights are only available as trained parameters, and their qualitative trend
 * (strong early forgetting, gentle late discounting) is what is mirrored here.
 * The anchors below are hand-chosen to bracket DCFR(1.5, 0, 2); the trajectory
 * interpolates LINEARLY between them in the iteration fraction p = t/T with a
 * single midpoint break at 'mid_fraction'. Fully deterministic.
 */
struct PiecewiseDDCFRPolicy {
   /// (alpha, beta, gamma) triple at one anchor point
   struct Anchor {
      double alpha = 0.;
      double beta = 0.;
      double gamma = 0.;
   };

   /// fraction of the horizon separating the early and the late segment
   double mid_fraction = 0.5;
   Anchor early{.alpha = 0.5, .beta = -2.5, .gamma = 4.0};
   Anchor mid{.alpha = 2.0, .beta = -1.0, .gamma = 2.0};
   Anchor late{.alpha = 3.0, .beta = -0.25, .gamma = 1.0};
   /// hold interval emitted for every query (tau itself is held flat here)
   size_t tau = 1;

   [[nodiscard]] DDCFRWeights operator()(const DDCFRFeatures& features, size_t /*index*/) const
   {
      const double p = std::clamp(features.iteration_fraction, 0., 1.);
      const auto& first_anchor = p <= mid_fraction ? early : mid;
      const auto& second_anchor = p <= mid_fraction ? mid : late;
      const double lo = p <= mid_fraction ? 0. : mid_fraction;
      const double hi = p <= mid_fraction ? mid_fraction : 1.;
      const double u = hi > lo ? (p - lo) / (hi - lo) : 0.;  // segment-local progress
      const auto lerp = [&](double a, double b) { return a + (b - a) * u; };
      return sanitize_weights(DDCFRWeights{
         .alpha = lerp(first_anchor.alpha, second_anchor.alpha),
         .beta = lerp(first_anchor.beta, second_anchor.beta),
         .gamma = lerp(first_anchor.gamma, second_anchor.gamma),
         .tau = tau});
   }
};

/**
 * @brief exploitability-proxy-adaptive scheme: every K iterations a cheap full-
 * tree exploitability estimate of the CURRENT average strategy profile is taken
 * (on a SMALL game -- the whole approach presumes the solver runs on a bed
 * game) and converted into the paper's convergence proxy; the discounts are
 * then interpolated between an AGGRESSIVE setting applied while far from
 * equilibrium (proxy -> 1: fast turnover, heavy history discounting) and a CALM
 * one applied near equilibrium (proxy -> 0). When no probe is bound the policy
 * falls back to 1 - strategy_entropy as a coarse distance-to-equilibrium
 * surrogate (documented deviation). Deterministic: the probe is a complete-tree
 * walk, no sampling anywhere.
 */
struct ExploitabilityProxyPolicy {
   size_t query_interval = 25;  ///< K: re-query/probe cadence AND emitted tau
   double aggressive_gamma = 3.5;
   double calm_gamma = 1.0;
   double aggressive_alpha = 1.0;
   double calm_alpha = 3.0;
   double aggressive_beta = -2.0;
   double calm_beta = -0.5;

   [[nodiscard]] DDCFRWeights operator()(const DDCFRFeatures& features, size_t /*index*/) const
   {
      const double proxy = std::invoke([&] {
         if(std::isfinite(features.convergence_proxy)) {
            return std::clamp(features.convergence_proxy, 0., 1.);
         }
         // fallback surrogate when no exploitability probe is wired
         return std::clamp(1. - features.strategy_entropy, 0., 1.);
      });
      const auto lerp = [&](double calm, double aggressive) {
         return calm + (aggressive - calm) * proxy;
      };
      return sanitize_weights(DDCFRWeights{
         .alpha = lerp(calm_alpha, aggressive_alpha),
         .beta = lerp(calm_beta, aggressive_beta),
         .gamma = lerp(calm_gamma, aggressive_gamma),
         .tau = query_interval});
   }
};

[[nodiscard]] inline DDCFRWeightPolicy piecewise_ddcfr_policy(PiecewiseDDCFRPolicy policy = {})
{
   return [policy](const DDCFRFeatures& f, size_t idx) { return policy(f, idx); };
}

[[nodiscard]] inline DDCFRWeightPolicy exploitability_proxy_policy(
   ExploitabilityProxyPolicy policy = {}
)
{
   return [policy](const DDCFRFeatures& f, size_t idx) { return policy(f, idx); };
}

///////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////// runtime feature extraction ///////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief mean over all REGISTERED infostates of the average strategy's entropy
 * normalized by ln|A(I)| ("strategy entropy" feature, in [0,1]).
 *
 * Average-policy entries are stored UNNORMALIZED (cumulative reach-weighted
 * mass); each visited action distribution is normalized locally first.
 * Infostates whose accumulated mass vanished entirely are SKIPPED (their
 * distribution is undefined, and treating them as uniform would fake maximal
 * entropy). Deterministic for a given run (unordered-map order only permutes
 * the mean's summands up to fp associativity).
 */
template < typename Solver >
double mean_normalized_strategy_entropy(const Solver& solver)
{
   double entropy_sum = 0.;
   size_t n_infostates = 0;
   for(const auto& [player, table_policy] : solver.average_policy()) {
      for(const auto& [infostate, action_policy] : table_policy.table()) {
         ++n_infostates;
         const double norm = std::ranges::fold_left(
            action_policy | std::views::values, double(0.), std::plus{}
         );
         if(norm <= 0.) {
            continue;  // never-updated infostate: skipped, see above
         }
         const double log_norm = std::log(static_cast< double >(action_policy.size()));
         if(log_norm <= 0.) {
            continue;  // single-action infostates carry zero entropy
         }
         double h = 0.;
         for(const auto& [action, mass] : action_policy) {
            const double p = mass / norm;
            if(p > 0.) {
               h -= p * std::log(p);
            }
         }
         entropy_sum += h / log_norm;
      }
   }
   return n_infostates == 0 ? 0. : entropy_sum / static_cast< double >(n_infostates);
}

/// aggregate cumulative-regret statistics over every registered infostate
/// (mean L1 norm + strictly-positive entry share); requires the read-only
/// VanillaCFR::visit_regret_tables visitor
template < typename Solver >
auto regret_statistics(const Solver& solver)
{
   struct RegretStats {
      double mean_l1_norm = 0.;
      double positive_fraction = 0.;
   };
   double l1_sum = 0.;
   size_t n_tables = 0;
   size_t positive_entries = 0;
   size_t total_entries = 0;
   solver.visit_regret_tables([&](const auto& table) {
      for(double entry : table) {
         l1_sum += std::abs(entry);
         positive_entries += entry > 0. ? size_t{1} : size_t{0};
      }
      total_entries += table.size();
      ++n_tables;
   });
   return RegretStats{
      .mean_l1_norm = n_tables == 0
                         ? 0.
                         : l1_sum / static_cast< double >(std::max(total_entries, size_t{1})),
      .positive_fraction = total_entries == 0 ? 0.
                                              : static_cast< double >(positive_entries)
                                                   / static_cast< double >(total_entries)};
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// the discounting controller ///////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

/// knobs of the runtime-adaptive layer. 'total_iterations' is the T of the
/// iteration-fraction feature and MUST match the planned iterate() count (a
/// mismatch merely rescales the fraction, exactly like the HS schedules'
/// horizon binding). 'exploitability_interval' is the probe cadence K of
/// bind(solver, env, root); 0 disables probing even there.
struct Options {
   size_t total_iterations = 0;
   size_t exploitability_interval = 25;
   /// log10 anchor e_floor of the paper's conv_frac normalization
   double exploitability_floor = 1e-12;
};

/**
 * @brief the shared brain behind the three B1 schedule hooks: queries the
 * injected weight policy once per NEW index (respecting the policy's tau-hold),
 * extracts runtime features from the bound solver at query time, sanitizes the
 * answer into the paper's ranges and memoizes it for every hook lookup.
 *
 * LIFETIME & THREADING: intended to live on a std::shared_ptr captured by the
 * schedule closures (see ddcfr_parameters) AND to outlive the solver's iterate()
 * calls; bind() stores references into the solver, so the controller must never
 * outlast them. Single-threaded use only (the engine's sweeps are serial).
 */
class DDCFRController {
  public:
   DDCFRController(DDCFRWeightPolicy policy, Options options = {})
       : m_policy(std::move(policy)), m_options(options)
   {
      if(not m_policy) {
         throw std::invalid_argument("DDCFRController requires a non-empty weight policy");
      }
   }

   /// B1-conformant schedule entry points (raw 0-based index -> parameter)
   [[nodiscard]] double alpha_at(size_t index) { return resolve(index).alpha; }
   [[nodiscard]] double beta_at(size_t index) { return resolve(index).beta; }
   [[nodiscard]] double gamma_at(size_t index) { return resolve(index).gamma; }

   /**
    * @brief binds the runtime feature extraction to a constructed solver
    * (entropy + regret-table statistics; convergence_proxy stays NaN).
    * Call AFTER solver construction, BEFORE the first iterate().
    */
   template < typename Solver >
   void bind(Solver& solver)
   {
      m_feature_provider = [&solver, this]([[maybe_unused]] size_t index) {
         return collect_features(solver, m_options.total_iterations);
      };
   }

   /**
    * @brief binds feature extraction PLUS the paper's exploitability probe:
    * every 'Options::exploitability_interval'-th queried index (and the first)
    * runs a full-tree exploitability evaluation of the current NORMALIZED
    * average strategy profile and appends the resulting convergence proxy to
    * convergence_history(). 'root_like' is stored BY VALUE (cheap for the bed
    * games this layer targets); the environment is also copied.
    */
   template < typename Solver, typename Env, typename RootLike >
   void bind(Solver& solver, const Env& env, RootLike&& root_like)
   {
      // NOTE: the copies are deliberately NOT named like the parameters --
      // init-captures that shadow them trip -Wshadow
      m_feature_provider = [&solver,
                            probed_env = env,
                            probed_root = std::forward< RootLike >(root_like),
                            this](size_t index) mutable {
         DDCFRFeatures features = collect_features(solver, m_options.total_iterations);
         if(m_options.exploitability_interval > 0
            and (not m_last_probe_index.has_value() or index - *m_last_probe_index >= m_options.exploitability_interval)) {
            const double expl = probe_exploitability(solver, probed_env, probed_root);
            if(std::isfinite(expl)) {
               m_last_probe_index = index;
               if(expl > 0.) {
                  if(not m_start_log10_expl.has_value()) {
                     m_start_log10_expl = std::log10(expl);
                  }
                  const double denom = *m_start_log10_expl
                                       - std::log10(m_options.exploitability_floor);
                  m_last_convergence_proxy = denom > 0. ? std::clamp(
                                                (std::log10(expl)
                                                 - std::log10(m_options.exploitability_floor))
                                                   / denom,
                                                0.,
                                                1.
                                             )
                                                        : 0.;
               } else {
                  m_last_convergence_proxy = 0.;  // equilibrium reached
               }
               m_convergence_history.emplace_back(index, m_last_convergence_proxy);
            }
            // a SKIPPED probe (not-finite expl: profile not ready yet) records
            // nothing and leaves the cadence armed -- the next query retries
            // immediately instead of waiting another full interval
         }
         features.convergence_proxy = m_last_convergence_proxy;
         return features;
      };
   }

   /// whether a solver has been bound yet
   [[nodiscard]] bool bound() const { return m_feature_provider != nullptr; }

   /// recorded (query_index, convergence_proxy) pairs of every fresh probe run
   [[nodiscard]] const std::vector< std::pair< size_t, double > >& convergence_history() const
   {
      return m_convergence_history;
   }

   /// the most recent convergence proxy value (NaN before the first probe)
   [[nodiscard]] double last_convergence_proxy() const { return m_last_convergence_proxy; }

   /// one entry per POLICY QUERY (not per resolved index -- holds reuse the
   /// cached weights silently): query index, features seen, sanitized weights
   struct TrajectoryEntry {
      size_t query_index = 0;
      DDCFRFeatures features{};
      DDCFRWeights weights{};
   };

   [[nodiscard]] const std::vector< TrajectoryEntry >& trajectory() const { return m_trajectory; }

  private:
   template < typename Solver >
   DDCFRFeatures collect_features(const Solver& solver, size_t total_iterations)
   {
      DDCFRFeatures features;
      features.iteration_fraction = total_iterations > 0
                                       ? static_cast< double >(solver.iteration())
                                            / static_cast< double >(total_iterations)
                                       : 0.;
      features.strategy_entropy = mean_normalized_strategy_entropy(solver);
      const auto stats = regret_statistics(solver);
      features.regret_l1_norm = stats.mean_l1_norm;
      features.positive_regret_fraction = stats.positive_fraction;
      return features;
   }
   template < typename Solver, typename Env >
   double probe_exploitability(const Solver& solver, Env& env, const auto& root_state) const
   {
      using TablePolicy = std::remove_cvref_t< decltype(solver.average_policy().begin()->second) >;
      if(solver.average_policy().empty()) {
         return std::numeric_limits< double >::quiet_NaN();
      }
      // PROBE READINESS: under alternating updates the earliest sweeps leave
      // the not-yet-updated players' average-policy tables EMPTY, and a
      // best-response walk over missing infostate entries cannot be evaluated.
      // Such snapshots are skipped (NaN -> no history entry) instead of
      // throwing; probing starts at the first query whose profile is complete.
      for(const auto& [player, table_policy] : solver.average_policy()) {
         if(table_policy.size() == 0) {
            return std::numeric_limits< double >::quiet_NaN();
         }
      }
      player_hashmap< TablePolicy > profile;
      for(const auto& [player, table_policy] : solver.average_policy()) {
         profile.emplace(player, normalize_state_policy(table_policy));
      }
      try {
         return nor::exploitability(env, root_state, profile, false);
      } catch(const std::exception&) {
         // defensively treat any residual partial-profile evaluation failure
         // as 'no measurement' (deterministic across runs)
         return std::numeric_limits< double >::quiet_NaN();
      }
   }

   /// resolves the effective weights at 'index': a cache hit for any index
   /// covered by the active tau-HOLD, otherwise a fresh feature extraction +
   /// policy query (memoized for repeated hook lookups of the same index)
   [[nodiscard]] const DDCFRWeights& resolve(size_t index)
   {
      const auto cached = m_cache.find(index);
      if(cached != m_cache.end()) {
         return cached->second;
      }
      if(m_hold_start.has_value() and index >= *m_hold_start
         and index < *m_hold_start + m_hold_tau) {
         return m_cache.emplace(index, m_held_weights).first->second;
      }
      if(not m_feature_provider) {
         throw std::logic_error(
            "DDCFRController: no solver bound -- call bind() after constructing the "
            "solver and before iterate()"
         );
      }
      DDCFRFeatures features = m_feature_provider(index);
      DDCFRWeights weights = sanitize_weights(m_policy(features, index));
      m_hold_start = index;
      m_hold_tau = weights.tau;
      m_held_weights = weights;
      m_trajectory.push_back(TrajectoryEntry{
         .query_index = index, .features = features, .weights = weights});
      return m_cache.emplace(index, weights).first->second;
   }

   DDCFRWeightPolicy m_policy;
   Options m_options{};
   std::function< DDCFRFeatures(size_t) > m_feature_provider{};
   std::unordered_map< size_t, DDCFRWeights > m_cache{};
   std::optional< size_t > m_hold_start{};
   size_t m_hold_tau = 1;
   DDCFRWeights m_held_weights{};
   std::vector< TrajectoryEntry > m_trajectory{};
   std::optional< size_t > m_last_probe_index{};
   std::optional< double > m_start_log10_expl{};
   double m_last_convergence_proxy = std::numeric_limits< double >::quiet_NaN();
   std::vector< std::pair< size_t, double > > m_convergence_history{};
};

/**
 * @brief assembles the rm::CFRDiscountedParameters bundle whose three B1
 * schedule hooks route through 'controller'. Pass the result to the factory /
 * constructor of any discounted-carrier solver exactly like an HS bundle:
 *
 *    auto controller = std::make_shared<rm::ddcfr::DDCFRController>(
 *       rm::ddcfr::piecewise_ddcfr_policy(), rm::ddcfr::Options{.total_iterations = 600});
 *    auto params = rm::ddcfr::ddcfr_parameters(controller);
 *    auto solver = factory::make_cfr<cfg>(env, std::move(root), curr, avg, params);
 *    controller->bind(solver);           // attach runtime state access
 *    solver.iterate(600);
 *
 * The constants of the returned bundle stay at the DCFR defaults but are
 * shadowed by the schedules; the closures share the controller's memoization so
 * the policy is consulted at most once per (held) index regardless of how many
 * hooks fire per iteration.
 */
[[nodiscard]] inline CFRDiscountedParameters ddcfr_parameters(
   const std::shared_ptr< DDCFRController >& controller
)
{
   if(not controller) {
      throw std::invalid_argument("ddcfr_parameters requires a non-null controller");
   }
   CFRDiscountedParameters params{};
   params.alpha_schedule = [controller](size_t index) { return controller->alpha_at(index); };
   params.beta_schedule = [controller](size_t index) { return controller->beta_at(index); };
   params.gamma_schedule = [controller](size_t index) { return controller->gamma_at(index); };
   return params;
}

}  // namespace nor::rm::ddcfr

#endif  // NOR_RM_CFR_TABULAR_DDCFR_HPP
