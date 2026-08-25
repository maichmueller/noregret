
#ifndef NOR_RM_ICFR_HPP
#define NOR_RM_ICFR_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <ranges>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "common/common.hpp"
#include "nor/at_runtime.hpp"
#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "nor/policy/action_policy.hpp"
// NOTE include-order sensitive: node.hpp must be parsed FIRST so that its
// own 'minimizers.hpp' inclusion completes before node's default template
// argument names RegretMatching (rm_utils -> node -> minimizers would nest
// the guarded header and break). Mirrors the house ordering in cfr_base.hpp.
// clang-format off
#include "nor/rm/node.hpp"
#include "nor/rm/minimizers/minimizers.hpp"
// clang-format on
#include "nor/rm/rm_utils.hpp"
#include "nor/type_defs.hpp"
#include "nor/utils/utils.hpp"

namespace nor::rm {

// local env-alias helpers keeping the class-external static assertions readable
template < typename Env >
using icfr_action_type_of = auto_action_type< Env >;
template < typename Env >
using icfr_chance_outcome_type_of = auto_chance_outcome_type< Env >;

/// hash functor for terminal action paths (root-to-leaf action/chance-outcome lists)
template < typename ActionVariant >
struct TerminalPathHash {
   size_t operator()(const std::vector< ActionVariant >& path) const noexcept
   {
      size_t seed = path.size();
      for(const auto& edge : path) {
         std::visit(
            [&seed]< typename Alt >(const Alt& alternative) {
               common::hash_combine(seed, std::hash< Alt >{}(alternative));
            },
            edge
         );
      }
      return seed;
   }
};

/**
 * @brief Blum-Mansour internal-regret minimizer built from per-source external
 * regret-minimizer units (Blum & Mansour, "From External to Internal Regret",
 * JMLR 8, 2007).
 *
 * One external kernel E_i (default: the house CFR+/RM+ kernel) is attached to
 * every action i, interpreted as the post-processing "whenever the mixture
 * mass on source i is consumed, redirect it according to E_i". Per round:
 *   1. every E_i produces a target recommendation q_i;
 *   2. the play distribution w is the STATIONARY DISTRIBUTION of the target
 *      matrix Q (w_j = sum_i w_i q_i(j)), computed by Cesaro-averaged power
 *      iteration (the average is essential: RM+-style kernels emit pure rows,
 *      making the chain periodic, so plain final iterates oscillate forever);
 *   3. after the round's utility vector u_hat arrives, each E_i observes the
 *      shifted vector scaled by the SOURCE'S OWN MIXING MASS,
 *      w(i) * (u_hat(j) - u_hat(i)) -- exactly the loss partition of the BM
 *      reduction ("giving algorithm A_i a fraction p_i of the loss vector").
 * The cumulative swap regret is then bounded by sum_i Reg_ext(E_i) = O(sqrt T)
 * -- genuine no-internal-regret strength, which is what ICFR's local units
 * require (paper section 6: "an arbitrary no-internal-regret algorithm").
 *
 * WHY NOT rm::InternalRegretMatching. The house swap-basis kernel collapses
 * the pairwise regret matrix onto per-TARGET-column aggregates R(phi_a);
 * regret matching on those aggregates controls only the best FIXED constant
 * post-processing (external-regret strength), which is measurably insufficient
 * here: under simultaneous adaptation the collapsed dynamics lock into
 * Shapley's best-response cycle with LINEAR internal regret. The BM
 * construction restores true no-internal-regret at O(|A|^2) memory per
 * infoset -- negligible in the tabular regime.
 *
 * PROTOCOL. 'observe_utilities' buffers the round's utility vector (call iff
 * the unit was consulted this round, with the full u_hat vector); the next
 * 'recommend' folds it -- scaled by the weights under which it was incurred --
 * through every source unit and resolves the new stationary mixture.
 */
template < concepts::action Action, typename ExternalKernel = RegretMatchingPlus< Action > >
class BlumMansourInternalRegretMatching {
  public:
   struct node_data_type {
      detail::action_registry< Action > registry;
      /// one external unit per SOURCE action: E_i learns the values of
      /// redirecting the mass consumed at i
      std::vector< typename ExternalKernel::node_data_type > per_source;
      /// buffered utility vector u^t of the current round
      std::vector< double > instant_utility;
      /// whether an unconsumed utility vector is buffered
      bool has_pending = false;
      /// mixing weights under which the buffered utilities were incurred
      std::vector< double > last_weights;
      /// each source unit's last recommendation (advantage-form folds); uniform
      /// before the first recommend
      std::vector< std::vector< double > > last_q;

      void register_action(const Action& action)
      {
         // every existing source unit gains the new target ...
         for(auto& unit : per_source) {
            unit.register_action(action);
         }
         registry.register_action(action);
         const auto n = registry.actions.size();
         instant_utility.assign(n, 0.);
         last_weights.assign(n, 1. / double(n));
         // ... and the new source unit starts from all targets so far
         auto& unit = per_source.emplace_back();
         for(const auto& registered : registry.actions) {
            unit.register_action(registered);
         }
         last_q.emplace_back(registry.actions.size(), 1. / double(n));
      }

      [[nodiscard]] size_t index_of(const Action& action) const
      {
         return registry.index_of(action);
      }
   };

   /// buffers one active round's full parameterized utility vector
   static void observe_utilities(node_data_type& data, const std::vector< double >& utilities)
   {
      data.instant_utility = utilities;
      data.has_pending = true;
   }

   template < typename PolicyOut >
   static void recommend(node_data_type& data, PolicyOut& policy_out, size_t /*iteration*/)
   {
      const auto n = data.registry.actions.size();

      // fold the pending utilities through every source unit, shifted by the
      // cost of staying and scaled by the source's own mixing mass:
      //    E_i observes  w(i) * (u(j) - u(i))
      if(data.has_pending) {
         for(auto i : std::views::iota(size_t{0}, n)) {
            const double w_i = data.last_weights[i];
            // shifted utility vector of source i: u(j) - u(i); its expectation
            // under the source's own last recommendation defines the advantage
            // form the external kernels' cumulative tables are built on
            double shifted_expectation = 0.;
            for(auto j : std::views::iota(size_t{0}, n)) {
               shifted_expectation += data.last_q[i][j]
                                      * (data.instant_utility[j] - data.instant_utility[i]);
            }
            for(auto j : std::views::iota(size_t{0}, n)) {
               ExternalKernel::observe(
                  data.per_source[i],
                  data.registry.actions[j],
                  w_i * (data.instant_utility[j] - data.instant_utility[i] - shifted_expectation)
               );
            }
         }
         data.has_pending = false;
         std::ranges::fill(data.instant_utility, 0.);
      }

      // refresh every source's target recommendation
      std::vector< std::vector< double > > q(n, std::vector< double >(n, 0.));
      for(auto i : std::views::iota(size_t{0}, n)) {
         HashmapActionPolicy< Action > q_i{};
         ExternalKernel::recommend(data.per_source[i], q_i, /*iteration=*/0);
         for(auto j : std::views::iota(size_t{0}, n)) {
            q[i][j] = q_i.at(data.registry.actions[j]);
         }
         data.last_q[i] = q[i];
      }

      // stationary distribution of the row-stochastic target matrix:
      // Cesaro-averaged power iteration (the average is what makes periodic
      // chains converge; early exit once the iteration has settled anyway)
      std::vector< double > w(n, 1. / double(n));
      std::vector< double > acc(n, 0.);
      constexpr size_t k_power_iterations = 4096;
      constexpr size_t k_cesaro_burn_in = 1024;
      constexpr double k_convergence_tol = 1e-13;
      for(auto it : std::views::iota(size_t{0}, k_power_iterations)) {
         std::vector< double > next(n, 0.);
         for(auto i : std::views::iota(size_t{0}, n)) {
            const double w_i = w[i];
            if(w_i == 0.) {
               continue;
            }
            for(auto j : std::views::iota(size_t{0}, n)) {
               next[j] += w_i * q[i][j];
            }
         }
         const double mass = std::ranges::fold_left(next, 0., std::plus{});
         if(mass <= 0.) {  // defensive: degenerate recommendations -> stay put
            break;
         }
         double delta = 0.;
         for(auto j : std::views::iota(size_t{0}, n)) {
            const double updated = next[j] / mass;
            delta = std::max(delta, std::abs(updated - w[j]));
            w[j] = updated;
         }
         if(it >= k_cesaro_burn_in) {
            for(auto j : std::views::iota(size_t{0}, n)) {
               acc[j] += w[j];
            }
         }
         if(delta <= k_convergence_tol) {
            break;
         }
      }
      const double acc_sum = std::ranges::fold_left(acc, 0., std::plus{});
      if(acc_sum > 0.) {
         for(auto j : std::views::iota(size_t{0}, n)) {
            w[j] = acc[j] / acc_sum;
         }
      }
      data.last_weights = w;
      for(auto j : std::views::iota(size_t{0}, n)) {
         policy_out[data.registry.actions[j]] = w[j];
      }
   }

   static constexpr double policy_weight(size_t /*iteration*/) { return 1.; }
};

/**
 * @brief ICFR -- Internal Counterfactual Regret Minimization (Celli, Marchesi,
 * Farina & Gatti, "No-Regret Learning Dynamics for Extensive-Form Correlated
 * Equilibrium", NeurIPS 2020, arXiv:2004.00603; journal version JACM 69(6),
 * 2022).
 *
 * The first uncoupled no-regret dynamics converging to extensive-form
 * correlated equilibria (EFCE). Unlike CFR -- whose average profile is a
 * PRODUCT of per-infoset strategies and whose guarantees are Nash-flavored --
 * ICFR's output is the empirical frequency of play mu_bar^T over JOINT
 * normal-form plans (a correlation mediated through the game tree), which
 * converges almost surely to the set of EFCEs.
 *
 * MECHANISM (paper sections 4-6, Algorithm 1). Every player i owns, per own
 * infoset I:
 *   - one INTERNAL-regret minimizer R^int_I over A(I), responsible for the
 *     laminar subtree regrets of trigger sequences rooted AT I, and
 *   - one EXTERNAL-regret minimizer R^ext_{sigma,I} per CO-trigger sequence
 *     sigma in Sigma^c_i(I) = {(J,a) : J a strict own ancestor of I,
 *     a != required action at J along the chain to I}, responsible for the
 *     laminar subtree regrets of triggers that fired ABOVE I.
 * Each iteration t every player samples a normal-form plan pi^t_i top-down:
 * an infoset still reachable under the partial plan consults its internal
 * unit; an unreachable one consults the external unit keyed by the unique
 * deviating ancestor sequence (Algorithm 1, SampleInternal). A single
 * full-tree pass then computes the counterfactual stop-values
 *      u^t_i[I,a] = sum over terminals immediately below (I,a),
 *                   opponent-plan gated and chance weighted
 * from which the parameterized utilities
 *      u_hat^t_I(a) = u^t_i[I,a] + sum_{J in C(I,a)} V^t_J(pi^t_i)
 * (play a at I, then follow the sampled plan below) are assembled on the
 * infoset graph and fed to exactly ONE unit per infoset -- the one whose
 * recommendation was followed (the 1[pi^t in Pi(sigma)] gating of Algorithm 1
 * UpdateInternal). Because a unit's recommendations are consumed exactly when
 * it is active, each unit faces an ordinary online-learning problem and its
 * regret bound composes, via the laminar decomposition (paper Lemmas 1-2),
 * into a sublinear TRIGGER regret R^T_sigma for every sequence sigma. By
 * Theorem 1 the empirical frequency mu_bar^T is then an epsilon-EFCE with
 * epsilon = max_sigma R^T_sigma / T; by Theorem 2 it converges to an EFCE
 * almost surely.
 *
 * OUTPUT REPRESENTATION. The correlation mu_bar^T is accumulated as raw counts
 * keyed by TERMINAL ACTION PATHS (the ordered list of actions/chance outcomes
 * from the root): mu_bar^T(z) = count(z)/T. Terminal-path keying keeps both
 * consumers cheap: the realized trajectory of iteration t IS the terminal path
 * selected by the joint plan (deterministic plans + fixed chance law), and the
 * trigger-agent deviation analysis of 'evaluate_efce_gap' needs only on-path
 * information, which the path carries in full -- under perfect recall the
 * ancestor chain of any sequence sigma relevant to z is a prefix of z's path,
 * so the trigger condition x^t_i(sigma) is decidable from the path alone.
 * Note that this correlation is deliberately NOT a TabularPolicy: mu_bar is a
 * general distribution over joint plans and cannot be factored into
 * per-player behavioral strategies.
 *
 * COMPLEXITY. Per iteration: one full-tree traversal plus infoset-graph work
 * O(sum_i |I_i| * depth_i) and O(#realized terminals) hash-keyed frequency
 * accumulation. Memory: O(sum_i sum_I (1 + |Sigma^c(I)|) * |A(I)|) regret
 * tables, the same order again for the raw trigger co-statistics, and
 * O(#distinct realized terminals) for the frequency map.
 * 'evaluate_efce_gap' is pure table work: dynamic programs over the infoset
 * graphs, O(|Sigma| * max|C*(J)| * |A|) time, no tree traversal.
 *
 * DEVIATIONS FROM THE PAPER. None algorithmic. The paper leaves both the
 * internal ("an arbitrary no-internal-regret algorithm") and the external
 * regret minimizer unspecified: the defaults instantiate the house swap-basis
 * InternalRegretMatching kernel for the former and RegretMatchingPlus (CFR+)
 * kernels for the latter. Trigger regrets are additionally tracked in RAW
 * unclipped laminar accumulators -- independent of the minimizers' internal
 * state, whose clamping/snapshotting does not preserve exact cumulative sums --
 * so 'evaluate_efce_gap' can report BOTH the true EFCE gap delta(mu_bar^T)
 * (paper Eq. 7, evaluated against the empirical correlation by a dedicated
 * best-response-style tree evaluation) AND the Theorem-1 certificate
 * max_sigma R^T_sigma / T computed from the local accumulators through the
 * Lemma-1 recursion. The invariant delta <= certificate + fp slack holds at
 * all times and is cross-checked in the test suite.
 *
 * @tparam Env the FOSG environment type to run ICFR on
 * @tparam InternalRM minimizer kernel used for the per-infoset internal units
 * @tparam ExternalRM minimizer kernel used for the per-co-sequence external units
 */
template <
   typename Env,
   typename InternalRM = BlumMansourInternalRegretMatching< icfr_action_type_of< Env > >,
   typename ExternalRM = RegretMatchingPlus< icfr_action_type_of< Env > > >
class ICFR {
   /////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////// API: public typedefs
   ///////////////////////////////////////////
   /////////////////////////////////////////////////////////////////////////////////////////////////

  public:
   using env_type = Env;
   using internal_rm_type = InternalRM;
   using external_rm_type = ExternalRM;

   using action_type = icfr_action_type_of< Env >;
   using chance_outcome_type = icfr_chance_outcome_type_of< Env >;
   /// the policy output carrier handed to the minimizers' recommend() step
   using policy_out_type = HashmapActionPolicy< action_type >;

   using world_state_type = auto_world_state_type< Env >;
   using info_state_type = auto_info_state_type< Env >;
   using public_state_type = auto_public_state_type< Env >;
   using observation_type = auto_observation_type< Env >;
   using action_variant_type = std::variant<
      action_type,
      std::conditional_t<
         std::is_same_v< chance_outcome_type, void >,
         std::monostate,
         chance_outcome_type > >;

   static_assert(
      concepts::fosg< Env >,
      "ICFR requires the environment to fulfill the fosg concept"
   );

   static_assert(
      requires(
         typename InternalRM::node_data_type& data,
         const action_type& action,
         policy_out_type& policy_out,
         const std::vector< double >& utility_vector
      ) {
         data.register_action(action);
         InternalRM::observe_utilities(data, utility_vector);
         InternalRM::recommend(data, policy_out, size_t{0});
      },
      "InternalRM must provide the internal-unit protocol: node-data "
      "registration plus observe_utilities/recommend"
   );
   static_assert(
      requires(
         typename ExternalRM::node_data_type& data,
         const action_type& action,
         policy_out_type& policy_out
      ) {
         data.register_action(action);
         ExternalRM::observe(data, action, 1.);
         ExternalRM::recommend(data, policy_out, size_t{0});
      },
      "ExternalRM must follow the regret-minimizer node-data protocol"
   );

   /// the policy output carrier handed to the minimizers' recommend() step
   using policy_output_type = policy_out_type;

   ////////////////////////////
   /// Constructors         ///
   ////////////////////////////

   /// fixed default seed so runs are reproducible bit-for-bit (house determinism
   /// convention, cf. the serial-execution note in rm_utils.hpp)
   static constexpr uint64_t k_default_rng_seed = 0x9E3779B97F4A7C15ull;

   ICFR(Env env, uptr< world_state_type > root_state, uint64_t rng_seed = k_default_rng_seed)
       : m_env(std::move(env)), m_root_state(std::move(root_state)), m_rng(rng_seed)
   {
      assert_serialized_and_unrolled(m_env);
      _init_roster();
   }

   ICFR(Env env, uint64_t rng_seed = k_default_rng_seed)
   requires concepts::has::method::initial_world_state< Env >:
       ICFR(
          std::move(env),
          [&] {
             // evaluated before the base delegation moves anything out of 'env'
             return std::make_unique< world_state_type >(env.initial_world_state());
          }(),
          rng_seed
       )
   {
   }

   ICFR(const ICFR&) = delete;
   ICFR(ICFR&&) = default;
   ICFR& operator=(const ICFR&) = delete;
   ICFR& operator=(ICFR&&) noexcept = default;
   ~ICFR() = default;

   ///////////////////////////////
   /// public result types     ///
   ///////////////////////////////

   /// deviation analysis of one trigger sequence sigma = (infoset, action)
   struct TriggerGapEntry {
      /// owning player of the sequence
      Player player{};
      /// local index of the trigger infoset among the player's enumerated infosets
      size_t infoset_id = 0;
      /// the recommended action whose reception arms (triggers) the agent
      action_type action{};
      /// max over deviation plans of the triggered agent's expected utility
      double triggered_value = 0.;
      /// expected utility of obediently following the mediator
      double obedient_value = 0.;
      /// triggered_value - obedient_value (this sequence's contribution to delta(mu_bar))
      double gap = 0.;
      /// raw cumulative trigger regret R^T_sigma (paper Definition 3, via Lemma 1)
      double trigger_regret = 0.;
   };

   /// full EFCE-quality report of the current empirical frequency of play
   struct GapReport {
      /// delta(mu_bar^T): the largest trigger-agent deviation gain (paper Eq. 7);
      /// mu_bar^T is an epsilon-EFCE for every epsilon >= this quantity
      double efce_gap = 0.;
      /// max_{i,sigma} R^T_sigma / T: the Theorem-1 certificate; delta <= this + fp slack
      double trigger_regret_bound = 0.;
      /// per-sequence breakdown (every sequence of every player)
      std::vector< TriggerGapEntry > sequences{};
   };

   ///////////////////////////////////////////
   /// API: public member functions        ///
   ///////////////////////////////////////////

  public:
   /**
    * @brief runs 'n_iters' joint ICFR iterations (all players update every
    * iteration -- the paper's simultaneous scheme).
    */
   void iterate(size_t n_iters);

   /// number of completed iterations T
   [[nodiscard]] size_t iteration() const { return m_iteration; }

   /**
    * @brief the empirical frequency of play mu_bar^T as RAW TERMINAL COUNTS
    * keyed by terminal action paths. Normalize by 'iteration()' to obtain
    * probabilities (see the class documentation for the representation rationale).
    */
   [[nodiscard]] const auto& empirical_frequency_counts() const { return m_terminal_counts; }

   /**
    * @brief evaluates the quality of the current empirical frequency mu_bar^T
    * as an extensive-form correlated equilibrium.
    *
    * Assembles, per trigger sequence sigma=(J,a), BOTH sides of paper Eq. (7)
    * from the exact online co-statistics: the triggered side
    * max_{mu_hat} sum_t 1[x^t(sigma)] V^t_J(mu_hat)/T by a dynamic program over
    * the owner's infoset graph below J, and the obedient side
    * sum_t 1[x^t(sigma)] V^t_J(pi^t)/T as an accumulated scalar. No additional
    * tree traversal is required at evaluation time.
    * Also reports the Theorem-1 trigger-regret certificate
    * max_sigma R^T_sigma / T assembled from the same blocks through the
    * Lemma-1 recursion, so callers can verify delta <= certificate + fp slack.
    */
   [[nodiscard]] GapReport evaluate_efce_gap();

   /**
    * @brief the raw cumulative trigger regrets R^T_sigma of every sequence of
    * every player as of the last completed iteration (paper Definition 3,
    * assembled from the local laminar accumulators through the Lemma-1
    * recursion).
    */
   [[nodiscard]] std::vector< TriggerGapEntry > trigger_regrets() const;

   /// number of own infosets the enumeration discovered for 'player'
   /// (0 before the first iterate() call initialized the structure)
   [[nodiscard]] size_t num_infosets(Player player) const;
   /// number of legal actions at the player's infoset 'infoset_id'
   [[nodiscard]] size_t infoset_action_count(Player player, size_t infoset_id) const;
   /// number of external regret-minimizer units attached to the infoset == |Sigma^c_i(I)|
   [[nodiscard]] size_t external_unit_count(Player player, size_t infoset_id) const;
   /// the infoset's own ancestor chain, root-first: (ancestor infoset id, required action)
   [[nodiscard]] std::vector< std::pair< size_t, action_type > >
   infoset_chain(Player player, size_t infoset_id) const;
   /// the (chain position, deviating action) descriptor of an external unit
   [[nodiscard]] std::pair< size_t, action_type >
   external_unit_descriptor(Player player, size_t infoset_id, size_t unit_idx) const;

   /// distribution the internal unit of 'infoset_id' recommended during the most
   /// recent sampling phase (diagnostic instrumentation; empty before the first
   /// iterate() call)
   [[nodiscard]] const std::vector< double >&
   last_recommendation_distribution(Player player, size_t infoset_id) const;

   [[nodiscard]] const Env& env() const { return m_env; }
   [[nodiscard]] const world_state_type& root_state() const { return *m_root_state; }
   /// the root participant roster (chance excluded), in stable index order
   [[nodiscard]] const std::vector< Player >& players() const { return m_roster; }

  private:
   /////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////// private structure types
   ////////////////////////////////////////
   /////////////////////////////////////////////////////////////////////////////////////////////////

   /// one external regret-minimizer unit R^ext_{sigma,I}: sigma ranges over
   /// Sigma^c(I), encoded as (chain position of the deviating ancestor J, the
   /// plan's action index AT J)
   struct ExternalUnit {
      size_t chain_pos = 0;
      size_t action_idx = 0;
      typename ExternalRM::node_data_type rm;
   };

   /// per-infoset record of the static structure
   struct InfosetRecord {
      sptr< info_state_type > istate;
      std::vector< action_type > actions;
      /// own ancestors, root-first: (ancestor local id, required action INDEX there)
      std::vector< std::pair< size_t, size_t > > chain;
      /// C(I,a): infosets immediately following edge (I,a); aligned with actions
      std::vector< std::vector< size_t > > children;
      /// the internal-regret unit R^int_I
      typename InternalRM::node_data_type internal_rm;
      /// the external-regret units R^ext_{sigma,I}, sigma in Sigma^c(I)
      std::vector< ExternalUnit > ext_units;
   };

   /// undo record of one edge advance (see _advance_edge)
   struct EdgeUndo {
      bool flushes = false;
      Player flush_target = Player::unknown;
      sptr< info_state_type > saved_infostate{};
      std::optional< std::vector< std::pair< observation_type, observation_type > > >
         saved_flush_buffer{};
      std::vector< std::pair< Player, size_t > > saved_sizes{};
   };

   /// the complete per-player structure built by the enumeration pass
   struct PlayerStructure {
      /// per-infoset records in first-discovery (= topological) order
      std::vector< InfosetRecord > infosets;
      /// infostate value -> local id
      std::unordered_map<
         info_state_type,
         size_t,
         common::value_hasher< info_state_type >,
         common::value_comparator< info_state_type > >
         ids;
      /// prefix sums linearizing (infoset id, action idx) -> dense slot
      std::vector< size_t > action_offsets;
      /// prefix sums of external-unit counts per infoset (flat indexing)
      std::vector< size_t > ext_unit_offsets;
      // ---- raw trigger-sequence co-statistics ----
      // For every sequence sigma and every infoset I' at-or-below sigma's
      // trigger point, ICFR accumulates the exact online sums
      //    raw(sigma,I',b)  = sum_t 1[pi^t in Pi_i(sigma)] * u_hat^t_{I'}(b)
      //    obed(sigma,I',b) = sum_t 1[pi^t in Pi_i(sigma)] * u_hat^t_{I'}(pi^t(I'))
      //                       (only the b == pi^t(I') column is ever nonzero;
      //                        the block shape mirrors 'raw' verbatim)
      // Three slice families exist, mirroring which unit owns the condition
      // 1[pi^t in Pi_i(sigma)]:
      //   internal: sigma=(I,a*) rooted AT I        -> [I][a*][b] blocks
      //   external: sigma in Sigma^c(I)             -> [I][unit][b] blocks
      //   on-path:  sigma=(J, required-at-J) above I -> [I][k][b] blocks
      //            (k = chain position of J; no minimizer unit owns this case)
      // From these, BOTH sides of paper Eq. (7) -- the triggered side via the
      // dynamic program max_{pi_hat} sum_t 1[x] V(pi_hat) over 'raw', the
      // obedient side via the 'obed' scalar along the DP root -- AND the
      // Definition-3 trigger regrets R^T_sigma (per-slice differences
      // raw - obed composed through the Lemma-1 recursion) are assembled
      // exactly. Note delta(mu_bar^T) is NOT computable from terminal-path-
      // keyed frequencies alone: the deviation plan's payoffs live on
      // unrealized terminals whose opponent compatibility the realized paths
      // do not reveal below the trigger point.
      std::vector< double > raw_internal;
      std::vector< double > raw_external;
      std::vector< double > raw_onpath;
      /// STOP-value twins of the raw blocks (same shapes): they accumulate
      /// u^t_{I'}[b] instead of u_hat -- the immediate-terminal values WITHOUT
      /// continuation terms. The EFCE-gap triggered-side DP must compose THESE
      /// (the deviation plan's value is a sum of stop-values along its own
      /// path); composing u_hat there would double-count continuations.
      std::vector< double > stop_internal;
      std::vector< double > stop_external;
      std::vector< double > stop_onpath;
      std::vector< double > obed_internal;
      std::vector< double > obed_external;
      std::vector< double > obed_onpath;
      std::vector< size_t > raw_internal_offsets;
      std::vector< size_t > raw_external_offsets;
      std::vector< size_t > raw_onpath_offsets;
      // ---- per-iteration scratch ----
      std::vector< size_t > choice;  ///< sampled action index per infoset
      std::vector< char > reached;  ///< plan-reachability per infoset
      std::vector< char > ext_active;  ///< activity flag per (infoset, external unit)
      std::vector< size_t > prefix_match;  ///< #leading chain positions matched by pi^t
      std::vector< double > v_memo;  ///< V^t_J(pi^t) of the current iteration
      std::vector< double > u_stop;  ///< u^t[I,a] of the current iteration (dense)
      std::vector< double > x_sums;  ///< X(i,I,a) leaf aggregates (dense)
      std::vector< double > s_sums;  ///< S(i,J) leaf aggregates
      /// instrumentation: last recommendation distribution per infoset
      std::vector< std::vector< double > > last_recommendation;
   };

   /////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////// private member functions ///////////////////////////////////////
   /////////////////////////////////////////////////////////////////////////////////////////////////

   /// protected-style environment access used throughout the traversals (house convention)
   [[nodiscard]] Env& _env() { return m_env; }
   [[nodiscard]] const Env& _env() const { return m_env; }

   /// seeds roster and per-player index from the root participant set
   void _init_roster();

   /// lazily builds all per-player structures via one full-tree enumeration
   void _ensure_initialized();
   /// recursive worker of the enumeration pass
   void _enumerate_visit(world_state_type& state, size_t depth);
   /// post-enumeration allocation of offsets/laminar buffers/external units
   void _finalize_enumeration();

   /// Algorithm 1 SampleInternal: top-down plan sampling for one player
   void _sample_plan(size_t p_idx);
   /// draws an action index from a normalized recommendation distribution
   [[nodiscard]] size_t
   _sample_action(const policy_out_type& dist, const std::vector< action_type >& actions);

   /**
    * @brief the single full-tree counterfactual pass of iteration t.
    *
    * Computes, for all players simultaneously, the opponent-gated stop-value
    * aggregates X(i,I,a) = sum_{h in I} M_i(h*a) and S(i,J) = sum_{h in J}
    * M_i(h) -- where M_i(h) is the chance-weighted payoff mass below h gated by
    * the OPPONENTS' sampled plans only -- and accumulates mu_bar's terminal
    * counts. Results land in the per-player scratch buffers.
    */
   void _traverse_values();
   [[nodiscard]] std::vector< double > _values_visit(world_state_type& state, size_t depth);

   /// shared edge-advance mechanics of the traversals (arena slot + observation
   /// buffering + infostate flush); the caller restores via '_undo_edge'
   template < typename ActionOrOutcome >
   world_state_type& _advance_edge(
      const world_state_type& state,
      size_t depth,
      const ActionOrOutcome& edge,
      EdgeUndo& undo
   );
   /// restores everything the matching '_advance_edge' mutated
   void _undo_edge(const EdgeUndo& undo);

   /// Algorithm 1 UpdateInternal for one player: assembles u/u_hat/V on the
   /// infoset graph and feeds the active minimizer units + laminar accumulators
   void _update_player(size_t p_idx);

   /// Lemma-1 dynamic program: raw cumulative trigger regret R^T_sigma of
   /// sequence sigma=(J,a*) over the infoset subtree rooted at
   /// 'subtree_infoset' (Definition 3, assembled from the raw/obed blocks)
   [[nodiscard]] double _trigger_regret_dp(
      const PlayerStructure& st,
      size_t trigger_infoset,
      size_t trigger_action_idx,
      size_t subtree_infoset,
      std::vector< double >& memo,
      std::vector< char >& computed
   ) const;

   /// triggered-side dynamic program of the EFCE-gap evaluation:
   /// max_{pi_hat in Delta_{Pi_i(J)}} sum_t 1[x^t(sigma)] V^t_J(pi_hat) for the
   /// trigger slot sigma=(J,a*), over the infoset subtree rooted at
   /// 'subtree_infoset' -- the max term of paper Eq. (7), unnormalized by T
   [[nodiscard]] double _gap_triggered_dp(
      const PlayerStructure& st,
      size_t trigger_infoset,
      size_t trigger_action_idx,
      size_t subtree_infoset,
      std::vector< double >& memo,
      std::vector< char >& computed
   ) const;

   /////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////// private member variables
   ///////////////////////////////////////
   /////////////////////////////////////////////////////////////////////////////////////////////////

   env_type m_env;
   uptr< world_state_type > m_root_state;
   std::mt19937_64 m_rng;

   std::vector< Player > m_roster{};
   player_hashmap< size_t > m_player_index{};

   bool m_initialized = false;
   size_t m_iteration = 0;

   std::vector< PlayerStructure > m_structures{};

   // ---- shared traversal machinery (arena + bookkeeping, reused across traversals) ----
   std::deque< utils::ReusableSlot< world_state_type > > m_arena;
   player_hashmap< std::vector< std::pair< observation_type, observation_type > > > m_obs_buffers{};
   player_hashmap< sptr< info_state_type > > m_infostates{};

   // ---- value-pass scratch ----
   /// per roster index: 1 iff the player's sampled plan matches the descent path so far
   std::vector< double > m_own_match{};
   /// running product of chance probabilities along the current descent (the
   /// p_c prefix); kept explicit so counterfactual aggregates at every infoset
   /// share the paper's absolute p_c(z) normalization
   double m_chance_product = 1.;
   /// the full action path of the current descent (terminal-frequency key)
   std::vector< action_variant_type > m_path{};
   /// per roster index: (infoset id, action idx) decision stack of the descent
   std::vector< std::vector< std::pair< size_t, size_t > > > m_decision_stacks{};

   /// the correlation accumulator: realized-trajectory counts keyed by terminal path
   std::unordered_map<
      std::vector< action_variant_type >,
      double,
      TerminalPathHash< action_variant_type > >
      m_terminal_counts{};
};

}  // namespace nor::rm

// include the actual template implementations of this class
#include "icfr.tcc"

#endif  // NOR_RM_ICFR_HPP
