
#ifndef NOR_RM_TEAM_DAG_CFR_HPP
#define NOR_RM_TEAM_DAG_CFR_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/common.hpp"
#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "nor/policy/action_policy.hpp"
#include "nor/rm/team/team_belief_dag.hpp"

namespace nor::rm::team {

/**
 * @brief DAG-CFR over Team Belief DAGs (Zhang, Farina & Sandholm, ICML 2022,
 * arXiv:2202.00789 -- paper Theorem 4.3 / Appendix A Algorithm 2).
 *
 * Solves an adversarial TEAM game, i.e. a coalition ('team members') coordinating ex ante but
 * not during play against the remaining participants of the root roster (the 'adversary
 * block'), by running vanilla counterfactual regret minimization ON the TB-DAG of BOTH sides:
 *
 *  - Both coalitions instantiate their own TeamBeliefDAG decision problem. For a single-agent
 *    adversary the construction degenerates gracefully (beliefs collapse across worlds the
 *    agent itself cannot distinguish), so one uniform learner covers team-vs-team and
 *    team-vs-player alike.
 *  - Each iteration runs Algorithm 2 verbatim per side: a TOP-DOWN pass computes the current
 *    flow (x[s] = sum over parents; RM+ recommendations split the flow over prescriptions)
 *    while accumulating the linearly-weighted average plan, followed by a BOTTOM-UP utility
 *    backpropagation R[s,a] += u[s,a] - u[s] over terminal weights g[z] that already carry the
 *    opposing side's current realization r_q(z): g_side(z) = p_c(z) * u_side(z) * prod_{q !=
 *    side} r_q(z). Because both DAGs exchange ONLY terminal-flow scalars, one iteration costs
 *    O(E_team + E_adversary) -- the complexity promised by Theorem 4.3.
 *  - Regret kernel: house RM+ (CFR+) arithmetic, bit-equivalent to rm::RegretMatchingPlus;
 *    averaging is LINEAR in the iteration index (LCFR convention; the paper additionally tried
 *    quadratic/Danger-discounted variants).
 *
 * GUARANTEE SCOPE. Under the paper's finite alternating-DAG assumptions, each side is a
 * CFR-style regret minimizer over its realization polytope (Corollary A.4: regret O(N sqrt(T))).
 * The learner therefore targets the correlated team max-min problem (TMECor). The extracted
 * `decentralized_policies` view is only a marginal projection and is not a TME solver: sampling
 * those member policies independently can lose the coordinator's correlation payoff.
 *
 * EVALUATION (`evaluate()`). Three co-reported instruments, mirroring the suite conventions of
 * e.g. ICFR:
 *   - average_pair_value: E(x_bar, y_bar) under normalized average flows of both sides;
 *   - regret_certificate_per_side/saddle_gap_proxy: (sum of positive cumulative regrets)/T per
 *     plane, whose half-sum upper-bounds the achieved exploitability up to constants (folklore
 *     bound eq. (6) of the paper applied to the two planes);
 *   - `br_surrogate_value`: an EMPIRICAL lower bracket on the opponent's best-response value
 *     against the frozen team average plan x_bar, obtained by re-training ONLY the queried
 *     plane for a few thousand extra iterations against the fixed opposing realization vector
 *     (rolling the kernels back afterwards, i.e. non-destructive). A true closed-form BR would
 *     require decomposing DAG realizations onto individual worlds -- implemented neither here
 *     nor needed below the certification thresholds asserted in the test-suite; see also the
 *     caveat note in `Evaluation`.
 */
template < typename Env >
class AdversarialTeamDagCfr {
  public:
   using dag_type = TeamBeliefDAG< Env >;
   using action_type = auto_action_type< Env >;
   using info_state_type = auto_info_state_type< Env >;

   using NodeId = rm::team::NodeId;
   using BeliefId = rm::team::BeliefId;
   using InactiveId = rm::team::InactiveId;

   static_assert(
      concepts::fosg< Env >,
      "AdversarialTeamDagCfr requires the environment to fulfill the fosg concept"
   );

   static constexpr size_t k_team_plane = 0;
   static constexpr size_t k_adversary_plane = 1;

   struct Config {
      /// the cooperating team members; every remaining root-roster seat joins the adversary
      std::vector< Player > team_members{};
      /// averaging weight exponent: weight(iteration) = pow(iteration + 1, linear_weight_power)
      double linear_weight_power = 1.;
      /// exact-construction bound applied independently to both team-belief DAGs
      size_t max_dag_nodes = dag_type::k_default_max_dag_nodes;
   };

   ////////////////////////////////////////
   /// Constructors                    ///
   ////////////////////////////////////////

   AdversarialTeamDagCfr(Env env, Config config)
      requires concepts::has::method::initial_world_state< Env >
       : AdversarialTeamDagCfr(
          std::move(env),
          std::make_unique< auto_world_state_type< Env > >(env.initial_world_state()),
          std::move(config)
       )
   {
   }

   AdversarialTeamDagCfr(Env env, uptr< auto_world_state_type< Env > > root_state, Config config)
       : m_env(std::move(env)), m_config(std::move(config))
   {
      if(not root_state) {
         throw std::invalid_argument("AdversarialTeamDagCfr: root_state must not be null");
      }
      if(m_config.team_members.empty()) {
         throw std::invalid_argument("AdversarialTeamDagCfr: empty team");
      }
      if(not std::isfinite(m_config.linear_weight_power)) {
         throw std::invalid_argument("AdversarialTeamDagCfr: linear_weight_power must be finite");
      }
      if(m_config.max_dag_nodes == 0) {
         throw std::invalid_argument("AdversarialTeamDagCfr: max_dag_nodes must be positive");
      }
      for(auto [index, member] : std::views::enumerate(m_config.team_members)) {
         if(std::ranges::find(
               m_config.team_members.begin(), m_config.team_members.begin() + index, member
            )
            != m_config.team_members.begin() + index) {
            throw std::invalid_argument("AdversarialTeamDagCfr: team_members must be unique");
         }
      }

      // derive the adversary block from the root roster
      std::vector< Player > all_players;
      std::vector< Player > root_roster;
      for(auto player : m_env.players(*root_state)) {
         if(player == Player::chance) {
            continue;
         }
         root_roster.emplace_back(player);
      }
      for(auto member : m_config.team_members) {
         if(std::ranges::find(root_roster, member) == root_roster.end()) {
            throw std::invalid_argument(
               "AdversarialTeamDagCfr: configured team member is not in the root roster"
            );
         }
      }
      for(auto player : root_roster) {
         bool is_team = false;
         for(auto member : m_config.team_members) {
            if(member == player) {
               is_team = true;
               break;
            }
         }
         if(not is_team) {
            all_players.emplace_back(player);
         }
      }
      if(all_players.empty()) {
         throw std::invalid_argument(
            "AdversarialTeamDagCfr: every root participant belongs to the team -- no adversary"
         );
      }
      // NOTE: both planes enumerate the SAME underlying game tree, so both DAGs carry
      // identical leaf payoff rows; leaf chance weights and utilities are shared through
      // m_leaf_utilities computed from the team plane below.
      auto adversary_root = std::make_unique< auto_world_state_type< Env > >(*root_state);
      typename dag_type::Config team_cfg{};
      team_cfg.members = m_config.team_members;
      team_cfg.max_dag_nodes = m_config.max_dag_nodes;
      m_planes.emplace_back(dag_type(m_env, std::move(root_state), std::move(team_cfg)));
      typename dag_type::Config adversary_cfg{};
      adversary_cfg.members = std::move(all_players);
      adversary_cfg.max_dag_nodes = m_config.max_dag_nodes;
      m_planes.emplace_back(dag_type(m_env, std::move(adversary_root), std::move(adversary_cfg)));
      m_leaf_utilities[k_team_plane] = _coalition_leaf_utility(m_config.team_members);
      m_leaf_utilities[k_adversary_plane] = _coalition_leaf_utility(
         std::vector< Player >(m_planes[k_adversary_plane].members())
      );
      m_rt[k_team_plane].allocate(m_planes[k_team_plane]);
      m_rt[k_adversary_plane].allocate(m_planes[k_adversary_plane]);
   }

   AdversarialTeamDagCfr(const AdversarialTeamDagCfr&) = delete;
   AdversarialTeamDagCfr& operator=(const AdversarialTeamDagCfr&) = delete;
   AdversarialTeamDagCfr(AdversarialTeamDagCfr&&) = default;
   AdversarialTeamDagCfr& operator=(AdversarialTeamDagCfr&&) = default;
   ~AdversarialTeamDagCfr() = default;

   /////////////////////////////
   /// API: learner          ///
   /////////////////////////////

   void iterate(size_t n_iters);

   [[nodiscard]] size_t iteration() const { return m_iteration; }

   [[nodiscard]] const dag_type& dag(size_t side) const { return m_planes.at(side); }

   /////////////////////////////
   /// API: evaluation       ///
   /////////////////////////////

   struct Evaluation {
      /// iterations trained so far
      size_t iterations = 0;
      /// each coalition's own summed roster payoff under both average plans
      std::array< double, 2 > average_pair_values{};
      /// sum of POSITIVE cumulative regret entries divided by T, per plane (valid CFR-style
      /// average-exploitability certificates up to standard constants)
      std::array< double, 2 > regret_certificates{};
      /// naive duality-gap proxy: regret_certificates[0] + regret_certificates[1]
      double saddle_gap_proxy = 0.;
      /**
       * CAVEAT: 'br_surrogate_values[i]' brackets max_{q != i} E(x_bar_i, q) from BELOW (see
       * class documentation); it is NOT an exact best-response value.
       */
      std::array< double, 2 > br_surrogate_values{};
   };

   [[nodiscard]] Evaluation evaluate();

   /**
    * @brief non-destructive empirical best-response bracket (see Evaluation::br_surrogate_*):
    * retrains ONLY plane 'side' against the frozen opposing average realization vector for
    * 'replay_rounds' further iterations while tracking the best attained pairing value; all
    * kernel state is rolled back afterwards.
    */
   [[nodiscard]] double br_surrogate_value(size_t side, size_t replay_rounds = 2000);

   /////////////////////////////
   /// API: plan extraction  ///
   /////////////////////////////

   /**
    * @brief DECENTRALIZED policy view of plane 'side''s average correlation plan (the
    * coordinator-prescription structure realized as plain per-member behavioral policies).
    *
    * The returned table maps, per member of the plane, its canonical infostate objects to the
    * marginal action distributions obtained by sweeping the average prescription mass over all
    * beliefs. Keys are unique by construction (identical observation histories share one
    * entry), so identical-observation infostates necessarily carry IDENTICAL distributions --
    * the decentralization invariant.
    *
    * NOTE this marginal projection does NOT reproduce the correlation payoff in general (two
    * members' marginals drop coplanar coupling); use it for execution/decentralization
    * purposes, and `average_pair_values`/flows for valuation.
    */
   [[nodiscard]] player_hashmap<
      dense_hashmap< info_state_type, HashmapActionPolicy< action_type > > >
   decentralized_policies(size_t side) const;

   /**
    * @brief the coordinator program of plane 'side': per belief, the normalized average
    * prescription distribution (the correlated object retained beyond the marginal view).
    */
   [[nodiscard]] std::vector< std::vector< double > > coordinator_plan(size_t side) const;

   /// average realization weights of plane 'side' at its terminal beliefs (indexed like
   /// 'dag(side).terminal_weights()'). These are realization-form masses, not a leaf probability
   /// distribution; chance probabilities are carried by the terminal utility weights.
   [[nodiscard]] std::vector< double > average_realizations(size_t side) const;

  private:
   ///////////////////////////
   /// runtime structures  ///
   ///////////////////////////

   struct PlaneRuntime {
      /// cumulative counterfactual regrets, per belief / prescription (RM+ tables)
      std::vector< std::vector< double > > regret;
      /// cached current recommendations x'[B, .], refreshed top-down every iteration
      std::vector< std::vector< double > > current_edgep;
      /// current inflow x[B] of the iteration's plan
      std::vector< double > inflow;
      /// unnormalized average-strategy accumulators sum_t w(t) * x[B] * x'[B, .]
      std::vector< std::vector< double > > strategy_sum;
      /// iteration scratch: routed masses on the inactive layer
      std::vector< double > inactive_inflow;
      /// evaluation scratch: bottom-up value buffers of ObserveUtility
      std::vector< double > ubuf_belief;
      std::vector< double > ubuf_inactive;

      void allocate(const dag_type& dag)
      {
         const auto nb = dag.belief_count();
         regret.assign(nb, {});
         current_edgep.assign(nb, {});
         inflow.assign(nb, 0.);
         strategy_sum.assign(nb, {});
         for(auto b : std::views::iota(size_t{0}, nb)) {
            const auto k = dag.belief(b).prescriptions.size();
            regret[b].assign(k, 0.);
            current_edgep[b].assign(k, 0.);
            strategy_sum[b].assign(k, 0.);
         }
         inactive_inflow.assign(dag.inactive_count(), 0.);
         ubuf_belief.assign(nb, 0.);
         ubuf_inactive.assign(dag.inactive_count(), 0.);
      }
   };

   ///////////////////////////
   /// internals           ///
   ///////////////////////////

   /// fills runtime[side]'s current_edgep/inflow via the top-down Algorithm-2 pass
   void _next_strategy(size_t side);
   /// bottom-up ObserveUtility pass given terminal weights 'g' (leaf-indexed)
   void _observe_utility(size_t side, const std::vector< double >& g);
   /// recomputes normalized average edges/flows; returns them WITHOUT mutating learner state
   std::pair< std::vector< std::vector< double > >, std::vector< double > > _average_plan(
      size_t side
   ) const;
   /// derives a topological order from the explicit belief -> inactive -> belief edges
   [[nodiscard]] std::vector< BeliefId > _topological_beliefs(size_t side) const;
   /// forward-propagates a seeded belief flow through the explicit DAG constraints
   [[nodiscard]] std::vector< double > _belief_flows(
      size_t side,
      const std::vector< double >& inflow,
      const std::vector< std::vector< double > >* edgep_override = nullptr
   ) const;
   /// forward-propagates belief inflow (root-seeded) through edge probabilities and returns the
   /// singleton-terminal-belief realization masses aligned with the leaf rows
   [[nodiscard]] std::vector< double > _terminal_flows(
      size_t side,
      const std::vector< double >& inflow,
      const std::vector< std::vector< double > >* edgep_override = nullptr
   ) const;
   /// chance-weighted payoff column of 'coalition' aligned with the (shared) leaf tables
   [[nodiscard]] std::vector< double > _coalition_leaf_utility(
      const std::vector< Player >& coalition
   ) const;

   Env m_env;
   Config m_config{};
   std::vector< dag_type > m_planes{};  //< [0]=team, [1]=adversary
   std::array< PlaneRuntime, 2 > m_rt{};
   /// per plane: leaf-indexed g-utility column p_c(z) * u_coalition(z) used inside the
   /// bottom-up backpropagation and value evaluations (plane 1's column is derived from
   /// m_leaf_utilities[0] by coalition membership at evaluate time if ever needed)
   std::array< std::vector< double >, 2 > m_leaf_utilities{};
   size_t m_iteration = 0;
};

}  // namespace nor::rm::team

// include the actual template implementations of this class
#include "dag_cfr.tcc"

#endif  // NOR_RM_TEAM_DAG_CFR_HPP
