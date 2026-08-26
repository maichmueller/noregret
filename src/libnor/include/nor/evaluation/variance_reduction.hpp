
#ifndef NOR_EVALUATION_VARIANCE_REDUCTION_HPP
#define NOR_EVALUATION_VARIANCE_REDUCTION_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <functional>
#include <numeric>
#include <random>
#include <ranges>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/common.hpp"
#include "nor/concepts.hpp"
#include "nor/fosg_traits.hpp"
#include "nor/game_defs.hpp"
#include "nor/rm/rm_utils.hpp"
#include "nor/type_defs.hpp"
#include "nor/utils/utils.hpp"

/**
 * @file variance_reduction.hpp
 *
 * Evaluation-side (FIXED profile) variance reduction for sampled playouts:
 *
 * - MIVAT -- Martha White and Michael H. Bowling, "Learning a Value Analysis
 *   Tool for Agent Evaluation", IJCAI 2009 (pp. 1976-1981). Control variates on
 *   the chance-event randomness: per-playout rewards are corrected with terms
 *   whose expectation is zero, using value estimates that are linear in
 *   features observable at chance nodes. The linear coefficients are learned
 *   POST-HOC by variance-minimizing least squares regression on a disjoint
 *   training split of the samples (the paper's train/test protocol). The
 *   estimator is provably unbiased for ANY quality of the learned value
 *   function.
 *
 * - AIVAT -- Neil Burch, Martin Schmid, Matej Moravcik, Michael Bowling,
 *   "AIVAT: a variance reduction technique for agent evaluation in
 *   imperfect information games", AAAI 2018 (arXiv:1610.00671). Extends the
 *   control-variate idea to PLAYER actions using both players' explicit
 *   strategy probabilities: at every decision point h along an observed
 *   trajectory it adds the correction term
 *
 *       k_h = sum_m sigma(h,m) * u_h(m)  -  u_h(observed move)
 *
 *   where u_h(m) are (possibly heuristic) values of ALL successor states h.m --
 *   including the ones not taken ("imaginary observations") -- and sigma is
 *   the true next-move distribution of the fixed profile. With the singleton
 *   terminal-state partition of the paper's base-value construction (valid
 *   since P_a = P here) the base value is the observed reward itself, so that
 *   with exact successor values every correction telescopes to the exact root
 *   value and the estimator attains ZERO variance, while for arbitrary u_h
 *   each k_h has conditional mean zero (paper's Lemma 1 / Theorem 1) hence the
 *   estimator stays EXACTLY unbiased.
 */

namespace nor::evaluation {

/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// diagnostic types /////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

/// empirical variance comparison of one estimator against plain raw sampling
struct VarianceReport {
   /// sample variance (ddof=1) of the raw terminal rewards over the evaluated playouts
   double raw_variance = 0.;
   /// sample variance (ddof=1) of the reduced per-playout estimates
   double reduced_variance = 0.;
   [[nodiscard]] double variance_ratio() const
   {
      return raw_variance > 0. ? reduced_variance / raw_variance : 1.;
   }
};

/// per-player outcome of running one estimator over a batch of playouts
struct PlayerEvaluation {
   /// unbiased point estimate of E[u_player | profile]
   double estimate = 0.;
   /// plain Monte Carlo sample mean over the very same playouts
   double raw_mean = 0.;
   VarianceReport variance{};
};

using EvaluationResultMap = player_hashmap< PlayerEvaluation >;

/////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// game tree /////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief full game-tree enumeration under a FIXED policy profile.
 *
 * Builds the entire tree once from the root state: every node records its
 * legal moves together with their probabilities under the profile (true chance
 * distribution at chance nodes, explicit strategy probabilities at player
 * nodes), the exact expected values under the profile (backward pass), and --
 * at chance nodes -- global control-variate feature slots (one indicator slot
 * per (chance node, outcome) pair) together with their exact means under the
 * profile. The latter two feed the MIVAT regression machinery; the exact
 * values serve both as ground truth and as default AIVAT successor-value
 * function.
 */
template <
   typename Env,
   typename Policy,
   typename ActionPolicy = auto_action_policy_type< Policy > >
   requires concepts::fosg< std::remove_cvref_t< Env > > and (concepts::state_policy_no_default< Policy, auto_info_state_type< std::remove_cvref_t< Env > >, auto_action_type< std::remove_cvref_t< Env > >, ActionPolicy > or concepts::state_policy_view< Policy, auto_info_state_type< std::remove_cvref_t< Env > >, auto_action_type< std::remove_cvref_t< Env > > >)
class ProfileGameTree {
   using env_type = std::remove_cvref_t< Env >;

  public:
   using world_state_type = auto_world_state_type< env_type >;
   using info_state_type = auto_info_state_type< env_type >;
   using action_type = auto_action_type< env_type >;
   using observation_type = auto_observation_type< env_type >;
   using chance_outcome_type = auto_chance_outcome_type< env_type >;
   using action_variant_type = auto_action_variant_type< env_type >;

   /// one outgoing edge of a node: the move itself, its probability under the
   /// profile, the child node index, and (chance moves only) the global MIVAT
   /// feature slot assigned to this specific (node, outcome) combination
   struct Move {
      action_variant_type move{};
      double probability = 0.;
      size_t child = 0;
      size_t feature_id = 0;
   };

   struct Node {
      Player active_player = Player::chance;
      bool is_terminal = false;
      /// infostate of the acting player at this node (only meaningful at
      /// PLAYER decision nodes; chance/terminal nodes carry the sentinel)
      info_state_type infostate{Player::chance};
      std::vector< Move > moves{};
      /// terminal nodes only: rewards of all actual participants
      rm::PlayerValueTable rewards{};
      /// exact expected value of every participant when the profile plays from here
      rm::PlayerValueTable values{};
      /// product of profile probabilities along the path from the root
      double reach_probability = 1.;
   };

   ProfileGameTree(
      const Env& env,
      const world_state_type& root_state,
      const player_hashmap< Policy >& profile
   );

   /// the roster of actual players (chance excluded), in canonical order
   [[nodiscard]] const std::vector< Player >& players() const { return m_players; }
   [[nodiscard]] size_t player_index(Player player) const
   {
      auto found = std::find(m_players.begin(), m_players.end(), player);
      assert(found != m_players.end());
      return static_cast< size_t >(found - m_players.begin());
   }
   [[nodiscard]] const Node& node(size_t idx) const { return m_nodes[idx]; }
   [[nodiscard]] const Node& root() const { return m_nodes.front(); }
   [[nodiscard]] size_t node_count() const { return m_nodes.size(); }
   /// dimension of the MIVAT chance-event feature space
   [[nodiscard]] size_t feature_count() const { return m_feature_means.size(); }
   /// exact mean of each chance-event indicator feature under the profile
   [[nodiscard]] const std::vector< double >& feature_means() const { return m_feature_means; }
   /// exact profile value at the root per player (backward pass result)
   [[nodiscard]] const rm::PlayerValueTable& root_values() const { return m_nodes.front().values; }

  private:
   size_t _build(
      const env_type& env,
      uptr< world_state_type > state,
      double reach_probability,
      player_hashmap< std::vector< std::pair< observation_type, observation_type > > >
         observation_buffer,
      player_hashmap< info_state_type > infostate_map,
      const player_hashmap< Policy >& profile
   );
   void _accumulate_values(size_t node_idx);

   std::vector< Player > m_root_roster;
   std::vector< Player > m_players;
   std::vector< Node > m_nodes;
   std::vector< double > m_feature_means;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// sampler /////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

/// record of one randomly sampled playout through the profile tree
struct Playout {
   /// decision steps as (node index, chosen move index) pairs, including chance steps
   std::vector< std::pair< uint32_t, uint32_t > > path;
   /// terminal node index
   uint32_t terminal_node = 0;
};

template < typename Env, typename Policy >
class PlayoutSampler {
  public:
   using tree_type = ProfileGameTree< Env, Policy >;

   PlayoutSampler(const tree_type& tree, size_t seed = common::default_seed)
       : m_tree(tree), m_rng(common::create_rng(seed))
   {
   }

   void reseed(size_t seed) { m_rng = common::create_rng(seed); }

   /// draws n independent playouts (the internal RNG stream advances across calls)
   std::vector< Playout > sample(size_t n)
   {
      std::vector< Playout > out;
      out.reserve(n);
      for([[maybe_unused]] size_t i : std::views::iota(size_t{0}, n)) {
         out.push_back(_sample_one());
      }
      return out;
   }

  private:
   Playout _sample_one();
   size_t _draw_move(const typename tree_type::Node& node);

   const tree_type& m_tree;
   common::RNG m_rng;
   std::uniform_real_distribution< double > m_uniform_01{0., 1.};
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// estimators //////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

/// Welford accumulator for streaming mean/variance
class OnlineStats {
  public:
   void push(double x)
   {
      ++m_n;
      const double delta = x - m_mean;
      m_mean += delta / static_cast< double >(m_n);
      m_m2 += delta * (x - m_mean);
   }
   [[nodiscard]] size_t count() const { return m_n; }
   [[nodiscard]] double mean() const { return m_mean; }
   /// unbiased sample variance (0 for fewer than 2 observations)
   [[nodiscard]] double variance() const
   {
      return m_n > 1 ? m_m2 / static_cast< double >(m_n - 1) : 0.;
   }

  private:
   size_t m_n = 0;
   double m_mean = 0.;
   double m_m2 = 0.;
};

/**
 * @brief solves the linear system 'A x = b' (A square row-major) via Gaussian
 * elimination with partial pivoting.
 */
inline std::vector< double > solve_linear_system(std::vector< double > A, std::vector< double > b)
{
   const size_t n = b.size();
   for(size_t col : std::views::iota(size_t{0}, n)) {
      // partial pivot
      size_t pivot = col;
      for(size_t row : std::views::iota(col + 1, n)) {
         if(std::abs(A[row * n + col]) > std::abs(A[pivot * n + col])) {
            pivot = row;
         }
      }
      if(std::abs(A[pivot * n + col]) < 1e-300) {
         throw std::runtime_error("variance reduction: singular normal equations");
      }
      if(pivot != col) {
         for(size_t j : std::views::iota(size_t{0}, n)) {
            std::swap(A[col * n + j], A[pivot * n + j]);
         }
         std::swap(b[col], b[pivot]);
      }
      // eliminate below the diagonal
      for(size_t row : std::views::iota(col + 1, n)) {
         const double factor = A[row * n + col] / A[col * n + col];
         if(factor == 0.) {
            continue;
         }
         for(size_t j = col; j < n; ++j) {
            A[row * n + j] -= factor * A[col * n + j];
         }
         b[row] -= factor * b[col];
      }
   }
   // back substitution
   std::vector< double > x(n, 0.);
   for(size_t i = n; i-- > 0;) {
      double sum = b[i];
      for(size_t j = i + 1; j < n; ++j) {
         sum -= A[i * n + j] * x[j];
      }
      x[i] = sum / A[i * n + i];
   }
   return x;
}

}  // namespace detail

/**
 * @brief MIVAT estimator (White & Bowling, IJCAI 2009).
 *
 * Chance-event control variates with post-hoc learned linear value functions.
 * Each run draws 'n_playouts' playouts, fits the regression coefficients on
 * the FIRST half (training split) by least squares of the observed rewards
 * onto the centered chance-event indicator features, and evaluates on the
 * SECOND half with the frozen coefficients. Because the coefficients are
 * independent of the evaluation randomness and the exact feature means are
 * known from the full-tree enumeration, the resulting estimate remains exactly
 * unbiased regardless of the fit quality (even degenerate features), while
 * good features yield substantial variance reduction.
 */
template < typename Env, typename Policy >
class MivatEstimator {
  public:
   using tree_type = ProfileGameTree< Env, Policy >;

   MivatEstimator(const tree_type& tree, size_t seed = common::default_seed)
       : m_tree(tree), m_sampler(tree, seed)
   {
   }

   void reset()
   {
      m_results.clear();
      m_playout_estimates.clear();
   }

   void reseed(size_t seed) { m_sampler.reseed(seed); }

   /**
    * @brief samples 'n_playouts' fresh playouts, fits the regression on the
    * first half and evaluates on the second half. Diagnostics cover the
    * evaluation half only.
    */
   void run(size_t n_playouts);

   [[nodiscard]] const EvaluationResultMap& results() const { return m_results; }
   /// per-playout reduced estimates of the last run, indexed like tree().players()
   [[nodiscard]] std::span< const std::vector< double > > playout_estimates() const
   {
      return m_playout_estimates;
   }
   [[nodiscard]] const tree_type& tree() const { return m_tree; }

  private:
   const tree_type& m_tree;
   PlayoutSampler< Env, Policy > m_sampler;
   EvaluationResultMap m_results;
   std::vector< std::vector< double > > m_playout_estimates;
};

/**
 * @brief AIVAT estimator (Burch, Schmid, Moravcik & Bowling, AAAI 2018),
 * full-knowledge instance (P_a = P): the explicit strategies of all players
 * are known.
 *
 * Per sampled playout z with terminal reward u_p(z) of the evaluated player p
 * the estimate is
 *
 *     AIVAT_p(z) = u_p(z)
 *        + sum_{(h,o) in path(z)} [ sum_m sigma(h,m) * u_p(h.m) - u_p(h.o) ]
 *
 * with u_p(h.m) read from a (possibly heuristic) successor-value function --
 * by default the exact profile-continuation values computed by the tree's
 * backward pass. Each bracketed term has conditional mean zero given reaching
 * h (paper Lemma 1), hence the estimator is exactly unbiased for arbitrary
 * heuristics; with exact values the corrections telescope and the estimator
 * has zero variance.
 */
template < typename Env, typename Policy >
class AivatEstimator {
  public:
   using tree_type = ProfileGameTree< Env, Policy >;

   /// user-injectable successor-value heuristic u(player, node_idx, move_idx)
   /// -> value estimate for 'player' of the child reached by the move. If
   /// unset, the exact tree values are used.
   using HeuristicFn = std::function< double(Player, size_t, size_t) >;

   AivatEstimator(const tree_type& tree, size_t seed = common::default_seed)
       : AivatEstimator(tree, seed, nullptr)
   {
   }
   AivatEstimator(const tree_type& tree, size_t seed, HeuristicFn heuristic)
       : m_tree(tree), m_heuristic(std::move(heuristic)), m_sampler(tree, seed)
   {
   }

   void reset()
   {
      m_results.clear();
      m_playout_estimates.clear();
   }

   void reseed(size_t seed) { m_sampler.reseed(seed); }

   void run(size_t n_playouts);

   [[nodiscard]] const EvaluationResultMap& results() const { return m_results; }
   [[nodiscard]] std::span< const std::vector< double > > playout_estimates() const
   {
      return m_playout_estimates;
   }
   [[nodiscard]] const tree_type& tree() const { return m_tree; }

  private:
   double _successor_value(Player player, size_t node_idx, size_t move_idx) const
   {
      if(m_heuristic) {
         return m_heuristic(player, node_idx, move_idx);
      }
      return m_tree.node(m_tree.node(node_idx).moves[move_idx].child).values.at(player);
   }

   const tree_type& m_tree;
   HeuristicFn m_heuristic;
   PlayoutSampler< Env, Policy > m_sampler;
   EvaluationResultMap m_results;
   std::vector< std::vector< double > > m_playout_estimates;
};

}  // namespace nor::evaluation

// include the actual template definitions
#ifndef NOR_EVALUATION_VARIANCE_REDUCTION_TCC_INCLUDED
   #define NOR_EVALUATION_VARIANCE_REDUCTION_TCC_INCLUDED
   #include "variance_reduction.tcc"
#endif

#endif  // NOR_EVALUATION_VARIANCE_REDUCTION_HPP
