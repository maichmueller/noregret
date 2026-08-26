
#ifndef NOR_RM_CORRELATED_CFR_JR_HPP
#define NOR_RM_CORRELATED_CFR_JR_HPP

#include <algorithm>
#include <cstdint>
#include <ranges>
#include <vector>

#include "nor/rm/cfr_tabular/cfr.hpp"
#include "nor/rm/correlated/joint_distribution.hpp"
#include "nor/rm/correlated/sequence_form.hpp"

namespace nor::rm::correlated {

/**
 * @brief CFR-Jr: CFR with joint distribution reconstruction (Celli, Marchesi, Bianchi &
 *        Gatti, NeurIPS 2019, arXiv:1910.06228, Algorithm 3).
 *
 * Wraps the repo's vanilla CFR kernel and adds ONE step per iteration: every player's
 * currently PLAYED behavioral strategy pi^t_i -- exactly the recommendation table the
 * kernel is about to traverse with -- is reconstructed into its realization-equivalent
 * normal-form strategy x^t_i over reduced plans (Algorithm 2), and the PRODUCT
 * distribution x^t = (x^t_1 x ... x^t_n) is accumulated into a sparse joint average.
 * The returned x̄^T = (1/T) sum_t x^t is an epsilon-CCE whenever every player's played
 * sequence satisfies the external-regret bound T^-1 R_i^T <= epsilon (Theorem 5 op.
 * cit.); the kernel's own behavioral state is left bit-for-bit untouched.
 *
 * NOTE on update modes: both simultaneous and alternating kernels are supported. The
 * epsilon-CCE guarantee is INHERITED from the wrapped kernel's regret decay: on the
 * deterministic bed games (Shapley, centipede) and on goofspiel the measured CCE gap of
 * x̄^T decays as Theorem 5 predicts, while on kuhn poker this repo's vanilla iterates
 * plateau at a constant plan-space regret rate for BOTH schedules (verified down to
 * machine-level against an independent brute-force evaluator; see test_cfr_jr.cpp).
 * The wrapper faithfully reconstructs whatever the kernel plays; it cannot manufacture
 * regret decay the kernel does not provide.
 */
template < rm::CFRConfig cfg, typename Env, typename Policy, typename AveragePolicy >
class CFRJr {
  public:
   static_assert(
      cfg.weighting_mode == CFRWeightingMode::uniform,
      "CFR-Jr requires uniform (vanilla) averaging."
   );
   static_assert(
      cfg.regret_minimizing_mode == RegretMinimizingMode::regret_matching,
      "CFR-Jr wraps VANILLA CFR (local regret matching) per Celli et al. 2019."
   );
   static_assert(
      cfg.pruning_mode == CFRPruningMode::none and cfg.lazy_update_mode == CFRLazyUpdateMode::off
         and cfg.warm_start_iterations == 0,
      "CFR-Jr wraps the unmodified full-traversal vanilla CFR kernel; pruning, lazy "
      "updates and warm starts are not composed here."
   );

   using env_type = Env;
   using world_state_type = auto_world_state_type< Env >;
   using policy_type = Policy;
   using average_policy_type = AveragePolicy;
   using oracle_type = SequenceFormOracle< Env >;
   using solver_type = VanillaCFR< cfg, Env, Policy, AveragePolicy >;

   /// activity counters of the reconstruction engine
   struct Stats {
      /// plan-tuples added to the average by the last iteration
      size_t last_product_support = 0;
      /// largest per-iteration tuple count observed so far
      size_t max_product_support = 0;
      /// total accumulated support of the averaged joint distribution
      size_t average_support = 0;
   };

   CFRJr(
      Env env,
      uptr< world_state_type > root_state,
      const Policy& policy,
      const AveragePolicy& avg_policy
   )
       : solver_(std::move(env), std::move(root_state), policy, avg_policy),
         oracle_(solver_.env(), solver_.root_state())
   {
      average_joint_.reserve(1024);
   }

   ////////////////////////////
   /// iteration            ///
   ////////////////////////////

   void iterate(size_t n_iters)
   {
      auto behavioral = detail::make_behavioral_query(solver_, oracle_);
      for([[maybe_unused]] auto _ : std::views::iota(size_t{0}, n_iters)) {
         // reconstruct each player's realization-equivalent normal-form strategy from
         // the behavioral strategy this CFR iteration is about to play
         std::vector< typename oracle_type::NormalFormStrategy > reconstructed;
         reconstructed.reserve(oracle_.player_count());
         for(const Player player : oracle_.players()) {
            reconstructed.push_back(oracle_.reconstruct(player, behavioral));
         }
         if(std::ranges::any_of(reconstructed, [](const auto& strategy) {
               return strategy.empty();
            })) {
            throw std::runtime_error("CFRJr: empty reconstruction of a normal-form strategy");
         }
         // accumulate the product distribution over joint plan tuples
         const size_t n_players = reconstructed.size();
         JointDistribution::key_type key(n_players);
         std::vector< size_t > odometer(n_players, 0);
         auto refresh_product = [&] {
            double mass = 1.;
            for(auto j : std::views::iota(size_t{0}, n_players)) {
               const auto& [plan, plan_mass] = reconstructed.at(j).at(odometer.at(j));
               key.at(j) = plan;
               mass *= plan_mass;
            }
            return mass;
         };
         average_joint_.begin_iteration();
         size_t tuples = 0;
         while(true) {
            average_joint_.accumulate(key, refresh_product());
            ++tuples;
            // advance the odometer to the next combination
            size_t digit = 0;
            for(; digit < n_players; ++digit) {
               if(++odometer.at(digit) < reconstructed.at(digit).size()) {
                  break;
               }
               odometer.at(digit) = 0;
            }
            if(digit == n_players) {
               break;
            }
         }
         m_stats.last_product_support = tuples;
         m_stats.max_product_support = std::max(m_stats.max_product_support, tuples);
         m_stats.average_support = average_joint_.support_size();

         // advance the behavioral CFR kernel (regrets + recommendations)
         solver_.iterate(1);
      }
   }

   ////////////////////////////
   /// accessors            ///
   ////////////////////////////

   [[nodiscard]] CCEMetrics metrics() const { return evaluate_cce(oracle_, average_joint_); }

   [[nodiscard]] const JointDistribution& average_joint() const { return average_joint_; }
   [[nodiscard]] const oracle_type& oracle() const { return oracle_; }
   [[nodiscard]] solver_type& cfr() { return solver_; }
   [[nodiscard]] const solver_type& cfr() const { return solver_; }
   /// completed CFR-Jr iterations
   [[nodiscard]] size_t iteration() const { return average_joint_.draws(); }
   [[nodiscard]] const Stats& stats() const { return m_stats; }

  private:
   solver_type solver_;
   oracle_type oracle_;
   JointDistribution average_joint_{};
   Stats m_stats{};
};

}  // namespace nor::rm::correlated

#endif  // NOR_RM_CORRELATED_CFR_JR_HPP
