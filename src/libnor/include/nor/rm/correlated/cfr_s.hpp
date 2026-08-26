
#ifndef NOR_RM_CORRELATED_CFR_S_HPP
#define NOR_RM_CORRELATED_CFR_S_HPP

#include <cstdint>
#include <random>
#include <ranges>
#include <vector>

#include "common/common.hpp"
#include "nor/rm/cfr_tabular/cfr.hpp"
#include "nor/rm/correlated/joint_distribution.hpp"
#include "nor/rm/correlated/sequence_form.hpp"

namespace nor::rm::correlated {

/**
 * @brief CFR-S: CFR with sampling (Celli, Marchesi, Bianchi & Gatti, NeurIPS 2019,
 *        arXiv:1910.06228, section 4.2 and Algorithm 1).
 *
 * The naive sibling of CFR-Jr: instead of reconstructing realization-equivalent
 * normal-form strategies, every player SAMPLES one pure plan per iteration from their
 * current behavioral strategy -- each infoset independently, so plan sigma is drawn
 * with probability prod_I pi_I(sigma(I)), exactly the paper's RECOMMEND rule (section
 * 4.2, Algorithm 1) -- and the empirical frequency of the sampled JOINT plans is
 * tracked. The empirical frequency of play converges almost surely to a CCE
 * (Theorem 3 op. cit.) but only stochastically at an O(T^-1/2) rate, hence the looser
 * convergence behaviour compared to CFR-Jr. As with CFR-Jr the guarantee is inherited
 * from the wrapped kernel's played-sequence regret decay; see the CFRJr notes.
 */
template < rm::CFRConfig cfg, typename Env, typename Policy, typename AveragePolicy >
class CFRS {
  public:
   static_assert(
      cfg.weighting_mode == CFRWeightingMode::uniform,
      "CFR-S requires uniform (vanilla) averaging."
   );
   static_assert(
      cfg.regret_minimizing_mode == RegretMinimizingMode::regret_matching,
      "CFR-S wraps VANILLA CFR (local regret matching) per Celli et al. 2019."
   );
   static_assert(
      cfg.pruning_mode == CFRPruningMode::none and cfg.lazy_update_mode == CFRLazyUpdateMode::off
         and cfg.warm_start_iterations == 0,
      "CFR-S wraps the unmodified full-traversal vanilla CFR kernel; pruning, lazy "
      "updates and warm starts are not composed here."
   );

   using env_type = Env;
   using world_state_type = auto_world_state_type< Env >;
   using policy_type = Policy;
   using average_policy_type = AveragePolicy;
   using oracle_type = SequenceFormOracle< Env >;
   using solver_type = VanillaCFR< cfg, Env, Policy, AveragePolicy >;
   using rng_type = common::RNG;

   CFRS(
      Env env,
      uptr< world_state_type > root_state,
      const Policy& policy,
      const AveragePolicy& avg_policy,
      size_t seed = common::default_seed
   )
       : solver_(std::move(env), std::move(root_state), policy, avg_policy),
         oracle_(solver_.env(), solver_.root_state()),
         rng_(seed)
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
         JointDistribution::key_type key{};
         key.reserve(oracle_.player_count());
         // sample one pure plan per player from the strategy this CFR iteration plays
         for(const Player player : oracle_.players()) {
            key.push_back(sample_plan(player, behavioral));
         }
         average_joint_.add_sample(key);
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
   /// completed CFR-S iterations
   [[nodiscard]] size_t iteration() const { return average_joint_.draws(); }

  private:
   /// draws a pure plan for 'player': one action per registered infoset, sampled from
   /// the current behavioral distribution
   template < typename Behavioral >
   [[nodiscard]] Plan sample_plan(Player player, Behavioral&& behavioral)
   {
      const auto& player_structure = oracle_.structure(player);
      Plan plan(player_structure.size());
      for(auto iid : std::views::iota(size_t{0}, player_structure.size())) {
         const std::vector< double > distribution = behavioral(player, iid);
         std::discrete_distribution< size_t > action_draw(distribution.begin(), distribution.end());
         plan.at(iid) = uint32_t(action_draw(rng_));
      }
      return plan;
   }

   solver_type solver_;
   oracle_type oracle_;
   JointDistribution average_joint_{};
   rng_type rng_;
};

}  // namespace nor::rm::correlated

#endif  // NOR_RM_CORRELATED_CFR_S_HPP
