
#ifndef NOR_RM_CORRELATED_JOINT_DISTRIBUTION_HPP
#define NOR_RM_CORRELATED_JOINT_DISTRIBUTION_HPP

#include <algorithm>
#include <bit>
#include <cstdint>
#include <functional>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "common/common.hpp"
#include "nor/game_defs.hpp"
#include "nor/rm/correlated/sequence_form.hpp"

namespace nor::rm::correlated {

/**
 * @brief sparse accumulator of a joint (correlated) distribution over tuples of
 *        normal-form plans -- one plan per root player, stored in roster order.
 *
 * The support is keyed by tuples that have received nonzero mass so far; its
 * worst-case size is the product |S_1| x ... x |S_n| of the per-player reduced plan
 * spaces (see SequenceFormOracle's memory notes). All reported masses are normalized
 * by the number of accumulated draws.
 */
class JointDistribution {
  public:
   using key_type = std::vector< Plan >;

  private:
   struct key_hasher {
      [[nodiscard]] size_t operator()(const key_type& key) const noexcept
      {
         size_t seed = 0;
         for(const Plan& plan : key) {
            common::hash_combine(seed, plan_hasher{}(plan));
         }
         return seed;
      }
   };

  public:
   JointDistribution() = default;

   /// opens a new averaging iteration: all subsequent 'accumulate' calls belong to
   /// this iteration and together carry total mass one (the product distribution)
   void begin_iteration() { ++draws_; }

   /// accumulates 'mass' under 'key' WITHOUT advancing the iteration counter -- a
   /// single iteration contributes MANY tuples (the support of its product
   /// distribution)
   void accumulate(const key_type& key, double mass)
   {
      if(not (mass >= 0.)) {
         throw std::domain_error("JointDistribution::accumulate: negative mass");
      }
      if(draws_ == 0) {
         throw std::logic_error("JointDistribution::accumulate called before begin_iteration");
      }
      masses_[key] += mass;
   }

   /// counts one empirical sample of 'key' as its own iteration (CFR-S style)
   void add_sample(const key_type& key)
   {
      begin_iteration();
      accumulate(key, 1.);
   }

   /// number of accumulated joint draws (the normalization denominator)
   [[nodiscard]] size_t draws() const { return draws_; }
   [[nodiscard]] size_t support_size() const { return masses_.size(); }

   void reserve(size_t n) { masses_.reserve(n); }

   template < typename F >
   void for_each_normalized(F&& functor) const
   {
      if(draws_ == 0) {
         return;
      }
      const double norm = double(draws_);
      for(const auto& [key, mass] : masses_) {
         functor(key, mass / norm);
      }
   }

  private:
   std::unordered_map< key_type, double, key_hasher > masses_{};
   size_t draws_ = 0;
};

/// summary metrics of a correlated profile over normal-form plans
struct CCEMetrics {
   /// max unilateral deviation gain: max_i max_{sigma'_i} u_i(sigma'_i, x̄_{-i}) - u_i(x̄);
   /// an epsilon-CCE requires this to be at most epsilon
   double cce_gap = 0.;
   /// sum of player values under x̄
   double social_welfare = 0.;
   player_hashmap< double > player_values{};
   player_hashmap< double > deviation_gains{};
};

namespace detail {

/**
 * builds a behavioral-distribution query compatible with SequenceFormOracle::reconstruct:
 * reads the CURRENT policy table of the wrapped solver -- the recommendation table the
 * simultaneous vanilla kernel traverses with (the dispatch's '_iterate<false>' fills
 * 'initializing_run', while 'use_current_policy' stays defaulted to true; see cfr.tcc)
 * -- so the query returns exactly the strategy the upcoming iteration plays, as
 * required by Theorem 5 (Celli et al. 2019). Missing entries default to uniform,
 * mirroring the solvers' default play; rows are defensively normalized.
 */
template < typename Solver, typename Oracle >
[[nodiscard]] auto make_behavioral_query(const Solver& solver, const Oracle& oracle)
{
   return [&solver, &oracle](Player player, uint32_t infoset_id) -> std::vector< double > {
      const auto& player_structure = oracle.structure(player);
      const auto& infostate = player_structure.infostates.at(infoset_id);
      const auto& actions = player_structure.actions.at(infoset_id);
      std::vector< double > distribution{};
      distribution.reserve(actions.size());
      const auto& table = solver.policy().at(player);
      double sum = 0.;
      if(auto found = table.find(infostate); found != table.end()) {
         for(const auto& action : actions) {
            double value = std::max(0., found->second.at(action));
            distribution.push_back(value);
            sum += value;
         }
      } else {
         distribution.assign(actions.size(), 0.);
      }
      if(sum <= 1e-12) {
         std::ranges::fill(distribution, 1. / double(actions.size()));
      } else {
         for(double& value : distribution) {
            value /= sum;
         }
      }
      return distribution;
   };
}

/// iterates the set bits of a terminal-membership bitmask
template < typename F >
void for_each_set_bit(const std::vector< uint64_t >& mask, F&& functor)
{
   for(size_t w : std::views::iota(size_t{0}, mask.size())) {
      uint64_t word = mask.at(w);
      while(word != 0) {
         uint64_t bit = word & (~word + 1);
         functor(w * 64 + size_t(std::countr_zero(word)));
         word ^= bit;
      }
   }
}

}  // namespace detail

/**
 * @brief evaluates CCE gap and social welfare of an averaged correlated profile
 *        DIRECTLY over the stored joint distribution (no product-policy best-response
 *        machinery): terminal expectations are assembled from memoized plan masks,
 *        and the deviation term maximizes u_i(sigma'_i, x̄_{-i}) over the enumerated
 *        reduced plans of each player against the OPPONENT-JOINT terminal marginal.
 *
 * Definition 1 of Celli et al. (2019): x* is a CCE iff for every player i and every
 * sigma'_i: sum over sigma of x*(sigma) * (u_i(sigma) - u_i(sigma'_i, sigma_{-i})) >= 0.
 */
template < typename Env >
[[nodiscard]] CCEMetrics
evaluate_cce(const SequenceFormOracle< Env >& oracle, const JointDistribution& average)
{
   const size_t n_terminals = oracle.terminal_count();
   const auto& roster = oracle.players();
   const size_t n_players = roster.size();

   // joint terminal marginal pbar[z] and per-player opponent-joint marginals W_i[z]
   std::vector< double > pbar(n_terminals, 0.);
   std::vector< std::vector< double > > opponent_marginal(
      n_players, std::vector< double >(n_terminals, 0.)
   );

   // a joint plan tuple does NOT pin down a single terminal: conditional on the tuple,
   // the terminals are distributed by the CHANCE probabilities -- they must weight the
   // marginal accumulation (chance_prob == 1 recovers the deterministic-game case)
   std::vector< double > chance_prob(n_terminals, 1.);
   for(auto z : std::views::iota(size_t{0}, n_terminals)) {
      chance_prob.at(z) = oracle.terminal(z).chance_prob;
   }

   const size_t n_words = SequenceFormOracle< Env >::words_for(n_terminals);

   // reusable bitmask scratch: [full conjunction | opponents-of-i conjunction]
   std::vector< uint64_t > conjunction(n_words, 0);
   std::vector< uint64_t > opponent_conjunction(n_words, 0);
   const uint64_t tail_mask = n_terminals % 64 == 0 ? ~uint64_t(0)
                                                    : (uint64_t(1) << (n_terminals % 64)) - 1;

   average.for_each_normalized([&](const JointDistribution::key_type& key, double mass) {
      if(n_words > 0) {
         std::ranges::fill(conjunction, ~uint64_t(0));
         conjunction.back() &= tail_mask;
      }
      for(auto j : std::views::iota(size_t{0}, n_players)) {
         const auto& mask = oracle.plan_mask(roster.at(j), key.at(j));
         for(auto w : std::views::iota(size_t{0}, n_words)) {
            conjunction.at(w) &= mask.at(w);
         }
      }
      detail::for_each_set_bit(conjunction, [&](size_t z) {
         pbar.at(z) += mass * chance_prob.at(z);
      });
      for(auto i : std::views::iota(size_t{0}, n_players)) {
         if(n_words > 0) {
            std::ranges::fill(opponent_conjunction, ~uint64_t(0));
            opponent_conjunction.back() &= tail_mask;
         }
         for(auto j : std::views::iota(size_t{0}, n_players)) {
            if(j == i) {
               continue;
            }
            const auto& mask = oracle.plan_mask(roster.at(j), key.at(j));
            for(auto w : std::views::iota(size_t{0}, n_words)) {
               opponent_conjunction.at(w) &= mask.at(w);
            }
         }
         detail::for_each_set_bit(opponent_conjunction, [&](size_t z) {
            opponent_marginal.at(i).at(z) += mass * chance_prob.at(z);
         });
      }
   });

   CCEMetrics out{};
   for(auto i : std::views::iota(size_t{0}, n_players)) {
      const Player player = roster.at(i);
      double value = 0.;
      for(auto z : std::views::iota(size_t{0}, n_terminals)) {
         value += pbar.at(z) * oracle.terminal_reward(z, player);
      }
      out.player_values.emplace(player, value);
      out.social_welfare += value;

      // max unilateral deviation gain against the opponents' JOINT marginal
      const auto& weighted = opponent_marginal.at(i);
      double best_deviation = -std::numeric_limits< double >::infinity();
      for(const Plan& candidate : oracle.reduced_plans(player)) {
         const auto& mask = oracle.plan_mask(player, candidate);
         double deviation_value = 0.;
         detail::for_each_set_bit(mask, [&](size_t z) {
            deviation_value += weighted.at(z) * oracle.terminal_reward(z, player);
         });
         best_deviation = std::max(best_deviation, deviation_value);
      }
      const double gain = best_deviation - value;
      out.deviation_gains.emplace(player, gain);
      out.cce_gap = std::max(out.cce_gap, gain);
   }
   return out;
}

}  // namespace nor::rm::correlated

#endif  // NOR_RM_CORRELATED_JOINT_DISTRIBUTION_HPP
