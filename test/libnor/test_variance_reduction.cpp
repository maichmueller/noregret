
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <ranges>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/common.hpp"
#include "nor/env/kuhn.hpp"
#include "nor/env/leduc.hpp"
#include "nor/evaluation/variance_reduction.hpp"
#include "nor/policy/policy.hpp"
#include "nor/rm/policy_value.hpp"

// goofspiel is a standalone header-only game library
#include "goofspiel/goofspiel.hpp"

namespace {

using namespace nor;

/////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// generic test machinery ////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief depth-first discovery of every reachable (player, infostate) pair
 * together with its (infostate-invariant) legal actions.
 *
 * The traversal replicates the infostate/observation-buffer bookkeeping of the
 * production traversals (rm::policy_value, best_response) so that discovered
 * keys match exactly what ProfileGameTree looks up when it consults a profile.
 */
template < typename Env >
class ReachableProfileDiscovery {
  public:
   using world_state_type = auto_world_state_type< Env >;
   using info_state_type = auto_info_state_type< Env >;
   using action_type = auto_action_type< Env >;

   using discovered_map = player_hashmap<
      std::unordered_map< info_state_type, std::vector< action_type > > >;
   using obs_buffer_type = player_hashmap<
      std::vector< std::pair< auto_observation_type< Env >, auto_observation_type< Env > > > >;
   using infostate_map_type = player_hashmap< info_state_type >;

   ReachableProfileDiscovery(const Env& env, const world_state_type& root)
       : m_discovered(std::invoke([&] {
            discovered_map map{};
            for(auto player : env.players(root) | utils::is_actual_player_filter) {
               map.try_emplace(player);
            }
            return map;
         }))
   {
      obs_buffer_type root_buffer{};
      for(auto player : m_discovered | std::views::keys) {
         root_buffer.try_emplace(player);
      }
      infostate_map_type root_infostates{};
      for(auto player : m_discovered | std::views::keys) {
         root_infostates.emplace(player, info_state_type{player});
      }
      _dfs(
         env,
         utils::static_unique_ptr_downcast< world_state_type >(utils::clone_any_way(root)),
         std::move(root_buffer),
         std::move(root_infostates)
      );
   }

   [[nodiscard]] const discovered_map& result() const { return m_discovered; }

  private:
   void _dfs(
      const Env& env,
      uptr< world_state_type > state,
      obs_buffer_type observation_buffer,
      infostate_map_type infostate_map
   )
   {
      if(env.is_terminal(*state)) {
         return;
      }
      Player active_player = env.active_player(*state);

      if(active_player != Player::chance) {
         auto& slot = m_discovered.at(active_player)[infostate_map.at(active_player)];
         for(const auto& action : env.actions(active_player, *state)) {
            if(std::find(slot.begin(), slot.end(), action) == slot.end()) {
               slot.push_back(action);
            }
         }
      }

      auto descend = [&](const auto& move_or_outcome) {
         uptr< world_state_type > next_uptr = child_state(env, *state, move_or_outcome);
         auto [child_buffer, child_infostates] = next_infostate_and_obs_buffers(
            env, observation_buffer, infostate_map, *state, move_or_outcome, *next_uptr
         );
         _dfs(env, std::move(next_uptr), std::move(child_buffer), std::move(child_infostates));
      };

      if constexpr(concepts::stochastic_env< Env >) {
         if(active_player == Player::chance) {
            for(const auto& outcome : env.chance_actions(*state)) {
               descend(outcome);
            }
            return;
         }
      }
      for(const auto& action : env.actions(active_player, *state)) {
         descend(action);
      }
   }

   discovered_map m_discovered;
};

/// per-game typedef bundle for building tabular profiles and estimators
template < typename Env >
struct GameProfileTypes {
   using info_state_type = auto_info_state_type< Env >;
   using action_type = auto_action_type< Env >;
   using action_policy_type = HashmapActionPolicy< action_type >;
   using table_type = std::unordered_map< info_state_type, action_policy_type >;
   using policy_type = TabularPolicy< info_state_type, action_policy_type, table_type >;
   using profile_type = player_hashmap< policy_type >;
   using tree_type = evaluation::ProfileGameTree< Env, policy_type >;
   using exact_values_fn = std::function< double(Player) >;
};

/**
 * @brief builds a tabular policy profile from discovered infostates. 'weight_of'
 * maps (player, action index within the discovered legal-action list) to a
 * relative weight; rows are normalized afterwards.
 */
template < typename Env >
typename GameProfileTypes< Env >::profile_type build_profile(
   const ReachableProfileDiscovery< Env >& discovery,
   const std::function< double(Player, size_t) >& weight_of
)
{
   using GT = GameProfileTypes< Env >;
   typename GT::profile_type profile{};
   for(const auto& [player, istate_actions] : discovery.result()) {
      typename GT::table_type table{};
      for(const auto& [istate, actions] : istate_actions) {
         typename GT::action_policy_type row{};
         double weight_sum = 0.;
         for(auto [aidx, action] : std::views::enumerate(actions)) {
            const double w = weight_of(player, static_cast< size_t >(aidx));
            row.emplace(action, w);
            weight_sum += w;
         }
         assert(weight_sum > 0.);
         for(auto& [action, prob] : row) {
            (void) action;
            prob /= weight_sum;
         }
         table.emplace(istate, std::move(row));
      }
      profile.emplace(player, typename GT::policy_type{std::move(table)});
   }
   return profile;
}

/// uniform-random-action relative weights
inline double uniform_weight(Player, size_t)
{
   return 1.;
}

/// geometrically decaying weights: strongly skewed towards earlier listed actions
double skewed_weight(Player, size_t action_idx)
{
   return std::pow(4., -static_cast< double >(action_idx));
}

/**
 * @brief partially informed AIVAT successor-value heuristic: the EXACT profile
 * continuation values distorted by shrinkage + constant bias. Deliberately
 * WRONG values demonstrate unbiasedness robustness while still exercising real
 * variance reduction.
 */
template < typename TreeT >
auto make_distorted_exact_heuristic(const TreeT& tree, double shrinkage = 0.5, double bias = 0.25)
{
   return [&tree, shrinkage, bias](Player player, size_t node_idx, size_t move_idx) -> double {
      return shrinkage * tree.node(tree.node(node_idx).moves[move_idx].child).values.at(player)
             + bias;
   };
}

struct SuiteConfig {
   size_t n_batches = 2200;  // seeded independent batches per estimator
   size_t batch_playouts = 32;  // playouts per batch
   size_t variance_playouts = 12000;  // single long run for variance diagnostics
   /// confidence half-width in standard errors of the pooled mean
   double unbiased_z = 5.0;
   /// strict upper bound imposed on measured variance ratios vs raw sampling
   double ratio_limit = 0.85;
};

std::string report_line(
   const std::string& label,
   Player player,
   double raw_var,
   double reduced_var,
   double ratio
)
{
   return label + " [" + common::to_string(player) + "] raw-var=" + std::to_string(raw_var)
          + " reduced-var=" + std::to_string(reduced_var) + " ratio=" + std::to_string(ratio);
}

void check_unbiasedness(
   const std::string& label,
   Player player,
   double exact_value,
   size_t n_observations,
   double mean,
   double sample_variance,
   double z
)
{
   const double s = std::sqrt(std::max(sample_variance, 0.));
   const double bound = std::max(
      z * s / std::sqrt(static_cast< double >(std::max(n_observations, size_t{1}))), 1e-9
   );
   EXPECT_NEAR(mean, exact_value, bound)
      << label << " [" << common::to_string(player) << "]: mean=" << mean
      << " exact=" << exact_value << " n=" << n_observations << " sd=" << s << " allowed=" << bound;
}

/**
 * @brief full validation suite for one fixed profile:
 *  - ground-truth gate: tree backward pass matches an independent policy_value traversal,
 *  - unbiasedness of MIVAT/AIVAT/distorted-heuristic-AIVAT and raw sampling over 'n_batches'
 *    independently seeded batches,
 *  - measured variance ratios (reported numerically, asserted substantially below 1),
 *  - zero-variance special case of AIVAT with exact counterfactual value functions.
 */
template < typename Env >
void run_evaluation_suite(
   const std::string& label,
   const Env& env,
   const auto_world_state_type< Env >& root,
   const typename GameProfileTypes< Env >::profile_type& profile,
   const SuiteConfig& cfg
)
{
   using GT = GameProfileTypes< Env >;
   using policy_type = typename GT::policy_type;
   using tree_type = typename GT::tree_type;

   const tree_type tree(env, root, profile);

   // ---- ground-truth gate -------------------------------------------------
   const auto traverse_values = rm::policy_value(env, root, profile).get();
   for(const auto& player : tree.players()) {
      ASSERT_NEAR(tree.root_values().at(player), traverse_values.at(player), 1e-9)
         << label << ": enumerated tree value disagrees with policy_value traversal";
   }

   const size_t n_players = tree.players().size();
   const auto exact_of = [&](Player player) { return traverse_values.at(player); };

   // ---- unbiasedness: pool per-playout estimates over seeded batches ------
   std::vector< evaluation::detail::OnlineStats > raw_pool(n_players);
   std::vector< evaluation::detail::OnlineStats > mivat_pool(n_players);
   std::vector< evaluation::detail::OnlineStats > aivat_pool(n_players);
   std::vector< evaluation::detail::OnlineStats > aivat_distorted_pool(n_players);

   evaluation::MivatEstimator< Env, policy_type > mivat(tree);
   evaluation::AivatEstimator< Env, policy_type > aivat(tree);
   evaluation::AivatEstimator< Env, policy_type > aivat_distorted(
      tree, common::default_seed, make_distorted_exact_heuristic(tree)
   );

   evaluation::PlayoutSampler< Env, policy_type > sampler(tree);
   for(size_t b : std::views::iota(size_t{0}, cfg.n_batches)) {
      const size_t seed = 104729u * b + 7919u;

      sampler.reseed(seed ^ 0xA17AB1E5u);
      for(const auto& playout : sampler.sample(cfg.batch_playouts)) {
         const auto& rewards = tree.node(playout.terminal_node).rewards;
         for(auto [pidx, player] : std::views::enumerate(tree.players())) {
            raw_pool[pidx].push(rewards.at(player));
         }
      }

      mivat.reseed(seed);
      mivat.run(cfg.batch_playouts);
      for(auto [pidx, estimates] : std::views::enumerate(mivat.playout_estimates())) {
         for(double estimate : estimates) {
            mivat_pool[pidx].push(estimate);
         }
      }

      aivat.reseed(seed ^ 0x51A57D11u);
      aivat.run(cfg.batch_playouts);
      for(auto [pidx, estimates] : std::views::enumerate(aivat.playout_estimates())) {
         for(double estimate : estimates) {
            aivat_pool[pidx].push(estimate);
         }
      }

      aivat_distorted.reseed(seed ^ 0xBADC0DE7u);
      aivat_distorted.run(cfg.batch_playouts);
      for(auto [pidx, estimates] : std::views::enumerate(aivat_distorted.playout_estimates())) {
         for(double estimate : estimates) {
            aivat_distorted_pool[pidx].push(estimate);
         }
      }
   }

   for(auto [pidx, player] : std::views::enumerate(tree.players())) {
      const double exact = exact_of(player);
      check_unbiasedness(
         label + "/raw",
         player,
         exact,
         raw_pool[pidx].count(),
         raw_pool[pidx].mean(),
         raw_pool[pidx].variance(),
         cfg.unbiased_z
      );
      check_unbiasedness(
         label + "/MIVAT",
         player,
         exact,
         mivat_pool[pidx].count(),
         mivat_pool[pidx].mean(),
         mivat_pool[pidx].variance(),
         cfg.unbiased_z
      );
      check_unbiasedness(
         label + "/AIVAT",
         player,
         exact,
         aivat_pool[pidx].count(),
         aivat_pool[pidx].mean(),
         aivat_pool[pidx].variance(),
         cfg.unbiased_z
      );
      check_unbiasedness(
         label + "/AIVAT(distorted heuristic)",
         player,
         exact,
         aivat_distorted_pool[pidx].count(),
         aivat_distorted_pool[pidx].mean(),
         aivat_distorted_pool[pidx].variance(),
         cfg.unbiased_z
      );

      ASSERT_TRUE(mivat.results().contains(player));
      ASSERT_TRUE(aivat.results().contains(player));
   }

   // ---- variance reduction diagnostics ------------------------------------
   mivat.run(cfg.variance_playouts);
   aivat_distorted.run(cfg.variance_playouts);
   aivat.run(cfg.variance_playouts);
   for(const auto& [pidx, player_ref] : std::views::enumerate(tree.players())) {
      const Player player{player_ref};
      const auto& miv = mivat.results().at(player);
      const auto& aiv = aivat.results().at(player);
      const auto& dis = aivat_distorted.results().at(player);

      std::cout << report_line(
         label + "/raw-vs-MIVAT",
         player,
         miv.variance.raw_variance,
         miv.variance.reduced_variance,
         miv.variance.variance_ratio()
      ) << "\n";
      std::cout << report_line(
         label + "/raw-vs-AIVAT(exact)",
         player,
         aiv.variance.raw_variance,
         aiv.variance.reduced_variance,
         aiv.variance.variance_ratio()
      ) << "\n";
      std::cout << report_line(
         label + "/raw-vs-AIVAT(distorted)",
         player,
         dis.variance.raw_variance,
         dis.variance.reduced_variance,
         dis.variance.variance_ratio()
      ) << "\n";

      EXPECT_LT(miv.variance.variance_ratio(), cfg.ratio_limit);
      EXPECT_LT(aiv.variance.variance_ratio(), cfg.ratio_limit);
      EXPECT_LT(dis.variance.variance_ratio(), cfg.ratio_limit);
   }
}

/**
 * @brief AIVAT's zero-variance special case: with exact counterfactual value
 * functions (injected explicitly AND used by default) every correction term
 * telescopes and all playout estimates must equal the exact root value.
 */
template < typename Env >
void assert_aivat_zero_variance(
   const std::string& label,
   const typename GameProfileTypes< Env >::tree_type& tree,
   const std::function< double(Player) >& exact_of
)
{
   using GT = GameProfileTypes< Env >;
   using policy_type = typename GT::policy_type;

   constexpr size_t n_playouts = 4096;
   constexpr double value_tol = 1e-9;
   constexpr double var_tol = 1e-18;

   // variant 1: default successor values (exact by construction)
   evaluation::AivatEstimator< Env, policy_type > estimator_default(tree, 987654321u);
   estimator_default.run(n_playouts);

   // variant 2: explicit heuristic that returns the exact continuation values
   evaluation::AivatEstimator< Env, policy_type > estimator_injected(
      tree,
      13579u,
      [&tree](Player player, size_t node_idx, size_t move_idx) -> double {
         return tree.node(tree.node(node_idx).moves[move_idx].child).values.at(player);
      }
   );
   estimator_injected.run(n_playouts);

   for(const auto& player : tree.players()) {
      const double exact = exact_of(player);
      SCOPED_TRACE(label + " [" + common::to_string(player) + "]");
      for(const auto* estimator : {&estimator_default, &estimator_injected}) {
         const auto& result = estimator->results().at(player);
         EXPECT_LT(result.variance.reduced_variance, var_tol);
         EXPECT_NEAR(result.estimate, exact, value_tol);

         const auto& estimates = estimator->playout_estimates().at(
            static_cast< size_t >(estimator->tree().player_index(player))
         );
         const auto [lowest, highest] = std::ranges::minmax(estimates);
         EXPECT_NEAR(highest - lowest, 0., value_tol);
      }
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// kuhn poker //////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(VarianceReduction_KuhnPoker, UnbiasednessAndVarianceReduction_TwoProfiles)
{
   using GameTypes = GameProfileTypes< games::kuhn::Environment >;
   games::kuhn::Environment env{};
   games::kuhn::State root{};

   ReachableProfileDiscovery< games::kuhn::Environment > discovery(env, root);
   SuiteConfig cfg{.n_batches = 2200, .batch_playouts = 32, .variance_playouts = 12000};

   auto uniform_profile = build_profile(discovery, uniform_weight);
   run_evaluation_suite("kuhn/uniform", env, root, uniform_profile, cfg);

   auto skewed_profile = build_profile(discovery, skewed_weight);
   run_evaluation_suite("kuhn/skewed", env, root, skewed_profile, cfg);

   const auto traverse_values = rm::policy_value(env, root, uniform_profile).get();
   assert_aivat_zero_variance< games::kuhn::Environment >(
      "kuhn/zero-variance",
      GameTypes::tree_type(env, root, uniform_profile),
      [&](Player player) { return traverse_values.at(player); }
   );
}

//////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(VarianceReduction_LeducShort, UnbiasednessAndVarianceReduction_TwoProfiles)
{
   using GameTypes = GameProfileTypes< games::leduc::Environment >;
   games::leduc::Environment env{};
   games::leduc::State root{};  // default LeducConfig: 2 players, J/Q/K deck

   ReachableProfileDiscovery< games::leduc::Environment > discovery(env, root);
   SuiteConfig cfg{.n_batches = 2200, .batch_playouts = 32, .variance_playouts = 10000};

   auto uniform_profile = build_profile(discovery, uniform_weight);
   run_evaluation_suite("leduc-short/uniform", env, root, uniform_profile, cfg);

   auto skewed_profile = build_profile(discovery, skewed_weight);
   run_evaluation_suite("leduc-short/skewed", env, root, skewed_profile, cfg);

   const auto traverse_values = rm::policy_value(env, root, uniform_profile).get();
   assert_aivat_zero_variance< games::leduc::Environment >(
      "leduc-short/zero-variance",
      GameTypes::tree_type(env, root, uniform_profile),
      [&](Player player) { return traverse_values.at(player); }
   );
}

////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(VarianceReduction_Goofspiel, UnbiasednessAndVarianceReduction_TwoProfiles)
{
   using GameTypes = GameProfileTypes< games::goofspiel::Environment >;
   const games::goofspiel::GoofspielConfig config{.deck_size = 3, .imp_info = true};
   games::goofspiel::Environment env{config};
   games::goofspiel::State root{config};

   ReachableProfileDiscovery< games::goofspiel::Environment > discovery(env, root);
   SuiteConfig cfg{.n_batches = 2200, .batch_playouts = 32, .variance_playouts = 12000};

   auto uniform_profile = build_profile(discovery, uniform_weight);
   run_evaluation_suite("goofspiel/uniform", env, root, uniform_profile, cfg);

   auto skewed_profile = build_profile(discovery, skewed_weight);
   run_evaluation_suite("goofspiel/skewed", env, root, skewed_profile, cfg);

   const auto traverse_values = rm::policy_value(env, root, uniform_profile).get();
   assert_aivat_zero_variance< games::goofspiel::Environment >(
      "goofspiel/zero-variance",
      GameTypes::tree_type(env, root, uniform_profile),
      [&](Player player) { return traverse_values.at(player); }
   );
}

}  // namespace
