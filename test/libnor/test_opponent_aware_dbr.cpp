
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <ranges>
#include <string>
#include <unordered_map>

#include "nor/env.hpp"
#include "nor/nor.hpp"
#include "rm_specific_testing_utils.hpp"

/// Tests for DATA BIASED ROBUST COUNTER-STRATEGIES (Johanson & Bowling, "Data Biased Robust
/// Counter Strategies", AISTATS 2009): per-infostate confidence-weighted blending of empirical
/// frequency models into CFR self-play.

namespace {

using namespace nor;
using namespace nor::opponent_aware;
using namespace games::kuhn;

constexpr auto kuhn_dbr_config = rm::CFRConfig{
   .update_mode = rm::UpdateMode::alternating,
   .regret_minimizing_mode = rm::RegretMinimizingMode::regret_matching_plus};

using counts_type = opponent_aware::FrequencyTable< Infostate, Action >;

/// builds observation counts of 'n' hands at '?jc' split 'check_fraction'/'1-check_fraction'
counts_type jc_counts(const auto& istates, int n, double check_fraction)
{
   counts_type counts{};
   const int n_check = static_cast< int >(std::lround(n * check_fraction));
   if(n_check > 0) {
      counts[istates.bob.at("?jc")][Action::check] = static_cast< double >(n_check);
   }
   if(n - n_check > 0) {
      counts[istates.bob.at("?jc")][Action::bet] = static_cast< double >(n - n_check);
   }
   return counts;
}

/// materializes a frequency table into the fully specified behavioral table that evaluation
/// helpers expect (zero mass for legal actions never observed)
auto to_policy_table(const auto& istate_range, const counts_type& counts)
{
   std::unordered_map< Infostate, HashmapActionPolicy< Action > > table{};
   for(const auto& [_, infostate] : istate_range) {
      table.emplace(
         infostate,
         HashmapActionPolicy< Action >{opponent_aware::normalized_frequencies(
            counts, infostate, std::vector< Action >{Action::check, Action::bet}
         )}
      );
   }
   return table;
}

double linfty_policy_distance(const auto& left, const auto& right)
{
   double distance = 0.;
   for(const auto& [infostate, action_policy] : left) {
      const auto& other = right.at(infostate);
      auto normalized_left = normalize_action_policy(action_policy);
      auto normalized_right = normalize_action_policy(other);
      for(const auto& [action, prob] : normalized_left) {
         distance = std::max(distance, std::abs(prob - normalized_right.at(action)));
      }
   }
   return distance;
}

}  // namespace

TEST(OpponentAwareDBR, constant_confidence_degenerates_to_rnr)
{
   auto env = Environment{};
   auto root = State{};
   auto istates = make_kuhn_named_infostates();

   // frequency model identical (per hand count) to the fixed {always-check} RNR model
   counts_type counts{};
   for(const auto& [_, infostate] : istates.bob) {
      counts[infostate][Action::check] = 100.;
   }
   auto model_table = to_policy_table(istates.bob, counts);

   auto rnr_result = rnr_response< kuhn_dbr_config >(
      env,
      root,
      nor::Player::alex,
      model_table,
      /*forcing_probability=*/0.7,
      3000
   );
   auto dbr_result = opponent_aware::dbr_response< kuhn_dbr_config >(
      env,
      root,
      nor::Player::alex,
      counts,
      /*pconf=*/[](double) { return 0.7; },
      3000
   );

   // DBR with a globally constant Pconf IS an RNR run on the frequency model: same behavioral
   // outcome up to stochastic tie-breaking noise ...
   EXPECT_LT(linfty_policy_distance(rnr_result.policy, dbr_result.policy), 0.15);
   // ... and indistinguishable realized-play value against the shared model
   double rnr_value = opponent_aware::value_against(
      env, root, nor::Player::alex, rnr_result.policy, model_table
   );
   double dbr_value = opponent_aware::value_against(
      env, root, nor::Player::alex, dbr_result.policy, model_table
   );
   EXPECT_NEAR(rnr_value, dbr_value, 1e-3);
}

TEST(OpponentAwareDBR, confident_data_steers_blending_differently_from_sparse_data)
{
   auto env = Environment{};
   auto root = State{};
   auto istates = make_kuhn_named_infostates();

   // IDENTICAL betting tendency at jack vs bet (85% passive), wildly different sample sizes
   constexpr double check_fraction = 0.85;
   auto confident_counts = jc_counts(istates, /*n=*/200, check_fraction);
   auto sparse_counts = jc_counts(istates, /*n=*/4, check_fraction);

   auto confident_result = opponent_aware::dbr_response< kuhn_dbr_config >(
      env,
      root,
      nor::Player::alex,
      confident_counts,
      opponent_aware::linear_confidence(/*p_max=*/1., /*ramp=*/10.),
      3000
   );
   auto sparse_result = opponent_aware::dbr_response< kuhn_dbr_config >(
      env,
      root,
      nor::Player::alex,
      sparse_counts,
      opponent_aware::linear_confidence(/*p_max=*/1., /*ramp=*/10.),
      3000
   );

   // the paper's core claim: HOW MUCH the data pulls the response away from equilibrium play
   // depends on the confidence at the observed infostate -- heavily trusted data and barely
   // trusted data must produce visibly different responses ...
   double distance = linfty_policy_distance(confident_result.policy, sparse_result.policy);
   std::cout << "[          ] dbr conf-vs-sparse behavioral distance: " << distance << "\n";
   EXPECT_GT(distance, 0.03);

   // ... and of the two only the confident one may rely on its exploitation direction:
   // against the TRUE underlying always-passive habit the trusted-data response extracts at
   // least as much value (the sparse-data response hedges toward equilibrium play instead)
   std::unordered_map< Infostate, HashmapActionPolicy< Action > > true_passive_table{};
   for(const auto& [_, infostate] : istates.bob) {
      true_passive_table.emplace(
         infostate, HashmapActionPolicy< Action >{std::pair{Action::check, 1.}}
      );
   }
   double confident_extraction = opponent_aware::value_against(
      env, root, nor::Player::alex, confident_result.policy, true_passive_table
   );
   double sparse_extraction = opponent_aware::value_against(
      env, root, nor::Player::alex, sparse_result.policy, true_passive_table
   );
   std::cout << "[          ] dbr extraction vs true passive habit: confident="
             << confident_extraction << " sparse=" << sparse_extraction << "\n";
   EXPECT_GE(confident_extraction, sparse_extraction - 1e-9);
}

TEST(OpponentAwareDBR, unobserved_infostates_free_the_opponent)
{
   auto env = Environment{};
   auto root = State{};
   auto istates = make_kuhn_named_infostates();

   // data recorded ONLY at '?jb' (bob facing a bet holding a jack)
   counts_type counts{};
   counts[istates.bob.at("?jb")][Action::check] = 40.;
   counts[istates.bob.at("?jb")][Action::bet] = 10.;

   // everywhere else there is nothing to trust, so the run must solve essentially as plain CFR
   // -- converging towards the game's equilibrium value rather than trusting fabricated data
   auto result = opponent_aware::dbr_response< kuhn_dbr_config >(
      env, root, nor::Player::alex, counts, opponent_aware::step_confidence(/*p_max=*/1.), 4000
   );
   EXPECT_NEAR(result.final_root_values.at(nor::Player::alex), -1. / 18., 4e-2);
}

TEST(OpponentAwareDBR, leduc_short_horizon_confidence_steers_solving)
{
   using istate_type = games::leduc::Infostate;
   using action_type = games::leduc::Action;
   games::leduc::Environment env{};
   games::leduc::State root{};
   auto [_, history_to_istate] = map_histories_to_infostates(env, root);

   // pseudo-counts of a heavily passive habit at EVERY bob decision. The canonical
   // {check, fold, bet} key triples need not match the exact legal sets: the zero-filling
   // normalizer adapts them to whatever actions each infostate actually offers
   opponent_aware::FrequencyTable< istate_type, action_type > counts{};
   for(const auto& [_, per_player] : history_to_istate | std::views::values) {
      auto found = per_player.find(nor::Player::bob);
      if(found == per_player.end()) {
         continue;
      }
      auto& entry = counts[*found->second];
      entry[action_type{games::leduc::ActionType::check, 0.}] += 8.;
      entry[action_type{games::leduc::ActionType::fold, 0.}] += 1.;
      entry[action_type{games::leduc::ActionType::bet, 2.}] += 1.;
   }
   ASSERT_FALSE(counts.empty());

   constexpr size_t n_iters = 150;
   auto low_trust = dbr_response< kuhn_dbr_config >(
      env, root, nor::Player::alex, counts, opponent_aware::step_confidence(/*p_max=*/0.2), n_iters
   );
   auto high_trust = dbr_response< kuhn_dbr_config >(
      env,
      root,
      nor::Player::alex,
      counts,
      opponent_aware::step_confidence(/*p_max=*/1., /*min_observations=*/10.),
      n_iters
   );

   double v_low = low_trust.final_root_values.at(nor::Player::alex);
   double v_high = high_trust.final_root_values.at(nor::Player::alex);
   std::cout << "[          ] leduc dbr: value@lowconf=" << v_low << " value@highconf=" << v_high
             << "\n";
   EXPECT_TRUE(std::isfinite(v_low));
   EXPECT_TRUE(std::isfinite(v_high));

   // confidence must steer SOLVING on the short-horizon leduc tree, not merely reweight the
   // same trajectory: the two responses are behaviorally distinguishable ...
   double distance = linfty_policy_distance(high_trust.policy, low_trust.policy);
   std::cout << "[          ] leduc dbr behavioral distance: " << distance << "\n";
   EXPECT_GT(distance, 0.01);

   // ... and confidence safety holds directionally: near-equilibrium play under LOW trust is
   // not easier to punish than the exploitation-heavy HIGH-trust response
   double nemesis_low = best_response_value_against(env, root, nor::Player::alex, low_trust.policy);
   double nemesis_high = best_response_value_against(
      env, root, nor::Player::alex, high_trust.policy
   );
   std::cout << "[          ] leduc dbr nemesis-value: low=" << nemesis_low
             << " high=" << nemesis_high << "\n";
   EXPECT_GT(nemesis_high, nemesis_low - 0.02);
}
