
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <ranges>
#include <string>
#include <unordered_map>
#include <vector>

#include "nor/env.hpp"
#include "nor/nor.hpp"
#include "rm_specific_testing_utils.hpp"

/// Tests for DEV-BASED BEST RESPONSE (Ganzfried & Sandholm, "Game Theory-Based Opponent
/// Modeling in Large Imperfect-Information Games", AAMAS 2011): projection of observed action
/// frequencies onto a fully specified consistent strategy followed by an exact best response.

namespace {

using namespace nor;
using namespace games::kuhn;

using counts_type = opponent_aware::FrequencyTable< Infostate, Action >;

/// per-infostate check frequencies of a hand-built 'true' bob profile (in deciles so that a
/// multiple-of-ten observation count reproduces it EXACTLY)
constexpr std::array< std::pair< const char*, double >, 6 > k_true_check_fractions{
   std::pair{"?jc", 0.6},
   std::pair{"?jb", 0.5},
   std::pair{"?qc", 0.9},
   std::pair{"?qb", 0.5},
   std::pair{"?kc", 0.1},
   std::pair{"?kb", 1.}};

auto true_profile_table(const auto& istates)
{
   opponent_aware::policy_table_type< Infostate, Action > table{};
   for(const auto& [name, check_fraction] : k_true_check_fractions) {
      table.emplace(
         istates.bob.at(name),
         HashmapActionPolicy< Action >{
            std::pair{Action::check, check_fraction}, std::pair{Action::bet, 1. - check_fraction}}
      );
   }
   return table;
}

/// hand-built observation counts that are CONSISTENT with the true profile above: 'n' draws per
/// infostate in exact multiples of ten following its fractions
counts_type consistent_counts(const auto& istates)
{
   counts_type counts{};
   for(const auto& [name, check_fraction] : k_true_check_fractions) {
      auto& entry = counts[istates.bob.at(name)];
      entry[Action::check] = static_cast< double >(static_cast< int >(10 * check_fraction));
      entry[Action::bet] = static_cast< double >(10 - static_cast< int >(10 * check_fraction));
   }
   return counts;
}

double value_of(
   const auto& env,
   const auto& root,
   nor::Player responder,
   const auto& policy_table,
   const auto& opponent_table
)
{
   return opponent_aware::value_against(env, root, responder, policy_table, opponent_table);
}

}  // namespace

TEST(OpponentAwareDBBR, projection_recovers_consistent_strategy_and_beats_it)
{
   auto env = Environment{};
   auto root = State{};
   auto istates = make_kuhn_named_infostates();

   auto truth = true_profile_table(istates);
   auto counts = consistent_counts(istates);

   auto result = opponent_aware::dbbr_response(env, root, nor::Player::alex, counts);

   // PROJECTION: with fully observed, internally consistent data the recovered model IS the
   // generating strategy -- exactly, up to floating noise on the normalized frequencies
   ASSERT_EQ(result.model.size(), truth.size());
   for(const auto& [name, check_fraction] : k_true_check_fractions) {
      const auto& projected = result.model.at(istates.bob.at(name));
      EXPECT_NEAR(projected.at(Action::check), check_fraction, 1e-9);
      EXPECT_NEAR(projected.at(Action::bet), 1. - check_fraction, 1e-9);
   }

   // RESPONSE: against the same strategy the computed best response is optimal ...
   double dbr_value = value_of(env, root, nor::Player::alex, result.policy, truth);
   double oracle_value = best_response_value_against(env, root, nor::Player::bob, truth);
   std::cout << "[          ] dbbr value vs truth: " << dbr_value << " (oracle " << oracle_value
             << ")\n";
   EXPECT_NEAR(dbr_value, oracle_value, 1e-6);

   // ... and clearly beats what equilibrium play would guarantee against an exploitable
   // opponent
   EXPECT_GT(dbr_value, -1. / 18. + 0.02);

   // the root-value bookkeeping of the result matches the evaluation above
   EXPECT_NEAR(result.root_values.at(nor::Player::alex), dbr_value, 1e-6);
}

TEST(OpponentAwareDBBR, unobserved_infostates_complete_through_base_policy)
{
   auto env = Environment{};
   auto root = State{};
   auto istates = make_kuhn_named_infostates();

   // observe ONLY the jack-vs-bet decision ...
   counts_type partial_counts{};
   partial_counts[istates.bob.at("?jb")][Action::check] = 5.;
   partial_counts[istates.bob.at("?jb")][Action::bet] = 5.;

   // ... and anchor completions at a deliberately recognizable always-bet base policy
   opponent_aware::policy_table_type< Infostate, Action > anchor_table{};
   for(const auto& [_, infostate] : istates.bob) {
      anchor_table.emplace(infostate, HashmapActionPolicy< Action >{std::pair{Action::bet, 1.}});
   }

   auto result = opponent_aware::dbbr_response(
      env, root, nor::Player::alex, partial_counts, anchor_table
   );

   // coverage: every modeled infostate carries a completed distribution ...
   ASSERT_EQ(result.model.size(), istates.bob.size());
   // ... the OBSERVED one holds the raw empirical projection ...
   EXPECT_NEAR(result.model.at(istates.bob.at("?jb")).at(Action::check), 0.5, 1e-12);
   EXPECT_NEAR(result.model.at(istates.bob.at("?jb")).at(Action::bet), 0.5, 1e-12);
   // ... while every UNOBSERVED one completes through the supplied anchor verbatim
   for(const auto& [name, _] : k_true_check_fractions) {
      if(std::string_view{name} == "?jb") {
         continue;
      }
      const auto& completed = result.model.at(istates.bob.at(name));
      EXPECT_NEAR(completed.at(Action::bet), 1., 1e-12);
      EXPECT_NEAR(completed.at(Action::check), 0., 1e-12);
   }

   // sanity: the returned policy beats the completed anchor by a best response's margin
   double vs_anchor = value_of(env, root, nor::Player::alex, result.policy, result.model);
   double oracle_vs_anchor = best_response_value_against(env, root, nor::Player::bob, result.model);
   EXPECT_NEAR(vs_anchor, oracle_vs_anchor, 1e-6);
}
