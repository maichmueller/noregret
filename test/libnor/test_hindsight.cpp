#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <ranges>
#include <unordered_map>
#include <utility>
#include <vector>

#include "nor/env.hpp"
#include "nor/hindsight.hpp"
#include "nor/nor.hpp"
#include "rm_specific_testing_utils.hpp"
#include "shapley/shapley.hpp"

// Tests for the hindsight-rationality / correlated-play evaluator (Morrill et
// al., AAAI 2021, arXiv:2012.05874): phi-deviation gaps of the empirical
// distribution of play over the taxonomy
//   external (CCE) | blind causal (EFCCE) | action (AFCCE)
//   | blind counterfactual (CFCCE) | informed counterfactual (CFCE)
//
// Numerically verified relations (see the header's documentation for why
// causal-vs-action / causal-vs-counterfactual orderings are provably absent):
//   - every gap is non-negative;
//   - gap(blind CF) >= gap(informed CF): on a single recommendation draw the
//     informed transformation coincides with a blind one whose target action
//     is deflated by the recommendation-matching weight <= 1;
//   - on single-decision games (rps, shapley) the first four families coincide
//     exactly and informed CF is deflated by the trigger weight;
//   - Nash product profiles are gap-free in all families (interior Nash:
//     indifference makes every counterfactual gain vanish);
//   - deliberately correlated hand-built play exhibits strictly positive
//     trigger/counterfactual gaps while the external gap stays pinned at zero
//     (the recommendations are a best response to the marginal play).

using namespace nor;

namespace {

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// shared helpers //////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

template < typename Env >
using HT = hindsight::EmpiricalPlay< Env >;

template < typename Env >
using ProfileTable = hindsight::hindsight_table< Env >;

constexpr double kGapSlack = 1e-9;

/// collect the realized support (infoset -> legal actions) of every actual
/// player by running the module's collector against a uniform dummy profile
template < typename Env >
static std::unordered_map<
   Player,
   std::unordered_map< auto_info_state_type< Env >, std::vector< auto_action_type< Env > > > >
realized_support(
   const Env& env,
   const auto_world_state_type< Env >& root,
   const std::vector< Player >& players
)
{
   using table_type = ProfileTable< Env >;
   player_hashmap< table_type > uniform_profile{};
   for(auto player : players) {
      uniform_profile.emplace(player, table_type{});
   }
   std::unordered_map<
      Player,
      std::unordered_map< auto_info_state_type< Env >, std::vector< auto_action_type< Env > > > >
      support{};
   for(auto player : players) {
      typename hindsight::detail::ProfileWalker< Env, table_type >::Config config{
         .target_player = player, .profile = &uniform_profile, .record_entries = true};
      const auto result = hindsight::detail::ProfileWalker< Env, table_type >::run(
         env, root, config
      );
      auto& per_player = support[player];
      for(const auto& [infostate, entry] : result.entries) {
         auto keys = entry.action_values | std::views::keys;
         per_player.emplace(infostate, std::ranges::to< std::vector >(keys));
      }
   }
   return support;
}

/// a uniformly-random (seeded) behavioral profile over the given support
template < typename Env >
static player_hashmap< ProfileTable< Env > > random_profile(
   const std::unordered_map<
      Player,
      std::unordered_map< auto_info_state_type< Env >, std::vector< auto_action_type< Env > > > >&
      support,
   std::mt19937_64& rng
)
{
   using ActionPolicy = HashmapActionPolicy< auto_action_type< Env > >;
   std::uniform_real_distribution< double > uniform01{0., 1.};
   player_hashmap< ProfileTable< Env > > profile{};
   for(const auto& [player, per_infostate] : support) {
      ProfileTable< Env > table{};
      for(const auto& [infostate, actions] : per_infostate) {
         std::vector< double > weights{};
         weights.reserve(actions.size());
         double total = 0.;
         for([[maybe_unused]] auto _ : actions) {
            double w = uniform01(rng);
            weights.push_back(w);
            total += w;
         }
         ActionPolicy row{};
         for(const auto [idx, action] : std::views::enumerate(actions)) {
            row.emplace(action, weights[idx] / total);
         }
         table.emplace(infostate, std::move(row));
      }
      profile.emplace(player, std::move(table));
   }
   return profile;
}

static void expect_valid_gap_vector(
   const char* label,
   Player player,
   const std::array< double, hindsight::deviation_family_count() >& gaps
)
{
   SCOPED_TRACE(label);
   for(const auto family :
       {hindsight::DeviationFamily::external,
        hindsight::DeviationFamily::blind_causal,
        hindsight::DeviationFamily::action,
        hindsight::DeviationFamily::blind_counterfactual,
        hindsight::DeviationFamily::informed_counterfactual}) {
      EXPECT_GE(gaps[hindsight::family_index(family)], -kGapSlack)
         << label << ": negative gap for player " << common::to_string(player) << " family "
         << static_cast< int >(family);
   }
   // the provable counterfactual-family inclusion
   EXPECT_GE(
      gaps[hindsight::family_index(hindsight::DeviationFamily::blind_counterfactual)],
      gaps[hindsight::family_index(hindsight::DeviationFamily::informed_counterfactual)] - kGapSlack
   ) << label
     << ": informed CF gap exceeds blind CF gap for player " << common::to_string(player);
}

static void print_gaps(
   const char* label,
   const player_hashmap< std::array< double, hindsight::deviation_family_count() > >& report
)
{
   for(const auto& [player, gaps] : report) {
      std::cout << "[" << label << "] " << common::to_string(player) << " | ext: " << gaps[0]
                << " blind-causal: " << gaps[1] << " action: " << gaps[2]
                << " blind-CF: " << gaps[3] << " informed-CF: " << gaps[4] << "\n";
   }
}

}  // namespace

/////////////////////////////////////////////////////////////////////////////////////////////
//////////////////// (a) taxonomy properties on random empirical plays //////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

TEST(HindsightTaxonomy, RandomKuhnPlaysRespectTheProvableInclusions)
{
   using Env = games::kuhn::Environment;
   Env env{};
   const Env::world_state_type root{};
   const std::vector< Player > players{Player::alex, Player::bob};

   const auto support = realized_support(env, root, players);

   // single-snapshot random plays: the provable inclusions must hold exactly
   for(size_t seed = 0; seed < 20; ++seed) {
      std::mt19937_64 rng{seed};
      HT< Env > play{};
      play.add(1., random_profile< Env >(support, rng));

      const auto report = hindsight::hindsight_gaps(env, root, play);
      print_gaps("kuhn-random-single", report);
      for(const auto& [player, gaps] : report) {
         expect_valid_gap_vector("kuhn-random-single", player, gaps);
      }
   }

   // multi-snapshot random plays: gaps stay well-defined and non-negative
   for(size_t seed = 100; seed < 105; ++seed) {
      std::mt19937_64 rng{seed};
      HT< Env > play{};
      for([[maybe_unused]] auto _ : std::views::iota(0, 3)) {
         play.add(1., random_profile< Env >(support, rng));
      }
      const auto report = hindsight::hindsight_gaps(env, root, play);
      print_gaps("kuhn-random-multi", report);
      for(const auto& [player, gaps] : report) {
         for(auto gap : gaps) {
            EXPECT_GE(gap, -kGapSlack);
         }
      }
   }
}

TEST(HindsightTaxonomy, RandomThreePlayerKuhnPlaysRespectTheProvableInclusions)
{
   using Env = games::kuhn::Environment;
   Env env{};
   const Env::world_state_type root{
      std::vector< games::kuhn::Card >{
         games::kuhn::Card::jack, games::kuhn::Card::queen, games::kuhn::Card::king},
      /*player_count=*/3};
   const std::vector< Player > players{Player::alex, Player::bob, Player::cedric};

   {
      auto root_players = env.players(root);
      std::erase(root_players, Player::chance);
      ASSERT_EQ(root_players.size(), size_t{3});
   }
   const auto support = realized_support(env, root, players);

   for(size_t seed = 0; seed < 10; ++seed) {
      std::mt19937_64 rng{seed};
      HT< Env > play{};
      play.add(1., random_profile< Env >(support, rng));
      const auto report = hindsight::hindsight_gaps(env, root, play);
      print_gaps("kuhn-3p-random", report);
      for(const auto& [player, gaps] : report) {
         expect_valid_gap_vector("kuhn-3p-random", player, gaps);
      }
   }
}

TEST(HindsightTaxonomy, SingleDecisionGamesAllNonInformedFamiliesCoincide)
{
   // with a single infoset per player, external == blind causal == action ==
   // blind CF (all reduce to forcing one action at the only decision), and the
   // informed variant is the same quantity deflated by sigma(I, a) <= 1.
   {
      using Env = games::rps::Environment;
      Env env{};
      const Env::world_state_type root{};
      const auto support = realized_support(env, root, {Player::alex, Player::bob});

      for(size_t seed = 0; seed < 10; ++seed) {
         std::mt19937_64 rng{seed};
         HT< Env > play{};
         play.add(1., random_profile< Env >(support, rng));
         const auto report = hindsight::hindsight_gaps(env, root, play);
         print_gaps("rps-random", report);
         for(const auto& [player, gaps] : report) {
            const double
               reference = gaps[hindsight::family_index(hindsight::DeviationFamily::external)];
            for(const auto family :
                {hindsight::DeviationFamily::blind_causal,
                 hindsight::DeviationFamily::action,
                 hindsight::DeviationFamily::blind_counterfactual}) {
               EXPECT_NEAR(gaps[hindsight::family_index(family)], reference, kGapSlack);
            }
            EXPECT_LE(
               gaps[hindsight::family_index(hindsight::DeviationFamily::informed_counterfactual)],
               reference + kGapSlack
            );
         }
      }
   }
   {
      using Env = games::shapley::Environment;
      Env env{};
      const Env::world_state_type root{};
      const auto support = realized_support(env, root, {Player::alex, Player::bob});

      for(size_t seed = 0; seed < 10; ++seed) {
         std::mt19937_64 rng{seed};
         HT< Env > play{};
         play.add(1., random_profile< Env >(support, rng));
         const auto report = hindsight::hindsight_gaps(env, root, play);
         print_gaps("shapley-random", report);
         for(const auto& [player, gaps] : report) {
            const double
               reference = gaps[hindsight::family_index(hindsight::DeviationFamily::external)];
            for(const auto family :
                {hindsight::DeviationFamily::blind_causal,
                 hindsight::DeviationFamily::action,
                 hindsight::DeviationFamily::blind_counterfactual}) {
               EXPECT_NEAR(gaps[hindsight::family_index(family)], reference, kGapSlack);
            }
            EXPECT_LE(
               gaps[hindsight::family_index(hindsight::DeviationFamily::informed_counterfactual)],
               reference + kGapSlack
            );
         }
      }
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////// (b) Nash product profiles are gap-free /////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

TEST(HindsightNash, InteriorNashProductProfileHasVanishingGapsInKuhn)
{
   using Environment = games::kuhn::Environment;
   using State = games::kuhn::State;
   using Infostate = games::kuhn::Infostate;
   using Action = games::kuhn::Action;
   using Card = games::kuhn::Card;
   Environment env{};
   const State root{};

   // interior member of the kuhn Nash family (alpha in (0, 1/3)): every action
   // is played with positive probability, so all infosets are reached and the
   // indifference property zeroes every counterfactual gain, while Nash
   // optimality zeroes every plain deviation gain
   constexpr double alpha = 0.15;
   auto [optimal_alex, optimal_bob] = kuhn_optimal(alpha);

   HT< Environment > play{};
   play.add(
      1.,
      player_hashmap< ProfileTable< Environment > >{
         std::pair{Player::alex, std::move(optimal_alex)},
         std::pair{Player::bob, std::move(optimal_bob)}}
   );

   const auto report = hindsight::hindsight_gaps(env, root, play);
   print_gaps("kuhn-nash", report);
   for(const auto& [player, gaps] : report) {
      for(auto gap : gaps) {
         EXPECT_NEAR(gap, 0., 1e-8) << "non-zero gap for player " << common::to_string(player);
      }
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////
//////////// (c) deliberately correlated hand-built play exploits triggers //////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

namespace {

enum class BobBehavior { always_check, always_bet };

template < typename Env, typename Pred >
static ProfileTable< Env > forced_behavior_table(
   const std::unordered_map< auto_info_state_type< Env >, std::vector< auto_action_type< Env > > >&
      infostate_actions,
   Pred&& forced_action_of
)
{
   using ActionPolicy = HashmapActionPolicy< auto_action_type< Env > >;
   ProfileTable< Env > table{};
   for(const auto& [infostate, actions] : infostate_actions) {
      ActionPolicy row{};
      for(const auto& action : actions) {
         row.emplace(action, forced_action_of(action) ? 1. : 0.);
      }
      table.emplace(infostate, std::move(row));
   }
   return table;
}

}  // namespace

TEST(HindsightCorrelated, TriggerGapsPositiveWhileExternalPinnedToZero)
{
   using Environment = games::kuhn::Environment;
   using State = games::kuhn::State;
   using Infostate = games::kuhn::Infostate;
   using Action = games::kuhn::Action;
   using Card = games::kuhn::Card;
   using ActionPolicy = HashmapActionPolicy< Action >;
   Environment env{};
   const State root{};

   // Two equally-weighted joint plays in which ALEX's behavioral recommendation
   // is IDENTICAL -- the average best response against the 50/50 mixture of
   // BOB's behaviors (computed below by brute-force enumeration) -- while BOB's
   // recommendation flips between diametrically opposed behaviors:
   //   play 1: bob always checks (never bets, never calls)
   //   play 2: bob always bets (opens every time, calls every bet)
   // Alex's external gap is therefore exactly zero (his recommendation is a
   // best response to the marginal correlation), yet the correlation is
   // sequentially exploitable: bob's own recommendation is far from any
   // sequentially rational one, giving strictly positive trigger /
   // counterfactual gaps.

   const auto support = realized_support(env, root, {Player::alex, Player::bob});
   const auto& alex_support = support.at(Player::alex);
   const auto& bob_support = support.at(Player::bob);

   auto bob_table_of = [&](BobBehavior behavior) {
      return forced_behavior_table< Environment >(
         bob_support,
         [&](const games::kuhn::Action& action) {
            return behavior == BobBehavior::always_check ? action == games::kuhn::Action::check
                                                         : action == games::kuhn::Action::bet;
         }
      );
   };

   // brute-force alex's best response against the bob mixture: enumerate all
   // deterministic alex strategies over the realized support and evaluate the
   // mixture value through the module's plain evaluator
   auto mixture_value_against = [&](const ProfileTable< Environment >& alex_table) {
      namespace hs = hindsight;
      double acc = 0.;
      for(auto behavior : {BobBehavior::always_check, BobBehavior::always_bet}) {
         player_hashmap< ProfileTable< Environment > > profile{};
         profile.emplace(Player::alex, alex_table);
         profile.emplace(Player::bob, bob_table_of(behavior));
         typename hs::detail::ProfileWalker< Environment, ProfileTable< Environment > >::Config
            config{.target_player = Player::alex, .profile = &profile};
         acc += hs::detail::ProfileWalker< Environment, ProfileTable< Environment > >::run(
                   env, root, config
         )
                   .value;
      }
      return acc / 2.;
   };

   std::vector< games::kuhn::Infostate > alex_infosets{};
   for(const auto& [infostate, actions] : alex_support) {
      (void) actions;
      alex_infosets.push_back(infostate);
   }

   auto build_candidate = [&](const std::vector< size_t >& cursor) {
      ProfileTable< Environment > candidate{};
      for(const auto [idx, infostate] : std::views::enumerate(alex_infosets)) {
         const auto& actions = alex_support.at(infostate);
         ActionPolicy row{};
         for(const auto [aidx, action] : std::views::enumerate(actions)) {
            row.emplace(action, aidx == cursor[idx] ? 1. : 0.);
         }
         candidate.emplace(infostate, std::move(row));
      }
      return candidate;
   };
   auto enumerate_deterministic_alex = [&](auto&& sink) {
      std::vector< size_t > cursor(alex_infosets.size(), 0);
      bool running = true;
      while(running) {
         std::invoke(sink, build_candidate(cursor));
         running = false;
         for(size_t idx = 0; idx < cursor.size(); ++idx) {
            if(++cursor[idx] < alex_support.at(alex_infosets[idx]).size()) {
               running = true;
               break;
            }
            cursor[idx] = 0;
         }
      }
   };

   ProfileTable< Environment > best_alex_table{};
   double best_value = -std::numeric_limits< double >::infinity();
   enumerate_deterministic_alex([&](const ProfileTable< Environment >& candidate) {
      const double value = mixture_value_against(candidate);
      if(value > best_value) {
         best_value = value;
         best_alex_table = candidate;
      }
   });
   ASSERT_NEAR(mixture_value_against(best_alex_table), best_value, 1e-12);

   HT< Environment > play{};
   {
      player_hashmap< ProfileTable< Environment > > profile{};
      profile.emplace(Player::alex, best_alex_table);
      profile.emplace(Player::bob, bob_table_of(BobBehavior::always_check));
      play.add(1., std::move(profile));
   }
   {
      player_hashmap< ProfileTable< Environment > > profile{};
      profile.emplace(Player::alex, best_alex_table);
      profile.emplace(Player::bob, bob_table_of(BobBehavior::always_bet));
      play.add(1., std::move(profile));
   }

   const auto report = hindsight::hindsight_gaps(env, root, play);
   print_gaps("kuhn-correlated", report);

   const auto& alex_gaps = report.at(Player::alex);
   const auto& bob_gaps = report.at(Player::bob);

   // alex follows a best response to the marginal correlation: no beneficial
   // EXTERNAL deviation exists at all
   EXPECT_NEAR(alex_gaps[hindsight::family_index(hindsight::DeviationFamily::external)], 0., 1e-9);

   // the correlated play is nevertheless trigger/counterfactually exploitable
   EXPECT_GT(bob_gaps[hindsight::family_index(hindsight::DeviationFamily::action)], 1e-6);
   EXPECT_GT(
      bob_gaps[hindsight::family_index(hindsight::DeviationFamily::blind_counterfactual)], 1e-6
   );
}

/////////////////////////////////////////////////////////////////////////////////////////////
///////////////////// (d) integration: VanillaCFR iterates on kuhn //////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

TEST(HindsightIntegration, VanillaCfrIteratesOnKuhnShowDecreasingGaps)
{
   using Environment = games::kuhn::Environment;
   using State = games::kuhn::State;
   using Infostate = games::kuhn::Infostate;
   using Action = games::kuhn::Action;
   Environment env{};
   auto root_state = std::make_unique< State >();

   auto current = factory::make_tabular_policy(
      std::unordered_map< Infostate, HashmapActionPolicy< Action > >{}
   );
   auto average = factory::make_tabular_policy(
      std::unordered_map< Infostate, HashmapActionPolicy< Action > >{}
   );

   auto solver = factory::make_cfr< rm::CFRConfig{}, true >(
      Environment{}, std::move(root_state), current, average
   );

   hindsight::EmpiricalPlay< Environment > early_play{};
   hindsight::EmpiricalPlay< Environment > late_play{};

   for(size_t iteration = 1; iteration <= 30000; ++iteration) {
      solver.iterate(1);
      if(iteration == 2000 || iteration == 2500 || iteration == 3000 || iteration == 3500
         || iteration == 4000) {
         early_play.record_average(solver);
      }
      if(iteration >= 25000 && (iteration - 25000) % 1000 == 0) {
         late_play.record_average(solver);
      }
   }

   ASSERT_FALSE(early_play.empty());
   ASSERT_FALSE(late_play.empty());

   const auto early_report = hindsight::hindsight_gaps(env, State{}, early_play);
   const auto late_report = hindsight::hindsight_gaps(env, State{}, late_play);
   print_gaps("kuhn-vanilla-cfr-early(2-4k)", early_report);
   print_gaps("kuhn-vanilla-cfr-late(25-30k)", late_report);

   const auto total_of = [](const auto& report) {
      double acc = 0.;
      for(const auto& [player, gaps] : report) {
         (void) player;
         acc += std::accumulate(gaps.begin(), gaps.end(), 0.);
      }
      return acc;
   };
   const double early_total = total_of(early_report);
   const double late_total = total_of(late_report);
   std::cout << "[hindsight integration] total gap mass early: " << early_total
             << " late: " << late_total << "\n";
   EXPECT_GT(early_total, 1e-4);
   EXPECT_LT(late_total, early_total);
}
