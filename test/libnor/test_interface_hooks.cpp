
#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <unordered_map>
#include <vector>

#include "nor/concepts/has.hpp"
#include "nor/env.hpp"
#include "nor/factory.hpp"
#include "nor/nor.hpp"
#include "nor/rm/sampling_rules.hpp"
#include "nor/utils/infostate_unpacker.hpp"

using namespace nor;

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// B1: discount schedules //////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(B1DiscountSchedules, constants_reproduce_default_behavior)
{
   rm::CFRDiscountedParameters params{};
   EXPECT_DOUBLE_EQ(params.alpha_at(0), 1.5);
   EXPECT_DOUBLE_EQ(params.alpha_at(100), 1.5);
   EXPECT_DOUBLE_EQ(params.beta_at(7), 0.);
   EXPECT_DOUBLE_EQ(params.gamma_at(3), 2.);
}

TEST(B1DiscountSchedules, schedules_override_constants)
{
   rm::CFRDiscountedParameters params{};
   params.alpha_schedule = [](size_t t) { return 1. + static_cast< double >(t) / 10.; };
   params.gamma_schedule = [](size_t) { return 1.25; };

   EXPECT_DOUBLE_EQ(params.alpha_at(0), 1.);
   EXPECT_DOUBLE_EQ(params.alpha_at(10), 2.);
   EXPECT_DOUBLE_EQ(params.gamma_at(999), 1.25);
   // unscheduled parameters keep constant behavior
   EXPECT_DOUBLE_EQ(params.beta_at(4), 0.);
}

TEST(B1DiscountSchedules, scheduled_discounted_cfr_converges_on_kuhn)
{
   using namespace nor;
   games::kuhn::Environment env{};
   auto root_state = std::make_unique< games::kuhn::State >();
   auto curr = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{}
   );
   auto avg = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{}
   );

   constexpr auto cfg = rm::CFRDiscountedConfig{};
   rm::CFRDiscountedParameters params{};
   params.alpha_schedule = [](size_t t) { return std::pow(static_cast< double >(t) + 1., -0.5); };
   auto solver = factory::make_cfr_discounted< cfg, true >(
      env, std::move(root_state), curr, avg, params
   );
   solver.iterate(200);

   const auto& ap = solver.average_policy();
   const double expl = exploitability(
      env,
      games::kuhn::State{},
      player_hashmap< std::decay_t< decltype(ap.at(Player::alex)) > >{
         {Player::alex, normalize_state_policy(ap.at(Player::alex))},
         {Player::bob, normalize_state_policy(ap.at(Player::bob))}}
   );
   EXPECT_LT(expl, 0.05);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// B3: cycle-based weighting ////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(B3CycleWeighting, cycle_index_is_iteration_over_players)
{
   using namespace nor;
   games::kuhn::Environment env{};
   auto root_state = std::make_unique< games::kuhn::State >();
   auto curr = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{}
   );
   auto avg = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{}
   );

   auto solver = factory::
      make_cfr_vanilla< rm::CFRConfig{.update_mode = rm::UpdateMode::alternating}, true >(
         env, std::move(root_state), curr, avg
      );
   solver.iterate(5);
   EXPECT_EQ(solver.iteration(), size_t{5});
   EXPECT_EQ(solver.cycle(), size_t{2});  // 5 / 2 players
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// B4: payoff bounds trait ////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

struct WithBounds {
   [[nodiscard]] std::pair< double, double > payoff_bounds(Player) const { return {-1., 2.}; }
};

struct WithoutBounds {};

}  // namespace

static_assert(
   nor::concepts::has::method::payoff_bounds< WithBounds >,
   "payoff_bounds detection failed"
);
static_assert(
   not nor::concepts::has::method::payoff_bounds< WithoutBounds >,
   "payoff_bounds detection produced a false positive"
);

TEST(B4PayoffBounds, concept_detects_method_presence)
{
   EXPECT_TRUE((nor::concepts::has::supports_payoff_bounds< WithBounds >) );
   EXPECT_FALSE((nor::concepts::has::supports_payoff_bounds< WithoutBounds >) );
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// B5: InfostateUnpacker skeleton //////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(B5InfostateUnpacker, records_representatives_and_prefix_probs)
{
   games::kuhn::Environment env{};
   games::kuhn::State wstate{};
   wstate.apply_action(games::kuhn::ChanceOutcome{games::kuhn::Player::one, games::kuhn::Card::king}
   );
   wstate.apply_action(games::kuhn::ChanceOutcome{
      games::kuhn::Player::two, games::kuhn::Card::queen});

   InfostateUnpacker< games::kuhn::Environment > unpacker(/*record_representatives=*/true);
   games::kuhn::Infostate istate(Player::alex);
   unpacker.record(istate, wstate, 0.25);
   unpacker.record(istate, wstate, 0.125);

   ASSERT_EQ(unpacker.representatives(istate).size(), size_t{2});
   EXPECT_DOUBLE_EQ(unpacker.chance_prefix_prob(istate), 0.375);

   // unknown infostates yield empty defaults
   games::kuhn::Infostate other(Player::bob);
   EXPECT_TRUE(unpacker.representatives(other).empty());
   EXPECT_DOUBLE_EQ(unpacker.chance_prefix_prob(other), 0.);
}

TEST(B5InfostateUnpacker, recording_disabled_by_default)
{
   InfostateUnpacker< games::kuhn::Environment > unpacker;
   EXPECT_FALSE(unpacker.recording());
   games::kuhn::State wstate{};
   games::kuhn::Infostate istate(Player::alex);
   unpacker.record(istate, wstate, 1.);
   EXPECT_TRUE(unpacker.representatives(istate).empty());
}

/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// B6: custom sampling rules ////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

template < typename Action >
struct FixedChoiceRule {
   Action forced;
   double prob;

   template < typename WeightOf >
   rm::SampledChoice< Action >
   operator()(common::RNG&, const std::vector< Action >& actions, const WeightOf&) const
   {
      for(const auto& action : actions) {
         if(action == forced) {
            return {action, prob};
         }
      }
      throw std::out_of_range("forced action not legal");
   }
};

}  // namespace

static_assert(
   rm::sampling_rule_for<
      rm::EpsilonOnPolicySamplingRule,
      games::kuhn::Action,
      decltype([](const games::kuhn::Action&) { return .5; }) >,
   "default rule must satisfy the sampling rule concept"
);

TEST(B6SamplingRules, default_rule_matches_epsilon_mixture_probability)
{
   common::RNG rng{common::default_seed};
   std::vector< int > actions{0, 1, 2, 3};
   rm::EpsilonOnPolicySamplingRule rule{0.6};
   const auto choice = rule(rng, actions, [](int) { return 0.5; });
   // mixture probability of any action with uniform policy:
   //    eps * 1/|A| + (1-eps) * 0.5 = 0.6*0.25 + 0.4*0.5 = 0.35
   EXPECT_DOUBLE_EQ(choice.sample_prob, 0.35);
   EXPECT_GE(choice.action, 0);
   EXPECT_LT(choice.action, 4);
}

TEST(B6SamplingRules, custom_rule_drives_outcome_sampling)
{
   using namespace nor;
   games::kuhn::Environment env{};
   auto root_state = std::make_unique< games::kuhn::State >();
   auto curr = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{}
   );
   auto avg = factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{}
   );

   constexpr auto cfg = rm::MCCFRConfig{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .exploration = rm::MCCFRExplorationMode::custom_sampling_policy};

   FixedChoiceRule< games::kuhn::Action > rule{games::kuhn::Action::check, 0.9};
   rm::MCCFR<
      cfg,
      games::kuhn::Environment,
      std::decay_t< decltype(curr) >,
      std::decay_t< decltype(avg) >,
      FixedChoiceRule< games::kuhn::Action > >
      solver{
         std::move(env),
         std::move(root_state),
         std::move(curr),
         std::move(avg),
         /*epsilon=*/0.6,
         /*seed=*/common::default_seed,
         rule};
   solver.iterate(50);
   SUCCEED() << "custom sampling rule drove 50 iterations without failure";
}
