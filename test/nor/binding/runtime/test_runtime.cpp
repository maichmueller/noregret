#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "nor/binding/runtime/runtime.hpp"
#include "nor/meta/enum_names.hpp"

using namespace nor::binding::runtime;
using nor::Player;
using nor::Stochasticity;

namespace {

enum class ProviderMode {
   deterministic,
   chance,
   sample,
   max_count_zero,
   wrong_count,
   duplicate_players,
   bad_active,
   bad_actions,
   bad_chance,
   not_serialized,
   not_unrolled
};

class TestDynamicProvider final: public DynamicEnvironmentProvider {
  public:
   using DynamicEnvironmentProvider::private_observation;
   using DynamicEnvironmentProvider::public_observation;
   using DynamicEnvironmentProvider::transition;

   explicit TestDynamicProvider(
      ProviderMode mode,
      std::shared_ptr< size_t > calls = std::make_shared< size_t >(0)
   )
       : m_mode(mode), m_calls(std::move(calls))
   {
   }

   [[nodiscard]] size_t max_player_count() const final
   {
      return m_mode == ProviderMode::max_count_zero ? 0 : 2;
   }
   [[nodiscard]] size_t player_count() const final
   {
      return m_mode == ProviderMode::wrong_count ? 1 : 2;
   }
   [[nodiscard]] Stochasticity stochasticity() const final
   {
      if(m_mode == ProviderMode::sample)
         return Stochasticity::sample;
      return is_chance_mode() ? Stochasticity::choice : Stochasticity::deterministic;
   }
   [[nodiscard]] bool serialized() const final { return m_mode != ProviderMode::not_serialized; }
   [[nodiscard]] bool unrolled() const final { return m_mode != ProviderMode::not_unrolled; }

   [[nodiscard]] DynamicWorldState initial_world_state() const final
   {
      return world(is_chance_mode() ? "chance_root" : "root");
   }

   [[nodiscard]] std::vector< DynamicAction > actions(Player, const DynamicWorldState& state)
      const final
   {
      touch();
      if(state.value().identity() == "terminal")
         return {};
      if(is_chance_mode() and state.value().identity() == "chance_root")
         return {};
      if(m_mode == ProviderMode::bad_actions)
         return {DynamicAction{}};
      return {action("left"), action("right")};
   }

   [[nodiscard]] std::vector< Player > players(const DynamicWorldState&) const final
   {
      touch();
      if(m_mode == ProviderMode::duplicate_players)
         return {Player::alex, Player::alex};
      return {Player::alex, Player::bob};
   }

   [[nodiscard]] Player active_player(const DynamicWorldState& state) const final
   {
      touch();
      if(m_mode == ProviderMode::bad_active)
         return Player::unknown;
      if(is_chance_mode() and state.value().identity() == "chance_root")
         return Player::chance;
      if(state.value().identity() == "root")
         return Player::alex;
      if(state.value().identity() == "after")
         return Player::bob;
      return Player::bob;
   }

   [[nodiscard]] bool is_terminal(const DynamicWorldState& state) const final
   {
      touch();
      return state.value().identity() == "terminal";
   }

   [[nodiscard]] bool is_partaking(const DynamicWorldState&, Player player) const final
   {
      touch();
      return player == Player::alex or player == Player::bob;
   }

   [[nodiscard]] double reward(Player player, const DynamicWorldState& state) const final
   {
      touch();
      if(state.value().identity() != "terminal")
         return 0.;
      return player == Player::alex ? 1. : -1.;
   }

   void transition(DynamicWorldState& state, const DynamicAction&) const final
   {
      touch();
      if(state.value().identity() == "root" or state.value().identity() == "chance_root")
         state.set(world_value("after"));
      else
         state.set(world_value("terminal"));
   }

   [[nodiscard]] DynamicObservation
   private_observation(Player, const DynamicWorldState&, const DynamicAction& action_value, const DynamicWorldState&)
      const final
   {
      touch();
      return observation(std::string(action_value.identity()));
   }

   [[nodiscard]] DynamicObservation
   public_observation(const DynamicWorldState&, const DynamicAction& action_value, const DynamicWorldState&)
      const final
   {
      touch();
      return observation(std::string(action_value.identity()));
   }

   [[nodiscard]] std::vector< DynamicChanceOutcome > chance_actions(const DynamicWorldState& state
   ) const final
   {
      touch();
      if(not is_chance_mode() or state.value().identity() != "chance_root")
         return {};
      return {outcome("heads"), outcome("tails")};
   }

   [[nodiscard]] double chance_probability(const DynamicWorldState&, const DynamicChanceOutcome&)
      const final
   {
      touch();
      return m_mode == ProviderMode::bad_chance ? 0.6 : 0.5;
   }

   void transition(DynamicWorldState& state, const DynamicChanceOutcome&) const final
   {
      touch();
      state.set(world_value("after"));
   }

   [[nodiscard]] DynamicObservation
   private_observation(Player, const DynamicWorldState&, const DynamicChanceOutcome& outcome_value, const DynamicWorldState&)
      const final
   {
      touch();
      return observation(std::string(outcome_value.identity()));
   }

   [[nodiscard]] DynamicObservation
   public_observation(const DynamicWorldState&, const DynamicChanceOutcome& outcome_value, const DynamicWorldState&)
      const final
   {
      touch();
      return observation(std::string(outcome_value.identity()));
   }

   [[nodiscard]] const std::shared_ptr< size_t >& calls() const noexcept { return m_calls; }

  private:
   [[nodiscard]] bool is_chance_mode() const noexcept
   {
      return m_mode == ProviderMode::chance or m_mode == ProviderMode::bad_chance;
   }

   static DynamicWorldState world(std::string identity)
   {
      return DynamicWorldState{world_value(std::move(identity))};
   }

   static DynamicWorldValue world_value(std::string identity)
   {
      return DynamicWorldValue::named("test.world", std::move(identity));
   }

   static DynamicAction action(std::string identity)
   {
      return DynamicAction::named("test.action", std::move(identity));
   }

   static DynamicChanceOutcome outcome(std::string identity)
   {
      return DynamicChanceOutcome::named("test.outcome", std::move(identity));
   }

   static DynamicObservation observation(std::string identity)
   {
      return DynamicObservation::named("test.observation", std::move(identity));
   }

   void touch() const { ++*m_calls; }

   ProviderMode m_mode;
   std::shared_ptr< size_t > m_calls;
};

[[nodiscard]] Result< SolverSession > make_rps_session(
   SolverId solver = SolverId::vanilla_cfr,
   ProfileId profile = ProfileId::vanilla_alternating
)
{
   auto game = make_game(GameSpec::defaults(GameId::rps));
   if(not game)
      return std::unexpected(game.error());
   return make_session(*game, solver, profile);
}

void expect_one_static_iteration(GameId game_id, SolverId solver, ProfileId profile)
{
   auto handle = make_game(GameSpec::defaults(game_id));
   ASSERT_TRUE(handle) << handle.error().message;
   ASSERT_NE(find_capability(game_id, solver, profile), nullptr);
   auto session = make_session(*handle, solver, profile);
   ASSERT_TRUE(session) << session.error().message;
   auto result = session->iterate();
   ASSERT_TRUE(result) << result.error().message;
   EXPECT_EQ(result->iteration, 0u);
   EXPECT_FALSE(result->root_values.empty());
}

template < typename Mutator >
void expect_reentrant_mutation_is_rejected(Mutator mutator)
{
   auto session = make_rps_session();
   ASSERT_TRUE(session) << session.error().message;
   ASSERT_TRUE(session->iterate());
   auto lookup_result = session->policy_lookup();
   ASSERT_TRUE(lookup_result) << lookup_result.error().message;
   auto lookup = std::move(*lookup_result);

   std::optional< CapabilityError > nested_error;
   ASSERT_GT(
      lookup.visit([&](const PolicyNodeView&) {
         auto nested = std::invoke(mutator, *session);
         if(nested)
            ADD_FAILURE() << "reentrant mutation unexpectedly succeeded";
         else
            nested_error = nested.error();
      }),
      0u
   );
   ASSERT_TRUE(nested_error.has_value());
   EXPECT_EQ(nested_error->code, CapabilityErrorCode::session_failure);
   EXPECT_NE(nested_error->message.find("mutation during policy visitation"), std::string::npos);
   EXPECT_TRUE(lookup.valid());
}

}  // namespace

TEST(Catalog, DescriptorsReflectAndAreUnique)
{
   const auto& static_catalog = catalog();
   ASSERT_FALSE(static_catalog.games.empty());
   ASSERT_FALSE(static_catalog.solvers.empty());
   ASSERT_FALSE(static_catalog.profiles.empty());
   ASSERT_FALSE(static_catalog.combinations.empty());

   std::set< GameId > game_ids;
   for(const auto& game : static_catalog.games) {
      EXPECT_TRUE(game_ids.insert(game.id).second);
      EXPECT_EQ(game.name, nor::meta::enum_name(game.id));
      std::set< GameFieldId > field_ids;
      for(const auto& field : game.fields) {
         EXPECT_TRUE(field_ids.insert(field.id).second);
         EXPECT_EQ(field.name, nor::meta::enum_name(field.id));
      }
   }

   std::set< ProfileId > profile_ids;
   for(const auto& profile : static_catalog.profiles) {
      EXPECT_TRUE(profile_ids.insert(profile.id).second);
      EXPECT_EQ(profile.name, nor::meta::enum_name(profile.id));
      ASSERT_NE(find_solver(profile.solver), nullptr);
   }

   std::set< std::tuple< GameId, SolverId, ProfileId > > capabilities;
   for(const auto& capability : static_catalog.combinations) {
      EXPECT_TRUE(
         capabilities.insert({capability.game, capability.solver, capability.profile}).second
      );
      EXPECT_NE(capability.create, nullptr);
      EXPECT_NE(find_game(capability.game), nullptr);
      EXPECT_NE(find_profile(capability.profile), nullptr);
   }
}

TEST(Catalog, ConstructsEveryAdmittedCapability)
{
   for(const auto& game : catalog().games) {
      auto handle = make_game(GameSpec::defaults(game.id));
      ASSERT_TRUE(handle) << game.name << ": " << handle.error().message;
      for(const auto& capability : capabilities_for(game.id)) {
         auto session = make_session(*handle, capability.solver, capability.profile);
         ASSERT_TRUE(session) << game.name << "/" << nor::meta::enum_name(capability.profile)
                              << ": " << session.error().message;
         EXPECT_TRUE(*session);
      }
   }
}

TEST(Session, CoarseOperationsDoNotCollectByDefault)
{
   auto session = make_rps_session();
   ASSERT_TRUE(session) << session.error().message;

   auto first = session->iterate();
   ASSERT_TRUE(first) << first.error().message;
   EXPECT_EQ(first->iteration, 0u);
   EXPECT_FALSE(first->root_values.empty());

   auto advanced = session->advance(2);
   ASSERT_TRUE(advanced) << advanced.error().message;
   ASSERT_TRUE(session->stats());
   EXPECT_EQ(session->stats()->iteration, 3u);

   auto no_last = session->advance_last(0);
   ASSERT_TRUE(no_last);
   EXPECT_FALSE(no_last->has_value());

   auto last = session->advance_last(2);
   ASSERT_TRUE(last) << last.error().message;
   ASSERT_TRUE(last->has_value());
   EXPECT_EQ(last->value().iteration, 4u);

   auto trace = session->trace(5, 2);
   ASSERT_TRUE(trace) << trace.error().message;
   EXPECT_EQ(trace->first_iteration, 5u);
   EXPECT_EQ(trace->last_iteration, 10u);
   ASSERT_EQ(trace->size(), 2u);
   EXPECT_EQ(trace->iterations[0].iteration, 6u);
   EXPECT_EQ(trace->iterations[1].iteration, 8u);

   auto bad_trace = session->trace(1, 0);
   ASSERT_FALSE(bad_trace);
   EXPECT_EQ(bad_trace.error().code, CapabilityErrorCode::invalid_spec);
}

TEST(Session, BorrowedViewsBecomeStaleAfterMutationAndDestruction)
{
   std::optional< PolicyLookup > lookup;
   std::optional< PolicyNodeView > row;
   std::optional< ErasedInfoState > key;

   {
      auto session = make_rps_session();
      ASSERT_TRUE(session) << session.error().message;
      ASSERT_TRUE(session->iterate());
      auto lookup_result = session->policy_lookup();
      ASSERT_TRUE(lookup_result) << lookup_result.error().message;
      lookup.emplace(std::move(*lookup_result));
      ASSERT_GT(
         lookup->visit([&](const PolicyNodeView& node) {
            if(not key) {
               key = node.info_state();
               row = node;
            }
         }),
         0u
      );
      ASSERT_TRUE(key.has_value());
      ASSERT_TRUE(row.has_value());
      EXPECT_TRUE(lookup->valid());
      EXPECT_TRUE(row->valid());

      auto mutation = session->advance(1);
      ASSERT_TRUE(mutation) << mutation.error().message;
      EXPECT_FALSE(lookup->valid());
      EXPECT_FALSE(row->valid());
      EXPECT_THROW((void) lookup->find(*key), std::logic_error);
      EXPECT_THROW((void) row->size(), std::logic_error);
   }

   ASSERT_TRUE(lookup.has_value());
   ASSERT_TRUE(row.has_value());
   EXPECT_FALSE(lookup->valid());
   EXPECT_FALSE(row->valid());
   EXPECT_THROW((void) lookup->find(*key), std::logic_error);
   EXPECT_THROW((void) row->size(), std::logic_error);
}

TEST(Session, PolicyVisitorsRejectReentrantMutations)
{
   expect_reentrant_mutation_is_rejected([](SolverSession& session) { return session.iterate(); });
   expect_reentrant_mutation_is_rejected([](SolverSession& session) { return session.advance(1); });
   expect_reentrant_mutation_is_rejected([](SolverSession& session) { return session.trace(1); });
}

TEST(Session, StaticSmokeAcrossDimensions)
{
   expect_one_static_iteration(GameId::rps, SolverId::vanilla_cfr, ProfileId::vanilla_alternating);
   expect_one_static_iteration(GameId::rps, SolverId::vanilla_cfr, ProfileId::vanilla_simultaneous);
   expect_one_static_iteration(
      GameId::kuhn_poker, SolverId::vanilla_cfr, ProfileId::vanilla_alternating
   );
   expect_one_static_iteration(
      GameId::goofspiel, SolverId::vanilla_cfr, ProfileId::vanilla_alternating
   );
}

TEST(Dynamic, ValidationRejectsMalformedProviders)
{
   const auto expect_invalid = [](ProviderMode mode, std::string_view message) {
      auto result = make_dynamic_game(
         GameSpec{GameId::dynamic}, std::make_shared< TestDynamicProvider >(mode)
      );
      ASSERT_FALSE(result);
      EXPECT_EQ(result.error().code, CapabilityErrorCode::invalid_dynamic_provider);
      EXPECT_NE(result.error().message.find(message), std::string::npos);
   };

   expect_invalid(ProviderMode::sample, "stochasticity::sample");
   expect_invalid(ProviderMode::max_count_zero, "max_player_count");
   expect_invalid(ProviderMode::wrong_count, "does not match player_count");
   expect_invalid(ProviderMode::duplicate_players, "duplicate players");
   expect_invalid(ProviderMode::bad_active, "not in the player roster");
   expect_invalid(ProviderMode::bad_actions, "legal actions must have");
   expect_invalid(ProviderMode::bad_chance, "sum approximately");
   expect_invalid(ProviderMode::not_serialized, "serialized() == true");
   expect_invalid(ProviderMode::not_unrolled, "unrolled() == true");
}

TEST(Dynamic, DeterministicProviderUsesUnifiedSession)
{
   auto calls = std::make_shared< size_t >(0);
   auto provider = std::make_shared< TestDynamicProvider >(ProviderMode::deterministic, calls);
   auto game = make_dynamic_game(GameSpec{GameId::dynamic}, provider);
   ASSERT_TRUE(game) << game.error().message;
   ASSERT_TRUE(*game);
   ASSERT_TRUE(game->game_spec());
   EXPECT_EQ(game->game_spec()->game_id(), GameId::dynamic);

   ASSERT_NE(
      find_dynamic_capability(SolverId::vanilla_cfr, ProfileId::vanilla_alternating), nullptr
   );
   auto session = game->make_session(SolverId::vanilla_cfr, ProfileId::vanilla_alternating);
   ASSERT_TRUE(session) << session.error().message;
   const auto calls_before_iteration = *calls;
   auto iteration = session->iterate();
   ASSERT_TRUE(iteration) << iteration.error().message;
   EXPECT_GT(*calls, calls_before_iteration);
   EXPECT_EQ(iteration->iteration, 0u);

   auto stats = session->stats();
   ASSERT_TRUE(stats) << stats.error().message;
   EXPECT_EQ(stats->game, GameId::dynamic);
   EXPECT_EQ(stats->player_count, 2u);

   auto trace = session->trace(2);
   ASSERT_TRUE(trace) << trace.error().message;
   ASSERT_EQ(trace->size(), 2u);
   EXPECT_EQ(trace->iterations[0].iteration, 1u);
   EXPECT_EQ(trace->iterations[1].iteration, 2u);

   auto lookup = session->policy_lookup();
   ASSERT_TRUE(lookup) << lookup.error().message;
   size_t visited = 0;
   lookup->visit([&](const PolicyNodeView& node) {
      ++visited;
      ASSERT_GT(node.size(), 0u);
      EXPECT_TRUE(node.action_at(0).valid());
      EXPECT_EQ(node.to_entries().size(), node.size());
   });
   EXPECT_GT(visited, 0u);
}

TEST(Dynamic, ChanceProviderUsesChoiceSuperset)
{
   auto calls = std::make_shared< size_t >(0);
   auto provider = std::make_shared< TestDynamicProvider >(ProviderMode::chance, calls);
   auto game = make_dynamic_game(GameSpec{GameId::dynamic}, provider);
   ASSERT_TRUE(game) << game.error().message;

   auto session = game->make_session(SolverId::vanilla_cfr, ProfileId::vanilla_alternating);
   ASSERT_TRUE(session) << session.error().message;
   auto iteration = session->iterate();
   ASSERT_TRUE(iteration) << iteration.error().message;
   EXPECT_GT(*calls, 0u);
   auto stats = session->stats();
   ASSERT_TRUE(stats) << stats.error().message;
   EXPECT_EQ(stats->player_count, 2u);
}

TEST(Values, IdentityPresentationAndTensorSemantics)
{
   const auto represented = DynamicAction::named(
      "test.action",
      "left",
      std::string{"display-left"},
      TensorData{.values = {1., 2.}, .shape = {2}}
   );
   const auto identity_only = DynamicAction::named("test.action", "left");
   const auto different = DynamicAction::named("test.action", "right");

   EXPECT_TRUE(represented.valid());
   EXPECT_EQ(represented, identity_only);
   EXPECT_NE(represented, different);
   EXPECT_EQ(represented.hash(), identity_only.hash());
   ASSERT_TRUE(represented.to_string().has_value());
   EXPECT_EQ(*represented.to_string(), "display-left");
   ASSERT_TRUE(represented.to_tensor().has_value());
   EXPECT_EQ(represented.to_tensor()->values, (std::vector< double >{1., 2.}));
   EXPECT_EQ(represented.to_tensor()->shape, (std::vector< size_t >{2}));
   EXPECT_FALSE(identity_only.to_string().has_value());
   EXPECT_FALSE(identity_only.to_tensor().has_value());
   EXPECT_FALSE(DynamicAction::named("test.action", "").valid());
   EXPECT_FALSE(DynamicAction::named("", "left").valid());

   const auto erased = ErasedValue::action(represented);
   ASSERT_TRUE(erased);
   EXPECT_EQ(erased.kind(), ValueKind::action);
   EXPECT_EQ(erased, ErasedValue::action(identity_only));
   EXPECT_EQ(erased.hash(), ErasedValue::action(identity_only).hash());
}
