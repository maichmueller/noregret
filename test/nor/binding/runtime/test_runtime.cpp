#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
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

/**
 * Modes prefixed with `late_` are well formed at the initial state and only malformed at a state
 * the solver reaches later. They exist because validating admission alone would accept them.
 */
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
   not_unrolled,
   empty_initial_world,
   late_empty_actions,
   late_duplicate_actions,
   late_invalid_action,
   late_bad_active,
   late_short_roster,
   late_duplicate_roster,
   late_foreign_roster,
   late_bad_reward,
   late_bad_world,
   late_bad_observation,
   late_bad_chance_sum,
   late_bad_chance_probability
};

class TestDynamicProvider: public DynamicEnvironmentProvider {
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

   [[nodiscard]] DynamicWorldState initial_world_state() const override
   {
      if(m_mode == ProviderMode::empty_initial_world)
         return DynamicWorldState{DynamicWorldValue::named("test.world", "")};
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
      if(is_later_state(state)) {
         switch(m_mode) {
            case ProviderMode::late_empty_actions: return {};
            case ProviderMode::late_duplicate_actions: return {action("left"), action("left")};
            case ProviderMode::late_invalid_action: return {action("left"), DynamicAction{}};
            default: break;
         }
      }
      return {action("left"), action("right")};
   }

   [[nodiscard]] std::vector< Player > players(const DynamicWorldState& state) const final
   {
      touch();
      if(m_mode == ProviderMode::duplicate_players)
         return {Player::alex, Player::alex};
      if(is_later_state(state)) {
         switch(m_mode) {
            case ProviderMode::late_short_roster: return {Player::alex};
            case ProviderMode::late_duplicate_roster: return {Player::alex, Player::alex};
            case ProviderMode::late_foreign_roster: return {Player::alex, Player::cedric};
            default: break;
         }
      }
      return {Player::alex, Player::bob};
   }

   [[nodiscard]] Player active_player(const DynamicWorldState& state) const final
   {
      touch();
      if(m_mode == ProviderMode::bad_active)
         return Player::unknown;
      if(m_mode == ProviderMode::late_bad_active and is_later_state(state))
         return Player::unknown;
      if(is_chance_mode()
         and (state.value().identity() == "chance_root" or state.value().identity() == "second_chance"))
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
      if(m_mode == ProviderMode::late_bad_reward and state.value().identity() == "terminal")
         return std::numeric_limits< double >::infinity();
      if(state.value().identity() != "terminal")
         return 0.;
      return player == Player::alex ? 1. : -1.;
   }

   void transition(DynamicWorldState& state, const DynamicAction&) const final
   {
      touch();
      if(m_mode == ProviderMode::late_bad_world and state.value().identity() == "after") {
         state.set(DynamicWorldValue::named("test.world", ""));
         return;
      }
      if(state.value().identity() == "root" or state.value().identity() == "chance_root")
         state.set(world_value("after"));
      else
         state.set(world_value("terminal"));
   }

   [[nodiscard]] DynamicObservation
   private_observation(Player, const DynamicWorldState& state, const DynamicAction& action_value, const DynamicWorldState&)
      const final
   {
      touch();
      if(m_mode == ProviderMode::late_bad_observation and is_later_state(state))
         return DynamicObservation{};
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
      if(not is_chance_mode())
         return {};
      if(state.value().identity() == "chance_root" or state.value().identity() == "second_chance")
         return {outcome("heads"), outcome("tails")};
      return {};
   }

   [[nodiscard]] double chance_probability(
      const DynamicWorldState& state,
      const DynamicChanceOutcome& outcome_value
   ) const final
   {
      touch();
      if(m_mode == ProviderMode::bad_chance)
         return 0.6;
      if(state.value().identity() == "second_chance") {
         if(m_mode == ProviderMode::late_bad_chance_sum)
            return 0.6;
         if(m_mode == ProviderMode::late_bad_chance_probability)
            return outcome_value.identity() == "heads" ? -0.5
                                                       : std::numeric_limits< double >::quiet_NaN();
      }
      return 0.5;
   }

   void transition(DynamicWorldState& state, const DynamicChanceOutcome&) const final
   {
      touch();
      if(has_second_chance_node() and state.value().identity() == "chance_root") {
         state.set(world_value("second_chance"));
         return;
      }
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

  protected:
   [[nodiscard]] bool is_chance_mode() const noexcept
   {
      return m_mode == ProviderMode::chance or m_mode == ProviderMode::bad_chance
             or has_second_chance_node();
   }

   /// A chance node that the initial admission pass never reaches.
   [[nodiscard]] bool has_second_chance_node() const noexcept
   {
      return m_mode == ProviderMode::late_bad_chance_sum
             or m_mode == ProviderMode::late_bad_chance_probability;
   }

   /// True for every state past the root, which is where the `late_` modes misbehave.
   [[nodiscard]] static bool is_later_state(const DynamicWorldState& state) noexcept
   {
      const auto identity = state.value().identity();
      return identity != "root" and identity != "chance_root";
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

TEST(Catalog, ReportsExactlyTheCompiledGameRoster)
{
   // The partition roster is generated from one macro list and cross-checked against the static
   // game type list at compile time. This test pins the *result* of that generation, so dropping
   // or renaming a game cannot pass silently just because both places were edited together.
   const std::set< GameId > expected_games{
      GameId::kuhn_poker,
      GameId::leduc_poker,
      GameId::rock_paper_scissors,
      GameId::stratego,
      GameId::texas_holdem_poker,
      GameId::goofspiel,
      GameId::three_player_goofspiel,
      GameId::battleship,
      GameId::battleship_gs,
      GameId::dark_hex,
      GameId::pursuit_evasion,
      GameId::oshi_zumo,
      GameId::shapley,
      GameId::centipede,
      GameId::colonel_blotto,
      GameId::sheriff,
      GameId::liars_dice};

   std::set< GameId > actual_games;
   for(const auto& game : games())
      actual_games.insert(game.id);
   EXPECT_EQ(actual_games, expected_games);
   EXPECT_EQ(games().size(), expected_games.size());

   // The reserved dynamic identifier is not a static game and must never be constructible as one.
   EXPECT_EQ(find_game(GameId::dynamic), nullptr);
   auto dynamic_as_static = make_game(GameSpec{GameId::dynamic});
   ASSERT_FALSE(dynamic_as_static);
   EXPECT_EQ(dynamic_as_static.error().code, CapabilityErrorCode::unknown_game);

   size_t counted = 0;
   for(const auto& game : games()) {
      const auto admitted = capabilities_for(game.id);
      EXPECT_FALSE(admitted.empty()) << game.name << " admits no solver profile";
      counted += admitted.size();
      for(const auto& capability : admitted) {
         const auto* profile = find_profile(capability.profile);
         ASSERT_NE(profile, nullptr);
         EXPECT_EQ(profile->solver, capability.solver);
      }
   }
   EXPECT_EQ(counted, capabilities().size());

   // Every admitted pair is reachable through the public lookup and no other pair is.
   for(const auto& game : games()) {
      for(const auto& profile : profiles()) {
         const auto* capability = find_capability(game.id, profile.solver, profile.id);
         const auto admitted = capabilities_for(game.id);
         const bool listed = std::ranges::any_of(admitted, [&](const CapabilityDescriptor& entry) {
            return entry.solver == profile.solver and entry.profile == profile.id;
         });
         EXPECT_EQ(capability != nullptr, listed);
      }
   }
}

TEST(Catalog, RejectsUnsupportedAndMismatchedCombinations)
{
   auto handle = make_game(GameSpec::defaults(GameId::rock_paper_scissors));
   ASSERT_TRUE(handle) << handle.error().message;

   auto unknown_solver = make_session(
      *handle, static_cast< SolverId >(0x7fff), ProfileId::vanilla_alternating
   );
   ASSERT_FALSE(unknown_solver);
   EXPECT_EQ(unknown_solver.error().code, CapabilityErrorCode::unknown_solver);

   auto unknown_profile = make_session(
      *handle, SolverId::vanilla_cfr, static_cast< ProfileId >(0x7fff)
   );
   ASSERT_FALSE(unknown_profile);
   EXPECT_EQ(unknown_profile.error().code, CapabilityErrorCode::unknown_profile);

   auto mismatched = make_session(*handle, SolverId::mccfr, ProfileId::vanilla_alternating);
   ASSERT_FALSE(mismatched);
   EXPECT_EQ(mismatched.error().code, CapabilityErrorCode::profile_solver_mismatch);
}

TEST(Catalog, EveryStaticThunkValidatesEpsilonItself)
{
   // A CapabilityDescriptor is a raw function pointer. A caller holding one never passes through
   // make_session(), so the outer epsilon check is not the only one that may exist.
   auto handle = make_game(GameSpec::defaults(GameId::rock_paper_scissors));
   ASSERT_TRUE(handle) << handle.error().message;

   for(const double epsilon :
       {-0.5,
        1.5,
        std::numeric_limits< double >::quiet_NaN(),
        std::numeric_limits< double >::infinity()}) {
      for(const auto& capability : capabilities_for(GameId::rock_paper_scissors)) {
         auto session = capability.create(*handle, SessionOptions{.epsilon = epsilon, .seed = 0});
         ASSERT_FALSE(session) << "a thunk accepted epsilon " << epsilon;
         EXPECT_EQ(session.error().code, CapabilityErrorCode::invalid_spec);
      }
      // The same rejection reaches callers of the public entry point.
      auto outer = make_session(
         *handle,
         SolverId::vanilla_cfr,
         ProfileId::vanilla_alternating,
         SessionOptions{.epsilon = epsilon, .seed = 0}
      );
      ASSERT_FALSE(outer);
      EXPECT_EQ(outer.error().code, CapabilityErrorCode::invalid_spec);
   }

   const auto* dynamic_capability = find_dynamic_capability(
      SolverId::vanilla_cfr, ProfileId::vanilla_alternating
   );
   ASSERT_NE(dynamic_capability, nullptr);
   auto dynamic_game = make_dynamic_game(
      GameSpec{GameId::dynamic},
      std::make_shared< TestDynamicProvider >(ProviderMode::deterministic)
   );
   ASSERT_TRUE(dynamic_game) << dynamic_game.error().message;
   auto dynamic_session = dynamic_capability->create(
      *dynamic_game, SessionOptions{.epsilon = 2., .seed = 0}
   );
   ASSERT_FALSE(dynamic_session);
   EXPECT_EQ(dynamic_session.error().code, CapabilityErrorCode::invalid_spec);
}

TEST(Catalog, StrategoDefaultIsPlayableRatherThanMerelyConstructible)
{
   // The registered default used to be an empty setup: it constructed, and then failed on the
   // first traversal. Registering a capability at all means the first iteration has to work.
   auto handle = make_game(GameSpec::defaults(GameId::stratego));
   ASSERT_TRUE(handle) << handle.error().message;
   auto session = make_session(*handle, SolverId::vanilla_cfr, ProfileId::vanilla_alternating);
   ASSERT_TRUE(session) << session.error().message;

   auto iteration = session->iterate();
   ASSERT_TRUE(iteration) << iteration.error().message;
   EXPECT_EQ(iteration->iteration, 0u);
   ASSERT_EQ(iteration->root_values.size(), 2u);

   auto stats = session->stats();
   ASSERT_TRUE(stats) << stats.error().message;
   EXPECT_EQ(stats->player_count, 2u);
   // A playable opening has at least one decision node for the player to move first.
   EXPECT_GT(stats->current_policy_entries, 0u);
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
   expect_invalid(ProviderMode::bad_active, "neither chance nor an admitted actual player");
   expect_invalid(ProviderMode::bad_actions, "legal actions must have");
   expect_invalid(ProviderMode::bad_chance, "sum approximately");
   expect_invalid(ProviderMode::not_serialized, "serialized() == true");
   expect_invalid(ProviderMode::not_unrolled, "unrolled() == true");
   expect_invalid(ProviderMode::empty_initial_world, "initial_world_state()");
}

TEST(Dynamic, ValidationRejectsProvidersThatOnlyBreakLater)
{
   // The initial state of each of these providers is perfectly admissible. Only a state the solver
   // reaches during traversal is malformed, which is exactly the case an admission-time-only check
   // would let through into the concrete solver.
   const auto expect_invalid_at_run =
      [](ProviderMode mode, std::string_view label, std::string_view message) {
         SCOPED_TRACE(std::string{label});
         auto game = make_dynamic_game(
            GameSpec{GameId::dynamic}, std::make_shared< TestDynamicProvider >(mode)
         );
         ASSERT_TRUE(game) << "provider was expected to be admissible at its initial state: "
                           << game.error().message;
         auto session = game->make_session(SolverId::vanilla_cfr, ProfileId::vanilla_alternating);
         ASSERT_TRUE(session) << session.error().message;
         auto iteration = session->iterate();
         ASSERT_FALSE(iteration) << "a malformed later state was accepted by the solver";
         EXPECT_EQ(iteration.error().code, CapabilityErrorCode::invalid_dynamic_provider);
         EXPECT_NE(iteration.error().message.find(message), std::string::npos)
            << iteration.error().message;
         EXPECT_EQ(iteration.error().game, GameId::dynamic);
         EXPECT_EQ(iteration.error().profile, ProfileId::vanilla_alternating);
      };

   expect_invalid_at_run(
      ProviderMode::late_empty_actions, "late_empty_actions", "no legal actions"
   );
   expect_invalid_at_run(
      ProviderMode::late_duplicate_actions, "late_duplicate_actions", "legal actions must be unique"
   );
   expect_invalid_at_run(
      ProviderMode::late_invalid_action, "late_invalid_action", "legal actions must have"
   );
   expect_invalid_at_run(
      ProviderMode::late_bad_active, "late_bad_active", "neither chance nor an admitted"
   );
   expect_invalid_at_run(
      ProviderMode::late_short_roster, "late_short_roster", "different size than the admitted"
   );
   expect_invalid_at_run(
      ProviderMode::late_duplicate_roster, "late_duplicate_roster", "duplicate players"
   );
   expect_invalid_at_run(
      ProviderMode::late_foreign_roster, "late_foreign_roster", "outside the admitted roster"
   );
   expect_invalid_at_run(ProviderMode::late_bad_reward, "late_bad_reward", "non-finite");
   expect_invalid_at_run(
      ProviderMode::late_bad_world, "late_bad_world", "world state without a type name"
   );
   expect_invalid_at_run(
      ProviderMode::late_bad_observation, "late_bad_observation", "observation without a type name"
   );
   expect_invalid_at_run(
      ProviderMode::late_bad_chance_sum, "late_bad_chance_sum", "sum approximately to one"
   );
   expect_invalid_at_run(
      ProviderMode::late_bad_chance_probability,
      "late_bad_chance_probability",
      "finite and nonnegative"
   );
}

TEST(Dynamic, HandleAndSolverConstructionShareOneInitialSnapshot)
{
   // A provider whose initial_world_state() answers differently on a later call would let a
   // session start from a state nothing validated. The handle's admission certificate is the
   // session root, so session construction must not ask for a second snapshot.
   class DriftingProvider final: public TestDynamicProvider {
     public:
      using TestDynamicProvider::TestDynamicProvider;

      [[nodiscard]] DynamicWorldState initial_world_state() const final
      {
         ++m_calls;
         return TestDynamicProvider::initial_world_state();
      }

      [[nodiscard]] size_t initial_calls() const noexcept { return m_calls; }

     private:
      mutable size_t m_calls = 0;
   };

   auto provider = std::make_shared< DriftingProvider >(ProviderMode::deterministic);
   auto game = make_dynamic_game(GameSpec{GameId::dynamic}, provider);
   ASSERT_TRUE(game) << game.error().message;
   const auto calls_after_admission = provider->initial_calls();
   auto session = game->make_session(SolverId::vanilla_cfr, ProfileId::vanilla_alternating);
   ASSERT_TRUE(session) << session.error().message;
   // The session reuses the handle's admission certificate rather than re-admitting the provider.
   EXPECT_EQ(provider->initial_calls(), calls_after_admission);
   ASSERT_TRUE(session->iterate());
}

TEST(Dynamic, ChanceDistributionsAreValidatedOncePerState)
{
   // Summing a chance node's distribution costs one probability query per outcome. Doing that on
   // every visit would double the provider traffic of the hottest dynamic code path, so the check
   // is memoized per state; the first iteration pays for it and later ones do not.
   auto calls = std::make_shared< size_t >(0);
   auto provider = std::make_shared< TestDynamicProvider >(ProviderMode::chance, calls);
   auto game = make_dynamic_game(GameSpec{GameId::dynamic}, provider);
   ASSERT_TRUE(game) << game.error().message;
   auto session = game->make_session(SolverId::vanilla_cfr, ProfileId::vanilla_alternating);
   ASSERT_TRUE(session) << session.error().message;

   const auto before_first = *calls;
   ASSERT_TRUE(session->iterate());
   const auto first_iteration_calls = *calls - before_first;

   const auto before_second = *calls;
   ASSERT_TRUE(session->iterate());
   const auto second_iteration_calls = *calls - before_second;

   EXPECT_GT(first_iteration_calls, 0u);
   EXPECT_LE(second_iteration_calls, first_iteration_calls);
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

TEST(Session, BulkPolicyRowOrderIsUnspecifiedButContentIsNot)
{
   // to_entries() is documented to produce rows in the solver's hash-map order. The contract this
   // test pins is therefore about the row *set* and each row's internal order, never the row
   // sequence itself.
   auto session = make_rps_session();
   ASSERT_TRUE(session) << session.error().message;
   ASSERT_TRUE(session->advance(3));

   auto lookup = session->policy_lookup();
   ASSERT_TRUE(lookup) << lookup.error().message;
   const auto entries = lookup->to_entries();
   ASSERT_FALSE(entries.empty());

   std::set< Player > owners;
   for(const auto& entry : entries) {
      owners.insert(entry.player);
      ASSERT_FALSE(entry.actions.empty());
      // Within one row the action order is the node's deterministic registry order, which the
      // ordinal accessors must agree with.
      auto row = lookup->find(entry.info_state);
      ASSERT_TRUE(row.has_value());
      ASSERT_EQ(row->size(), entry.actions.size());
      for(size_t index = 0; index < entry.actions.size(); ++index) {
         EXPECT_EQ(row->action_at(index), entry.actions[index].action);
         EXPECT_DOUBLE_EQ(row->value_at(index), entry.actions[index].probability);
      }
   }
   EXPECT_EQ(owners, (std::set< Player >{Player::alex, Player::bob}));

   // Repeating the copy on an unmutated policy yields the same rows, whatever order they arrive in.
   const auto repeated = lookup->to_entries();
   ASSERT_EQ(repeated.size(), entries.size());
   for(const auto& entry : entries) {
      const auto found = std::ranges::find(repeated, entry.info_state, &PolicyEntry::info_state);
      ASSERT_NE(found, repeated.end());
      EXPECT_EQ(found->player, entry.player);
      EXPECT_EQ(found->actions.size(), entry.actions.size());
   }
}

TEST(Session, SolverDestructionDuringVisitationStopsTheTraversal)
{
   // The erased boundary hands out a lookup that borrows solver-owned storage. A visitor callback
   // that releases the session frees that storage mid-traversal; the traversal must stop rather
   // than step to the next node.
   std::optional< SolverSession > session = std::nullopt;
   auto created = make_rps_session();
   ASSERT_TRUE(created) << created.error().message;
   session.emplace(std::move(*created));
   ASSERT_TRUE(session->iterate());

   auto lookup_result = session->policy_lookup();
   ASSERT_TRUE(lookup_result) << lookup_result.error().message;
   auto lookup = std::move(*lookup_result);

   size_t visited = 0;
   EXPECT_THROW(
      {
         (void) lookup.visit([&](const PolicyNodeView&) {
            ++visited;
            session.reset();
         });
      },
      std::logic_error
   );
   EXPECT_EQ(visited, 1u);
   EXPECT_FALSE(lookup.valid());
}

TEST(Session, SessionMoveAssignmentDuringVisitationStopsTheTraversal)
{
   auto destination_result = make_rps_session();
   ASSERT_TRUE(destination_result) << destination_result.error().message;
   auto destination = std::move(*destination_result);
   ASSERT_TRUE(destination.iterate());

   auto source_result = make_rps_session();
   ASSERT_TRUE(source_result) << source_result.error().message;
   auto source = std::move(*source_result);
   ASSERT_TRUE(source.iterate());

   auto lookup_result = destination.policy_lookup();
   ASSERT_TRUE(lookup_result) << lookup_result.error().message;
   auto lookup = std::move(*lookup_result);

   size_t visited = 0;
   EXPECT_THROW(
      {
         (void) lookup.visit([&](const PolicyNodeView&) {
            ++visited;
            destination = std::move(source);
         });
      },
      std::logic_error
   );
   EXPECT_EQ(visited, 1u);
   EXPECT_FALSE(lookup.valid());

   auto fresh = destination.policy_lookup();
   ASSERT_TRUE(fresh) << fresh.error().message;
   EXPECT_TRUE(fresh->valid());
   EXPECT_GT(fresh->visit([](const PolicyNodeView&) {}), 0u);
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
