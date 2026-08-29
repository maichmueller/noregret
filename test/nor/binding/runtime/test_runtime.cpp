#include <cassert>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "nor/binding/runtime/runtime.hpp"

using namespace nor::binding::runtime;
using nor::Player;
using nor::Stochasticity;

namespace {

class FakeDynamicProvider final: public DynamicEnvironmentProvider {
  public:
   using DynamicEnvironmentProvider::private_observation;
   using DynamicEnvironmentProvider::public_observation;
   using DynamicEnvironmentProvider::transition;

   explicit FakeDynamicProvider(std::shared_ptr< size_t > calls) : m_calls(std::move(calls)) {}

   [[nodiscard]] size_t max_player_count() const final { return 2; }
   [[nodiscard]] size_t player_count() const final { return 2; }
   [[nodiscard]] Stochasticity stochasticity() const final { return Stochasticity::deterministic; }

   [[nodiscard]] DynamicWorldState initial_world_state() const final { return world("root"); }

   [[nodiscard]] std::vector< DynamicAction > actions(Player, const DynamicWorldState& state)
      const final
   {
      touch();
      if(state.value().identity() == "terminal")
         return {};
      return {action("left"), action("right")};
   }

   [[nodiscard]] std::vector< Player > players(const DynamicWorldState&) const final
   {
      touch();
      return {Player::alex, Player::bob};
   }

   [[nodiscard]] Player active_player(const DynamicWorldState& state) const final
   {
      touch();
      if(state.value().identity() == "root")
         return Player::alex;
      if(state.value().identity() == "after")
         return Player::bob;
      return Player::unknown;
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
      if(state.value().identity() == "root")
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

  private:
   static DynamicWorldState world(std::string identity)
   {
      return DynamicWorldState{world_value(std::move(identity))};
   }

   static DynamicWorldValue world_value(std::string identity)
   {
      return DynamicWorldValue::named("fake.world", std::move(identity));
   }

   static DynamicAction action(std::string identity)
   {
      return DynamicAction::named("fake.action", std::move(identity));
   }

   static DynamicObservation observation(std::string identity)
   {
      return DynamicObservation::named("fake.observation", std::move(identity));
   }

   void touch() const { ++*m_calls; }

   std::shared_ptr< size_t > m_calls;
};

}  // namespace

int main()
{
   const auto& static_catalog = catalog();
   assert(static_catalog.games.size() == 17);
   assert(static_catalog.solvers.size() == 11);
   assert(static_catalog.profiles.size() == 15);

   size_t generated_capabilities = 0;
   for(const auto& game : static_catalog.games)
      generated_capabilities += capabilities_for(game.id).size();
   assert(generated_capabilities == static_catalog.combinations.size());
   assert(generated_capabilities > 0);

   for(const auto& game : static_catalog.games) {
      auto handle = make_game(GameSpec::defaults(game.id));
      if(not handle) {
         std::cerr << "failed to construct catalog game " << game.name << ": "
                   << handle.error().message << '\n';
         return 1;
      }
      assert(handle->game_id() == game.id);
      assert(handle->spec().fields().size() == game.fields.size());
   }

   auto rps = make_game(GameSpec::defaults(GameId::rps));
   assert(rps.has_value());

   auto first = make_session(*rps, SolverId::vanilla_cfr, ProfileId::vanilla_alternating);
   auto second = make_session(*rps, SolverId::vanilla_cfr, ProfileId::vanilla_alternating);
   assert(first.has_value());
   assert(second.has_value());

   auto first_iteration = first->iterate();
   auto second_iteration = second->iterate();
   assert(first_iteration.has_value());
   assert(second_iteration.has_value());
   assert(first_iteration->iteration == 0);
   assert(second_iteration->iteration == 0);
   assert(not first_iteration->root_values.empty());

   auto first_stats = first->stats();
   auto second_stats = second->stats();
   assert(first_stats.has_value());
   assert(second_stats.has_value());
   assert(first_stats->iteration == 1);
   assert(second_stats->iteration == 1);

   auto policy = first->policy_lookup();
   assert(policy.has_value());
   assert(policy->valid());
   assert(policy->generation() != 0);

   std::optional< ErasedInfoState > remembered_info_state;
   std::optional< ErasedAction > remembered_action;
   size_t visited_nodes = policy->visit([&](const PolicyNodeView& node) {
      if(not remembered_info_state) {
         remembered_info_state = node.info_state();
         remembered_action = node.action_at(0);
         assert(node.contains(*remembered_action));
         assert(node.find(*remembered_action).has_value());
         assert(node.to_entries().size() == node.size());
         assert(node.to_tensor().shape.size() == 1);
      }
   });
   assert(visited_nodes > 0);
   assert(remembered_info_state.has_value());
   assert(remembered_action.has_value());
   auto row = policy->find(*remembered_info_state);
   assert(row.has_value());
   assert(row->find(*remembered_action).has_value());
   assert(row->info_state() == *remembered_info_state);

   assert(first_stats->current_policy_entries == 6);
   assert(first_stats->average_policy_entries == 6);

   // Erased values are exact, type-aware, and do not invent optional representations.
   assert(*remembered_action == *remembered_action);
   assert(remembered_action->hash() == remembered_action->hash());
   assert(not remembered_action->type_name().empty());
   assert(not remembered_action->to_tensor().has_value());

   const auto represented_dynamic_action = DynamicAction::named(
      "fake.action", "left", std::string{"display-left"}, TensorData{.values = {1.0}, .shape = {1}}
   );
   const auto identity_only_dynamic_action = DynamicAction::named("fake.action", "left");
   assert(represented_dynamic_action == identity_only_dynamic_action);
   assert(represented_dynamic_action.hash() == identity_only_dynamic_action.hash());

   auto held_lookup = std::move(policy);
   auto held_row = std::move(row);
   auto advanced = first->advance(2);
   assert(advanced.has_value());
   assert(first->stats()->iteration == 3);
   assert(not held_lookup->valid());
   assert(not held_row->valid());
   bool stale_lookup_threw = false;
   try {
      (void) held_lookup->find(*remembered_info_state);
   } catch(const std::logic_error&) {
      stale_lookup_threw = true;
   }
   assert(stale_lookup_threw);
   bool stale_node_threw = false;
   try {
      (void) held_row->size();
   } catch(const std::logic_error&) {
      stale_node_threw = true;
   }
   assert(stale_node_threw);

   auto last = first->advance_last(2);
   assert(last.has_value());
   assert(last->has_value());
   assert(last->value().iteration == 4);
   auto zero = first->advance_last(0);
   assert(zero.has_value());
   assert(not zero->has_value());
   assert(first->stats()->iteration == 5);

   auto trace = first->trace(5, 2);
   assert(trace.has_value());
   assert(trace->first_iteration == 5);
   assert(trace->last_iteration == 10);
   assert(trace->iterations.size() == 2);
   assert(trace->iterations[0].iteration == 6);
   assert(trace->iterations[1].iteration == 8);
   auto bad_trace = first->trace(1, 0);
   assert(not bad_trace.has_value());
   assert(bad_trace.error().code == CapabilityErrorCode::invalid_spec);

   auto mismatched_profile = make_session(
      *rps, SolverId::vanilla_cfr, ProfileId::cfr_plus_alternating
   );
   assert(not mismatched_profile.has_value());
   assert(mismatched_profile.error().code == CapabilityErrorCode::profile_solver_mismatch);

   auto unknown_solver = make_session(
      *rps, static_cast< SolverId >(0xffff), ProfileId::vanilla_alternating
   );
   assert(not unknown_solver.has_value());
   assert(unknown_solver.error().code == CapabilityErrorCode::unknown_solver);

   auto invalid_epsilon = make_session(
      *rps,
      SolverId::mccfr,
      ProfileId::mccfr_outcome_lazy,
      SessionOptions{.epsilon = 1.1, .seed = 7}
   );
   assert(not invalid_epsilon.has_value());
   assert(invalid_epsilon.error().code == CapabilityErrorCode::invalid_spec);

   // Every generated profile thunk is exercised through a fresh, reusable GameSpec.
   for(const auto& profile : static_catalog.profiles) {
      const auto* capability = find_capability(GameId::rps, profile.solver, profile.id);
      assert(capability != nullptr);
      auto session = make_session(*rps, profile.solver, profile.id);
      if(not session) {
         std::cerr << "failed to construct profile " << profile.name << ": "
                   << session.error().message << '\n';
         return 1;
      }
      auto iteration = session->iterate();
      if(not iteration) {
         std::cerr << "failed to iterate profile " << profile.name << ": "
                   << iteration.error().message << '\n';
         return 1;
      }
      assert(iteration->iteration == 0);
      auto profile_stats = session->stats();
      assert(profile_stats.has_value());
      assert(profile_stats->solver == profile.solver);
      assert(profile_stats->profile == profile.id);
   }

   for(const auto& game : static_catalog.games) {
      auto reusable = make_game(GameSpec::defaults(game.id));
      assert(reusable.has_value());
      for(const auto& capability : capabilities_for(game.id)) {
         auto session = make_session(*reusable, capability.solver, capability.profile);
         if(not session) {
            std::cerr << "failed to construct " << game.name << " + "
                      << static_cast< uint16_t >(capability.profile) << ": "
                      << session.error().message << '\n';
            return 1;
         }
         assert(not session->empty());
      }
   }

   GameSpec bad_spec{GameId::battleship};
   bad_spec.set(GameFieldId::rows, true);
   auto bad_game = make_game(bad_spec);
   assert(not bad_game.has_value());
   assert(bad_game.error().code == CapabilityErrorCode::invalid_spec);

   GameSpec malformed_spec{GameId::rps};
   malformed_spec.set(GameFieldId::rows, uint64_t{2});
   GameHandle malformed_handle{std::move(malformed_spec)};
   auto invalid_handle = make_session(
      malformed_handle, SolverId::vanilla_cfr, ProfileId::vanilla_alternating
   );
   assert(not invalid_handle.has_value());
   assert(invalid_handle.error().code == CapabilityErrorCode::invalid_handle);

   // Dynamic sessions use the same C++ solver loop while every game-tree operation crosses the
   // provider vtable. The static session above cannot touch this provider.
   auto provider_calls = std::make_shared< size_t >(0);
   auto provider = std::make_shared< FakeDynamicProvider >(provider_calls);
   assert(*provider_calls == 0);
   auto dynamic_game = make_dynamic_game(GameSpec{GameId::dynamic}, provider);
   assert(dynamic_game.has_value());
   assert(dynamic_capabilities().size() > 0);
   auto dynamic_session = dynamic_game->make_session(
      SolverId::vanilla_cfr, ProfileId::vanilla_alternating
   );
   assert(dynamic_session.has_value());
   const size_t calls_before_iteration = *provider_calls;
   auto dynamic_iteration = (*dynamic_session)->iterate();
   assert(dynamic_iteration.has_value());
   assert(dynamic_iteration->iteration == 0);
   assert(not dynamic_iteration->root_values.empty());
   assert(*provider_calls > calls_before_iteration);
   auto dynamic_stats = (*dynamic_session)->stats();
   assert(dynamic_stats.has_value());
   assert(dynamic_stats->game == GameId::dynamic);
   assert(dynamic_stats->solver == SolverId::vanilla_cfr);
   assert(dynamic_stats->iteration == 1);

   return 0;
}
