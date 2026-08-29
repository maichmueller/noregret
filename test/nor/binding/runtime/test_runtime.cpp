#include <cassert>
#include <iostream>

#include "nor/binding/runtime/runtime.hpp"

using namespace nor::binding::runtime;

int main()
{
   const auto& static_catalog = catalog();
   assert(static_catalog.games.size() == 17);
   assert(static_catalog.solvers.size() == 11);
   assert(static_catalog.profiles.size() == 15);
   assert(static_catalog.combinations.size() == 17 * 15);

   for(const auto& game : static_catalog.games) {
      auto handle = make_game(GameSpec::defaults(game.id));
      if(not handle) {
         std::cerr << "failed to construct catalog game " << game.name << ": "
                   << handle.error().message << '\n';
         return 1;
      }
      assert(handle.has_value());
      assert(handle->game_id() == game.id);
      assert(handle->spec().fields().size() == game.fields.size());
   }

   auto rps = make_game(GameSpec::defaults(GameId::rps));
   assert(rps.has_value());

   auto first = make_session(*rps, SolverId::vanilla_cfr, ProfileId::vanilla_alternating);
   auto second = make_session(*rps, SolverId::vanilla_cfr, ProfileId::vanilla_alternating);
   assert(first.has_value());
   assert(second.has_value());

   auto first_iteration = first->iterate(2);
   auto second_iteration = second->iterate(1);
   assert(first_iteration.has_value());
   assert(second_iteration.has_value());
   assert(first_iteration->iterations.size() == 2);
   assert(second_iteration->iterations.size() == 1);
   assert(first_iteration->iterations[0].iteration == 0);
   assert(second_iteration->iterations[0].iteration == 0);

   auto first_stats = first->stats();
   auto second_stats = second->stats();
   assert(first_stats.has_value());
   assert(second_stats.has_value());
   assert(first_stats->iteration == 2);
   assert(second_stats->iteration == 1);

   auto policy = first->policy_view(PolicyViewKind::current);
   assert(policy.has_value());
   assert(policy->temporary_adapter);
   assert(not policy->entries.empty());

   auto trace = first->trace();
   assert(not trace.has_value());
   assert(trace.error().code == CapabilityErrorCode::operation_unavailable);

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

   // Every named profile must have a generated capability thunk and be constructible through the
   // same reusable GameSpec. One small iteration exercises each factory entry point, including
   // the dedicated Lazy-CFR+ path whose config carrier is shared with Lazy-CFR.
   for(const auto& profile : static_catalog.profiles) {
      const auto* capability = find_capability(GameId::rps, profile.solver, profile.id);
      assert(capability != nullptr);
      auto session = make_session(*rps, profile.solver, profile.id);
      if(not session) {
         std::cerr << "failed to construct profile " << profile.name << ": "
                   << session.error().message << '\n';
         return 1;
      }
      auto iteration = session->iterate(1);
      if(not iteration) {
         std::cerr << "failed to iterate profile " << profile.name << ": "
                   << iteration.error().message << '\n';
         return 1;
      }
      assert(iteration->iterations.size() == 1);
      auto profile_stats = session->stats();
      assert(profile_stats.has_value());
      assert(profile_stats->solver == profile.solver);
      assert(profile_stats->profile == profile.id);
   }

   // Construction is also checked for every generated game/profile pair. Keeping one handle
   // while creating these sessions specifies that a GameSpec is reusable and that each thunk
   // reconstructs fresh concrete state rather than borrowing mutable registry state.
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

   return 0;
}
