
#ifndef NOR_FOSG_HELPERS_HPP
#define NOR_FOSG_HELPERS_HPP

#include <unordered_set>

#include "nor/concepts.hpp"
#include "nor/fosg_traits.hpp"
#include "nor/rm/forest.hpp"

namespace nor {

template < concepts::fosg Env >
void enumerate_infostates(Env& env, const typename fosg_auto_traits< Env >::world_state_type& root)
{
   using world_state_type = typename fosg_auto_traits< Env >::world_state_type;
   using info_state_type = typename fosg_auto_traits< Env >::info_state_type;

   auto game_tree_traverser = forest::GameTreeTraverser{env};
   using action_variant = typename decltype(game_tree_traverser)::action_variant_type;
   // this hasher may be low quality given that boost's hash_combine paired with std::hash is not
   // necessarily a good hashing function (and long vectors may lead to many collisions)
   using av_vector_sptr_hasher = decltype([](const sptr< std::vector< action_variant > >& av_vec_ptr
                                          ) {
      size_t start = 0;
      return ranges::for_each(*av_vec_ptr, [&](const action_variant& av) {
         return common::hash_combine(
            start,
            std::visit(
               [&](const auto& action_or_outcome) {
                  return std::hash< decltype(action_or_outcome) >{}(action_or_outcome);
               },
               av
            )
         );
      });
   });

   using av_sequence_infostate_map = std::unordered_map<
      sptr< std::vector< action_variant > >,
      info_state_type,
      av_vector_sptr_hasher,
      common::sptr_value_comparator< std::vector< action_variant > > >;

   av_sequence_infostate_map active_infostate_set;
   av_sequence_infostate_map inactive_infostate_set;

   using infostate_map_type = std::unordered_map< Player, info_state_type >;

   struct VisitData {
      infostate_map_type istate_map;
      sptr< std::vector< action_variant > > action_sequence;
   };

   auto root_action_sequence = std::make_shared< std::vector< action_variant > >();

   auto child_hook = [&](
                        const VisitData& visit_data,
                        action_variant* curr_action,
                        world_state_type* curr_state,
                        world_state_type* next_state
                     ) {
      VisitData child_visit_data{
         .istate_map = visit_data.istate_map,
         .action_sequence = std::make_shared(visit_data.action_sequence)};
      // add the last action to the action sequence vector
      child_visit_data.action_sequence->emplace_back(&curr_action);
      // emplace private and public observation to each player's information states copies
      for(auto& [player, infostate] : child_visit_data.istate_map) {
         infostate.append(std::visit([&](const auto& action_or_outcome) {
            return env.private_observation(player, action_or_outcome);
         }));
         infostate.append(env.private_observation(player, *next_state));

         // add new infostates to either active or inactive sets depending on whether their owner is
         // the active player
         (env.active_player(*next_state) == player ? active_infostate_set : inactive_infostate_set)
            .emplace(child_visit_data.action_sequence, infostate);
      }
      return child_visit_data;
   };

   game_tree_traverser.walk(utils::static_unique_ptr_downcast< world_state_type >(
      utils::clone_any_way(root),
      VisitData{
         .istate_map =
            [&] {
               auto root_istate_map = infostate_map_type{};
               for(auto player : env.players(root)) {
                  if(player != Player::chance) {
                     root_istate_map.emplace(player, info_state_type{player});
                  }
               }
               return root_istate_map;
            }(),
         .action_sequence = std::make_shared< std::vector< action_variant > >()},
      forest::TraversalHooks{.child_hook = std::move(child_hook)}
   ));

   return std::tuple{std::move(active_infostate_set), std::move(inactive_infostate_set)};
}

}  // namespace nor

#endif  // NOR_FOSG_HELPERS_HPP
