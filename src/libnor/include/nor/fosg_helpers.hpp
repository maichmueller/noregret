
#ifndef NOR_FOSG_HELPERS_HPP
#define NOR_FOSG_HELPERS_HPP

#include <unordered_set>

#include "nor/concepts.hpp"
#include "nor/fosg_traits.hpp"
#include "nor/rm/forest.hpp"

namespace nor {

template < concepts::fosg Env >
auto map_histories_to_infostates(
   Env& env,
   const typename fosg_auto_traits< Env >::world_state_type& root,
   bool include_inactive_player_states = false
)
{
   using world_state_type = typename fosg_auto_traits< Env >::world_state_type;
   using info_state_type = typename fosg_auto_traits< Env >::info_state_type;

   auto game_tree_traverser = forest::GameTreeTraverser{env};
   using action_variant = typename decltype(game_tree_traverser)::action_variant_type;
   // this hasher may be low quality given that boost's hash_combine paired with std::hash is not
   // necessarily a good hashing function (and long vectors may lead to many collisions)
   using history_type = std::vector< action_variant >;
   using history_sptr_hasher = decltype([](const sptr< history_type >& hist_vec_ptr) {
      size_t hash = 0;
      ranges::for_each(*hist_vec_ptr, [&](const action_variant& av) {
         common::hash_combine(
            hash,
            std::visit(
               [&]< typename T >(const T& action_or_outcome) {
                  return std::hash< T >{}(action_or_outcome);
               },
               av
            )
         );
      });
      return hash;
   });

   using infostate_map_type = std::unordered_map< Player, info_state_type >;

   std::unordered_map<
      sptr< history_type >,
      info_state_type,
      history_sptr_hasher,
      common::sptr_value_comparator< history_type > >
      active_infostate_set;

   std::unordered_map<
      sptr< history_type >,
      infostate_map_type,
      history_sptr_hasher,
      common::sptr_value_comparator< history_type > >
      inactive_infostate_set;

   // terminal states do not have an information state associated
   std::vector< sptr< history_type > > terminal_histories;

   struct VisitData {
      infostate_map_type istate_map;
      sptr< history_type > action_sequence;
   };

   auto root_action_sequence = std::make_shared< std::vector< action_variant > >();

   auto child_hook = [&](
                        const VisitData& visit_data,
                        const action_variant* curr_action,
                        world_state_type*,
                        world_state_type* next_state
                     ) {
      VisitData child_visit_data{
         .istate_map = visit_data.istate_map,
         .action_sequence = std::make_shared< history_type >(*(visit_data.action_sequence))};
      // add the last action to the action sequence vector
      child_visit_data.action_sequence->emplace_back(*curr_action);
      // emplace private and public observation to each player's information states copies
      for(auto& [player, infostate] : child_visit_data.istate_map) {
         infostate.append(std::visit(
            common::Overload{
               [&](const auto& action_or_outcome) {
                  return env.private_observation(player, action_or_outcome);
               },
               [&](const std::monostate&) {
                  return typename fosg_auto_traits< Env >::observation_type{};
               }},
            *curr_action
         ));
         infostate.append(env.private_observation(player, *next_state));

         // add new infostates to either active or inactive sets depending on whether their owner is
         // the active player
         if(env.is_terminal(*next_state)) {
            terminal_histories.emplace_back(child_visit_data.action_sequence);
         } else if(env.active_player(*next_state) == player) {
            active_infostate_set.emplace(child_visit_data.action_sequence, infostate);
         } else if(include_inactive_player_states) {
            inactive_infostate_set[child_visit_data.action_sequence].emplace(player, infostate);
         }
      }
      return child_visit_data;
   };

   game_tree_traverser.walk(
      utils::static_unique_ptr_downcast< world_state_type >(utils::clone_any_way(root)),
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
   );

   return std::tuple{
      terminal_histories, std::move(active_infostate_set), std::move(inactive_infostate_set)};
}

}  // namespace nor

#endif  // NOR_FOSG_HELPERS_HPP
