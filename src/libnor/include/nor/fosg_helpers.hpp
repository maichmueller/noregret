
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

   using action_variant_type = typename fosg_auto_traits< Env >::action_variant_type;
   // this hasher may be low quality given that boost's hash_combine paired with std::hash is not
   // necessarily a good hashing function (and long vectors may lead to many collisions)
   using history_type = std::vector< action_variant_type >;

   using history_hasher = decltype([](const history_type& hist_vec) {
      size_t hash = 0;
      ranges::for_each(hist_vec, [&](const action_variant_type& av) {
         common::hash_combine(
            hash,
            std::visit(
               [&]< typename U >(const U& action_or_outcome) {
                  return std::hash< U >{}(action_or_outcome);
               },
               av
            )
         );
      });
      return hash;
   });

   using infostate_map_type = std::unordered_map< Player, sptr< info_state_type > >;

   std::unordered_set<
      sptr< info_state_type >,
      common::value_hasher< info_state_type >,
      common::value_comparator< info_state_type > >
      existing_infostates;

   std::unordered_map<
      history_type,
      std::pair< std::vector< Player >, infostate_map_type >,
      common::value_hasher< history_type, history_hasher >,
      common::value_comparator< history_type > >
      hist_to_infostate_map{};

   // terminal states do not have an information state associated
   std::vector< history_type > terminal_histories;

   struct VisitData {
      infostate_map_type istate_map;
      history_type action_sequence;
   };

   auto root_action_sequence = std::make_shared< history_type >();

   auto child_hook = [&](
                        const VisitData& visit_data,
                        const action_variant_type* curr_action,
                        world_state_type* curr_state,
                        world_state_type* next_state
                     ) {
      auto child_action_sequence = visit_data.action_sequence;
      infostate_map_type child_istates_map;
      std::vector< info_state_type > copied_istates;
      for(const auto& istate_ptr : visit_data.istate_map | ranges::views::values) {
         copied_istates.emplace_back(*istate_ptr);
      }
      // add the last action to the action sequence vector
      child_action_sequence.emplace_back(*curr_action);
      // emplace private and public observation to each player's information states copies
      if(env.is_terminal(*next_state)) {
         terminal_histories.emplace_back(child_action_sequence);
      } else {
         auto active_player_next_state = env.active_player(*next_state);
         for(auto& infostate : copied_istates) {
            auto player = infostate.player();
            auto [pub_obs, priv_obs] = std::visit(
               common::Overload{
                  [&](const auto& action_or_outcome) {
                     return std::pair{
                        env.public_observation(*curr_state, action_or_outcome, *next_state),
                        env.private_observation(
                           player, *curr_state, action_or_outcome, *next_state
                        )};
                  },
                  [&](const std::monostate&) {
                     return std::pair{
                        typename fosg_auto_traits< Env >::observation_type{},
                        typename fosg_auto_traits< Env >::observation_type{}};
                  }},
               *curr_action
            );

            auto infostate_sptr = std::invoke([&] {
               infostate.update(std::move(pub_obs), std::move(priv_obs));
               auto find_iter = existing_infostates.find(infostate);
               return find_iter != existing_infostates.end()
                         ? *find_iter
                         : std::make_shared< info_state_type >(std::move(infostate));
            });
            child_istates_map.emplace(player, infostate_sptr);

            // add new infostates to either active or inactive sets depending on whether their owner
            // is the active player
            auto& active_players_istate_map_pair = hist_to_infostate_map[child_action_sequence];
            if(active_player_next_state == player) {
               active_players_istate_map_pair.first.emplace_back(player);
               active_players_istate_map_pair.second.emplace(player, infostate_sptr);
            } else if(include_inactive_player_states) {
               active_players_istate_map_pair.second.emplace(player, infostate_sptr);
            }
         }
      }
      return VisitData{
         .istate_map = std::move(child_istates_map),
         .action_sequence = std::move(child_action_sequence)};
      ;
   };

   forest::GameTreeTraverser{env}.walk(
      utils::static_unique_ptr_downcast< world_state_type >(utils::clone_any_way(root)),
      VisitData{
         .istate_map =
            [&] {
               auto root_istate_map = infostate_map_type{};
               for(auto player : env.players(root)) {
                  if(player != Player::chance) {
                     auto [iter, _] = root_istate_map.emplace(player, std::make_shared< info_state_type >(player));
                     existing_infostates.emplace(iter->second);
                  }
               }
               return root_istate_map;
            }(),
         .action_sequence = history_type{}},
      forest::TraversalHooks{.child_hook = std::move(child_hook)}
   );

   return std::tuple{terminal_histories, std::move(hist_to_infostate_map)};
}

}  // namespace nor

#endif  // NOR_FOSG_HELPERS_HPP
