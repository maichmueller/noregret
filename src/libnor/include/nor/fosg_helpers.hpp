
#ifndef NOR_FOSG_HELPERS_HPP
#define NOR_FOSG_HELPERS_HPP

#include <unordered_set>

#include "nor/at_runtime.hpp"
#include "nor/concepts.hpp"
#include "nor/fosg_traits.hpp"
#include "nor/rm/forest.hpp"

namespace nor {

template < typename Env >
   requires concepts::fosg< std::remove_cvref_t< Env > >
auto map_histories_to_infostates(
   Env env,
   const WorldstateHolder< auto_world_state_type< Env > >& root,
   bool include_inactive_player_states = false
)
{
   assert_serialized_and_unrolled(env);
   using env_type = Env;
   using world_state_type = auto_world_state_type< env_type >;
   using info_state_type = auto_info_state_type< env_type >;

   using action_variant_type = auto_action_variant_type< env_type >;
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

   using infostate_map_type = std::
      unordered_map< Player, InfostateHolder< info_state_type, std::true_type > >;

   std::unordered_set<
      InfostateHolder< info_state_type, std::true_type >,
      common::value_hasher< info_state_type >,
      common::value_comparator< info_state_type > >
      existing_infostates;

   std::unordered_map<
      history_type,
      std::pair< Player, infostate_map_type >,
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
                        const WorldstateHolder< world_state_type >* curr_state,
                        const WorldstateHolder< world_state_type >* next_state
                     ) {
      auto child_action_sequence = visit_data.action_sequence;
      infostate_map_type child_istates_map;
      std::vector< InfostateHolder< info_state_type > > copied_istates;
      for(const auto& istate_holder : visit_data.istate_map | ranges::views::values) {
         copied_istates.emplace_back(
            istate_holder.template copy< InfostateHolder< info_state_type > >()
         );
      }
      // add the last action to the action sequence vector
      child_action_sequence.emplace_back(*curr_action);
      // emplace private and public observation to each player's information states copies
      if(env.is_terminal(*next_state)) {
         terminal_histories.emplace_back(child_action_sequence);
      } else {
         auto child_active_player = env.active_player(*next_state);
         auto& infostate_map = std::invoke([&]() -> auto& {
            auto hist_to_infostate_map_iter = hist_to_infostate_map.find(child_action_sequence);
            if(hist_to_infostate_map_iter == hist_to_infostate_map.end()) {
               return hist_to_infostate_map
                  .emplace(
                     child_action_sequence, std::pair{child_active_player, infostate_map_type{}}
                  )
                  .first->second;
            }
            return hist_to_infostate_map_iter->second;
         });
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
                        ObservationHolder< auto_observation_type< env_type > >{},
                        ObservationHolder< auto_observation_type< env_type > >{}};
                  }},
               *curr_action
            );

            auto shared_infostate = std::invoke([&]() -> auto& {
               infostate.update(std::move(pub_obs), std::move(priv_obs));
               auto find_iter = existing_infostates.find(infostate);
               return find_iter != existing_infostates.end()
                         ? *find_iter
                         : *existing_infostates.emplace(std::move(*infostate)).first;
            });
            // the shared infostate holder will be emplace in the child infostates map for further
            // tree traversal
            child_istates_map.emplace(player, shared_infostate);
            // we store this infostate also in the mapping of histories to infostates
            if(child_active_player == player or include_inactive_player_states) {
               infostate_map.second.emplace(player, shared_infostate);
            }
         }
      }
      return VisitData{
         .istate_map = std::move(child_istates_map),
         .action_sequence = std::move(child_action_sequence)};
   };

   forest::GameTreeTraverser{env}.walk(
      root.copy(),
      VisitData{
         .istate_map = std::invoke([&] {
            auto root_istate_map = infostate_map_type{};
            auto root_player = env.active_player(root);
            auto& infostate_map = std::get< 1 >(*(hist_to_infostate_map
                                                     .emplace(
                                                        history_type{},
                                                        std::pair{root_player, infostate_map_type{}}
                                                     )
                                                     .first))
                                     .second;
            for(auto player : env.players(root)) {
               InfostateHolder< info_state_type, std::true_type >* infostate = nullptr;
               if(player != Player::chance) {
                  auto [iter, _] = root_istate_map.emplace(
                     player, InfostateHolder< info_state_type, std::true_type >(player)
                  );
                  infostate = &iter->second;
                  existing_infostates.emplace(*infostate);
                  if(root_player == player or include_inactive_player_states) {
                     infostate_map.emplace(player, *infostate);
                  }
               }
            }
            return root_istate_map;
         }),
         .action_sequence = {}},
      forest::TraversalHooks{.child_hook = std::move(child_hook)}
   );

   return std::tuple{terminal_histories, std::move(hist_to_infostate_map)};
}

}  // namespace nor

#endif  // NOR_FOSG_HELPERS_HPP
