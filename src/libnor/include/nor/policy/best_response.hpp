
#ifndef NOR_BEST_RESPONSE_HPP
#define NOR_BEST_RESPONSE_HPP

#include <unordered_map>
#include <unordered_set>

#include "default_policy.hpp"
#include "nor/concepts.hpp"
#include "nor/rm/forest.hpp"
#include "nor/rm/rm_utils.hpp"
#include "policy_view.hpp"

namespace nor {

template < concepts::info_state Infostate, concepts::action Action >
class BestResponsePolicy {
  public:
   using info_state_type = Infostate;
   using action_type = Action;

  private:
   template < typename Env >
   struct WorldNode {
      using action_variant_type = typename fosg_auto_traits< Env >::action_variant_type;
      using info_state_type = typename fosg_auto_traits< Env >::info_state_type;

      /// the likelihood that the opponents play to this world state.
      double opp_reach_prob;
      /// the state value of this node.
      std::optional< double > state_value = std::nullopt;
      /// the child nodes reachable from this infostate the child node pointer that the action leads
      std::unordered_map<
         action_variant_type,
         WorldNode*,
         common::variant_hasher< action_variant_type > >
         children{};
      /// the player that takes the actions at this node. (Could be the chance player too!)
      Player active_player;
      /// a backreferencing pointer to the infostate that this worldstate belongs to
      const info_state_type* infostate_ptr;
   };

  public:
   BestResponsePolicy(Player best_response_player) : m_br_player(best_response_player) {}

   BestResponsePolicy(
      Player best_response_player,
      std::unordered_map< info_state_type, std::pair< action_type, double > > best_response_map = {}
   )
       : m_br_player(best_response_player), m_best_response(std::move(best_response_map))
   {
   }

   template < concepts::fosg Env >
   decltype(auto) allocate(
      Env& env,
      std::unordered_map< Player, StatePolicyView< info_state_type, action_type > > player_policies,
      uptr< typename fosg_auto_traits< Env >::world_state_type > root_state,
      std::unordered_map< Player, info_state_type > root_infostates = {}
   )
   {
      using world_state_type = typename fosg_auto_traits< Env >::world_state_type;
      using observation_type = typename fosg_auto_traits< Env >::observation_type;
      using action_variant_type = typename fosg_auto_traits< Env >::action_variant_type;

      for(auto player : env.players(*root_state)) {
         auto istate_iter = root_infostates.find(player);
         if(istate_iter == root_infostates.end()) {
            // if one of the infostates is missing then all must be missing or empty, otherwise we
            // have different information states for each player which would lead to inconsistencies
            if(not root_infostates.empty()
               or ranges::any_of(
                  root_infostates | ranges::views::values,
                  [](const auto& infostate) { return infostate.size() > 0; }
               )) {
               throw std::invalid_argument(
                  "The given infostate map has inconsistent infostates (some player states are "
                  "missing, but others are given)."
               );
            }
         }
      }
      for(auto player : env.players(*root_state)) {
         if(not root_infostates.contains(player)) {
            root_infostates.emplace(player, info_state_type{player});
         }
      }

      auto root_node = [&] {
         auto root_player = env.active_player(*root_state);
         return WorldNode< Env >{
            .opp_reach_prob = 1.,
            .active_player = root_player,
            .infostate_ptr = &root_infostates.at(root_player)};
      }();

      reachable_childnodes_map< Env > consistent_worldstates;

      struct VisitData {
         std::unordered_map< Player, info_state_type > infostates;
         std::
            unordered_map< Player, std::vector< std::pair< observation_type, observation_type > > >
               observation_buffer;
         WorldNode< Env >* parent = nullptr;
      };

      auto child_hook = [&](
                           const VisitData& visit_data,
                           const action_variant_type* curr_action,
                           world_state_type* curr_state,
                           world_state_type* next_state
                        ) {
         auto curr_player = env.active_player(*curr_state);
         // emplace private and public observation to each player's information states copies and
         // get the action probability for the current scenario
         auto [action_prob, child_observation_buffer, child_infostate_map] = std::visit(
            common::Overload{
               [&]< typename ActionT >(const ActionT& action_or_outcome) {
                  auto [child_obs_buffer, child_istate_map] = [&] {
                     if(not env.is_terminal(*next_state)) {
                        return next_infostate_and_obs_buffers(
                           env,
                           visit_data.observation_buffer,
                           visit_data.infostates,
                           *curr_state,
                           action_or_outcome,
                           *next_state
                        );
                     } else {
                        return std::tuple{
                           decltype(visit_data.observation_buffer){},
                           decltype(visit_data.infostates){}};
                     }
                  }();
                  if constexpr(std::same_as< ActionT, action_type >) {
                     auto prob = curr_player == m_br_player
                                    ? 1.
                                    : player_policies.at(curr_player)
                                         .at(visit_data.infostates.at(curr_player))
                                         .at(action_or_outcome);
                     return std::tuple{
                        prob, std::move(child_obs_buffer), std::move(child_istate_map)};
                  } else {
                     // the constexpr check for the action type ensures that we will never reach
                     // this case for a deterministic environment! This is important since
                     // deterministic envs are not required to provide the chance_probability
                     // function and so this code would otherwise not compile for them
                     return std::tuple{
                        env.chance_probability(*curr_state, action_or_outcome),
                        std::move(child_obs_buffer),
                        std::move(child_istate_map)};
                  }
               },
               [&](std::monostate) {
                  // this should never be visited, but if so --> error
                  throw std::logic_error("We entered a std::monostate visit branch.");
                  return std::tuple{1., visit_data.observation_buffer, visit_data.infostates};
               }},
            *curr_action
         );

         auto [iter, _] = consistent_worldstates.try_emplace(visit_data.infostates.at(curr_player));
         auto& [infostate, child_nodes] = *iter;

         WorldNode< Env >* child_node_ptr;

         child_node_ptr = child_nodes[*curr_action]
                             .emplace_back(std::make_unique< WorldNode< Env > >(WorldNode< Env >{
                                .opp_reach_prob = visit_data.parent->opp_reach_prob
                                                  * (1. + action_prob * (curr_player != m_br_player)
                                                  ),
                                .active_player = env.active_player(*next_state),
                                .infostate_ptr = &infostate}))
                             .get();

         visit_data.parent->children.emplace(*curr_action, child_node_ptr);

         if(env.is_terminal(*next_state)) {
            child_node_ptr->state_value = env.reward(m_br_player, *next_state);
         }
         return VisitData{
            .infostates = std::move(child_infostate_map),
            .observation_buffer = std::move(child_observation_buffer),
            .parent = child_node_ptr,
         };
      };
      forest::GameTreeTraverser(env).walk(
         std::move(root_state),
         VisitData{
            .infostates = std::move(root_infostates),
            .observation_buffer = {},
            .parent = &root_node},
         forest::TraversalHooks{.child_hook = std::move(child_hook)}
      );

      _compute_best_responses< Env >(std::move(consistent_worldstates));
      return *this;
   }

   auto operator()(const info_state_type& infostate) const
   {
      return HashmapActionPolicy< action_type >{
         std::array{std::pair{m_best_response.at(infostate).first, 1.}}};
   }
   auto operator()(const info_state_type& infostate, auto&&...) const
   {
      return operator()(infostate);
   }

   auto& map() const { return m_best_response; }
   auto value(const info_state_type& infostate) const
   {
      return m_best_response.at(infostate).second;
   }

  private:
   Player m_br_player;
   std::unordered_map< info_state_type, std::pair< action_type, double > > m_best_response;

   /// The map of infostates points to another map of actions -> all possible child worldstates
   /// reachable from this action.
   /// Such a map is necessary, since each infostate is produced by a collection of worldstates that
   /// are consistent with this infostate. Each of these worldstates has the same legal actions and
   /// thus offers the options to choose. But in each case a different child worldstate is reached.
   /// Precisely those are captured in these vectors.
   template < typename Env >
   using reachable_childnodes_map = std::unordered_map<
      info_state_type,
      std::unordered_map<
         typename fosg_auto_traits< Env >::action_variant_type,
         std::vector< uptr< WorldNode< Env > > >,
         common::variant_hasher< typename fosg_auto_traits< Env >::action_variant_type > > >;

   template < typename Env >
   void _compute_best_responses(reachable_childnodes_map< Env > istate_to_nodes);

   template < typename Env >
   auto
   _best_response(const info_state_type& infostate,
      reachable_childnodes_map< Env >& istate_to_nodes);

   template < typename Env >
   double _value(WorldNode< Env >& node, reachable_childnodes_map< Env >& istate_to_nodes);
};

template < concepts::info_state Infostate, concepts::action Action >
template < typename Env >
void BestResponsePolicy< Infostate, Action >::_compute_best_responses(
   reachable_childnodes_map< Env > istate_to_nodes
)
{
   for(const auto& infostate : istate_to_nodes | ranges::views::keys) {
      // we compute best respones only for the br player
      if(infostate.player() != m_br_player) {
         continue;
      }
      auto br = _best_response(infostate, istate_to_nodes);
      m_best_response.emplace(
         std::piecewise_construct,
         std::forward_as_tuple(infostate),
         std::forward_as_tuple(br.action, br.value)
      );
   }
}

template < concepts::info_state Infostate, concepts::action Action >
template < typename Env >
auto BestResponsePolicy< Infostate, Action >::_best_response(
   const info_state_type& infostate,
   reachable_childnodes_map< Env >& istate_to_nodes
)
{
   if(infostate.player() != m_br_player) {
      throw std::invalid_argument("Best response action requested at an opponent info state.");
   }
   // we can assume that this is an infostate of the best responding player
   std::optional< action_type > best_action = std::nullopt;
   double state_value = std::numeric_limits< double >::lowest();
   for(const auto& [action_variant, node_vec] : istate_to_nodes.at(infostate)) {
      double action_value = ranges::accumulate(
         node_vec | ranges::views::transform([&](const uptr< WorldNode< Env > >& child_node_uptr) {
            //            double v = _value(*child_node_uptr, istate_to_nodes);
            //            LOGD2("Action raw value", v);
            //            LOGD2("Action prob", _value(*child_node_uptr, istate_to_nodes));
            //            LOGD2(
            //               "Action expected value",
            //               _value(*child_node_uptr, istate_to_nodes) *
            //               child_node_uptr->opp_reach_prob
            //            );
            return _value(*child_node_uptr, istate_to_nodes) * child_node_uptr->opp_reach_prob;
         }),
         double(0.),
         std::plus{}
      );

      LOGD2("child-value", action_value);
      if(action_value > state_value) {
         //         if(best_action.has_value()) {
         //            LOGD2("BRP old best response action", best_action.value());
         //         } else {
         //            LOGD2("BRP old best response action", "None");
         //         }
         best_action = std::get< 0 >(action_variant);
         //         LOGD2("BRP new best response action", best_action.value());
         //         LOGD2("BRP old state-value", state_value);
         state_value = action_value;
         //         LOGD2("BRP new state-value", state_value);
      }
   }
   return [&] {
      struct {
         action_type action;
         double value;
      } value{.action = std::move(best_action.value()), .value = state_value};
      return value;
   }();
}

template < concepts::info_state Infostate, concepts::action Action >
template < typename Env >
double BestResponsePolicy< Infostate, Action >::_value(
   WorldNode< Env >& node,
   reachable_childnodes_map< Env >& istate_to_nodes
)
{
   // first check if this node's value hasn't been already computed by another visit or is
   // already in the br map
   if(node.state_value.has_value()) {
      return node.state_value.value();
   }

   double state_value = [&] {
      if(node.active_player == m_br_player) {
         // in a BR player state only the best response action is played and thus should be
         // considered
         auto best_response_action = _best_response(*node.infostate_ptr, istate_to_nodes).action;
         return _value(*(node.children.at(best_response_action)), istate_to_nodes);
      } else {
         // in an opponent state only we have to take the expected value as the child's value
         return ranges::accumulate(
            node.children | ranges::views::transform([&](const auto& action_child_pair) {
               const auto& [action_variant, child_node_ptr] = action_child_pair;
               return _value(*child_node_ptr, istate_to_nodes)
                      * (child_node_ptr->opp_reach_prob / node.opp_reach_prob);
            }),
            double(0.),
            std::plus{}
         );
      }
   }();
   node.state_value = state_value;
   return state_value;
}

}  // namespace nor

#endif  // NOR_BEST_RESPONSE_HPP
