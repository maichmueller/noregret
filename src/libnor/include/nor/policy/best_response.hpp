
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

namespace detail {

template < concepts::fosg Env >
struct best_response_impl {
   using world_state_type = auto_world_state_type< Env >;
   using info_state_type = auto_info_state_type< Env >;
   using observation_type = auto_observation_type< Env >;
   using action_type = auto_action_type< Env >;
   using action_variant_type = auto_action_variant_type< Env >;

   struct WorldNode {
      /// the state value of this node.
      std::optional< double > state_value = std::nullopt;
      /// the likelihood that the opponents play to this world state.
      double opp_reach_prob;
      /// the child nodes reachable from this infostate the child node pointer that the action leads
      std::unordered_map<
         action_variant_type,
         uptr< WorldNode >,
         common::variant_hasher< action_variant_type > >
         children{};
      /// whether this node is a node of the best responding player or general opponent
      bool is_br_node;
      /// a backreferencing pointer to the infostate that this worldstate belongs to
      const info_state_type* infostate_ptr = nullptr;
   };

   template < typename EnvT, typename... Args >
   best_response_impl(Player player, EnvT&& env, Args&&... args)
       : m_br_player(player), m_infostate_children_map()
   {
      _run(env, std::forward< Args >(args)...);
   }

  private:
   Player m_br_player;

   /// The map of infostates points to another map of actions -> all possible child worldstates
   /// reachable from this action.
   /// Such a map is necessary, since each infostate is produced by a collection of worldstates that
   /// are consistent with this infostate. Each of these worldstates has the same legal actions and
   /// thus offers the options to choose. But in each case a different child worldstate is reached.
   /// Precisely those are captured in these vectors.
   using child_node_map = std::unordered_map<
      action_variant_type,
      std::vector< WorldNode* >,
      common::variant_hasher< action_variant_type > >;

   std::unordered_map< info_state_type, child_node_map > m_infostate_children_map;

   void _run(
      Env& env,
      std::unordered_map< Player, StatePolicyView< info_state_type, action_type > > player_policies,
      const world_state_type& root_state,
      std::unordered_map< info_state_type, std::pair< action_type, double > >& best_response_map,
      std::unordered_map< Player, info_state_type > root_infostates = {}
   );

   void _compute_best_responses(
      std::unordered_map< info_state_type, std::pair< action_type, double > >& best_response_map
   );

   auto _best_response(const info_state_type& infostate);

   double _value(WorldNode& node);
};

// deduction guide
template < typename EnvT, typename... Args >
best_response_impl(Player player, EnvT&& env, Args&&... args)
   -> best_response_impl< std::remove_cvref_t< EnvT > >;

template < concepts::fosg Env >
void best_response_impl< Env >::_run(
   Env& env,
   std::unordered_map< Player, StatePolicyView< info_state_type, action_type > > player_policies,
   const world_state_type& root_state,
   std::unordered_map< info_state_type, std::pair< action_type, double > >& best_response_map,
   std::unordered_map< Player, info_state_type > root_infostates
)
{
   auto players = env.players(root_state);
   for(auto player : players) {
      auto istate_iter = root_infostates.find(player);
      if(istate_iter == root_infostates.end()) {
         // if one of the infostates is missing then all must be missing or empty, otherwise we
         // have different information states for each player which would lead to inconsistencies
         if(not root_infostates.empty()
            or ranges::any_of(root_infostates | ranges::views::values, [](const auto& infostate) {
                  return infostate.size() > 0;
               })) {
            throw std::invalid_argument(
               "The given infostate map has inconsistent infostates (some player states are "
               "missing, but others are given)."
            );
         }
      }
   }
   for(auto player : players) {
      if(not root_infostates.contains(player)) {
         root_infostates.emplace(player, info_state_type{player});
      }
   }

   struct VisitData {
      double opp_reach_prob;
      std::unordered_map< Player, info_state_type > infostates;
      std::unordered_map< Player, std::vector< std::pair< observation_type, observation_type > > >
         observation_buffer;
      WorldNode* parent = nullptr;
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
               auto [child_obs_buffer, child_istate_map] = std::invoke([&] {
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
               });
               double prob = std::invoke([&] {
                  if constexpr(std::same_as< ActionT, action_type >) {
                     return prob = curr_player == m_br_player
                                      ? 1.
                                      : player_policies.at(curr_player)
                                           .at(visit_data.infostates.at(curr_player))
                                           .at(action_or_outcome);
                  } else {
                     // the constexpr check for the action type ensures that we will never reach
                     // this case for a deterministic environment! This is important since
                     // deterministic envs are not required to provide the chance_probability
                     // function and so this code would otherwise not compile for them
                     return env.chance_probability(*curr_state, action_or_outcome);
                  }
               });
               return std::tuple{
                  std::move(prob), std::move(child_obs_buffer), std::move(child_istate_map)};
            },
            [&](std::monostate) {
               // this should never be visited, but if so --> error
               throw std::logic_error("We entered a std::monostate visit branch.");
               return std::tuple{1., visit_data.observation_buffer, visit_data.infostates};
            }},
         *curr_action
      );

      double child_reach_prob = visit_data.opp_reach_prob
                                * (double(curr_player == m_br_player)
                                   + double(curr_player != m_br_player) * action_prob);

      auto next_parent = visit_data.parent->children
                            .try_emplace(
                               *curr_action,
                               std::make_unique< WorldNode >(WorldNode{
                                  .state_value = env.is_terminal(*next_state) ? std::optional(
                                                    env.reward(m_br_player, *next_state)
                                                 )
                                                                              : std::nullopt,
                                  .opp_reach_prob = child_reach_prob,
                                  .is_br_node = env.active_player(*next_state) == m_br_player})
                            )
                            .first->second.get();

      // check if we should try to emplace the infostate into the infostate-to-children map and then
      // add the child pointer if so.
      if(curr_player == m_br_player) {
         // we only emplace BR player infostates
         auto [iter, _] = m_infostate_children_map.try_emplace(visit_data.infostates.at(m_br_player)
         );
         auto& [infostate, child_nodes] = *iter;
         // assign this infostate to the node
         visit_data.parent->infostate_ptr = &infostate;
         // if the parent is a BR node then we have to emplace all potential child worldnodes for
         // this infostate
         child_nodes[*curr_action].emplace_back(next_parent);
      }

      return VisitData{
         .opp_reach_prob = child_reach_prob,
         .infostates = std::move(child_infostate_map),
         .observation_buffer = std::move(child_observation_buffer),
         .parent = next_parent,
      };
   };

   auto root_node = std::invoke([&] {
      auto root_player = env.active_player(root_state);
      const info_state_type* infostate_ptr = nullptr;
      if(root_player == m_br_player) {
         auto [iter, _] = m_infostate_children_map.try_emplace(root_infostates.at(root_player));
         infostate_ptr = &(iter->first);
      }
      return WorldNode{
         .opp_reach_prob = 1.,
         .is_br_node = root_player == m_br_player,
         .infostate_ptr = infostate_ptr};
   });
   forest::GameTreeTraverser(env).walk(
      utils::dynamic_unique_ptr_cast< world_state_type >(utils::clone_any_way(root_state)),
      VisitData{
         .opp_reach_prob = 1.,
         .infostates = std::move(root_infostates),
         .observation_buffer = {},
         .parent = &root_node},
      forest::TraversalHooks{.child_hook = std::move(child_hook)}
   );

   _compute_best_responses(best_response_map);
}

template < concepts::fosg Env >
void best_response_impl< Env >::_compute_best_responses(
   std::unordered_map< info_state_type, std::pair< action_type, double > >& best_response_map
)
{
   for(const auto& infostate : m_infostate_children_map | ranges::views::keys) {
      // we compute best respones only for the br player
#ifndef NDEBUG
      if(infostate.player() != m_br_player) {
         throw std::invalid_argument("Best response action requested at an opponent info state.");
      }
#endif
      auto br = _best_response(infostate);
      best_response_map.emplace(
         std::piecewise_construct,
         std::forward_as_tuple(infostate),
         std::forward_as_tuple(br.action, br.value)
      );
   }
}

template < concepts::fosg Env >
auto best_response_impl< Env >::_best_response(const info_state_type& infostate)
{
   // we can assume that this is an infostate of the best responding player
   std::optional< action_type > best_action = std::nullopt;
   double state_value = std::numeric_limits< double >::lowest();
   for(const auto& [action_variant, node_vec] : m_infostate_children_map.at(infostate)) {
      double action_value = ranges::accumulate(
         node_vec | ranges::views::transform([&](WorldNode* child_node_uptr) {
            return _value(*child_node_uptr) * child_node_uptr->opp_reach_prob;
         }),
         double(0.),
         std::plus{}
      );
      if(action_value > state_value) {
         best_action = std::get< 0 >(action_variant);
         state_value = action_value;
      }
   }
   return std::invoke([&] {
      struct {
         action_type action;
         double value;
      } value{.action = std::move(best_action.value()), .value = state_value};
      return value;
   });
}

template < concepts::fosg Env >
double best_response_impl< Env >::_value(WorldNode& node)
{
   // first check if this node's value hasn't been already computed by another visit or is
   // already in the br map
   if(node.state_value.has_value()) {
      return node.state_value.value();
   }

   double state_value = std::invoke([&] {
      if(node.is_br_node) {
         // in a BR player state only the best response action is played and thus should be
         // considered
         auto best_response_action = _best_response(*node.infostate_ptr).action;
         return _value(*(node.children.at(best_response_action)));
      } else {
         // in an opponent state only we have to take the expected value as the child's value
         // If the node has reach prob of 0. by the opponent then we don't even need to check the
         // children values, since we are in a trajectory that won't be reached in play by the
         // opponents and therefore our best-response at associated infostates will be arbitrary
         // anyway. the exact comparison of doubles here should be fine since we are actually

         // asking whether this number is precisely +-0 and not whether it is close to 0. When the
         // value is exactly 0 we will generate a nan in the following code.
         return node.opp_reach_prob == 0.
                   ? 0.
                   : ranges::accumulate(
                      node.children | ranges::views::transform([&](const auto& action_child_pair) {
                         const auto& [action_variant, child_node_ptr] = action_child_pair;
                         return _value(*child_node_ptr)
                                * (child_node_ptr->opp_reach_prob / node.opp_reach_prob);
                      }),
                      double(0.),
                      std::plus{}
                   );
      }
   });
   node.state_value = state_value;
   return state_value;
}

}  // namespace detail

template < concepts::info_state Infostate, concepts::action Action >
class BestResponsePolicy {
  public:
   using info_state_type = Infostate;
   using action_type = Action;
   using action_policy_type = HashmapActionPolicy< action_type >;

   BestResponsePolicy(Player best_response_player) : m_br_player(best_response_player) {}

   BestResponsePolicy(
      Player best_response_player,
      std::unordered_map< info_state_type, std::pair< action_type, double > > best_response_map = {}
   )
       : m_br_player(best_response_player), m_best_response(std::move(best_response_map))
   {
   }

   template < typename Env >
      requires concepts::fosg< std::remove_cvref_t< Env > > decltype(auto)
   allocate(
      Env&& env,
      std::unordered_map< Player, StatePolicyView< info_state_type, action_type > > player_policies,
      const auto_world_state_type< std::remove_cvref_t< Env > >& root_state,
      std::unordered_map< Player, info_state_type > root_infostates = {}
   )
   {
      detail::best_response_impl(
         m_br_player,
         std::forward< Env >(env),
         std::move(player_policies),
         root_state,
         m_best_response,
         std::move(root_infostates)
      );
      return *this;
   }

   [[nodiscard]] auto operator()(const info_state_type& infostate, auto&&...) const
   {
      return HashmapActionPolicy< action_type >{
         std::array{std::pair{m_best_response.at(infostate).first, 1.}}};
   }

   [[nodiscard]] auto at(const info_state_type& infostate, auto&&...) const
   {
      return operator()(infostate);
   }

   [[nodiscard]] auto& map() const { return m_best_response; }
   [[nodiscard]] auto value(const info_state_type& infostate) const
   {
      return m_best_response.at(infostate).second;
   }
   [[nodiscard]] auto size() const { return m_best_response.size(); }

  private:
   Player m_br_player;
   std::unordered_map< info_state_type, std::pair< action_type, double > > m_best_response;
};

}  // namespace nor

#endif  // NOR_BEST_RESPONSE_HPP
