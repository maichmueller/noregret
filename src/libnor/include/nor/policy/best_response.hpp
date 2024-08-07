
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

struct BRConfig {
   bool store_infostate_values = false;
};

namespace detail {

template < BRConfig config, typename Action >
using mapped_br_type = std::
   conditional_t< config.store_infostate_values, std::pair< Action, double >, Action >;

template < BRConfig config, concepts::fosg Env >
struct best_response_impl {
   using world_state_type = auto_world_state_type< Env >;
   using info_state_type = auto_info_state_type< Env >;
   using observation_type = auto_observation_type< Env >;
   using action_type = auto_action_type< Env >;
   using action_variant_type = auto_action_variant_type< Env >;

   using mapped_type = mapped_br_type< config, action_type >;

   struct WorldNode {
      /// the state value of this node.
      std::optional< nor::player_hashmap< double > > state_value_map = std::nullopt;
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
      /// which player is the acting one at this node
      Player active_player;
      /// a backreferencing pointer to the infostate that this worldstate belongs to
      const info_state_type* infostate_ptr = nullptr;
   };

   template < typename EnvT, typename... Args >
   best_response_impl(std::vector< Player > br_players, EnvT&& env, Args&&... args)
       : m_br_players(std::move(br_players)), m_infostate_children_map()
   {
      _run(env, std::forward< Args >(args)...);
   }

  private:
   std::vector< Player > m_br_players;

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

   template < typename StatePolicy >
   void _run(
      Env& env,
      player_hashmap< StatePolicy > player_policies,
      const world_state_type& root_state,
      player_hashmap< std::unordered_map< info_state_type, mapped_type > >&
         best_response_map_to_fill,
      player_hashmap< info_state_type > root_infostates = {}
   );

   void _compute_best_responses(
      player_hashmap< std::unordered_map< info_state_type, mapped_type > >&
         best_response_map_to_fill
   );

   auto _best_response(const info_state_type& infostate);

   const player_hashmap< double >& _value(WorldNode& node);
};

template < BRConfig config, concepts::fosg Env >
template < typename StatePolicy >
void best_response_impl< config, Env >::_run(
   Env& env,
   player_hashmap< StatePolicy > player_policies,
   const world_state_type& root_state,
   player_hashmap< std::unordered_map< info_state_type, mapped_type > >& best_response_map_to_fill,
   player_hashmap< info_state_type > root_infostates
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
      bool curr_player_is_br = common::isin(curr_player, m_br_players);
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
                     return prob = curr_player_is_br ? 1.
                                                     : player_policies.at(curr_player)
                                                          .at(visit_data.infostates.at(curr_player))
                                                          .at(action_or_outcome);
                  } else {
                     // the constexpr check for the action type ensures that we will never reach
                     // this case for a deterministic environment! This is important since
                     // deterministic envs are not required to provide the chance_probability
                     // function and so this code would otherwise not compile for them
                     static_assert(
                        concepts::stochastic_fosg< Env >,
                        "Only stochastic environments should reach this point."
                     );
                     // workaround for clang not recognizing the constexpr check above correctly and
                     // still throwing an error when a deterministic env does not have a
                     // chance_probability function...
                     if constexpr(concepts::stochastic_fosg< Env >) {
                        return env.chance_probability(*curr_state, action_or_outcome);
                     } else {
                        // theoretically unreachable, but if so we throw an error
                        throw std::logic_error(
                           "This should never be reached for deterministic envs."
                        );
                     }
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
                                * (double(curr_player_is_br)
                                   + double(not curr_player_is_br) * action_prob);
      auto next_player = env.active_player(*next_state);
      auto next_parent = visit_data.parent->children
                            .try_emplace(
                               *curr_action,
                               std::make_unique< WorldNode >(WorldNode{
                                  .state_value_map = env.is_terminal(*next_state)
                                                        ? std::optional(rm::collect_rewards(
                                                           env, *next_state, m_br_players
                                                        ))
                                                        : std::nullopt,
                                  .opp_reach_prob = child_reach_prob,
                                  .is_br_node = common::isin(next_player, m_br_players),
                                  .active_player = next_player})
                            )
                            .first->second.get();

      // check if we should try to emplace the infostate into the infostate-to-children map and then
      // add the child pointer if so.
      if(curr_player_is_br) {
         // we only emplace BR player infostates
         auto [iter, _] = m_infostate_children_map.try_emplace(visit_data.infostates.at(curr_player)
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
      bool root_is_br = common::isin(root_player, m_br_players);
      if(root_is_br) {
         auto [iter, _] = m_infostate_children_map.try_emplace(root_infostates.at(root_player));
         infostate_ptr = &(iter->first);
      }
      return WorldNode{
         .opp_reach_prob = 1.,
         .is_br_node = root_is_br,
         .active_player = root_player,
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

   _compute_best_responses(best_response_map_to_fill);
}

template < BRConfig config, concepts::fosg Env >
void best_response_impl< config, Env >::_compute_best_responses(
   player_hashmap< std::unordered_map< info_state_type, mapped_type > >& best_response_map_to_fill
)
{
   auto build_mapped_type_args = [&](auto& br) {
      if constexpr(config.store_infostate_values)
         return std::tuple{std::move(br.action), std::move(br.value)};
      else
         return std::tuple{std::move(br.action)};
   };
   auto& player_br_map = best_response_map_to_fill[m_br_players.front()];
   auto fetch_player_br_map = [&, n_br_players = m_br_players.size()](Player player) -> auto& {
      if(n_br_players == 1) {
         return player_br_map;
      } else {
         return best_response_map_to_fill[player];
      }
   };
   for(const auto& infostate :
       m_infostate_children_map | ranges::views::keys
          | ranges::views::filter([&](const auto& istate) {
               return not fetch_player_br_map(istate.player()).contains(istate);
            })) {
#ifndef NDEBUG
      // we compute best-responses only for the br player
      if(not common::isin(infostate.player(), m_br_players)) {
         throw std::invalid_argument("Best response action requested at an opponent info state.");
      }
#endif
      auto br = _best_response(infostate);
      fetch_player_br_map(infostate.player())
         .emplace(
            std::piecewise_construct, std::forward_as_tuple(infostate), build_mapped_type_args(br)
         );
   }
}

template < BRConfig config, concepts::fosg Env >
auto best_response_impl< config, Env >::_best_response(const info_state_type& infostate)
{
   // we can assume that this is an infostate of the best responding player
   Player best_responder = infostate.player();
   std::optional< action_type > best_action = std::nullopt;
   double state_value = std::numeric_limits< double >::lowest();
   for(const auto& [action_variant, node_vec] : m_infostate_children_map.at(infostate)) {
      double action_value = ranges::accumulate(
         node_vec | ranges::views::transform([&](WorldNode* child_node_uptr) {
            return _value(*child_node_uptr).at(best_responder) * child_node_uptr->opp_reach_prob;
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

template < BRConfig config, concepts::fosg Env >
const player_hashmap< double >& best_response_impl< config, Env >::_value(WorldNode& node)
{
   // first check if this node's value hasn't been already computed by another visit or is
   // already in the br map
   if(node.state_value_map.has_value()) {
      return *node.state_value_map;
   }

   node.state_value_map = std::invoke([&] {
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
         player_hashmap< double > running_values;
         for(auto player : m_br_players) {
            running_values.emplace(player, 0.);  // fill the map with 0 for best responders as init
         }
         // asking whether this number is precisely +-0 and not whether it is close to 0. For a
         // value that is exactly 0 we would generate a nan in the following code.
         if(node.opp_reach_prob != 0.) {
            for(const auto& action_child_pair : node.children) {
               const auto& [action_variant, child_node_ptr] = action_child_pair;
               const auto& value_map = _value(*child_node_ptr);
               for(auto player : m_br_players) {
                  running_values[player] += value_map.at(player) * child_node_ptr->opp_reach_prob
                                            / node.opp_reach_prob;
               }
            }
         }
         return running_values;
      }
   });
   return *node.state_value_map;
}

template < BRConfig config, typename Env, typename... Args >
best_response_impl< config, std::remove_cvref_t< Env > >
make_best_response_impl(std::vector< Player > br_players, Env&& env, Args&&... args)
{
   return {std::move(br_players), std::forward< Env >(env), std::forward< Args >(args)...};
}

}  // namespace detail

template <
   concepts::info_state Infostate,
   concepts::action Action,
   BRConfig config = BRConfig{.store_infostate_values = false} >
class BestResponsePolicy {
  public:
   using info_state_type = Infostate;
   using action_type = Action;
   using action_policy_type = HashmapActionPolicy< action_type >;

   using mapped_type = detail::mapped_br_type< config, action_type >;

   BestResponsePolicy(Player best_response_player) : m_best_responders{best_response_player} {}
   BestResponsePolicy(std::vector< Player > best_response_players)
       : m_best_responders{std::move(best_response_players)}
   {
   }

   BestResponsePolicy(
      Player best_response_player,
      const std::unordered_map< info_state_type, mapped_type >& cached_br_map = {}
   )
       : m_best_responders{best_response_player}, m_best_response()
   {
      _fill_from_cached_map(cached_br_map);
   }
   BestResponsePolicy(
      std::vector< Player > best_response_players,
      const std::unordered_map< info_state_type, mapped_type >& cached_br_map = {}
   )
       : m_best_responders{std::move(best_response_players)},
         m_best_response(std::move(cached_br_map))
   {
      _fill_from_cached_map(cached_br_map);
   }

   template < typename Env, typename StatePolicy >
      requires concepts::fosg< std::remove_cvref_t< Env > >
               and concepts::state_policy_view<
                  StatePolicy,
                  auto_info_state_type< std::remove_cvref_t< Env > >,
                  auto_action_type< std::remove_cvref_t< Env > > >
   decltype(auto) allocate(
      Env&& env,
      const auto_world_state_type< std::remove_cvref_t< Env > >& root_state,
      const player_hashmap< StatePolicy >& player_policies,
      player_hashmap< info_state_type > root_infostates = {}
   )
   {
      detail::make_best_response_impl< config >(
         m_best_responders,
         std::forward< Env >(env),
         player_policies,
         root_state,
         m_best_response,
         std::move(root_infostates)
      );
      return *this;
   }

   [[nodiscard]] auto operator()(const info_state_type& infostate, auto&&...) const
   {
      return HashmapActionPolicy< action_type >{
         std::pair{_get_action(m_best_response.at(infostate.player()).at(infostate)), 1.}};
   }

   [[nodiscard]] auto at(const info_state_type& infostate, auto&&...) const
   {
      return operator()(infostate);
   }

   /// ref-qualifiers for when one has an rvalue of this class and simply wants to extract the
   /// player tables from the
   [[nodiscard]] const auto& table() const& { return m_best_response; }
   [[nodiscard]] const auto& table(Player player) const& { return m_best_response.at(player); }
   [[nodiscard]] auto&& table() && { return std::move(m_best_response); }
   [[nodiscard]] auto&& table(Player player) && { return std::move(m_best_response.at(player)); }

   [[nodiscard]] auto value(const info_state_type& infostate) const
      requires(config.store_infostate_values)
   {
      return m_best_response.at(infostate.player()).at(infostate).second;
   }
   [[nodiscard]] auto size() const { return m_best_response.size(); }

  private:
   std::vector< Player > m_best_responders;
   player_hashmap< std::unordered_map< info_state_type, mapped_type > > m_best_response;

   [[nodiscard]] const auto& _get_action(const mapped_type& mapped_elem) const
   {
      if constexpr(config.store_infostate_values)
         return mapped_elem.first;
      else
         return mapped_elem;
   }

   void _fill_from_cached_map(const auto& cached_br_map)
   {
      for(const auto& [infostate, mapped_elem] : cached_br_map) {
         m_best_response[infostate.player()].emplace(infostate, mapped_elem);
      }
   }
};

}  // namespace nor

#endif  // NOR_BEST_RESPONSE_HPP
