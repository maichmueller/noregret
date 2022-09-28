
#ifndef NOR_BEST_RESPONSE_HPP
#define NOR_BEST_RESPONSE_HPP

#include <queue>

#include "default_policy.hpp"
#include "nor/concepts.hpp"
#include "nor/rm/rm_utils.hpp"

namespace nor::rm {

template <
   concepts::fosg Env,
   typename StatePolicy,
   typename ValueTable = std::unordered_map<
      typename fosg_auto_traits< Env >::info_state_type,
      std::unordered_map< typename fosg_auto_traits< Env >::action_type, double > > >
class BestResponsePolicy {
  public:
   using world_state_type = typename fosg_auto_traits< Env >::world_state_type;
   using info_state_type = typename fosg_auto_traits< Env >::info_state_type;
   using action_type = typename fosg_auto_traits< Env >::action_type;
   using chance_outcome_type = typename fosg_auto_traits< Env >::chance_outcome_type;
   using observation_type = typename fosg_auto_traits< Env >::observation_type;
   using value_table_type = ValueTable;

   using chance_outcome_conditional_type = std::conditional_t<
      std::is_same_v< chance_outcome_type, void >,
      std::monostate,
      chance_outcome_type >;
   using action_variant_type = std::variant< action_type, chance_outcome_conditional_type >;
   using action_variant_hasher = decltype([](const auto& action_variant) {
      return std::visit(
         []< typename VarType >(const VarType& var_element) {
            return std::hash< VarType >{}(var_element);
         },
         action_variant);
   });

   BestResponsePolicy(
      Player best_response_player,
      Env& env,
      std::unordered_map< Player, StatePolicy* > player_policies)
      requires concepts::state_policy< StatePolicy, info_state_type, action_type >
   : m_br_player(best_response_player), m_env(&env), m_player_policies(std::move(player_policies))
   {
   }

   action_type operator[](const info_state_type& infostate);

   template < typename Any >
   action_type operator[](std::pair< const info_state_type&, Any > infostate_and_rest)
   {
      return operator[](std::get< 0 >(infostate_and_rest));
   }

   template < typename... Any >
   action_type operator[](std::tuple< const info_state_type&, Any... > infostate_and_rest)
   {
      return operator[](std::get< 0 >(infostate_and_rest));
   }

   action_type operator[](const info_state_type& infostate) const;

   template < typename Any >
   action_type operator[](std::pair< const info_state_type&, Any > infostate_and_rest) const
   {
      return operator[](std::get< 0 >(infostate_and_rest));
   }

   template < typename... Any >
   action_type operator[](std::tuple< const info_state_type&, Any... > infostate_and_rest) const
   {
      return operator[](std::get< 0 >(infostate_and_rest));
   }

  private:
   Player m_br_player;
   Env* m_env;
   std::unordered_map< Player, StatePolicy* > m_player_policies;
   std::unordered_map< info_state_type, action_type > m_best_response;
   double m_root_value;

   auto& _best_response_values(
      uptr< world_state_type > wstate,
      std::unordered_map< Player, info_state_type > infostates = {});

   double _fill_best_response_values(
      uptr< world_state_type > wstate,
      std::unordered_map< Player, info_state_type > infostates,
      std::unordered_map< Player, std::vector< observation_type > > observation_buffer,
      Probability opp_reach_prob);

   /**
    * @brief Traverses all selected game actions and connects the nodes along the way.
    */

   inline void walk(uptr< world_state_type > root_state)
   {
      struct Node {
         // the parent from which this infostate came
         Node* parent;
         /// the children reachable from this infostate
         // the mapped tuple represents...
         // 1. the child node pointer that the action leads to
         // 2. the policy probability of playing this action the player at this node had according
         // to his policy
         // 3. the optional value of this action. This value is only set once the terminal values
         // have been filled for this action, or once the downstream value of the next infostate
         // node, that the action points to, has been found.
         std::unordered_map<
            action_variant_type,
            std::tuple< uptr< Node >, std::optional< Probability >, std::optional< double > >,
            action_variant_hasher >
            children;
         /// the player that takes the actions at this node. (Could be the chance player too!)
         Player active_player;
         /// the state value of this node. It can only be computed once the entire tree has been
         /// traversed and all trajectories terminal values were found
         std::optional< double > state_value = std::nullopt;
      };
      std::unordered_map< info_state_type, Node* > istate_to_node;

      struct VisitationData {
         // the infostates are meant to be maintained by the istate-to-node map which lives longer
         // than the visitation data objects. Hence, we can safely refer to reference wrappers of
         // those without seg-faulting
         std::unordered_map< Player, std::reference_wrapper< info_state_type > > infostates;
         std::unordered_map< Player, std::vector< observation_type > > observation_buffer;
      };

      // the tree needs to be traversed. To do so, every node (starting from the root node aka
      // the current game state) will emplace its child states - as generated from its possible
      // actions
      // - into the queue. This queue is Last-In-First-Out (LIFO), hence referred to as 'stack',
      // which will guarantee that we perform a depth-first-traversal of the game tree (FIFO
      // would lead to breadth-first). This is necessary since any state-value of a given node is
      // computed via the probability of each action multiplied by their successor state-values,
      // i.e. v(s) = \sum_a \pi(s,a) * v(s'), the same holds for reach probabilities

      // we first define the action emplacing lambda since it will be used in the beginning for the
      // root node and later for all child nodes again
      auto action_emplacer = [&](Node& infostate_node, world_state_type& state) {
         if(infostate_node.children.empty()) {
            if constexpr(concepts::stochastic_fosg< Env, world_state_type, action_type >) {
               if(m_env->active_player(state) == Player::chance) {
                  for(auto&& outcome : m_env->chance_actions(state)) {
                     infostate_node.children.emplace(outcome, {});
                  }

               } else {
                  for(auto&& act : m_env->actions(state)) {
                     infostate_node.children.emplace(act, {});
                  }
               }
            } else {
               for(auto&& act : m_env->actions(state)) {
                  infostate_node.children.emplace(act, {});
               }
            }
         }
      };

      // we first emplace the root infostate node. We need to keep the variable of it, in order to
      // start the information state update DFS later in the function in which we find the actual
      // best response values
      auto root_player = m_env->active_player(*root_state);
      auto root_node = Node{.infostate = nullptr, .children = {}, .active_player = root_player};
      if(root_player != Player::chance) {
         root_node.infostate = &(
            *(istate_to_node.emplace(info_state_type{root_player}, &root_node).first));
      }
      action_emplacer(root_node, *root_state);
      // the visitation stack. Each node in this stack will be visited once according to the
      // traversal strategy selected.
      std::stack< std::tuple< uptr< world_state_type >, VisitationData, Node* > > visit_stack;

      // emplace the root node into the visitation stack
      visit_stack.emplace(
         utils::static_unique_ptr_downcast< world_state_type >(utils::clone_any_way(root_state)),
         {.infostates =
             [&] {
                std::unordered_map< Player, info_state_type > infostates;
                for(auto player : m_env->players()) {
                   infostates.emplace(player, info_state_type{player});
                }
                return infostates;
             }(),
          .observation_buffer = {}});

      auto outcome_traverser = [&](
                                  world_state_type& curr_state,
                                  VisitationData& visit_data,
                                  Node& curr_node) {
         auto curr_player = m_env->active_player(*curr_state);
         for(auto& [action_variant, uptr_prob_value_tuple] : curr_node.children) {
            // emplace the child node for this action if it doesn't already exist.
            // Another trajectory could have already emplaced it and that is fine, since every
            // trajectory contained in an information state has the same cf. reach probability
            auto& [next_node_uptr, action_prob, action_value] = uptr_prob_value_tuple;

            auto [next_state, curr_action_prob] = std::visit(
               [&](const auto& action_or_outcome) {
                  // since we are only interested in counterfactual reach probabilities for
                  // the pure best response of the given player, we do not account for that
                  // player's action probability, but for each opponent, we do fetch their
                  // policy probability
                  return std::pair{
                     child_state(*m_env, curr_state, action_or_outcome),
                     Probability{
                        action_prob.has_value()
                           ? action_prob.value().get()
                           : (curr_player == m_br_player
                                 ? 1.
                                 : m_player_policies[curr_player][visit_data.infostates.at(
                                    curr_player)][action_or_outcome])}};
               },
               [&](const chance_outcome_type& outcome) {
                  return std::pair{
                     child_state(*m_env, curr_state, outcome),
                     Probability{m_env->chance_probability(*curr_state, std::get< 0 >(outcome))}};
               },
               action_variant);
            // we overwrite the existing action_prob here since any worldstate pertaining to the
            // infostate is going to have the same action probability (since player's can only
            // choose according to the knowledge they have in the information state and chance
            // states will simply assign the same value again.)
            action_prob = curr_action_prob;
            Player next_active_player = m_env->is_terminal(*next_state);

            if(not next_node_uptr) {
               // create the child node unique ptr. The parent takes ownership of the child node.
               next_node_uptr = std::make_unique< Node >(nullptr, {}, next_active_player);
            }
            if(next_active_player != Player::chance) {
               next_node_uptr->infostate = &(
                  *(istate_to_node
                       .try_emplace(
                          visit_data.infostates.at(m_env->active_player(*next_state)),
                          next_node_uptr.get())
                       .first));
            }
            // if the actions/outcomes have not yet been emplaced into this infostate node
            // then we do this here. (They could have been already emplaced by another
            // trajectory passing through this infostate before and emplacing them there)
            // we also compute the probability of this action being played depending on whether
            // it is a chance action or a player action
            action_emplacer(*next_node_uptr, *next_state);

            if(m_env->is_terminal((next_state))) {
               // if the child is a terminal state then we would like to take the reward and add
               // that to the value of the infostate node
               action_value = action_value.value_or(0.) + m_env->reward(curr_player, *next_state);
            } else {
               // since it isn't a terminal state we emplace the child state to visit further
               auto [child_observation_buffer, child_infostate_map] = std::visit(
                  [&](const auto& action_or_outcome) {
                     m_env->transition(*next_state, action_or_outcome);
                     auto [child_obs_buffer, child_istate_map] = fill_infostate_and_obs_buffers(
                        *m_env,
                        visit_data.observation_buffer,
                        visit_data.infostates,
                        action_or_outcome,
                        *next_state);
                     return std::tuple{child_obs_buffer, child_istate_map};
                  },
                  action_variant);

               visit_stack.emplace(
                  std::move(next_state),
                  VisitationData{
                     .infostates = std::move(child_infostate_map),
                     .observation_buffer = std::move(child_observation_buffer)},
                  next_node_uptr.get());
            }
         }
      };

      while(not visit_stack.empty()) {
         // get the top node and world state from the stack and move them into our values.
         auto [curr_wstate, visit_data, node] = std::move(visit_stack.top());
         // remove those elements from the stack (there is no unified pop-and-return method for
         // stack)
         visit_stack.pop();

         action_traverser(*curr_wstate, visit_data, node);
      }

      const auto best_response_recursor = [&]() {
         auto best_response_recursor_impl = [&](Node& curr_node, auto& br_recursor_ref) mutable {
            // first check if this node's value hasn't been already computed by another visit
            if(curr_node.state_value.has_value()) {
               return curr_node.state_value.value();
            }

            auto child_traverser = [&](auto state_value_updater) {
               for(auto& [action_variant, child_tuple] : curr_node.children) {
                  auto& [child_node_uptr, action_prob, action_value_opt] = child_tuple;
                  // the action_value_optional will only be set if this action leads directly to a
                  // terminal state. In that case we know exactly what the value of the action is
                  // and can set fetch it. We don't set the action_value with the computed value
                  // otherwise, since we are not going to return to this infostate anymore.
                  double child_value = action_value_opt.has_value()
                                          ? action_value_opt.value()
                                          : br_recursor_ref(*child_node_uptr, br_recursor_ref);
                  state_value_updater(action_variant, action_prob, child_value);
               }
            };

            if(curr_node.active_player == m_br_player) {
               // if this is an infostate pertaining to the Best Response player then we need to
               // compute:
               // 1. value(I) = max_{a}(v(a | I))
               // 2. best_response(I) = argmax_{a}(v(a | I))
               // we return value(I) back up as this infostate's value to the BR player.
               std::optional< action_type > best_action = std::nullopt;
               double state_value = std::numeric_limits< double >::lowest();
               child_traverser([&](const auto& action_variant, Probability, double child_value) {
                  if(child_value > state_value) {
                     // the action variant holds the action type in the second slot
                     best_action = std::get< 1 >(action_variant);
                     state_value = child_value;
                  }
               });
               // store this value to not trigger a recomputation upon visit from another Infostate.
               curr_node.state_value = state_value;
               m_best_response.emplace(*(curr_node.infostate), best_action.value());
               return state_value;
            } else {
               // if this is not an infostate of the Best Response player then it is either a
               // chance node or an opponent node. In both cases we compute:
               //    value(I) = sum_{a}(policy(a | I) * v(a | I))
               // we return value(I) back up as this infostate's value to the BR player.
               double state_value = 0.;
               child_traverser([&](const auto&, Probability action_prob, double child_value) {
                  state_value += action_prob.get() * child_value;
               });
               // store this value to not trigger a recomputation upon visit from another Infostate.
               curr_node.state_value = state_value;
               return state_value;
            }
         };

         return best_response_recursor_impl(root_node, best_response_recursor_impl);
      }();
   }
};

//template < concepts::fosg Env, typename StatePolicy, typename ValueTable >
//auto& BestResponsePolicy< Env, StatePolicy, ValueTable >::_best_response_values(
//   uptr< world_state_type > wstate,
//   std::unordered_map< Player, info_state_type > infostates)
//{
//   if(infostates.empty()) {
//      // if the infostates are all empty then we will be filling them with empty ones, assuming
//      // this is the root of the game.
//      for(auto player : m_env->players(*wstate)) {
//         infostates.emplace(player, info_state_type{player});
//      }
//   } else {
//      if(m_env->players(*wstate) != ranges::to_vector(infostates | ranges::views::keys)) {
//         throw std::invalid_argument(
//            "The passed map of infostates does not align with the environment's partaking "
//            "player "
//            "list.");
//      }
//   }
//   std::unordered_map< info_state_type, std::unordered_map< action_type, double > >
//      action_value_table{};
//   detail::_fill_best_response_values(wstate, std::move(infostates), action_value_table);
//   return action_value_table;
//}
//
//template < concepts::fosg Env, typename StatePolicy, typename ValueTable >
//double BestResponsePolicy< Env, StatePolicy, ValueTable >::_fill_best_response_values(
//   uptr< world_state_type > wstate,
//   std::unordered_map< Player, info_state_type > infostates,
//   std::unordered_map< Player, std::vector< observation_type > > observation_buffer,
//   Probability opp_reach_prob)
//{
//   if(m_env->is_terminal(*wstate)) {
//      return m_env->reward(m_br_player, *wstate);
//   }
//
//   if constexpr(concepts::stochastic_fosg< Env, world_state_type, action_type >) {
//      if(m_env->active_player(*wstate) == Player::chance) {
//         // iterate over all chance outcomes and create the expected value of them. Pass this
//         // value back up.
//
//         double value = 0.;
//         for(auto outcome : m_env->chance_outcomes(*wstate)) {
//            double outcome_prob = m_env->chance_probability(*wstate, outcome);
//            auto next_state = child_state(*m_env, wstate, outcome);
//
//            auto child_opp_reach = opp_reach_prob.get() * outcome_prob;
//
//            auto [child_observation_buffer, child_infostate_map] = fill_infostate_and_obs_buffers(
//               *m_env, observation_buffer, infostates, outcome, *next_state);
//
//            auto child_value = _fill_best_response_values(
//
//               std::move(next_state),
//               child_infostate_map,
//               child_observation_buffer,
//               Probability{child_opp_reach});
//
//            value += outcome_prob * child_value;
//         }
//         return value;
//      }
//   }
//
//   Player active_player = m_env->active_player(*wstate);
//   const auto& curr_infostate = infostates.at(active_player);
//   auto& child_values = m_value_map[infostates.at(active_player)];
//   if(active_player == m_br_player) {
//      // the active player is the best response seeking player
//      std::optional< action_type > best_response_action = std::nullopt;
//      double best_response_value = std::numeric_limits< double >::lowest();
//      for(const auto& action : m_env->actions(active_player, *wstate)) {
//         auto next_state = child_state(*m_env, wstate, action);
//         auto [child_observation_buffer, child_infostate_map] = fill_infostate_and_obs_buffers(
//            *m_env, observation_buffer, infostates, action, *next_state);
//         double child_value = _fill_best_response_values(
//            std::move(next_state), child_infostate_map, child_observation_buffer, opp_reach_prob);
//         // memorize this action value, so that after the tree traversal we can iterate over them
//         // and find the maximum, once the values over each history in this information state
//         // have been accumulated. Note that if this action has never been emplaced before, then
//         // the bracket operator[] will emplace it with the default value 0., which is precisely
//         // what we want in this case.
//         child_values[action] += child_value;
//         // we only send back up the tree the best value that can be found in this history for
//         // the best response player
//         best_response_value = std::max(best_response_value, child_value);
//      }
//      return best_response_value;
//   } else {
//      // set up the action value and weight containers. These are computed regardless of whether
//      // the active player is a best responder or not.
//      auto legal_actions = m_env->actions(active_player, *wstate);
//      auto opp_action_policy = rm::normalize_action_policy(
//         m_player_policies.at(active_player)[{curr_infostate, legal_actions}]);
//      double value = 0.;
//      for(const auto& action : legal_actions) {
//         // we first have to compute the weight of this action and the new child reach
//         // probability from the opponent policy
//         auto action_prob = opp_action_policy[action];
//         auto child_opp_reach_prob = opp_reach_prob.get() * action_prob;
//
//         auto next_state = child_state(*m_env, wstate, action);
//         auto [child_observation_buffer, child_infostate_map] = fill_infostate_and_obs_buffers(
//            *m_env, observation_buffer, infostates, action, *next_state);
//         double child_value = _fill_best_response_values(
//            std::move(next_state), child_infostate_map, child_observation_buffer, opp_reach_prob);
//         // at opponent nodes we need to gather the counterfactual expected value of the
//         // trajectory (history) that we are currently at.
//         value += child_value * action_prob;
//      }
//      return value;
//   }
//}

}  // namespace nor::rm

#endif  // NOR_BEST_RESPONSE_HPP
