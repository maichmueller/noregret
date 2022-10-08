
#ifndef NOR_BEST_RESPONSE_HPP
#define NOR_BEST_RESPONSE_HPP

#include <queue>

#include "default_policy.hpp"
#include "nor/concepts.hpp"
#include "nor/rm/rm_utils.hpp"
#include "policy_view.hpp"

namespace nor {

namespace detail {

template < concepts::fosg Env >
class InfostateTree {
  public:
   using world_state_type = typename fosg_auto_traits< Env >::world_state_type;
   using info_state_type = typename fosg_auto_traits< Env >::info_state_type;
   using action_type = typename fosg_auto_traits< Env >::action_type;
   using chance_outcome_type = typename fosg_auto_traits< Env >::chance_outcome_type;
   using observation_type = typename fosg_auto_traits< Env >::observation_type;

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

   struct Node {
      /// the player that takes the actions at this node. (Could be the chance player too!)
      Player active_player;
      // the parent from which this infostate came
      Node* parent = nullptr;
      /// the children reachable from this infostate
      // the mapped tuple represents...
      // 1. the child node pointer that the action leads to
      // 2. the policy probability of playing this action the player at this node had according
      // to his policy
      // 3. the optional value of this action. This value is only set once the terminal values
      // have been filled for this action, or once the downstream value of the next infostate
      // node, that the action points to, has been found.
      using child_node_tuple = std::
         tuple< uptr< Node >, std::optional< rm::Probability >, std::optional< double > >;

      std::unordered_map< action_variant_type, child_node_tuple, action_variant_hasher > children{};
      /// the infostate that is associated with this node. Remains a nullptr if it's a chance node
      uptr< info_state_type > infostate = nullptr;
      /// the state value of this node. It can only be computed once the entire tree has been
      /// traversed and all trajectories terminal values were found
      std::optional< double > state_value = std::nullopt;
   };

   InfostateTree(
      Env& env,
      uptr< world_state_type > root_state,
      std::unordered_map< Player, info_state_type > root_infostates = {})
       : m_env(env),
         m_root_state(std::move(root_state)),
         m_root_node(Node{.active_player = env.active_player(*m_root_state)}),
         m_root_infostates(std::move(root_infostates))
   {
      if(m_root_node.active_player != Player::chance) {
         auto infostate_iter = m_root_infostates.find(m_root_node.active_player);
         if(infostate_iter != m_root_infostates.end()) {
            m_root_node.infostate = std::make_unique< info_state_type >(infostate_iter->second);
         }
      }
      for(auto player : env.players(*m_root_state)) {
         auto istate_iter = m_root_infostates.find(player);
         if(istate_iter == m_root_infostates.end()) {
            m_root_infostates.emplace(player, info_state_type{player});
         }
      }
      action_emplacer(m_root_node, *root_state);
   }

   void build(
      Player br_player,
      std::unordered_map< Player, StatePolicyView< info_state_type, action_type > >
         player_policies);

   auto& env() { return m_env; }
   auto& env() const { return m_env; }
   auto& root_state() { return *m_root_state; }
   auto& root_state() const { return *m_root_state; }
   auto& root_node() { return m_root_node; }
   auto& root_node() const { return m_root_node; }

  private:
   Env& m_env;
   uptr< world_state_type > m_root_state;
   Node m_root_node;
   std::unordered_map< Player, info_state_type > m_root_infostates;

   auto action_emplacer(Node& infostate_node, world_state_type& state);
};

template < concepts::fosg Env >
auto InfostateTree< Env >::action_emplacer(
   InfostateTree::Node& infostate_node,
   InfostateTree::world_state_type& state)
{
   if(infostate_node.children.empty()) {
      if constexpr(concepts::stochastic_fosg< Env, world_state_type, action_type >) {
         if(auto active_player = m_env.active_player(state); active_player == Player::chance) {
            for(auto&& outcome : m_env.chance_actions(state)) {
               infostate_node.children.emplace(outcome);
            }

         } else {
            for(auto&& act : m_env.actions(active_player, state)) {
               infostate_node.children.emplace(act);
            }
         }
      } else {
         for(auto&& act : m_env.actions(m_env.active_player(state), state)) {
            infostate_node.children.emplace(act);
         }
      }
   }
}

template < concepts::fosg Env >
void InfostateTree< Env >::build(
   Player br_player,
   std::unordered_map< Player, StatePolicyView< info_state_type, action_type > > player_policies)
{
   // the tree needs to be traversed. To do so, every node (starting from the root node aka
   // the current game state) will emplace its child states - as generated from its possible
   // actions - into the queue. This queue is Last-In-First-Out (LIFO), ie a 'stack', which will
   // guarantee that we perform a depth-first-traversal of the game tree (FIFO would lead to
   // breadth-first).

   struct VisitationData {
      // the infostates are meant to be maintained by the istate-to-node map which lives longer
      // than the visitation data objects. Hence, we can safely refer to reference wrappers of
      // those without seg-faulting
      std::unordered_map< Player, std::reference_wrapper< info_state_type > > infostates;
      std::unordered_map< Player, std::vector< observation_type > > observation_buffer;
   };

   // the visitation stack. Each node in this stack will be visited once according to the
   // traversal strategy selected.
   std::stack< std::tuple< uptr< world_state_type >, VisitationData, Node* > > visit_stack;

   // emplace the root node into the visitation stack
   visit_stack.emplace(
      utils::static_unique_ptr_downcast< world_state_type >(utils::clone_any_way(m_root_state)),
      VisitationData{
         .infostates =
            [&] {
               std::unordered_map< Player, std::reference_wrapper< info_state_type > > infostates;
               for(auto player : m_env.players(*m_root_state)) {
                  infostates.emplace(player, std::ref(m_root_infostates.at(player)));
               }
               return infostates;
            }(),
         .observation_buffer = {}},
      &m_root_node);

   while(not visit_stack.empty()) {
      // get the top node and world state from the stack and move them into our values.
      auto [curr_state, visit_data, curr_node] = std::move(visit_stack.top());
      // remove those elements from the stack (there is no unified pop-and-return method for
      // stack)
      visit_stack.pop();

      auto curr_player = m_env.active_player(*curr_state);
      for(auto& [action_variant, uptr_prob_value_tuple] : curr_node->children) {
         // emplace the child node for this action if it doesn't already exist.
         // Another trajectory could have already emplaced it and that is fine, since every
         // trajectory contained in an information state has the same cf. reach probability
         auto& [next_node_uptr, action_prob, action_value] = uptr_prob_value_tuple;
         auto [next_state, curr_action_prob] = std::visit(
            common::Overload{
               [&](const action_type& action) {
                  // since we are only interested in counterfactual reach probabilities for
                  // the pure best response of the given player, we do not account for that
                  // player's action probability, but for each opponent, we do fetch their
                  // policy probability
                  return std::pair{
                     child_state(m_env, curr_state, action),
                     rm::Probability{
                        action_prob.has_value()
                           ? action_prob.value().get()
                           : (curr_player == br_player
                                 ? 1.
                                 : player_policies.at(curr_player)[std::tuple{
                                    visit_data.infostates.at(curr_player).get(),
                                    std::vector< action_type >{}}][action])}};
               },
               [&](const chance_outcome_conditional_type& outcome) {
                  if constexpr(concepts::deterministic_fosg< Env >) {
                     // we shouldn't reach here if this is a deterministic fosg
                     throw std::logic_error(
                        "A deterministic environment traversed a chance outcome. "
                        "This should not occur.");
                     // this return is needed to silence the non-matching return types
                     return std::pair{uptr< world_state_type >{nullptr}, rm::Probability{1.}};
                  } else {
                     return std::pair{
                        child_state(m_env, curr_state, outcome),
                        rm::Probability{m_env.chance_probability(*curr_state, outcome)}};
                  }
               }},
            action_variant);
         // we overwrite the existing action_prob here since any worldstate pertaining to the
         // infostate is going to have the same action probability (since player's can only
         // choose according to the knowledge they have in the information state and chance
         // states will simply assign the same value again.)
         action_prob = curr_action_prob;
         Player next_active_player = m_env.active_player(*next_state);

         if(not next_node_uptr) {
            // create the child node unique ptr. The parent takes ownership of the child node.
            next_node_uptr = std::make_unique< Node >(
               Node{.active_player = next_active_player, .infostate = nullptr});
         }

         if(next_active_player != Player::chance) {
            next_node_uptr->infostate = std::make_unique< info_state_type >(
               visit_data.infostates.at(m_env.active_player(*next_state)));
         }
         // if the actions/outcomes have not yet been emplaced into this infostate node
         // then we do this here. (They could have been already emplaced by another
         // trajectory passing through this infostate before and emplacing them there)
         // we also compute the probability of this action being played depending on whether
         // it is a chance action or a player action
         action_emplacer(*next_node_uptr, *next_state);

         if(m_env.is_terminal(*next_state)) {
            // if the child is a terminal state then we would like to take the reward and add
            // that to the value of the infostate node
            action_value = action_value.value_or(0.) + m_env.reward(curr_player, *next_state);
         } else {
            // since it isn't a terminal state we emplace the child state to visit further
            auto [child_observation_buffer, child_infostate_map] = std::visit(
               common::Overload{
                  [&](std::monostate) {
                     // this will never be visited anyway, but if so --> error
                     throw std::logic_error("We entered a std::monostate visit branch.");
                     return std::tuple{visit_data.observation_buffer, visit_data.infostates};
                  },
                  [&](const auto& action_or_outcome) {
                     m_env.transition(*next_state, action_or_outcome);
                     auto [child_obs_buffer, child_istate_map] = fill_infostate_and_obs_buffers(
                        m_env,
                        visit_data.observation_buffer,
                        visit_data.infostates,
                        action_or_outcome,
                        *next_state);
                     return std::tuple{child_obs_buffer, child_istate_map};
                  }},
               action_variant);

            visit_stack.emplace(
               std::move(next_state),
               VisitationData{
                  .infostates = std::move(child_infostate_map),
                  .observation_buffer = std::move(child_observation_buffer)},
               next_node_uptr.get());
         }
      }
   }
};

}  // namespace detail

template < concepts::info_state Infostate, concepts::action Action >
class BestResponsePolicy {
  public:
   using info_state_type = Infostate;
   using action_type = Action;

   BestResponsePolicy(Player best_response_player) : m_br_player(best_response_player) {}

   BestResponsePolicy(
      Player best_response_player,
      std::unordered_map< info_state_type, action_type > best_response_map = {})
       : m_br_player(best_response_player), m_best_response(std::move(best_response_map))
   {
   }

   template < concepts::fosg Env >
   auto& allocate(
      Env& env,
      std::unordered_map< Player, StatePolicyView< info_state_type, action_type > > player_policies,
      uptr< typename fosg_auto_traits< Env >::world_state_type > root_state,
      std::unordered_map< Player, info_state_type > root_infostates = {})
   {
      auto istate_tree = detail::InfostateTree(
         env, std::move(root_state), std::move(root_infostates));
      istate_tree.build(m_br_player, std::move(player_policies));
      m_root_value = _compute_best_responses< Env >(istate_tree.root_node());
      return *this;
   }

   auto operator[](const info_state_type& infostate) const
   {
      return HashmapActionPolicy< action_type >{{m_best_response.at(infostate), 1.}};
   }

   template < typename Any >
   auto operator[](std::pair< const info_state_type&, Any > infostate_and_rest) const
   {
      return operator[](std::get< 0 >(infostate_and_rest));
   }

   template < typename... Any >
   auto operator[](std::tuple< const info_state_type&, Any... > infostate_and_rest) const
   {
      return operator[](std::get< 0 >(infostate_and_rest));
   }

   auto& map() const { return m_best_response; }

  private:
   Player m_br_player;
   std::unordered_map< info_state_type, action_type > m_best_response;
   double m_root_value = 0.;

   template < typename Env >
   double _compute_best_responses(typename detail::InfostateTree< Env >::Node& curr_node);
};

template < concepts::info_state Infostate, concepts::action Action >
template < typename Env >
double BestResponsePolicy< Infostate, Action >::_compute_best_responses(
   typename detail::InfostateTree< Env >::Node& curr_node)
{
   // first check if this node's value hasn't been already computed by another visit
   if(curr_node.state_value.has_value()) {
      return curr_node.state_value.value();
   }

   auto child_traverser = [&](auto state_value_updater) {
      for(const auto& [action_variant, child_tuple] : curr_node.children) {
         const auto& [child_node_uptr, action_prob, action_value_opt] = child_tuple;
         // the action_value_optional will only hold a value if this action leads directly
         // to a terminal state. In that case we know exactly what the value of the action
         // is and can set fetch it. Otherwise, we ask the downstream information state to
         // return to us its value in a recursive manner.
         // We don't set the action_value with the computed value afterwards, since we are
         // not going to query this infostate's actions for values anymore, as by that
         // point we already know the complete infostate value to return upon being
         // queried.
         double child_value = action_value_opt.has_value()
                                 ? action_value_opt.value()
                                 : _compute_best_responses< Env >(*child_node_uptr);
         // the state value is updated depending on the given update rule
         state_value_updater(action_variant, action_prob.value(), child_value);
      }
   };

   double state_value = 0.;
   if(curr_node.active_player == m_br_player) {
      // if this is an infostate pertaining to the Best Response player then we need to
      // compute:
      // 1. value(I) = max_{a}(v(a | I))
      // 2. best_response(I) = argmax_{a}(v(a | I))
      // we return value(I) back up as this infostate's value to the BR player.
      std::optional< action_type > best_action = std::nullopt;
      state_value = std::numeric_limits< double >::lowest();
      child_traverser([&](const auto& action_variant, rm::Probability, double child_value) {
         if(child_value > state_value) {
            // the action variant holds the action type in the second slot
            best_action = std::get< action_type >(action_variant);
            state_value = child_value;
         }
      });
      // store this value to full the BR policy with the answer for this infostate
      m_best_response.emplace(*(curr_node.infostate), best_action.value());
   } else {
      // if this is an opponent infostate or chance node. In both cases we compute:
      //    value(I) = sum_{a}(policy(a | I) * v(a | I))
      // we return value(I) back up as this infostate's value to the BR player.
      child_traverser([&](const auto&, rm::Probability action_prob, double child_value) {
         state_value += action_prob.get() * child_value;
      });
   }
   // store this value to not trigger a recomputation upon visit from another Infostate.
   curr_node.state_value = state_value;
   return state_value;
}

}  // namespace nor

#endif  // NOR_BEST_RESPONSE_HPP
