
#ifndef NOR_FOREST_HPP
#define NOR_FOREST_HPP

#include <execution>
#include <range/v3/all.hpp>

#include "common/common.hpp"
#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "nor/policy/policy_view.hpp"
#include "nor/type_defs.hpp"
#include "nor/utils/utils.hpp"
#include "rm_utils.hpp"

namespace nor::forest {

template <
   typename RootVisitHook = common::noop<>,
   typename PreChildVisitHook = common::noop<>,
   typename ChildVisitHook = common::noop<>,
   typename PostChildVisitHook = common::noop<> >
struct TraversalHooks {
   /// typedefs for access later down the line
   using pre_child_hook_type = PreChildVisitHook;
   using child_hook_type = ChildVisitHook;
   using post_child_hook_type = PostChildVisitHook;
   using root_hook_type = RootVisitHook;
   /// the stored functors for each hook
   RootVisitHook root_hook{};
   PreChildVisitHook pre_child_hook{};
   ChildVisitHook child_hook{};
   PostChildVisitHook post_child_hook{};
};

template <
   typename Env,
   typename VisitationData,
   typename PreChildVisitHook ,
   typename ChildVisitHook ,
   typename PostChildVisitHook ,
   typename RootVisitHook>
concept walk_requirements = concepts::fosg<Env> and
   ((std::is_invocable_r_v<
      VisitationData,  // the expected return type
      ChildVisitHook,  // the actual functor
      VisitationData&,  // current node's visitation data access
      auto_action_variant_type< Env >*,  // the action that lead to this child
      auto_world_state_type< Env >*,  // current world state
      auto_world_state_type< Env >*  // child world state
   > or std::is_invocable_r_v<
      void,
      ChildVisitHook,
      VisitationData&,
      auto_action_variant_type< Env >*,
      auto_world_state_type< Env >*,
      auto_world_state_type< Env >*
   >)
   and std::invocable<
      PreChildVisitHook,  // the actual functor
      VisitationData&,  // current node's visitation data access
      auto_world_state_type< Env >*  // current world state
   >
   and std::invocable<
      PostChildVisitHook,  // the actual functor
      auto_world_state_type< Env >*  // current world state
   >
   and std::invocable<
      RootVisitHook  // the actual functor
   >
   and std::is_move_constructible_v< VisitationData >  // the visit data needs to be at least moveable
);

template < concepts::fosg Env >
class GameTreeTraverser {
  public:
   using action_type = auto_action_type< Env >;
   using chance_outcome_type = auto_chance_outcome_type< Env >;
   using observation_type = auto_observation_type< Env >;
   using world_state_type = auto_world_state_type< Env >;
   using public_state_type = auto_public_state_type< Env >;
   using info_state_type = auto_info_state_type< Env >;
   using action_variant_type = auto_action_variant_type< Env >;

   GameTreeTraverser(Env& env) : m_env(&env) {}

   /**
    * @brief Traverses all selected game actions and connects the nodes along the way.
    *
    * This function should be called whenever one wants to traverse the tree again. If one emplaces
    * all nodes in the tree, then this function should not be called anymore. The method traverses
    * the game tree and emplaces all nodes and their assoicted data storage (if desired) into the
    * game tree.
    *
    * @param traversal_strategy the action selector strategy at a world state (this selects the
    * children of a node during traversal)
    */
   template <
      typename VisitationData = utils::empty,
      typename PreChildVisitHook = common::noop< VisitationData >,
      typename ChildVisitHook = common::noop< VisitationData >,
      typename PostChildVisitHook = common::noop< VisitationData >,
      typename RootVisitHook = common::noop< VisitationData > >
   // clang-format off
   requires walk_requirements< Env, VisitationData, PreChildVisitHook, ChildVisitHook, PostChildVisitHook, RootVisitHook >
   // clang-format on
   void walk(
      uptr< world_state_type > root_state,
      VisitationData vis_data = {},
      TraversalHooks< RootVisitHook, PreChildVisitHook, ChildVisitHook, PostChildVisitHook > hooks =
         {}
   )
   {
      constexpr bool child_visit_hook_returns_void = std::is_void_v< std::invoke_result_t<
         ChildVisitHook,
         VisitationData&,
         auto_action_variant_type< Env >*,
         auto_world_state_type< Env >*,
         auto_world_state_type< Env >* > >;
      // we need to fill the root node's data (if desired) before entering the loop, since the
      // loop assumes all entered nodes to have their data node emplaced already.
      hooks.root_hook(root_state.get());

      // the tree needs to be traversed. To do so, every node (starting from the root node aka
      // the current game state) will emplace its child states - as generated from its possible
      // actions
      // - into the queue. This queue is Last-In-First-Out (LIFO), hence referred to as 'stack',
      // which will guarantee that we perform a depth-first-traversal of the game tree (FIFO
      // would lead to breadth-first). This is necessary since any state-value of a given node is
      // computed via the probability of each action multiplied by their successor state-values,
      // i.e. v(s) = \sum_a \pi(s,a) * v(s').
      // The stack uses raw pointers to nodes, since nodes are first emplaced in the tree(s) and
      // then put on the stack for later visitation. Their lifetime management is thus handled by
      // shared pointers stored in the trees.
      // the info state and public state types need to be default constructible to be filled
      // along the trajectory

      // the visitation stack. Each node in this stack will be visited once according to the
      // traversal strategy selected.
      std::stack< std::tuple< uptr< world_state_type >, VisitationData > > visit_stack;
      // emplace the root node into the visitation stack
      visit_stack.emplace(
         utils::static_unique_ptr_downcast< world_state_type >(utils::clone_any_way(root_state)),
         std::move(vis_data)
      );

      while(not visit_stack.empty()) {
         // get the top node and world state from the stack and move them into our values.
         auto [curr_wstate_uptr, visit_data] = std::move(visit_stack.top());
         // remove those elements from the stack (there is no unified pop-and-return method for
         // stack)
         visit_stack.pop();

         auto curr_wstate_raw_ptr = curr_wstate_uptr.get();
         auto curr_player = m_env->active_player(*curr_wstate_uptr.get());
         hooks.pre_child_hook(curr_wstate_raw_ptr, visit_data);

         for(const action_variant_type& action_variant : [&] {
                auto to_variant_transform = ranges::views::transform([](const auto& any) {
                   return action_variant_type(any);
                });
                if constexpr(concepts::stochastic_env< Env >) {
                   if(curr_player == Player::chance) {
                      auto actions = m_env->chance_actions(*curr_wstate_uptr.get());
                      return ranges::to< std::vector >(actions | to_variant_transform);
                   }
                }
                auto actions = m_env->actions(curr_player, *curr_wstate_uptr.get());
                return ranges::to< std::vector >(actions | to_variant_transform);
             }()) {
            // beginning of for loop body

            auto next_wstate_uptr = utils::static_unique_ptr_downcast< world_state_type >(
               utils::clone_any_way(curr_wstate_uptr)
            );
            // move the new world state forward by the current action
            std::visit(
               common::Overload{
                  [&](const auto& any_action) { m_env->transition(*next_wstate_uptr, any_action); },
                  [&](const std::monostate&) {},
               },
               action_variant
            );

            // offer the caller to extract information for the currently visited node. We are
            // passing the worldstate ptrs even if we are traversing by states in order to maintain
            // consistency in our call signature
            VisitationData new_visitation_data = std::invoke([&] {
               if constexpr(not child_visit_hook_returns_void) {
                  return hooks.child_hook(
                     visit_data, &action_variant, curr_wstate_raw_ptr, next_wstate_uptr.get()
                  );
               } else {
                  return VisitationData{};
               }
            });

            if(not m_env->is_terminal(*next_wstate_uptr)) {
               // if the newly reached world state is not a terminal state, then we merely append
               // the new child node to the queue. This way we further explore its child states
               // as reachable from the next possible actions.
               visit_stack.emplace(std::move(next_wstate_uptr), std::move(new_visitation_data));
            }
         }
         hooks.post_child_hook(curr_wstate_raw_ptr);
      }
   }

  private:
   /// pointer to the environment used to traverse the tree
   Env* m_env;
};

// template < concepts::fosg Env >
// class InfostateTree {
//   public:
//    using world_state_type = auto_world_state_type< Env >;
//    using info_state_type = auto_info_state_type< Env >;
//    using action_type = auto_action_type< Env >;
//    using chance_outcome_type = auto_chance_outcome_type< Env >;
//    using observation_type = auto_observation_type< Env >;
//
//    using chance_outcome_conditional_type = std::conditional_t<
//       std::is_same_v< chance_outcome_type, void >,
//       std::monostate,
//       chance_outcome_type >;
//    using action_variant_type = std::variant< action_type, chance_outcome_conditional_type >;
//    using action_variant_hasher = decltype([](const auto& action_variant) {
//       return std::visit(
//          []< typename VarType >(const VarType& var_element) {
//             return std::hash< VarType >{}(var_element);
//          },
//          action_variant
//       );
//    });
//
//    struct WorldstateNode {
//       /// the state value of this node.
//       std::optional< double > state_value = std::nullopt;
//       /// the child nodes reachable from this infostate the child node pointer that the action
//       leads std::unordered_map<
//          action_variant_type,
//          std::tuple< uptr< WorldstateNode >, rm::Probability >,
//          action_variant_hasher >
//          children{};
//       /// the player that takes the actions at this node. (Could be the chance player too!)
//       Player active_player;
//    };
//
//    InfostateTree(
//       Env& env,
//       uptr< world_state_type > root_state,
//       std::unordered_map< Player, info_state_type > root_infostates = {}
//    )
//        : m_env(env),
//          m_root_state(std::move(root_state)),
//          m_root_node(InfoNode{.active_player = env.active_player(*m_root_state)}),
//          m_root_infostates(std::move(root_infostates))
//    {
//       if(m_root_node.active_player != Player::chance) {
//          auto infostate_iter = m_root_infostates.find(m_root_node.active_player);
//          if(infostate_iter != m_root_infostates.end()) {
//             m_root_node.infostate = std::make_unique< info_state_type >(infostate_iter->second);
//          } else {
//             m_root_node.infostate = std::make_unique< info_state_type
//             >(m_root_node.active_player);
//          }
//       }
//       for(auto player : env.players(*m_root_state)) {
//          auto istate_iter = m_root_infostates.find(player);
//          if(istate_iter == m_root_infostates.end()) {
//             m_root_infostates.emplace(player, info_state_type{player});
//          }
//       }
//       action_emplacer(m_root_node, *m_root_state);
//    }
//
//    /**
//     *
//     * @param owning_player, the player whose infostate tree is being built
//     * @param player_policies, a map of the policies each player uses
//     */
//    void build(
//       Player owning_player,
//       std::unordered_map< Player, StatePolicyView< info_state_type, action_type > >
//       player_policies
//    );
//
//    auto& env() { return m_env; }
//    auto& env() const { return m_env; }
//    auto& root_state() { return *m_root_state; }
//    auto& root_state() const { return *m_root_state; }
//    auto& root_node() { return m_root_node; }
//    auto& root_node() const { return m_root_node; }
//
//   private:
//    Env& m_env;
//    uptr< world_state_type > m_root_state;
//    InfoNode m_root_node;
//    std::unordered_map< Player, info_state_type > m_root_infostates;
//
//    auto action_emplacer(InfoNode& infostate_node, world_state_type& state);
// };
//
// template < concepts::fosg Env >
// auto InfostateTree< Env >::action_emplacer(
//    InfostateTree::InfoNode& infostate_node,
//    InfostateTree::world_state_type& state
//)
//{
//    if(infostate_node.children.empty()) {
//       if constexpr(concepts::stochastic_fosg< Env, world_state_type, action_type >) {
//          if(auto active_player = m_env.active_player(state); active_player == Player::chance) {
//             for(auto&& outcome : m_env.chance_actions(state)) {
//                infostate_node.children.emplace(outcome, typename InfoNode::child_node_tuple{});
//             }
//
//          } else {
//             for(auto&& act : m_env.actions(active_player, state)) {
//                infostate_node.children.emplace(act, typename InfoNode::child_node_tuple{});
//             }
//          }
//       } else {
//          for(auto&& act : m_env.actions(m_env.active_player(state), state)) {
//             infostate_node.children.emplace(act, typename InfoNode::child_node_tuple{});
//          }
//       }
//    }
// }
//
// struct InfoNode {};
//
// template < concepts::fosg Env >
// void InfostateTree< Env >::build(
//    Player owning_player,
//    std::unordered_map< Player, StatePolicyView< info_state_type, action_type > > player_policies
//)
//{
//    // the tree needs to be traversed. To do so, every node (starting from the root node aka
//    // the current game state) will emplace its child states - as generated from its possible
//    // actions - into the queue. This queue is Last-In-First-Out (LIFO), ie a 'stack', which will
//    // guarantee that we perform a depth-first-traversal of the game tree (FIFO would lead to
//    // breadth-first).
//
//    struct VisitationData {
//       std::unordered_map< Player, info_state_type > infostates;
//       std::unordered_map< Player, std::vector< std::pair< observation_type, observation_type > >
//       >
//          observation_buffer;
//    };
//
//    // the visitation stack. Each node in this stack will be visited once according to the
//    // traversal strategy selected.
//    std::stack< std::tuple< uptr< world_state_type >, VisitationData, InfoNode* > > visit_stack;
//
//    // emplace the root node into the visitation stack
//    visit_stack.emplace(
//       utils::static_unique_ptr_downcast< world_state_type >(utils::clone_any_way(m_root_state)),
//       VisitationData{
//          .infostates =
//             [&] {
//                std::unordered_map< Player, info_state_type > infostates;
//                for(auto player : m_env.players(*m_root_state)) {
//                   infostates.emplace(player, m_root_infostates.at(player));
//                }
//                return infostates;
//             }(),
//          .observation_buffer = {}},
//       &m_root_node
//    );
//
//    while(not visit_stack.empty()) {
//       // get the top node and world state from the stack and move them into our values.
//       auto [curr_state, visit_data, curr_node] = std::move(visit_stack.top());
//       // remove those elements from the stack (there is no unified pop-and-return method for
//       // stack)
//       visit_stack.pop();
//
//       auto curr_player = m_env.active_player(*curr_state);
//       for(auto& [action_variant, uptr_prob_value_tuple] : curr_node->children) {
//          // emplace the child node for this action if it doesn't already exist.
//          // Another trajectory could have already emplaced it and that is fine, since every
//          // trajectory contained in an information state has the same cf. reach probability
//          auto& [next_node_uptr, action_prob, action_value] = uptr_prob_value_tuple;
//          auto [next_state, curr_action_prob] = std::visit(
//             common::Overload{
//                [&](const action_type& action) {
//                   // since we are only interested in counterfactual reach probabilities for
//                   // the pure best response of the given player, we do not account for that
//                   // player's action probability, but for each opponent, we do fetch their
//                   // policy probability
//                   return std::tuple{
//                      child_state(m_env, *curr_state, action),
//                      rm::Probability{
//                         action_prob.has_value()
//                            ? action_prob.value().get()
//                            : (curr_player == owning_player
//                                  ? 1.
//                                  : player_policies.at(curr_player)
//                                       .at(visit_data.infostates.at(curr_player))
//                                       .at(action))}};
//                },
//                [&](const chance_outcome_conditional_type& outcome) {
//                   if constexpr(concepts::deterministic_fosg< Env >) {
//                      // we shouldn't reach here if this is a deterministic fosg
//                      throw std::logic_error(
//                         "A deterministic environment traversed a chance outcome. "
//                         "This should not occur."
//                      );
//                      // this return is needed to silence the non-matching return types
//                      return std::tuple{uptr< world_state_type >{nullptr}, rm::Probability{1.}};
//                   } else {
//                      return std::tuple{
//                         child_state(m_env, *curr_state, outcome),
//                         rm::Probability{m_env.chance_probability(*curr_state, outcome)}};
//                   }
//                }},
//             action_variant
//          );
//          // we overwrite the existing action_prob here since any worldstate pertaining to the
//          // infostate is going to have the same action probability (since player's can only
//          // choose according to the knowledge they have in the information state and chance
//          // states will simply assign the same value again.)
//          SPDLOG_DEBUG("Active player: {}", curr_player);
//          LOGD2(
//             "Action",
//             std::visit([](const auto& av) { return common::to_string(av); }, action_variant)
//          );
//          action_prob = curr_action_prob;
//          SPDLOG_DEBUG("Action prob after: {}", (action_prob.has_value() ?
//          action_prob.value().get() : 404.));
//
//          if(m_env.is_terminal(*next_state)) {
//             // if the child is a terminal state then we take the payoff of this history and add
//             // that to the overall payoff of the infostate (node)
//             SPDLOG_DEBUG("Terminal action value before update: {}", action_value.value_or(404));
//             action_value = action_value.value_or(0.) + m_env.reward(owning_player, *next_state);
//             SPDLOG_DEBUG("Terminal action value after update: {}", action_value.value_or(404));
//
//          } else {
//             // since it isn't a terminal state we emplace the child state to visit further
//             auto [child_observation_buffer, child_infostate_map] = std::visit(
//                common::Overload{
//                   [&](std::monostate) {
//                      // this will never be visited anyway, but if so --> error
//                      throw std::logic_error("We entered a std::monostate visit branch.");
//                      return std::tuple{visit_data.observation_buffer, visit_data.infostates};
//                   },
//                   [&](const auto& action_or_outcome) {
//                      auto [child_obs_buffer, child_istate_map] = next_infostate_and_obs_buffers(
//                         m_env,
//                         visit_data.observation_buffer,
//                         visit_data.infostates,
//                         *curr_state,
//                         action_or_outcome,
//                         *next_state
//                      );
//                      return std::tuple{child_obs_buffer, child_istate_map};
//                   }},
//                action_variant
//             );
//
//             Player next_active_player = m_env.active_player(*next_state);
//
//             if(not next_node_uptr) {
//                // create the child node unique ptr. The parent takes ownership of the child node.
//                next_node_uptr = std::make_unique< InfoNode >(InfoNode{
//                   .active_player = next_active_player,
//                   .parent = curr_node,
//                   .infostate = next_active_player != Player::chance
//                                   ? std::make_unique< info_state_type >(
//                                      child_infostate_map.at(next_active_player)
//                                   )
//                                   : nullptr});
//
//                // if the actions/outcomes have not yet been emplaced into this infostate node
//                // then we do this here. (They could have been already emplaced by another
//                // trajectory passing through this infostate before and emplacing them there)
//                action_emplacer(*next_node_uptr, *next_state);
//             }
//
//             visit_stack.emplace(
//                std::move(next_state),
//                VisitationData{
//                   .infostates = std::move(child_infostate_map),
//                   .observation_buffer = std::move(child_observation_buffer)},
//                next_node_uptr.get()
//             );
//          }
//       }
//    }
// };

// template < concepts::action Action, concepts::info_state Infostate, typename ChanceOutcome >
// struct PublicTreeNode {
//    static_assert(
//       std::conditional_t<
//          std::is_same_v< ChanceOutcome, void >,
//          std::true_type,
//          std::conditional_t<
//             concepts::chance_outcome< ChanceOutcome >,
//             std::true_type,
//             std::false_type > >::value,
//       "The passed chance outcome type either has to be void or fulfill the concept: "
//       "'chance_outcome'."
//    );
//
//    using action_type = Action;
//    using info_state_type = Infostate;
//    using chance_outcome_type = ChanceOutcome;
//
//    /// this node's id (or index within the node vector)
//    const size_t id;
//    /// the parent node from which this node stems
//    PublicTreeNode* parent = nullptr;
//    /// the children that each action maps to in the game tree.
//    /// Should be filled during the traversal.
//    // If the environment is deterministic, then ChanceOutcome should be void, and thus the map
//    only
//    // stores the action type itself. If the environment is stochastic however, then either
//    actions
//    // or chance outcomes can be stored.
//    using chance_outcome_conditional_type = std::
//       conditional_t< std::is_same_v< ChanceOutcome, void >, std::monostate, ChanceOutcome >;
//    using action_variant_type = std::variant< Action, chance_outcome_conditional_type >;
//    using variant_hasher = decltype([](const action_variant_type& action_variant) {
//       return std::visit(
//          []< typename VarType >(const VarType& variant_elem) {
//             return std::hash< VarType >{}(variant_elem);
//          },
//          action_variant
//       );
//    });
//    std::unordered_map< action_variant_type, PublicTreeNode*, variant_hasher > children{};
//    /// the action that was taken at the parent to get to this node (nullopt for the root)
//    std::optional< action_variant_type > action_from_parent = std::nullopt;
//    /// all the infostates that are associated with this public node
//    std::vector< sptr< Infostate > > contained_infostates;
// };

}  // namespace nor::forest
//
//
// namespace nor::rm::forest {
//
// template < typename Action >
// struct GameTreeNode {
//   /// the children that each action maps to in the game tree.
//   /// Should be filled during the traversal.
//   std::unordered_map< Action, GameTreeNode* > children{};
//};
//
// template < concepts::fosg Env, concepts::game_tree_node Node >
// class GameTree {
//  public:
//   using world_state_type = auto_world_state_type< Env >;
//   using action_type = auto_action_type< Env >;
//   using node_type = Node;
//
//   /**
//    * @brief Does the first full traversal of the game tree and emplaces the nodes along the way.
//    *
//    * This function should be called exactly once on iteration 0 and never thereafter. The method
//    * traverses the game tree as usual, but then also emplaces all node shared pointers into the
//    * individual game trees. In order to not let this traversal go to waste the update stack for
//    * regret and policy updating is also filled during this traversal.
//    * I decided to convert iteration 0 into its own function (instead of querying at runtime
//    * whether nodes already exist) to avoid multiple runtime overhead of searching for nodes in a
//    * hash table. Whether the slight code duplication will become a maintenance issue will need
//    * to be seen in the future.
//    */
//   GameTree(Env& env, uptr< const world_state_type >&& root_state)
//       : m_root_state(std::move(root_state)), m_nodes{}
//   {
//      auto fill_children = [&](node_type& node, world_state_type& wstate) {
//         for(const auto& action : env.actions(env.active_player(wstate), wstate)) {
//            node.children.emplace(action, nullptr);
//         }
//      };
//      auto index_pool = size_t(0);
//      // emplace the root node
//      auto& root = m_nodes.emplace_back(std::make_unique<node_type>());
//      fill_children(root, *root_state);
//      // the tree needs to be traversed. To do so, every node (starting from the root node aka
//      // the current game state) will emplace its child states - as generated from its possible
//      // actions
//      // - into the queue. This queue is Last-In-First-Out (LIFO), hence referred to as 'stack',
//      // which will guarantee that we perform a depth-first-traversal of the game tree (FIFO
//      // would lead to breadth-first). This is necessary since any state-value of a given node is
//      // computed via the probability of each action multiplied by their successor state-values,
//      // i.e. v(s) = \sum_a \pi(s,a) * v(s').
//      // The stack uses raw pointers to nodes, since nodes are first emplaced in the tree(s) and
//      // then put on the stack for later visitation. Their lifetime management is thus handled by
//      // shared pointers stored in the trees.
//      std::stack< std::tuple< uptr< world_state_type >, node_type* > > visit_stack;
//      // copy the root state into the visitation stack
//      visit_stack.emplace(std::move(root_state), root_node());
//
//      while(not visit_stack.empty()) {
//         // get the top node and world state from the stack and move them into our values.
//         auto [curr_wstate, curr_node] = std::move(visit_stack.top());
//         // remove those elements from the stack (there is no unified pop-and-return method for
//         // stack)
//         visit_stack.pop();
//
//         for(const auto& action : ranges::views::keys(curr_node->children)) {
//            // copy the current nodes world state
//            auto next_wstate_uptr = utils::static_unique_ptr_downcast< world_state_type >(
//               utils::clone_any_way(curr_wstate));
//            // move the new world state forward by the current action
//            env.transition(*next_wstate_uptr, action);
//            // the child node has shared ownership by each player's game tree
//            auto& child_node = m_nodes.emplace_back(
//               {.id = index_pool++, .parent = curr_node, .children = {}});
//            fill_children(child_node, *next_wstate_uptr);
//
//            // append the new tree node as a child of the currently visited node. We are using a
//            // raw pointer here to avoid the unnecessary sptr counter increase cost
//            curr_node->children(action) = &child_node;
//
//            if(not env.is_terminal(*next_wstate_uptr)) {
//               // if the newly reached world state is not a terminal state, then we merely append
//               // the new child node to the queue. This way we further explore its child states
//               // as reachable from the next possible actions.
//               visit_stack.emplace(std::move(next_wstate_uptr), &child_node);
//            }
//         }
//      }
//   }
//
//   [[nodiscard]] auto* root_node() const { return &(m_nodes[0]); }
//   [[nodiscard]] auto* root_state() const { return m_root_state.get(); }
//
//  private:
//   std::vector< uptr<node_type> > m_nodes;
//   sptr< world_state_type > m_root_state;
//};

#endif  // NOR_FOREST_HPP
