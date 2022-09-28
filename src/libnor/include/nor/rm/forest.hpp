
#ifndef NOR_FOREST_HPP
#define NOR_FOREST_HPP

#include <execution>
#include <range/v3/all.hpp>

#include "common/common.hpp"
#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "nor/type_defs.hpp"
#include "nor/utils/utils.hpp"

namespace nor::concepts {

template < typename Node >
concept game_tree_node = requires(Node n) {
                            // the node itself should be default constructible
                            std::is_default_constructible_v< Node >;
                            // the node needs to have a public member children
                            n.children;
                            // the children member needs to be a map
                            map< std::remove_cvref_t< decltype(n.children) > >;
                            // the mapped type of this map needs to be dereferencable (pointer-like)
                            is::dereferencable<
                               std::remove_cvref_t< typename decltype(n.children)::mapped_type > >;
                            // the object pointed to by the mapped values in the children map needs
                            // to be another Node
                            std::is_same_v<
                               decltype(*(
                                  std::declval< typename decltype(n.children)::mapped_type >())),
                               Node >;
                         };
}

namespace nor::rm::forest {

template <
   typename PreChildVisitHook = common::noop,
   typename ChildVisitHook = common::noop,
   typename PostChildVisitHook = common::noop,
   typename RootVisitHook = common::noop >
struct TraversalHooks {
   /// typedefs for access later down the line
   using pre_child_hook_type = PreChildVisitHook;
   using child_hook_type = ChildVisitHook;
   using post_child_hook_type = PostChildVisitHook;
   using root_hook_type = RootVisitHook;
   /// the stored functors for each hook
   root_hook_type root_hook{};
   pre_child_hook_type pre_child_hook{};
   child_hook_type child_hook{};
   post_child_hook_type post_child_hook{};
};

template < concepts::fosg Env >
class GameTreeTraverser {
  public:
   using action_type = typename fosg_auto_traits< Env >::action_type;
   using chance_outcome_type = typename fosg_auto_traits< Env >::chance_outcome_type;
   using observation_type = typename fosg_auto_traits< Env >::observation_type;
   using world_state_type = typename fosg_auto_traits< Env >::world_state_type;
   using public_state_type = typename fosg_auto_traits< Env >::public_state_type;
   using info_state_type = typename fosg_auto_traits< Env >::info_state_type;

   using chance_outcome_conditional_type = std::
      conditional_t< concepts::deterministic_fosg< Env >, std::monostate, chance_outcome_type >;
   using action_variant_type = std::variant< chance_outcome_conditional_type, action_type >;

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
      typename TraversalStrategy,
      typename VisitationData = utils::empty,
      typename PreChildVisitHook = common::noop,
      typename ChildVisitHook = common::noop,
      typename PostChildVisitHook = common::noop,
      typename RootVisitHook = common::noop >
   // clang-format off
   requires
      std::is_invocable_r_v<
         VisitationData,  // the expected return type
         ChildVisitHook,  // the actual functor
         VisitationData&,  // current node's visitation data access
         world_state_type*,  // current world state
         world_state_type*  // child world state
      >
      and std::invocable<
         PreChildVisitHook,  // the actual functor
         VisitationData&,  // current node's visitation data access
         world_state_type*  // current world state
      >
      and std::invocable<
         PostChildVisitHook,  // the actual functor
         VisitationData&,  // current node's visitation data access
         world_state_type*  // current world state
      >
      and std::is_move_constructible_v< VisitationData >
   // clang-format on
   inline void walk(
      uptr< world_state_type > root_state,
      TraversalStrategy traversal_strategy = &traverse_all_actions,
      VisitationData vis_data = {},
      TraversalHooks< PreChildVisitHook, ChildVisitHook, PostChildVisitHook, RootVisitHook >&&
         hooks = {})
   {
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
      std::stack< std::tuple< uptr< world_state_type > >, VisitationData > visit_stack;
      // emplace the root node into the visitation stack
      visit_stack.emplace(
         utils::static_unique_ptr_downcast< world_state_type >(utils::clone_any_way(root_state)),
         std::move(vis_data));

      while(not visit_stack.empty()) {
         // get the top node and world state from the stack and move them into our values.
         auto [curr_wstate_uptr, visit_data] = std::move(visit_stack.top());
         // remove those elements from the stack (there is no unified pop-and-return method for
         // stack)
         visit_stack.pop();

         auto curr_wstate_raw_ptr = curr_wstate_uptr.get();

         hooks.pre_child_hook(curr_wstate_raw_ptr, visit_data);

         for(const auto& action : traversal_strategy(curr_wstate_uptr.get())) {
            auto next_wstate_uptr = utils::static_unique_ptr_downcast< world_state_type >(
               utils::clone_any_way(curr_wstate_uptr));
            // move the new world state forward by the current action
            if constexpr(concepts::deterministic_fosg< Env >) {
               m_env->transition(*next_wstate_uptr, std::get< 1 >(action));
            } else {
               std::visit(
                  [&](const auto& any_action) { m_env->transition(*next_wstate_uptr, any_action); },
                  action);
            }

            // offer the caller to extract information for the currently visited node. We are
            // passing the worldstate ptrs even if we are traversing by states in order to maintain
            // consistency in our call signature

            auto new_visitation_data = hooks.child_hook(
               visit_data, &action, curr_wstate_raw_ptr, next_wstate_uptr.get());

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

   auto traverse_all_actions(world_state_type* wstate)
   {
      if constexpr(concepts::deterministic_fosg< Env >) {
         // if we have a deterministic environment then we don't need to enfore the existence of
         // a chance action member function
         return m_env->actions(m_env->active_player(*wstate), *wstate);
      } else {
         if(m_env->active_player == Player::chance) {
            return m_env->chance_actions(*wstate);
         }
         return m_env->actions(m_env->active_player(*wstate), *wstate);
      }
   }

  private:
   /// pointer to the environment used to traverse the tree
   Env* m_env;
};

template < typename Action >
struct GameTreeNode {
   /// the children that each action maps to in the game tree.
   /// Should be filled during the traversal.
   std::unordered_map< Action, GameTreeNode* > children{};
};

template <
   concepts::fosg Env,
   concepts::game_tree_node Node = GameTreeNode< typename fosg_auto_traits< Env >::action_type > >
class GameTree {
  public:
   using action_type = typename fosg_auto_traits< Env >::action_type;
   using chance_outcome_type = typename fosg_auto_traits< Env >::chance_outcome_type;
   using observation_type = typename fosg_auto_traits< Env >::observation_type;
   using world_state_type = typename fosg_auto_traits< Env >::world_state_type;
   using public_state_type = typename fosg_auto_traits< Env >::public_state_type;
   using info_state_type = typename fosg_auto_traits< Env >::info_state_type;

   using node_type = Node;

   using chance_outcome_conditional_type = std::
      conditional_t< concepts::deterministic_fosg< Env >, std::monostate, chance_outcome_type >;
   using action_variant_type = std::variant< chance_outcome_conditional_type, action_type >;

   GameTree(Env& env) : m_env(&env), m_tree_traverser(GameTreeTraverser(env)) {}

   template <
      typename NodeBuilder,
      typename VisitationData = utils::empty,
      typename PreChildVisitHook = common::noop,
      typename ChildVisitHook = common::noop,
      typename PostChildVisitHook = common::noop,
      typename RootVisitHook = common::noop >
      requires std::is_invocable_r_v<
         NodeBuilder,
         node_type,
         VisitationData,
         action_variant_type*,
         world_state_type*,
         world_state_type* >
   void build(NodeBuilder builder = [](auto&&...) { return std::make_unique< node_type >(); }){

   }

  private:
   /// pointer to the environment used to traverse the tree
   Env* m_env;
   /// the traverser instance to walk the game tree with
   GameTreeTraverser< Env > m_tree_traverser;
   /// the game tree nodes
   std::vector< uptr< node_type > > m_nodes;
};

template < concepts::action Action, concepts::info_state Infostate, typename ChanceOutcome >
struct PublicTreeNode {
   static_assert(
      std::conditional_t<
         std::is_same_v< ChanceOutcome, void >,
         std::true_type,
         std::conditional_t<
            concepts::chance_outcome< ChanceOutcome >,
            std::true_type,
            std::false_type > >::value,
      "The passed chance outcome type either has to be void or fulfill the concept: "
      "'chance_outcome'.");

   using action_type = Action;
   using info_state_type = Infostate;
   using chance_outcome_type = ChanceOutcome;

   /// this node's id (or index within the node vector)
   const size_t id;
   /// the parent node from which this node stems
   PublicTreeNode* parent = nullptr;
   /// the children that each action maps to in the game tree.
   /// Should be filled during the traversal.
   // If the environment is deterministic, then ChanceOutcome should be void, and thus the map only
   // stores the action type itself. If the environment is stochastic however, then either actions
   // or chance outcomes can be stored.
   using chance_outcome_conditional_type = std::
      conditional_t< std::is_same_v< ChanceOutcome, void >, std::monostate, ChanceOutcome >;
   using action_variant_type = std::variant< Action, chance_outcome_conditional_type >;
   using variant_hasher = decltype([](const action_variant_type& action_variant) {
      return std::visit(
         []< typename VarType >(const VarType& variant_elem) {
            return std::hash< VarType >{}(variant_elem);
         },
         action_variant);
   });
   std::unordered_map< action_variant_type, PublicTreeNode*, variant_hasher > children{};
   /// the action that was taken at the parent to get to this node (nullopt for the root)
   std::optional< action_variant_type > action_from_parent = std::nullopt;
   /// all the infostates that are associated with this public node
   std::vector< sptr< Infostate > > contained_infostates;
};

template < concepts::fosg Env >
class PublicTree {
  public:
   using action_type = typename fosg_auto_traits< Env >::action_type;
   using chance_outcome_type = typename fosg_auto_traits< Env >::chance_outcome_type;
   using observation_type = typename fosg_auto_traits< Env >::observation_type;
   using world_state_type = typename fosg_auto_traits< Env >::world_state_type;
   using public_state_type = typename fosg_auto_traits< Env >::public_state_type;
   using info_state_type = typename fosg_auto_traits< Env >::info_state_type;

   using chance_outcome_conditional_type = std::conditional_t<
      std::is_same_v< chance_outcome_type, void >,
      std::monostate,
      chance_outcome_type >;
   using action_variant_type = std::variant< action_type, chance_outcome_conditional_type >;
   using variant_hasher = decltype([](const auto& action_variant) {
      return std::visit(
         []< typename VarType >(const VarType& var_element) {
            return std::hash< VarType >{}(var_element);
         },
         action_variant);
   });

   PublicTree(Env& env, uptr< world_state_type >&& root_state)
       : m_env(&env), m_root_state(std::move(root_state))
   {
   }

   /**
    * @brief Traverses all selected game actions and connects the public nodes along the way.
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
      typename TraversalStrategy,
      typename VisitationData = utils::empty,
      typename PreChildVisitHook = common::noop,
      typename ChildVisitHook = common::noop,
      typename PostChildVisitHook = common::noop,
      typename RootVisitHook = common::noop >
   // clang-format off
   requires
      std::is_invocable_r_v<
         VisitationData,  // the expected return type
         ChildVisitHook,  // the actual functor
         VisitationData&,  // visitation data access
         world_state_type*,  // current world state
         world_state_type*  // child world state
      >
      and std::invocable<
         PreChildVisitHook,  // the actual functor
         VisitationData&,  // visitation data access
         world_state_type*  // current world state
      >
      and std::invocable<
         PostChildVisitHook,  // the actual functor
         VisitationData&,  // visitation data access
         world_state_type*  // current world state
      >
      and std::is_move_constructible_v< VisitationData >
   // clang-format on
   inline void traverse(
      TraversalStrategy traversal_strategy = &traverse_all_actions,
      VisitationData vis_data = {},
      TraversalHooks< PreChildVisitHook, ChildVisitHook, PostChildVisitHook, RootVisitHook >&&
         hooks = {},
      bool single_trajectory = false)
   {
      _traverse(
         std::move(traversal_strategy), std::move(vis_data), std::move(hooks), single_trajectory);
   }

   auto traverse_all_actions(world_state_type* wstate)
   {
      if constexpr(concepts::deterministic_fosg< Env >) {
         // if we have a deterministic environment then we don't need to enfore the existence of
         // a chance action member function
         return m_env->actions(m_env->active_player(*wstate), *wstate);
      } else {
         if(m_env->active_player == Player::chance) {
            return m_env->chance_actions(*wstate);
         }
         return m_env->actions(m_env->active_player(*wstate), *wstate);
      }
   }

   [[nodiscard]] auto& root_state() const { return *m_root_state; }

  private:
   /// pointer to the environment used to traverse the tree
   Env* m_env;
   /// the root game state from which the tree is built
   const uptr< world_state_type > m_root_state;

   template < typename TraversalStrategy, typename VisitationData, typename... HookFunctors >
   void _traverse(
      TraversalStrategy traversal_strategy,
      VisitationData&& initial_vis_data,
      TraversalHooks< HookFunctors... >&& hooks,
      bool single_trajectory)
   {
      // we need to fill the root node's data (if desired) before entering the loop, since the
      // loop assumes all entered nodes to have their data node emplaced already.
      hooks.root_hook(m_root_state.get());

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
      std::stack< std::tuple< uptr< world_state_type > >, VisitationData > visit_stack;
      // emplace the root node into the visitation stack
      visit_stack.emplace(
         utils::static_unique_ptr_downcast< world_state_type >(utils::clone_any_way(m_root_state)),
         std::move(initial_vis_data));

      while(not visit_stack.empty()) {
         // get the top node and world state from the stack and move them into our values.
         auto [curr_wstate_uptr, vis_data] = std::move(visit_stack.top());
         // remove those elements from the stack (there is no unified pop-and-return method for
         // stack)
         visit_stack.pop();

         auto curr_wstate_raw_ptr = curr_wstate_uptr.get();

         hooks.pre_child_hook(curr_wstate_raw_ptr, vis_data);

         for(const auto& action : traversal_strategy(curr_wstate_uptr.get())) {
            uptr< world_state_type > next_wstate_uptr = nullptr;

            if(single_trajectory) {
               // we cannot check the semantic correctness of the traversal strategy providing
               // only a single action to iterate over a single trajectory
               next_wstate_uptr = std::move(curr_wstate_uptr);
            } else {
               next_wstate_uptr = utils::static_unique_ptr_downcast< world_state_type >(
                  utils::clone_any_way(curr_wstate_uptr));
            }
            // move the new world state forward by the current action
            if constexpr(concepts::deterministic_fosg< Env >) {
               m_env->transition(*next_wstate_uptr, std::get< action_type >(action));
            } else {
               std::visit(
                  common::Overload{[&](const auto& any_action) {
                     m_env->transition(*next_wstate_uptr, any_action);
                  }},
                  action);
            }

            // offer the caller to extract information for the currently visited node. We are
            // passing the worldstate ptrs even if we are traversing by states in order to maintain
            // consistency in our call signature

            auto new_visitation_data = hooks.child_hook(
               vis_data, &action, curr_wstate_raw_ptr, next_wstate_uptr.get());

            if(not m_env->is_terminal(*next_wstate_uptr)) {
               // if the newly reached world state is not a terminal state, then we merely append
               // the new child node to the queue. This way we further explore its child states
               // as reachable from the next possible actions.
               visit_stack.emplace(std::move(next_wstate_uptr));
            }
         }
         hooks.post_child_hook(curr_wstate_raw_ptr);
      }
   }
};

}  // namespace nor::rm::forest
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
//   using world_state_type = typename fosg_auto_traits< Env >::world_state_type;
//   using action_type = typename fosg_auto_traits< Env >::action_type;
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
