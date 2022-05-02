
#ifndef NOR_FOREST_HPP
#define NOR_FOREST_HPP

#include <execution>
#include <range/v3/all.hpp>

#include "common/common.hpp"
#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "nor/type_defs.hpp"
#include "nor/utils/utils.hpp"

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

class GameTree {
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

   GameTree(Env& env, uptr< world_state_type >&& root_state)
       : m_env(&env), m_root_state(std::move(root_state))
   {
   }

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
    * @param action_emplacer the action selector strategy to emplace into a node at any world state
    * (this selects the number of children to store at a time)
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

         hooks.pre_child_hook(curr_wstate_uptr.get(), vis_data);

         for(const auto& action : traversal_strategy(curr_wstate_uptr.get())) {
            uptr< world_state_type > next_wstate_uptr = nullptr;

            if(single_trajectory) {
               // we cannot check the semantic correctness of the traversal strategy providing
               // only a single action to iterate over a single trajectory
               next_wstate_uptr = std::move(curr_wstate_uptr);
               curr_wstate_uptr = nullptr;
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
               vis_data, &action, curr_wstate_uptr.get(), next_wstate_uptr.get());

            if(not m_env->is_terminal(*next_wstate_uptr)) {
               // if the newly reached world state is not a terminal state, then we merely append
               // the new child node to the queue. This way we further explore its child states
               // as reachable from the next possible actions.
               visit_stack.emplace(std::move(next_wstate_uptr));
            }
         }
         hooks.post_child_hook(curr_wstate_uptr.get());
      }
   }
};
}  // namespace nor::rm::forest

#endif  // NOR_FOREST_HPP
