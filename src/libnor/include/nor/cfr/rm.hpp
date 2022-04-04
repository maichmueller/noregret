
#ifndef NOR_RM_HPP
#define NOR_RM_HPP

#include <execution>

#include "common/common.hpp"
#include "node.hpp"
#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "nor/type_defs.hpp"
#include "nor/utils/utils.hpp"

namespace nor::rm {

template < concepts::action Action, concepts::action_policy< Action > Policy >
void regret_matching(Policy& policy_map, const std::unordered_map< Action, double >& cumul_regret)
{
   // sum up the positivized regrets and store them in a new vector
   std::unordered_map< Action, double > pos_regrets;
   double pos_regret_sum{0.};
   for(const auto& [action, regret] : cumul_regret) {
      double pos_regret = std::max(0., regret);
      pos_regrets.emplace(action, pos_regret);
      pos_regret_sum += pos_regret;
   }
   // apply the new policy to the vector policy
   auto exec_policy{std::execution::par_unseq};
   if(pos_regret_sum > 0) {
      if(cumul_regret.size() != policy_map.size()) {
         throw std::invalid_argument(
            "Passed regrets and policy maps do not have the same number of elements");
      }
      std::for_each(exec_policy, policy_map.begin(), policy_map.end(), [&](auto& entry) {
         return std::get< 1 >(entry) = cumul_regret.at(std::get< 0 >(entry)) / pos_regret_sum;
      });
   } else {
      std::for_each(exec_policy, policy_map.begin(), policy_map.end(), [&](auto& entry) {
         return std::get< 1 >(entry) = 1. / static_cast< double >(policy_map.size());
      });
   }
}

namespace forest {

enum class NodeCategory {
   chance = 0,
   choice,
   terminal,
};

template < concepts::action Action, typename ChanceOutcome >
struct Node {
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
   /// this node's id (or index within the node vector)
   const size_t id;
   /// what type of node this is
   const NodeCategory category;
   /// the parent node from which this node stems
   Node* parent = nullptr;
   /// the children that each action maps to in the game tree.
   /// Should be filled during the traversal.
   // If the environment is deterministic, then ChanceOutcome should be void, and thus the map only
   // store the action type itself. If the environment is stochastic however, then either actions or
   // chance outcomes can be stored.
   static constexpr size_t action_type_access_index = 0;
   static constexpr size_t chance_outcome_type_access_index = 1;
   using chance_outcome_conditional_type = std::
      conditional_t< std::is_same_v< ChanceOutcome, void >, std::monostate, ChanceOutcome >;
   using action_variant_type = std::variant< Action, chance_outcome_conditional_type >;
   using variant_hasher = decltype([](const auto& action_variant) {
      return std::visit(
         []< typename VarType >(const VarType& var_element) {
            return std::hash< VarType >{}(var_element);
         },
         action_variant);
   });
   std::unordered_map< action_variant_type, Node*, variant_hasher > children{};
   /// the action that was taken at the parent to get to this node
   std::optional< action_variant_type > action_from_parent = std::nullopt;
};

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
   pre_child_hook_type pre_child_hook{};
   child_hook_type child_hook{};
   post_child_hook_type post_child_hook{};
   root_hook_type root_hook{};
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

   using node_type = Node< action_type, chance_outcome_type >;

   GameTree(Env& env, uptr< world_state_type >&& root_state)
       : m_env(&env), m_nodes{}, m_root_state(std::move(root_state))
   {
      // emplace the root node
      m_nodes.emplace(0, node_type{.id = 0, .category = _categorize(*m_root_state)});
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
      typename PreChildVisitHook = common::noop,
      typename ChildVisitHook = common::noop,
      typename PostChildVisitHook = common::noop,
      typename RootVisitHook = common::noop >
   // clang-format off
   requires
      std::invocable<
         ChildVisitHook,  // the actual functor
         const node_type&,  // child node input
         const typename node_type::action_variant_type*,  // action that led to the child
         world_state_type*,  // current world state
         world_state_type*  // child world state
      >
      and std::invocable<
         PreChildVisitHook,  // the actual functor
         const node_type&,  // child node input
         world_state_type*  // current world state
      >
      and std::invocable<
         PostChildVisitHook,  // the actual functor
         const node_type&,  // child node input
         world_state_type*  // current world state
      >
   // clang-format on
   inline void traverse(
      TraversalStrategy traversal_strategy = &traverse_all_actions,
      TraversalHooks< PreChildVisitHook, ChildVisitHook, PostChildVisitHook, RootVisitHook >&&
         hooks = {},
      bool traverse_via_worldstate = true,
      bool single_trajectory = false)
   {
      if(traverse_via_worldstate) {
         _traverse< true >(std::move(traversal_strategy), std::move(hooks), single_trajectory);
      } else {
         _traverse< false >(std::move(traversal_strategy), std::move(hooks), single_trajectory);
      }
   }

   auto traverse_all_actions(node_type& node, world_state_type* wstate)
   {
      if(wstate != nullptr) {
         auto action_emplacer = [&](auto&& action_container) {
            for(auto& action : action_container) {
               node.children.emplace(std::move(action), nullptr);
            }
         };
         if constexpr(concepts::deterministic_fosg< Env >) {
            // if we have a deterministic environment then we don't need to enfore the existence of
            // a chance action member function
            action_emplacer(m_env->actions(m_env->active_player(*wstate), *wstate));
         } else {
            if(node.category == NodeCategory::chance) {
               action_emplacer(m_env->chance_actions(*wstate));
            } else {
               action_emplacer(m_env->actions(m_env->active_player(*wstate), *wstate));
            }
         }
      } else {
         if(node.children.empty()) {
            throw std::logic_error(
               "No world state provided and no actions have been previously emplaced at this "
               "node. Cannot traverse over this node.");
         }
      }
      return ranges::views::keys(node.children);
   }
   /**
    * @brief walk up the tree until a non-chance node parent is found or the root is hit
    * @param node the node for which a non chance parent is sought
    */
   node_type* nonchance_parent(const node_type* node)
   {
      if(node->parent == nullptr) {
         // we have reached the root
         return nullptr;
      }
      if(node->parent->category == NodeCategory::chance) {
         return nonchance_parent(node->parent);
      }
      return node->parent;
   }
   /**
    * @brief delets all nodes except the root node. Resets the root node's child links.
    */
   void reset()
   {
      auto start = m_nodes.begin();
      std::advance(start, 1);
      m_nodes.erase(start, m_nodes.end());
      for(auto& [action, child_node] : m_nodes[0].children) {
         child_node = nullptr;
      }
      m_index_counter = 1;
   }

   [[nodiscard]] auto size() const { return m_nodes.size(); }
   [[nodiscard]] auto& root_state() const { return *m_root_state; }
   [[nodiscard]] auto& root_node() const { return m_nodes.at(0); }

  private:
   /// pointer to the environment used to traverse the tree
   Env* m_env;
   /// the node storage of all nodes in the tree. Entry 0 is the root.
   std::unordered_map< size_t, node_type > m_nodes;
   /// the root game state from which the tree is built
   const uptr< world_state_type > m_root_state;
   /// the index counter to assign to nodes as ids
   size_t m_index_counter{1};

   inline auto& root_node() { return m_nodes.at(0); }

   // current deciding mechanism for which category the current world state may belong to.
   auto _categorize(world_state_type& wstate)
   {
      if(m_env->active_player(wstate) == Player::chance) {
         return NodeCategory::chance;
      }
      if(m_env->is_terminal(wstate)) {
         return NodeCategory::terminal;
      }
      return NodeCategory::choice;
   };

   template < bool traverse_via_worldstate, typename TraversalStrategy, typename... HookFunctors >
   void _traverse(
      TraversalStrategy traversal_strategy,
      TraversalHooks< HookFunctors... >&& hooks,
      bool single_trajectory)
   {
//      static_assert(
//         // clang-format off
//         std::invocable< TraversalStrategy, node_type& >
//         and ranges::range< std::invoke_result_t< TraversalStrategy, node_type&, world_state_type* > >
//         and std::is_same_v<
//            typename node_type::action_variant_type,
//            decltype(*(
//               std::declval< TraversalStrategy >()(
//                  std::declval< node_type >(),
//                  std::declval< world_state_type* >()
//               )
//            ))
//         >,
//         // clang-format on
//         ""
//         "Traversal strategy needs to provide a range over action variants.");

      // we need to fill the root node's data (if desired) before entering the loop, since the
      // loop assumes all entered nodes to have their data node emplaced already.
      hooks.root_hook(root_node(), m_root_state.get());

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
      std::stack< std::tuple< node_type*, uptr< world_state_type > > > visit_stack;
      // emplace the root node into the visitation stack
      visit_stack.emplace(&root_node(), [&] {
         if constexpr(traverse_via_worldstate)
            return utils::static_unique_ptr_downcast< world_state_type >(
               utils::clone_any_way(m_root_state));
         else
            return nullptr;
      }());

      while(not visit_stack.empty()) {
         // get the top node and world state from the stack and move them into our values.
         auto [curr_node, curr_wstate_uptr] = std::move(visit_stack.top());
         // remove those elements from the stack (there is no unified pop-and-return method for
         // stack)
         visit_stack.pop();

         hooks.pre_child_hook(*curr_node, curr_wstate_uptr.get());

         for(const auto& action : traversal_strategy(*curr_node, curr_wstate_uptr.get())) {
            // get the child node pointer related to this action. This could potentially be a
            // nullptr still
            auto* child_node = curr_node->children[action];
            uptr< world_state_type > next_wstate_uptr = nullptr;

            if constexpr(traverse_via_worldstate) {
               // if we are traversing via worldstates, instead of pre-emplaced nodes, then we need
               // to copy and move the new state forward by the chosen action
               // copy the current node's world state
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
               if(child_node == nullptr) {
                  auto child_id = m_index_counter++;
                  child_node = &(m_nodes.emplace(
                     child_id,
                     node_type{
                        .id = child_id,
                        .category = _categorize(*next_wstate_uptr),
                        .parent = curr_node,
                        .children = {},
                        .action_from_parent = action}).first->second);
               }
               // append the new tree node as a child of the currently visited node. We are using a
               // raw pointer here since lifetime management is maintained by the game tree itself
               curr_node->children[action] = child_node;
            }
            // offer the caller to extract information for the currently visited node. We are
            // passing the worldstate ptrs even if we are traversing by states in order to maintain
            // consistency in our call signature

            hooks.child_hook(*child_node, &action, curr_wstate_uptr.get(), next_wstate_uptr.get());

            if(child_node->category != forest::NodeCategory::terminal) {
               // if the newly reached world state is not a terminal state, then we merely append
               // the new child node to the queue. This way we further explore its child states
               // as reachable from the next possible actions.
               visit_stack.emplace(child_node, std::move(next_wstate_uptr));
            }
         }
         hooks.post_child_hook(*curr_node, curr_wstate_uptr.get());
      }
   }
};
}  // namespace forest

}  // namespace nor::rm

namespace nor::utils {
constexpr CEBijection< nor::rm::forest::NodeCategory, std::string_view, 27 > nodecateogry_name_bij =
   {std::pair{nor::rm::forest::NodeCategory::chance, "chance"},
    std::pair{nor::rm::forest::NodeCategory::choice, "choice"},
    std::pair{nor::rm::forest::NodeCategory::terminal, "terminal"}};

}  // namespace nor::utils

namespace common {
template <>
inline std::string_view enum_name(nor::rm::forest::NodeCategory e)
{
   return nor::utils::nodecateogry_name_bij.at(e);
}
}  // namespace common

inline auto& operator<<(std::ostream& os, nor::rm::forest::NodeCategory e)
{
   os << common::enum_name(e);
   return os;
}

#endif  // NOR_RM_HPP
