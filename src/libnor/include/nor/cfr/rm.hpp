
#ifndef NOR_RM_HPP
#define NOR_RM_HPP

#include <execution>

#include "node.hpp"
#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "nor/type_defs.hpp"
#include "nor/utils/logging_macros.h"
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

template < typename Action >
struct Node {
   size_t id;
   /// the parent node from which this node stems
   Node* parent;
   /// the children that each action maps to in the game tree.
   /// Should be filled during the traversal.
   std::unordered_map< Action, uptr< Node > > children{};
};

template < concepts::fosg Env >
class GameTree {
  public:
   using world_state_type = typename fosg_auto_traits< Env >::world_state_type;
   using action_type = typename fosg_auto_traits< Env >::action_type ;
   using node_type = Node< action_type >;

   /**
    * @brief Does the first full traversal of the game tree and emplaces the nodes along the way.
    *
    * This function should be called exactly once on iteration 0 and never thereafter. The method
    * traverses the game tree as usual, but then also emplaces all node shared pointers into the
    * individual game trees. In order to not let this traversal go to waste the update stack for
    * regret and policy updating is also filled during this traversal.
    * I decided to convert iteration 0 into its own function (instead of querying at runtime
    * whether nodes already exist) to avoid multiple runtime overhead of searching for nodes in a
    * hash table. Whether the slight code duplication will become a maintenance issue will need
    * to be seen in the future.
    */
   GameTree(Env& env, uptr< const world_state_type >&& root_state)
       : m_root_state(std::move(root_state)), m_nodes{}
   {
      auto fill_children = [&](node_type& node, world_state_type& wstate) {
         for(const auto& action : env.actions(env.active_player(wstate), wstate)) {
            node.children.emplace(action, nullptr);
         }
      };
      auto index_pool = size_t(0);
      // emplace the root node
      auto& root = m_nodes.emplace_back({.id = index_pool++, .parent = nullptr, .children = {}});
      fill_children(root, *root_state);
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
      std::stack< std::tuple< uptr< Worldstate >, node_type* > > visit_stack;
      // copy the root state into the visitation stack
      visit_stack.emplace(std::move(root_state), root_node());

      while(not visit_stack.empty()) {
         // get the top node and world state from the stack and move them into our values.
         auto [curr_wstate, curr_node] = std::move(visit_stack.top());
         // remove those elements from the stack (there is no unified pop-and-return method for
         // stack)
         visit_stack.pop();

         for(const auto& action : ranges::views::keys(curr_node->children)) {
            // copy the current nodes world state
            auto next_wstate_uptr = utils::static_unique_ptr_cast< world_state_type >(
               utils::clone_any_way(curr_wstate));
            // move the new world state forward by the current action
            env.transition(*next_wstate_uptr, action);
            // the child node has shared ownership by each player's game tree
            auto& child_node = m_nodes.emplace_back(
               {.id = index_pool++, .parent = curr_node, .children = {}});
            fill_children(child_node, *next_wstate_uptr);

            // append the new tree node as a child of the currently visited node. We are using a
            // raw pointer here to avoid the unnecessary sptr counter increase cost
            curr_node->children(action) = &child_node;

            if(not env.is_terminal(*next_wstate_uptr)) {
               // if the newly reached world state is not a terminal state, then we merely append
               // the new child node to the queue. This way we further explore its child states
               // as reachable from the next possible actions.
               visit_stack.emplace(std::move(next_wstate_uptr), &child_node);
            }
         }
      }
   }

   [[nodiscard]] auto* root_node() const { return &(m_nodes[0]); }
   [[nodiscard]] auto* root_state() const { return m_root_state.get(); }

  private:
   std::vector< node_type > m_nodes;
   sptr< world_state_type > m_root_state;
};
}  // namespace forest

}  // namespace nor::rm

#endif  // NOR_RM_HPP
