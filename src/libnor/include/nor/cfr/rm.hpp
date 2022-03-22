
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

enum class NodeCategory {
   chance = 0,
   choice,
   terminal,
};

template < typename Action >
struct Node {
   /// this node's id (or index within the node vector)
   size_t id;
   /// what type of node this is
   NodeCategory category;
   /// the parent node from which this node stems
   Node* parent = nullptr;
   /// the children that each action maps to in the game tree.
   /// Should be filled during the traversal.
   std::unordered_map< Action, Node* > children{};
};

template < concepts::fosg Env, typename NodeDataType >
class GameTree {
  public:
   using action_type = typename fosg_auto_traits< Env >::action_type;
   using observation_type = typename fosg_auto_traits< Env >::observation_type;
   using world_state_type = typename fosg_auto_traits< Env >::world_state_type;
   using public_state_type = typename fosg_auto_traits< Env >::public_state_type;
   using info_state_type = typename fosg_auto_traits< Env >::info_state_type;

   using node_type = Node< action_type >;
   using node_data_type = NodeDataType;

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
   template < typename NodeDataExtractor >
   // clang-format off
   requires
      std::is_invocable_r_v<
         uptr< NodeDataType >,  // return type
         NodeDataExtractor,  // the actual functor
         const node_type&,  // child node input
         world_state_type*,  // current world state
         world_state_type&,  // child world state
         const NodeDataType*,  // current node's data
         const std::map< Player, info_state_type>&,  // current info states
         const typename fosg_auto_traits<NodeDataType>::public_state_type &,  // current public state
         const std::optional<action_type>&  // action that led to the child
      >
         // clang-format on
         GameTree(Env& env, uptr< world_state_type >&& root_state, NodeDataExtractor data_extractor)
       : m_nodes{},
   m_node_data{}, m_root_state(std::move(root_state))
   {
      _grow_tree< true >(env, m_root_state, std::move(data_extractor));
   }

   GameTree(Env& env, uptr< world_state_type >&& root_state)
       : m_nodes{}, m_node_data{}, m_root_state(std::move(root_state))
   {
      _grow_tree< false >(env, m_root_state);
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

   auto* data(const node_type& node)
   {
      uptr< node_data_type >& data = m_node_data[node.id];
      return data == nullptr ? nullptr : data.get();
   }
   [[nodiscard]] const auto* data(const node_type& node) const
   {
      const uptr< node_data_type >& data = m_node_data[node.id];
      return data == nullptr ? nullptr : data.get();
   }
   [[nodiscard]] auto& root_state() const { return *m_root_state; }
   [[nodiscard]] auto& root_node() const { return m_nodes[0]; }

  private:
   /// the node storage of all nodes in the tree. Entry 0 is the root.
   std::vector< node_type > m_nodes;
   /// the paired node data vector holds at entry i the data for node i
   std::vector< uptr< node_data_type > > m_node_data;
   /// the root game state from which the tree is built
   const uptr< world_state_type > m_root_state;

   inline auto& root_node() { return m_nodes[0]; }

   template < bool extract_data, typename NodeDataExtractor = utils::empty >
   void _grow_tree(
      Env& env,
      const uptr< world_state_type >& root_state,
      NodeDataExtractor data_extractor = {})
   {
      auto fill_children = [&](node_type& node, world_state_type& wstate) {
         for(const auto& action : env.actions(env.active_player(wstate), wstate)) {
            node.children.emplace(action, nullptr);
         }
      };
      auto index_pool = size_t(0);
      // current deciding mechanism for which category the current world state may belong to.
      auto categorizer = [&](world_state_type& wstate) {
         if(env.active_player(wstate) == Player::chance) {
            return NodeCategory::chance;
         }
         if(env.is_terminal(wstate)) {
            return NodeCategory::terminal;
         }
         return NodeCategory::choice;
      };
      // initialize the infostates and public states
      auto init_infostate = [&]() {
         std::map< Player, info_state_type > istate_map{};
         for(auto player : env.players()) {
            istate_map.emplace(player, info_state_type{});
         }
         return istate_map;
      }();
      auto init_publicstate = typename fosg_auto_traits< NodeDataType >::public_state_type{};
      // emplace the root node
      auto& root = m_nodes.emplace_back(
         node_type{.id = index_pool++, .category = categorizer(*root_state)});
      fill_children(root, *root_state);
      // we need to fill the root node's data (if desired) before entering the loop, since the loop
      // assumes all entered nodes to have their data node emplaced already.
      if constexpr(extract_data) {
         m_node_data.emplace_back(data_extractor(
            root, nullptr, *m_root_state, nullptr, init_infostate, init_publicstate, std::nullopt));
      }
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
      std::stack< std::tuple<
         node_type*,
         sptr< world_state_type >,
         std::map< Player, info_state_type >,
         typename fosg_auto_traits< NodeDataType >::public_state_type > >
         visit_stack;
      // the info state and public state types need to be default constructible to be filled along
      // the trajectory
      static_assert(
         all_predicate_v< std::is_default_constructible, info_state_type, public_state_type >,
         "The Infostate and Publicstate type need to be default constructible.");
      // copy the root state into the visitation stack
      visit_stack.emplace(
         &root,
         utils::static_unique_ptr_cast< world_state_type >(utils::clone_any_way(root_state)),
         std::move(init_infostate),
         std::move(init_publicstate));
      int simple_counter = 0;
      while(not visit_stack.empty()) {
         // get the top node and world state from the stack and move them into our values.
         auto [curr_node, curr_wstate_uptr, curr_infostates, curr_publicstate] = std::move(
            visit_stack.top());
         simple_counter += 1;
         // remove those elements from the stack (there is no unified pop-and-return method for
         // stack)
         visit_stack.pop();

         for(const auto& action : ranges::views::keys(curr_node->children)) {
            // copy the current nodes world state
            simple_counter += 1;
            auto next_wstate_uptr = utils::static_unique_ptr_cast< world_state_type >(
               utils::clone_any_way(curr_wstate_uptr));
            // move the new world state forward by the current action
            env.transition(*next_wstate_uptr, action);
            // copy the infostates to append the new observations to it
            auto next_infostates = curr_infostates;
            for(auto player : env.players()) {
               next_infostates[player].append(
                  env.private_observation(player, action),
                  env.private_observation(player, *next_wstate_uptr));
            }
            auto next_publicstate = curr_publicstate;
            if constexpr(not concepts::is::empty< typename node_data_type::public_state_type >) {
               next_publicstate.append(env.public_observation(*next_wstate_uptr));
            }
            auto curr_node_id = curr_node->id;
            auto& child_node = m_nodes.emplace_back(node_type{
               .id = index_pool++,
               .category = categorizer(*next_wstate_uptr),
               .parent = curr_node,
               .children = {}});
            fill_children(child_node, *next_wstate_uptr);
            // due to vectors dynamically growing their array the previous curr_node pointer can get
            // invalidated. We thus have to reassign it to be sure to have the correct reference
            // still.
            curr_node = &m_nodes[curr_node_id];
            // append the new tree node as a child of the currently visited node. We are using a
            // raw pointer here to avoid the unnecessary sptr counter increase cost
            auto hash = std::hash< action_type >{}(action);
            auto v = ranges::to< std::vector< action_type > >(
               ranges::views::keys(curr_node->children));
            curr_node->children[action] = &child_node;

            if constexpr(extract_data) {
               m_node_data.emplace_back(data_extractor(
                  child_node,
                  curr_wstate_uptr.get(),
                  *next_wstate_uptr,
                  data(*curr_node),
                  next_infostates,
                  next_publicstate,
                  action));
            }

            if(not env.is_terminal(*next_wstate_uptr)) {
               // if the newly reached world state is not a terminal state, then we merely append
               // the new child node to the queue. This way we further explore its child states
               // as reachable from the next possible actions.
               visit_stack.emplace(
                  &child_node,
                  std::move(next_wstate_uptr),
                  std::move(next_infostates),
                  std::move(next_publicstate));
            }
         }
      }
   }
};
}  // namespace forest

}  // namespace nor::rm

#endif  // NOR_RM_HPP
