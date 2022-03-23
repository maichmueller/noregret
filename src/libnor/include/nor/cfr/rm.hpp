
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

template < concepts::fosg Env >
class GameTree {
  public:
   using action_type = typename fosg_auto_traits< Env >::action_type;
   using observation_type = typename fosg_auto_traits< Env >::observation_type;
   using world_state_type = typename fosg_auto_traits< Env >::world_state_type;
   using public_state_type = typename fosg_auto_traits< Env >::public_state_type;
   using info_state_type = typename fosg_auto_traits< Env >::info_state_type;

   using node_type = Node< action_type >;

   GameTree(Env& env, uptr< world_state_type >&& root_state)
       : m_env(&env), m_nodes{}, m_root_state(std::move(root_state))
   {
   }

   /**
    * @brief Traverses all possible game choices and connects the nodes along the way.
    *
    * This function should be called exactly once after object construction and never thereafter.
    * The method traverses the game tree and emplaces all nodes and their assoicted data storage (if
    * desired) into the game tree.
    */
   template < typename NodeDataExtractor >
   // clang-format off
   requires
      std::is_invocable_v<
         NodeDataExtractor,  // the actual functor
         const node_type&,  // child node input
         world_state_type*,  // current world state
         world_state_type&,  // child world state
         const std::map< Player, info_state_type>&,  // current info states
         const public_state_type &,  // current public state
         const std::optional<action_type>&  // action that led to the child
      >
         // clang-format on
         inline void initialize(NodeDataExtractor data_extractor)
   {
      _grow_tree< true >(m_root_state, std::move(data_extractor));
   }
   inline void initialize() { _grow_tree< false >(m_root_state); }

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

   [[nodiscard]] auto size() const { return m_nodes.size(); }
   [[nodiscard]] auto& root_state() const { return *m_root_state; }
   [[nodiscard]] auto& root_node() const { return *(m_nodes[0]); }

  private:
   /// pointer to the environment used to initialize the tree
   Env* m_env;
   /// the node storage of all nodes in the tree. Entry 0 is the root.
   std::vector< uptr< node_type > > m_nodes;
   /// the root game state from which the tree is built
   const uptr< world_state_type > m_root_state;

   inline auto& root_node() { return *(m_nodes[0]); }

   template < bool extract_data, typename NodeDataExtractor = utils::empty >
   void _grow_tree(
      const uptr< world_state_type >& root_state,
      NodeDataExtractor data_extractor = {})
   {
      auto dummy_fill_children = [&](node_type& node, world_state_type& wstate) {
         for(const auto& action : m_env->actions(m_env->active_player(wstate), wstate)) {
            node.children.emplace(action, nullptr);
         }
      };
      auto index_pool = size_t(0);
      // current deciding mechanism for which category the current world state may belong to.
      auto categorizer = [&](world_state_type& wstate) {
         if(m_env->active_player(wstate) == Player::chance) {
            return NodeCategory::chance;
         }
         if(m_env->is_terminal(wstate)) {
            return NodeCategory::terminal;
         }
         return NodeCategory::choice;
      };
      // initialize the infostates and public states
      auto init_infostate = [&]() {
         std::map< Player, info_state_type > istate_map{};
         for(auto player : m_env->players()) {
            istate_map.emplace(player, info_state_type{});
         }
         return istate_map;
      }();
      public_state_type init_publicstate{};
      // emplace the root node
      auto& root = m_nodes.emplace_back(std::make_unique< node_type >(
         node_type{.id = index_pool++, .category = categorizer(*root_state)}));
      dummy_fill_children(*root, *root_state);
      // we need to fill the root node's data (if desired) before entering the loop, since the loop
      // assumes all entered nodes to have their data node emplaced already.
      if constexpr(extract_data) {
         data_extractor(
            *root, nullptr, *m_root_state, init_infostate, init_publicstate, std::nullopt);
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
         public_state_type > >
         visit_stack;
      // the info state and public state types need to be default constructible to be filled along
      // the trajectory
      static_assert(
         all_predicate_v< std::is_default_constructible, info_state_type, public_state_type >,
         "The Infostate and Publicstate type need to be default constructible.");
      // copy the root state into the visitation stack
      visit_stack.emplace(
         root.get(),
         utils::static_unique_ptr_cast< world_state_type >(utils::clone_any_way(root_state)),
         std::move(init_infostate),
         std::move(init_publicstate));
      while(not visit_stack.empty()) {
         // get the top node and world state from the stack and move them into our values.
         auto [curr_node, curr_wstate_uptr, curr_infostates, curr_publicstate] = std::move(
            visit_stack.top());
         // remove those elements from the stack (there is no unified pop-and-return method for
         // stack)
         visit_stack.pop();

         for(const auto& action : ranges::views::keys(curr_node->children)) {
            // copy the current nodes world state
            auto next_wstate_uptr = utils::static_unique_ptr_cast< world_state_type >(
               utils::clone_any_way(curr_wstate_uptr));
            // move the new world state forward by the current action
            m_env->transition(*next_wstate_uptr, action);
            // copy the infostates to append the new observations to it
            auto next_infostates = curr_infostates;
            for(auto player : m_env->players()) {
               next_infostates[player].append(
                  m_env->private_observation(player, action),
                  m_env->private_observation(player, *next_wstate_uptr));
            }
            auto next_publicstate = curr_publicstate;
            next_publicstate.append(
               m_env->public_observation(action), m_env->public_observation(*next_wstate_uptr));

            auto& child_node = m_nodes.emplace_back(std::make_unique< node_type >(node_type{
               .id = index_pool++,
               .category = categorizer(*next_wstate_uptr),
               .parent = curr_node,
               .children = {}}));
            dummy_fill_children(*child_node, *next_wstate_uptr);
            // append the new tree node as a child of the currently visited node. We are using a
            // raw pointer here since lifetime management is maintained by the game tree itself
            curr_node->children[action] = child_node.get();

            if constexpr(extract_data) {
               data_extractor(
                  *child_node,
                  curr_wstate_uptr.get(),
                  *next_wstate_uptr,
                  next_infostates,
                  next_publicstate,
                  action);
            }

            if(child_node->category != forest::NodeCategory::terminal) {
               // if the newly reached world state is not a terminal state, then we merely append
               // the new child node to the queue. This way we further explore its child states
               // as reachable from the next possible actions.
               visit_stack.emplace(
                  child_node.get(),
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

namespace nor::utils {
constexpr CEBijection< nor::rm::forest::NodeCategory, std::string_view, 27 > nodecateogry_name_bij =
   {std::pair{nor::rm::forest::NodeCategory::chance, "chance"},
    std::pair{nor::rm::forest::NodeCategory::choice, "choice"},
    std::pair{nor::rm::forest::NodeCategory::terminal, "terminal"}};
template <>
inline std::string_view enum_name(nor::rm::forest::NodeCategory e)
{
   return nodecateogry_name_bij.at(e);
}

}  // namespace nor::utils

inline auto& operator<<(std::ostream& os, nor::rm::forest::NodeCategory e)
{
   os << nor::utils::enum_name(e);
   return os;
}

#endif  // NOR_RM_HPP
