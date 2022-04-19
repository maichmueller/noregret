
#ifndef NOR_RM_HPP
#define NOR_RM_HPP

#include <execution>
#include <range/v3/all.hpp>

#include "common/common.hpp"
#include "node.hpp"
#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "nor/type_defs.hpp"
#include "nor/utils/utils.hpp"

namespace nor::rm {

template < ranges::range Policy >
auto& normalize_action_policy_inplace(Policy& policy)
{
   double sum = 0.;
   for(const auto& [action, prob] : policy) {
      sum += prob;
   }
   for(auto& [action, prob] : policy) {
      prob /= sum;
   }
   return policy;
};

template < ranges::range Policy >
auto normalize_action_policy(const Policy& policy)
{
   Policy copy = policy;
   return normalize_action_policy_inplace(copy);
};

template < ranges::range Policy >
auto& normalize_state_policy_inplace(Policy& policy)
{
   for(auto& [_, action_policy] : policy) {
      normalize_action_policy_inplace(action_policy);
   }
   return policy;
};

template < ranges::range Policy >
auto normalize_state_policy(const Policy& policy)
{
   auto copy = policy;
   return normalize_state_policy_inplace(copy);
};

template < typename MapLike >
concept kv_like_over_doubles = requires(MapLike m)
{
   // has to be key-value-like to iterate over values only
   ranges::views::values(m);
   // value type has to be convertible to double
   std::is_convertible_v< decltype(*(ranges::views::values(m).begin())), double >;
};
/**
 * @brief computes the reach probability of the node.
 *
 * Since each player's compounding likelihood contribution is stored in the nodes themselves, the
 * actual computation is nothing more than merely multiplying all player's individual
 * contributions.
 * @param reach_probability_contributions the compounded reach probability of each player for
 * this node
 * @return the reach probability of the nde
 */
template < kv_like_over_doubles KVdouble >
[[nodiscard]] inline double reach_probability(const KVdouble& reach_probability_contributions)
{
   auto values_view = reach_probability_contributions | ranges::views::values;
   return std::reduce(values_view.begin(), values_view.end(), double(1.), std::multiplies{});
}
/**
 * @brief computes the counterfactual reach probability of the player for this node.
 *
 * The method delegates the actual computation to the overload with an already provided reach
 * probability.
 * @param node the node at which the counterfactual reach probability is to be computed
 * @param player the player for which the value is computed
 * @return the counterfactual reach probability
 */
template < kv_like_over_doubles KVdouble >
requires requires(KVdouble m)
{
   // the keys have to of type 'Player' as well
   std::is_convertible_v< decltype(*(ranges::views::keys(m).begin())), Player >;
}
inline double cf_reach_probability(
   const KVdouble& reach_probability_contributions,
   const Player& player)
{
   auto values_view = reach_probability_contributions
                      | ranges::views::filter([&](const auto& player_rp_pair) {
                           return std::get< 0 >(player_rp_pair) != player;
                        })
                      | ranges::views::values;
   return std::reduce(values_view.begin(), values_view.end(), double(1.), std::multiplies{});
}

/**
 * @brief Performs regret-matching on the given policy with respect to the provided regret
 *
 * @tparam Action
 * @tparam Policy
 */
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
         return std::get< 1 >(entry) = std::max(0., cumul_regret.at(std::get< 0 >(entry)))
                                       / pos_regret_sum;
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
         const node_type&,  // child node input
         VisitationData&,  // visitation data access
         const typename node_type::action_variant_type*,  // action that led to the child
         world_state_type*,  // current world state
         world_state_type*  // child world state
      >
      and std::invocable<
         PreChildVisitHook,  // the actual functor
         const node_type&,  // child node input
         VisitationData&,  // visitation data access
         world_state_type*  // current world state
      >
      and std::invocable<
         PostChildVisitHook,  // the actual functor
         const node_type&,  // child node input
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
      bool traverse_via_worldstate = true,
      bool single_trajectory = false)
   {
      if(traverse_via_worldstate) {
         _traverse< true >(
            std::move(traversal_strategy),
            std::move(vis_data),
            std::move(hooks),
            single_trajectory);
      } else {
         _traverse< false >(
            std::move(traversal_strategy),
            std::move(vis_data),
            std::move(hooks),
            single_trajectory);
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
   [[nodiscard]] auto& node(size_t id) const { return m_nodes.at(id); }
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

   template <
      bool traverse_via_worldstate,
      typename TraversalStrategy,
      typename VisitationData,
      typename... HookFunctors >
   void _traverse(
      TraversalStrategy traversal_strategy,
      VisitationData&& initial_vis_data,
      TraversalHooks< HookFunctors... >&& hooks,
      bool single_trajectory)
   {

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
      std::stack< std::tuple< node_type*, uptr< world_state_type > >, VisitationData > visit_stack;
      // emplace the root node into the visitation stack
      visit_stack.emplace(
         &root_node(),
         [&] {
            if constexpr(traverse_via_worldstate)
               return utils::static_unique_ptr_downcast< world_state_type >(
                  utils::clone_any_way(m_root_state));
            else
               return nullptr;
         }(),
         std::move(initial_vis_data));

      while(not visit_stack.empty()) {
         // get the top node and world state from the stack and move them into our values.
         auto [curr_node, curr_wstate_uptr, vis_data] = std::move(visit_stack.top());
         // remove those elements from the stack (there is no unified pop-and-return method for
         // stack)
         visit_stack.pop();

         hooks.pre_child_hook(*curr_node, curr_wstate_uptr.get(), vis_data);

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
                  child_node = &(m_nodes
                                    .emplace(
                                       child_id,
                                       node_type{
                                          .id = child_id,
                                          .category = _categorize(*next_wstate_uptr),
                                          .parent = curr_node,
                                          .children = {},
                                          .action_from_parent = action})
                                    .first->second);
               }
               // append the new tree node as a child of the currently visited node. We are using a
               // raw pointer here since lifetime management is maintained by the game tree itself
               curr_node->children[action] = child_node;
            }
            // offer the caller to extract information for the currently visited node. We are
            // passing the worldstate ptrs even if we are traversing by states in order to maintain
            // consistency in our call signature

            auto new_visitation_data = hooks.child_hook(
               *child_node, vis_data, &action, curr_wstate_uptr.get(), next_wstate_uptr.get());

            if(child_node->category != forest::NodeCategory::terminal) {
               // if the newly reached world state is not a terminal state, then we merely append
               // the new child node to the queue. This way we further explore its child states
               // as reachable from the next possible actions.
               visit_stack.emplace(
                  child_node, std::move(next_wstate_uptr), std::move(new_visitation_data));
            }
         }
         hooks.post_child_hook(*curr_node, curr_wstate_uptr.get(), vis_data);
      }
   }
};
}  // namespace forest

}  // namespace nor::rm

namespace nor::utils {
constexpr CEBijection< nor::rm::forest::NodeCategory, std::string_view, 3 > nodecateogry_name_bij =
   {std::pair{nor::rm::forest::NodeCategory::chance, "chance"},
    std::pair{nor::rm::forest::NodeCategory::choice, "choice"},
    std::pair{nor::rm::forest::NodeCategory::terminal, "terminal"}};

}  // namespace nor::utils

namespace common {
template <>
inline std::string to_string(const nor::rm::forest::NodeCategory& e)
{
   return std::string(nor::utils::nodecateogry_name_bij.at(e));
}
}  // namespace common

// inline auto& operator<<(std::ostream& os, nor::rm::forest::NodeCategory e)
//{
//    os << common::to_string(e);
//    return os;
// }

#endif  // NOR_RM_HPP
