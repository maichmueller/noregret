
#ifndef NOR_FOREST_HPP
#define NOR_FOREST_HPP

#include <spdlog/spdlog.h>

#include <algorithm>
#include <execution>
#include <memory>
#include <optional>
#include <ranges>
#include <stack>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

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
                auto to_variant_transform = std::views::transform([](const auto& any) {
                   return action_variant_type(any);
                });
                if constexpr(concepts::stochastic_env< Env >) {
                   if(curr_player == Player::chance) {
                      auto actions = m_env->chance_actions(*curr_wstate_uptr.get());
                      return std::ranges::to< std::vector >(actions | to_variant_transform);
                   }
                }
                auto actions = m_env->actions(curr_player, *curr_wstate_uptr.get());
                return std::ranges::to< std::vector >(actions | to_variant_transform);
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

template < concepts::fosg Env >
class InfostateTree {
  public:
   using world_state_type = auto_world_state_type< Env >;
   using info_state_type = auto_info_state_type< Env >;
   using action_type = auto_action_type< Env >;
   using chance_outcome_type = auto_chance_outcome_type< Env >;
   using observation_type = auto_observation_type< Env >;
   using action_variant_type = auto_action_variant_type< Env >;
   using action_variant_hasher = common::variant_hasher< action_variant_type >;
   using player_policy_type = StatePolicyView< info_state_type, action_type >;
   using player_policies_type = player_hashmap< player_policy_type >;

   // Keep the historical name for callers that used the conditional outcome type directly. The
   // current action-variant generator also maps std::monostate and void to std::monostate.
   using chance_outcome_conditional_type = std::conditional_t<
      std::is_same_v< chance_outcome_type, void >,
      std::monostate,
      chance_outcome_type >;

   struct Node {
      /// the player that takes the actions at this node (possibly Player::chance)
      Player active_player;
      /// the parent from which this node stems; ownership remains with the parent's child tuple
      Node* parent = nullptr;

      // The completed pre-comment API used three tuple fields. Keep that shape so existing tree
      // consumers can inspect the child, local counterfactual reach multiplier, and accumulated
      // terminal action value independently.
      using child_node_tuple = std::
         tuple< uptr< Node >, std::optional< rm::Probability >, std::optional< double > >;

      std::unordered_map< action_variant_type, child_node_tuple, action_variant_hasher > children{};
      /// null for chance nodes; otherwise the information state at this node
      uptr< info_state_type > infostate = nullptr;
      /// reserved for the post-traversal state-value pass; build fills action values only
      std::optional< double > state_value = std::nullopt;
   };

   InfostateTree(
      Env& env,
      uptr< world_state_type > root_state,
      player_hashmap< info_state_type > root_infostates = {}
   )
       : m_env(env),
         m_root_state(checked_root_state(std::move(root_state))),
         m_root_node(std::make_unique< Node >(Node{
            .active_player = env.active_player(*m_root_state)})),
         m_root_infostates(std::move(root_infostates))
   {
      const auto root_player = m_root_node->active_player;
      if(utils::is_actual_player_pred(root_player)) {
         if(auto infostate_iter = m_root_infostates.find(root_player);
            infostate_iter != m_root_infostates.end()) {
            m_root_node->infostate = std::make_unique< info_state_type >(infostate_iter->second);
         } else {
            m_root_node->infostate = std::make_unique< info_state_type >(root_player);
         }
      }

      // Chance is part of a FOSG roster but has no information state. Keeping only actual players
      // here also means a chance entry can never be accidentally queried as a policy state.
      for(auto player : m_env.players(*m_root_state) | utils::is_actual_player_filter) {
         if(m_root_infostates.find(player) == m_root_infostates.end()) {
            m_root_infostates.emplace(player, info_state_type{player});
         }
      }
      action_emplacer(*m_root_node, *m_root_state);
   }

   InfostateTree(const InfostateTree&) = delete;
   InfostateTree& operator=(const InfostateTree&) = delete;

   // The root node is heap-owned so moving the tree preserves every child Node::parent pointer.
   // The environment is deliberately non-owning and must outlive the tree.
   InfostateTree(InfostateTree&& other)
       : m_env(other.m_env),
         m_root_state(std::move(other.m_root_state)),
         m_root_node(std::move(other.m_root_node)),
         m_root_infostates(std::move(other.m_root_infostates))
   {
   }
   InfostateTree& operator=(InfostateTree&&) = delete;

   /**
    * @brief Fill the tree with policy/chance reach probabilities and terminal action values.
    *
    * The owning player's action probability is always one: this is the pure best-response
    * counterfactual reach convention. Opponent probabilities come from the supplied policy views,
    * and chance probabilities come from the environment. Observation histories are advanced with
    * nor::next_infostate_and_obs_buffers, including its delayed-buffer flush semantics.
    *
    * A subsequent build reuses the existing node topology but clears derived probabilities and
    * values first, so changing the owning player or policy views cannot leave stale reach data.
    */
   void build(Player owning_player, player_policies_type player_policies);

   [[nodiscard]] Env& env() { return m_env; }
   [[nodiscard]] const Env& env() const { return m_env; }
   [[nodiscard]] world_state_type& root_state() { return *m_root_state; }
   [[nodiscard]] const world_state_type& root_state() const { return *m_root_state; }
   [[nodiscard]] Node& root_node() { return *m_root_node; }
   [[nodiscard]] const Node& root_node() const { return *m_root_node; }

  private:
   static uptr< world_state_type > checked_root_state(uptr< world_state_type > root_state)
   {
      if(not root_state) {
         throw std::invalid_argument("InfostateTree requires a non-null root state.");
      }
      return root_state;
   }

   static void reset_derived_data(Node& node)
   {
      node.state_value.reset();
      for(auto& child_tuple : node.children | std::views::values) {
         auto& [child_node, action_prob, action_value] = child_tuple;
         action_prob.reset();
         action_value.reset();
         if(child_node) {
            reset_derived_data(*child_node);
         }
      }
   }

   Env& m_env;
   uptr< world_state_type > m_root_state;
   uptr< Node > m_root_node;
   player_hashmap< info_state_type > m_root_infostates;

   void action_emplacer(Node& infostate_node, world_state_type& state);
};

template < concepts::fosg Env >
void InfostateTree< Env >::action_emplacer(Node& infostate_node, world_state_type& state)
{
   if(not infostate_node.children.empty() or m_env.is_terminal(state)) {
      return;
   }

   const auto active_player = m_env.active_player(state);
   if constexpr(concepts::stochastic_env< Env >) {
      if(active_player == Player::chance) {
         for(const auto& outcome : m_env.chance_actions(state)) {
            infostate_node.children.emplace(
               action_variant_type{outcome}, typename Node::child_node_tuple{}
            );
         }
         return;
      }
   }

   for(const auto& action : m_env.actions(active_player, state)) {
      infostate_node.children.emplace(
         action_variant_type{action}, typename Node::child_node_tuple{}
      );
   }
}

template < concepts::fosg Env >
void InfostateTree< Env >::build(Player owning_player, player_policies_type player_policies)
{
   if(not utils::is_actual_player_pred(owning_player)) {
      throw std::invalid_argument("InfostateTree::build requires an actual owning player.");
   }

   using observation_buffer_type = player_hashmap<
      std::vector< std::pair< observation_type, observation_type > > >;
   using infostate_map_type = player_hashmap< info_state_type >;

   struct VisitationData {
      infostate_map_type infostates;
      observation_buffer_type observation_buffer;
   };
   struct VisitFrame {
      uptr< world_state_type > state;
      VisitationData data;
      Node* node;
   };

   // Node topology is reusable, while these values depend on both the owner and policy views.
   reset_derived_data(*m_root_node);

   infostate_map_type root_infostates;
   for(auto player : m_env.players(*m_root_state) | utils::is_actual_player_filter) {
      root_infostates.emplace(player, m_root_infostates.at(player));
   }

   std::stack< VisitFrame > visit_stack;
   visit_stack.push(VisitFrame{
      .state = utils::static_unique_ptr_downcast< world_state_type >(
         utils::clone_any_way(m_root_state)
      ),
      .data = VisitationData{.infostates = std::move(root_infostates), .observation_buffer = {}},
      .node = m_root_node.get()});

   while(not visit_stack.empty()) {
      auto frame = std::move(visit_stack.top());
      visit_stack.pop();

      const auto current_player = m_env.active_player(*frame.state);
      for(auto& [action_variant, child_tuple] : frame.node->children) {
         auto& [child_node, action_prob, action_value] = child_tuple;
         auto [next_state, current_action_prob] = std::visit(
            common::Overload{
               [&](const action_type& action) {
                  // Do not multiply by the owning player's policy: this tree stores the pure
                  // best-response counterfactual reach of 'owning_player'.
                  const double probability = action_prob.has_value()
                                                ? action_prob.value().get()
                                                : (current_player == owning_player
                                                      ? 1.
                                                      : player_policies.at(current_player)
                                                           .at(frame.data.infostates.at(
                                                              current_player
                                                           ))
                                                           .at(action));
                  return std::tuple{
                     nor::child_state(m_env, *frame.state, action), rm::Probability{probability}};
               },
               [&](const chance_outcome_conditional_type& outcome) {
                  if constexpr(concepts::deterministic_fosg< Env >) {
                     throw std::logic_error(
                        "A deterministic environment traversed a chance outcome."
                     );
                     return std::tuple{uptr< world_state_type >{nullptr}, rm::Probability{1.}};
                  } else {
                     return std::tuple{
                        nor::child_state(m_env, *frame.state, outcome),
                        rm::Probability{m_env.chance_probability(*frame.state, outcome)}};
                  }
               }},
            action_variant
         );

         // The probability is local to this edge. It is intentionally not a cumulative product;
         // callers can recover counterfactual reach by multiplying edge probabilities along a
         // path, while the owning player's own edges remain one.
         action_prob = current_action_prob;

         if(m_env.is_terminal(*next_state)) {
            // Preserve the historical action-value contract: terminal rewards are accumulated per
            // edge/history, while the corresponding local reach multiplier remains in action_prob.
            action_value = action_value.value_or(0.) + m_env.reward(owning_player, *next_state);
            continue;
         }

         auto [child_observation_buffer, child_infostate_map] = std::visit(
            common::Overload{
               [&](const std::monostate&) {
                  throw std::logic_error("InfostateTree entered a std::monostate transition.");
                  return std::tuple{frame.data.observation_buffer, frame.data.infostates};
               },
               [&](const auto& action_or_outcome) {
                  return nor::next_infostate_and_obs_buffers(
                     m_env,
                     frame.data.observation_buffer,
                     frame.data.infostates,
                     *frame.state,
                     action_or_outcome,
                     *next_state
                  );
               }},
            action_variant
         );

         const auto next_active_player = m_env.active_player(*next_state);
         if(not child_node) {
            child_node = std::make_unique< Node >();
            child_node->active_player = next_active_player;
            child_node->parent = frame.node;
            if(utils::is_actual_player_pred(next_active_player)) {
               child_node->infostate = std::make_unique< info_state_type >(
                  child_infostate_map.at(next_active_player)
               );
            }
            action_emplacer(*child_node, *next_state);
         }

         visit_stack.push(VisitFrame{
            .state = std::move(next_state),
            .data =
               VisitationData{
                  .infostates = std::move(child_infostate_map),
                  .observation_buffer = std::move(child_observation_buffer)},
            .node = child_node.get()});
      }
   }
}

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
