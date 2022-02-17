
#ifndef NOR_VANILLA_HPP
#define NOR_VANILLA_HPP

#include <list>
#include <queue>
#include <utility>
#include <vector>

#include "nor/concepts.hpp"
#include "nor/policy.hpp"
#include "nor/utils/utils.hpp"

namespace nor {

template <
   concepts::action Action,
   concepts::observation Observation,
   concepts::info_state< Action, Observation > Infostate,
   concepts::info_state< Action, Observation > Publicstate,
   concepts::info_state< Action, Observation > Worldstate >
struct CFRNode {
   Worldstate world_state;
   Publicstate public_state;
   Infostate info_state;
   Player player;
   std::map< Player, double > reach_prob;
   std::map< Action, double > regret;
   double value = 0.;
   std::map< Action, CFRNode* > children;
   CFRNode* parent;

   CFRNode(
      const Worldstate& world_state,
      const Publicstate& public_state_,
      const Infostate& info_state_,
      Player player_,
      std::map< Player, double > reach_prob_,
      CFRNode* parent = nullptr)
       : world_state(world_state),
         public_state(public_state_),
         info_state(info_state_),
         player(player_),
         reach_prob(std::move(reach_prob_)),
         regret(),
         children(),
         parent(parent),
         m_hash_cache(std::hash< Infostate >{}(info_state))
   {
   }

   template < concepts::fosg Game >
   CFRNode(const Game& g, const std::map< Player, double >& reach_prob_)
       : CFRNode(
          g.world_state(),
          g.info_state(world_state, g.player()),
          g.public_state(world_state, g.world_state()),
          g.active_player(g.world_state()),
          reach_prob_)
   {
      static_assert(
         std::is_same_v< Action, typename Game::action_type >,
         "Provided action type and game action type are not the same.");
   }

   auto hash() const { return m_hash_cache; }
   auto operator==(const CFRNode& other) const { return world_state == other.world_state; }

  private:
   size_t m_hash_cache;
};

}  // namespace nor

namespace std {

template < typename... Args >
struct hash< nor::CFRNode< Args... > > {
   size_t operator()(const nor::CFRNode< Args... >& node) const { return node.hash(); }
};
}  // namespace std

namespace nor {
/**
 * A (Vanilla) Counterfactual Regret Minimization algorithm class following the
 * Factored-Observation Stochastic Games (FOSG) formalism.
 * @tparam Game, the game type to run CFR on.
 * or a neural network type for estimating values.
 *
 */
template <
   concepts::fosg Game,
   concepts::state_policy< typename Game::info_state_type, typename Game::action_type > Policy >
class CFR {
   /// define all aliases to be used in this class from the game type.
   using action_type = typename Game::action_type;
   using info_state_type = typename Game::info_state_type;
   using public_state_type = typename Game::public_state_type;
   using world_state_type = typename Game::world_state_type;
   // we configure the correct CFR Node type to store in the tree
   using cfr_node_type = decltype(CFRNode(
      std::declval< Game >(),
      std::declval< std::array< double, Game::player_count > >()));

   explicit CFR(Game&& game, Policy&& policy = Policy())
       : m_game(std::forward< Game >(game)),
         m_policy(std::forward< Policy >(policy)),
         m_avg_policy()
   {
      static_assert(
         Game::turn_dynamic == TurnDynamic::sequential,
         "CFR can only be performed on a sequential turn-based game.");
   }

   /**
    * @brief Executes n iterations of the CFR algorithm in unrolled form (no recursion).
    * @param n_iters, the number of iterations to perform.
    * @return the updated state_policy
    */
   const Policy* iterate(int n_iters = 1);

   template < concepts::vector_policy< action_type > VectorPolicy >
   void regret_matching(const VectorPolicy& policy_vec, std::vector< double > cumul_reg);

  private:
   Game m_game;
   std::map< info_state_type, cfr_node_type > m_game_tree;
   std::map< Player, Policy > m_policy;
   std::map< Player, Policy > m_avg_policy;
};

template <
   concepts::fosg Game,
   concepts::state_policy< typename Game::info_state_type, typename Game::action_type > Policy >
const Policy* CFR< Game, Policy >::iterate(int n_iters)
{
   auto& game_state = m_game.world_state();
   bool is_terminal = m_game.is_terminal(game_state);
   // the evaluated policy value. This value will be iteratively updated during the traversal with
   // new counterfactual (state, action)-values
   double policy_value = 0;

   // the tree needs to be traversed. To do so, every node (starting from the root node aka the
   // current game state) will emplace its child states - as generated from its possible actions -
   // into the queue and then set itself as the receiver of any updates later down the path in the
   // tree of its children.
   std::queue< cfr_node_type* > tree_queue;

   auto root_node = cfr_node_type(m_game, std::array< double, Game::player_count >(1.));
   m_game_tree[root_node.info_state] = root_node;
   tree_queue.push(root_node);

   std::map< info_state_type, std::map< action_type, double > > cf_action_values;
   while(not tree_queue.empty()) {
      // get the next tree node from the queue
      const auto& curr_node = *(tree_queue.pop());
      auto actions = m_game.actions(curr_node.data.world_state, m_game.player());

      // this needs to be computed only once in the for loop over actions
      auto is_chance_player = m_game.active_player(curr_node.data.world_state) == Player::chance;

      for(const auto& action : actions) {
         auto new_wstate = curr_node.world_state;
         m_game.transition(new_wstate, action);
         auto new_infostate = m_game.private_state(new_wstate, m_game.active_player());
         std::map< Player, double > node_reach_probs = curr_node.reach_prob;
         if(not is_chance_player) {
            // multiply the current reach probabilities of this state by the policy of choosing the
            // action by the active player.
            node_reach_probs[curr_node.player] *= m_policy[curr_node.player]
                                                          [{curr_node.info_state, action}];
         };
         // add this new world state into the tree by its active player's information state
         auto* emplaced_node = &(m_game_tree
                                    .emplace(
                                       new_infostate,
                                       new_wstate,
                                       m_game.public_state(new_wstate),
                                       new_infostate,
                                       m_game.active_player(new_wstate),
                                       node_reach_probs,
                                       &curr_node)
                                    .first->second);
         // append the new tree node as a child of the currently visited node.
         curr_node.children[action] = emplaced_node;
         if(not m_game.is_terminal(new_wstate)) {
            // if the newly reached world state is not a terminal state, then we append to the queue
            // to further explore its child states as reachable from the possible actions
            tree_queue.push(emplaced_node);
         } else {
            // otherwise we have reached a terminal state and can now backpropagate the state value
            // to all upstream nodes that lead to it
            cfr_node_type* node_ptr = emplaced_node;
            double reward = m_game.reward(new_wstate);
            while(node_ptr != nullptr) {
               node_ptr->value = reward;
               node_ptr = node_ptr->parent;
            }
         }
      }
   }

   //
   //   if() {
   //      return m_game.reward(game_state);
   //   }

   return &m_policy;
}

// template <
//    concepts::fosg Game,
//    concepts::state_policy< typename Game::info_state_type, typename Game::action_type > Policy >
// const Policy* CFR< Game, Policy >::iterate(world_state_type curr_wstate, int n_iters)
//{
//    if()
//    auto& game_state = m_game.world_state();
//    bool is_terminal = m_game.is_terminal(game_state);
//    // the evaluated policy value. This value will be iteratively updated during the traversal
//    with
//    // new counterfactual (state, action)-values
//    double policy_value = 0;
//
//    // the tree needs to be traversed. To do so, every node (starting from the root node aka the
//    // current game state) will emplace its child states - as generated from its possible actions
//    -
//    // into the queue and then set itself as the receives of any updates later down the path in
//    the
//    // tree of its children.
//    std::queue< cfr_node_type > tree_queue;
//    std::vector<std::list<>> backprop_streams;
//
//    tree_queue.push(cfr_node_type(m_game, {}, std::array< double, Game::player_count >(1.)));
//
//    std::map< info_state_type, std::map<action_type, double >> cf_action_values;
//    while(not tree_queue.empty()) {
//       const auto& curr_node = tree_queue.pop();
//       auto actions = m_game.actions(curr_node.data.world_state, m_game.player());
//       auto is_chance_player = m_game.active_player(curr_node.data.world_state) == Player::chance;
//
//       for(const auto& action : actions) {
//          auto new_wstate = curr_node.data.world_state;
//          m_game.transition(new_wstate, action);
//          if(not is_chance_player) {
//             tree_queue.push({m_game,})
//          }
//       }
//    }
//
//    //
//    //   if() {
//    //      return m_game.reward(game_state);
//    //   }
//
//    return &m_policy;
// }

template <
   concepts::fosg Game,
   concepts::state_policy< typename Game::info_state_type, typename Game::action_type > Policy >
template < concepts::vector_policy< typename CFR< Game, Policy >::action_type > VectorPolicy >
void CFR< Game, Policy >::regret_matching(
   const VectorPolicy& policy_vec,
   std::vector< double > cumul_reg)
{
}

///// helper function for building CFR from a game later maybe?
// template <typename...Args>
// CFR<Args...> make_cfr(Args&&... args);

}  // namespace nor

#endif  // NOR_VANILLA_HPP
