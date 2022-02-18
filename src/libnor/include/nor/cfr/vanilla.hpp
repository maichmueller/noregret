
#ifndef NOR_VANILLA_HPP
#define NOR_VANILLA_HPP

#include <execution>
#include <list>
#include <queue>
#include <utility>
#include <vector>

#include "nor/concepts.hpp"
#include "nor/policy.hpp"
#include "nor/utils/utils.hpp"

namespace nor {

template < concepts::action Action, concepts::vector_policy< Action > VectorPolicy >
void regret_matching(VectorPolicy& policy_vec, const std::vector< double >& cumul_reg)
{
   // sum up the positivized regrets and store them in a new vector
   std::vector< double > pos_regrets;
   double pos_regret_sum{0.};
   pos_regrets.reserve(cumul_reg.size());
   for(const auto& regret : cumul_reg) {
      double pos_regret = std::max(0., regret);
      pos_regrets.emplace_back(pos_regret);
      pos_regret_sum += pos_regret;
   }
   // apply the new policy to the vector policy
   auto exec_policy{std::execution::par_unseq};
   if(pos_regret_sum > 0) {
      std::transform(
         exec_policy,
         policy_vec.begin(),
         policy_vec.end(),
         policy_vec.begin(),
         [&](auto regret) { return regret / pos_regret_sum; });
   } else {
      std::transform(
         exec_policy,
         policy_vec.begin(),
         policy_vec.end(),
         policy_vec.begin(),
         [&](auto regret) { return 1. / static_cast< double >(policy_vec.size()); });
   }
}

struct CFRConfig {
   bool alternating_updates;
};

template <
   CFRConfig cfr_config,
   concepts::action Action,
   concepts::info_state Infostate,
   concepts::info_state Worldstate >
struct CFRNode {
   using state_value_type = std::conditional_t<
      cfr_config.alternating_updates,
      /*true:  */ double,  // we need only a scalar if a single player's values are updated
      /*false: */ std::map< Player, double > >;

   Worldstate world_state;
   Infostate info_state;
   Player player;
   state_value_type value{};
   std::map< Player, double > reach_prob;
   std::map< Action, double > regret;
   std::map< Action, CFRNode* > children;
   CFRNode* parent;

   CFRNode(
      const Worldstate& world_state,
      const Infostate& info_state_,
      Player player_,
      std::map< Player, double > reach_prob_,
      state_value_type values = {},
      CFRNode* parent = nullptr)
       : world_state(world_state),
         info_state(info_state_),
         player(player_),
         value(std::move(values)),
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

template < nor::CFRConfig cfg, typename... Args >
struct hash< nor::CFRNode< cfg, Args... > > {
   size_t operator()(const nor::CFRNode< cfg, Args... >& node) const { return node.hash(); }
};
}  // namespace std

namespace nor {

/**
 * A (Vanilla) Counterfactual Regret Minimization algorithm class following the
 * Factored-Observation Stochastic Games (FOSG) formalism.
 * @tparam Game, the game type to run CFRBase on.
 * or a neural network type for estimating values.
 *
 */
template <
   CFRConfig cfr_config,
   concepts::fosg Game,
   concepts::state_policy< typename Game::info_state_type, typename Game::action_type > Policy,
   typename Variant >
class CFRBase {
   /// define all aliases to be used in this class from the game type.
   using action_type = typename Game::action_type;
   using info_state_type = typename Game::info_state_type;
   using public_state_type = typename Game::public_state_type;
   using world_state_type = typename Game::world_state_type;
   // we configure the correct CFR Node type to store in the tree
   using cfr_node_type = CFRNode< cfr_config, action_type, info_state_type, world_state_type >;

   explicit CFRBase(Game&& game, Policy&& policy = Policy())
       : m_game(std::forward< Game >(game)),
         m_policy(std::forward< Policy >(policy)),
         m_avg_policy()
   {
      static_assert(
         Game::turn_dynamic == TurnDynamic::sequential,
         "CFRBase can only be performed on a sequential turn-based game.");
   }

   /**
    * @brief Executes n iterations of the CFRBase algorithm in unrolled form (no recursion).
    * @param n_iters, the number of iterations to perform.
    * @return the updated state_policy
    */

   const Policy* iterate(int n_iters = 1) requires(cfr_config.alternating_updates);

   const Policy* iterate([[maybe_unused]] Player player_to_update, int n_iters = 1);

   template < concepts::vector_policy< action_type > VectorPolicy >
   void policy_update(VectorPolicy& policy_vec, const std::vector< double >& regrets)
   {
      derived_cast()->_policy_update(policy_vec, regrets);
   }

  private:
   Game m_game;
   std::map< info_state_type, cfr_node_type > m_game_tree;
   std::map< Player, Policy > m_policy;
   std::map< Player, Policy > m_avg_policy;

   auto derived_cast() { return static_cast< Variant* >(this); }

   void _collect_state_value(cfr_node_type* node, Player player);
};

template <
   CFRConfig cfr_config,
   concepts::fosg Game,
   concepts::state_policy< typename Game::info_state_type, typename Game::action_type > Policy,
   typename Variant >
const Policy* CFRBase< cfr_config, Game, Policy, Variant >::iterate(int n_iters) requires(
   cfr_config.alternating_updates)
{
   return iterate(Player::chance, n_iters);
}

template <
   CFRConfig cfr_config,
   concepts::fosg Game,
   concepts::state_policy< typename Game::info_state_type, typename Game::action_type > Policy,
   typename Variant >
const Policy* CFRBase< cfr_config, Game, Policy, Variant >::iterate(
   Player player_to_update,
   int n_iters)
{
   auto& game_state = m_game.world_state();
   bool is_terminal = m_game.is_terminal(game_state);

   // the tree needs to be traversed. To do so, every node (starting from the root node aka the
   // current game state) will emplace its child states - as generated from its possible actions -
   // into the queue. This queue is a First-In-First-Out (FIFO) stack which will guarantee that we
   // perform a depth-first-traversal of the game tree. This is necessary since any state-value of a
   // given node are computed via the probability of each action multiplied by their successor
   // state-values, i.e. v(s) = \sum_a \pi(s,a) * v(s')
   std::queue< std::tuple< cfr_node_type*, std::vector< action_type >, bool > > tree_queue;

   // emplace the root node in the queue to start the traversal from
   auto root_node = cfr_node_type(m_game, std::array< double, Game::player_count >(1.));
   m_game_tree[root_node.info_state] = root_node;
   tree_queue.emplace(root_node, m_game.actions(m_game.active_player()), false);

   std::map< info_state_type, std::map< action_type, double > > cf_action_values;
   while(not tree_queue.empty()) {
      // get the next tree node from the queue:
      // curr_node: stands for the popped node from the queue
      // actions: all the possible actions the active player has in this node
      // trigger_value_collection: a boolean indicating if this is the 'rightmost' child node of
      // the parent and thus should trigger the collection of state values
      const auto& [curr_node, actions, trigger_value_collection] = tree_queue.pop();
      // the last action index decides over whose child of curr_node will be the one to trigger the
      // curr_node's value collection from its child states
      size_t last_action_idx = actions.size() - 1;

      for(size_t action_idx = 0; action_idx < actions.size(); action_idx++) {
         const auto& action = actions[action_idx];
         auto new_wstate = curr_node->world_state;
         m_game.transition(new_wstate, action);
         auto new_infostate = m_game.private_state(new_wstate, m_game.active_player());
         std::map< Player, double > node_reach_probs = curr_node->reach_prob;

         // multiply the current reach probabilities of this state by the policy of choosing the
         // action by the active player.
         node_reach_probs[curr_node->player] *= m_policy[curr_node->player]
                                                        [{curr_node->info_state, action}];
         // add this new world state into the tree under its active player's information state as
         // key
         auto& child_node = m_game_tree
                               .emplace(
                                  /*key=*/new_infostate,
                                  /*value_args...=*/new_wstate,
                                  new_infostate,
                                  m_game.active_player(new_wstate),
                                  node_reach_probs,
                                  curr_node)
                               .first->second;
         // append the new tree node as a child of the currently visited node.
         curr_node->children[action] = child_node;
         if(m_game.is_terminal(new_wstate)) {
            // we have reached a terminal state and can save the reward as the value of this node
            std::map< Player, double > values;
            child_node->value = m_game.reward(new_wstate);
         } else {
            // if the newly reached world state is not a terminal state, then we append to the queue
            // to further explore its child states as reachable from the possible actions
            tree_queue.emplace(
               &child_node,
               m_game.actions(child_node.world_state, m_game.active_player(child_node.player)),
               action_idx == last_action_idx);
         }
      }
      if(trigger_value_collection) {
         // TODO: time this implementation difference! Probably faster to simply have the for
         //  loop.
         // the unparallelized version is a simple for loop
         //            for(auto player : m_game.players()) {
         //               _collect_state_value(curr_node, player);
         //            }
         // the parallelized version has to use std transform
         const auto& players = m_game.players();
         std::transform(
            std::execution::par_unseq,
            players.begin(),
            players.end(),
            [&curr_node = curr_node, this](Player player) {
               _collect_state_value(curr_node, player);
            });
      }

      if constexpr(not cfr_config.alternating_updates) {
         if(player_to_update == curr_node->player) {}
      } else {
         const auto& players = m_game.players();
         std::transform(
            std::execution::par_unseq,
            players.begin(),
            players.end(),
            [&curr_node = curr_node, this](Player player) {
               regret_matching(m_policy[player][curr_node.infostate], curr_node.regret);
            });
      }
   }
   return &m_policy;
}

template <
   CFRConfig cfr_config,
   concepts::fosg Game,
   concepts::state_policy< typename Game::info_state_type, typename Game::action_type > Policy,
   typename Variant >
void CFRBase< cfr_config, Game, Policy, Variant >::_collect_state_value(
   CFRBase::cfr_node_type* node,
   Player player)
{
   double state_value = 0;
   double& curr_node_value = node->value[player];
   for(const auto& [applied_action, child_node] : node->children) {
      // v <- v + pi_{player}(s, a) * v(s'(a))
      curr_node_value += m_policy[player][{node->info_state, applied_action}]
                         * child_node->value[player];
   }
}

}  // namespace nor

#endif  // NOR_VANILLA_HPP
