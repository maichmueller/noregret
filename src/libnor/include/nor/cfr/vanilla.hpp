
#ifndef NOR_VANILLA_HPP
#define NOR_VANILLA_HPP

#include <cppitertools/enumerate.hpp>
#include <execution>
#include <list>
#include <stack>
#include <utility>
#include <vector>

#include "node.hpp"
#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "nor/policy.hpp"
#include "nor/utils/utils.hpp"

namespace nor::rm {

template < concepts::action Action, concepts::action_policy< Action > Policy >
void regret_matching(Policy& policy_map, const std::map< Action, double >& cumul_regret)
{
   // sum up the positivized regrets and store them in a new vector
   std::map< Action, double > pos_regrets;
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
      std::transform(
         exec_policy,
         policy_map.begin(),
         policy_map.end(),
         policy_map.begin(),
         [&](const auto& entry) { return cumul_regret[std::get< 0 >(entry)] / pos_regret_sum; });
   } else {
      std::transform(
         exec_policy, policy_map.begin(), policy_map.end(), policy_map.begin(), [&](const auto&) {
            return 1. / static_cast< double >(policy_map.size());
         });
   }
}

struct CFRConfig {
   bool alternating_updates = true;
   bool store_public_states = false;
};

/**
 * A (Vanilla) Counterfactual Regret Minimization algorithm class following the
 * Factored-Observation Stochastic Games (FOSG) formalism.
 * @tparam Game, the game type to run VanillaCFR on.
 *
 */
template <
   CFRConfig cfr_config,
   concepts::fosg Game,
   concepts::state_policy< typename Game::info_state_type, typename Game::action_type > Policy >
class VanillaCFR {
   /// define all aliases to be used in this class from the game type.
   using game_type = Game;
   using action_type = typename Game::action_type;
   using world_state_type = typename Game::world_state_type;
   using info_state_type = typename Game::info_state_type;
   using public_state_type = typename Game::public_state_type;
   // the CFR Node type to store in the tree
   using cfr_node_type = CFRNode<
      action_type,
      world_state_type,
      std::conditional_t< cfr_config.store_public_states, public_state_type, NEW_EMPTY_TYPE >,
      info_state_type >;

   explicit VanillaCFR(Game&& game, Policy&& policy = Policy())
       : m_game(std::forward< Game >(game)),
         m_curr_policy(std::forward< Policy >(policy)),
         m_avg_policy()
   {
      static_assert(
         Game::turn_dynamic == TurnDynamic::sequential,
         "VanillaCFR can only be performed on a sequential turn-based game.");
   }

   /**
    * @brief Executes n iterations of the VanillaCFR algorithm in unrolled form (no recursion).
    * @param n_iterations, the number of iterations to perform.
    * @return the updated state_policy
    */
   const Policy* iterate(size_t n_iterations = 1) requires(cfr_config.alternating_updates);
   const Policy* iterate([[maybe_unused]] Player player_to_update, size_t n_iters = 1);

   void update_regret_and_policy(cfr_node_type& node, Player player);
   double reach_probability(const cfr_node_type& node) const;
   double cf_reach_probability(const cfr_node_type& node, const Player& player) const;
   double cf_reach_probability(const cfr_node_type& node, double reach_prob, const Player& player)
      const;

  private:
   /// the game object to maneuver the states with
   Game m_game;
   /// the game tree mapping information states to the associated game nodes.
   std::map< info_state_type, cfr_node_type > m_game_tree;
   /// the current policy $\sigma^t$ that each player is following in this iteration.
   std::map< Player, Policy > m_curr_policy;
   /// the average policy table.
   std::map< Player, Policy > m_avg_policy;
   size_t m_iteration = 0;

   template < typename ResultType, std::invocable< cfr_node_type*, ResultType > Functor >
   auto child_collector(cfr_node_type& node, Player player, Functor f);
};

template <
   CFRConfig cfr_config,
   concepts::fosg Game,
   concepts::state_policy< typename Game::info_state_type, typename Game::action_type > Policy >
const Policy* VanillaCFR< cfr_config, Game, Policy >::iterate(size_t n_iterations) requires(
   cfr_config.alternating_updates)
{
   return iterate(Player::chance, n_iterations);
}

template <
   CFRConfig cfr_config,
   concepts::fosg Game,
   concepts::state_policy< typename Game::info_state_type, typename Game::action_type > Policy >
const Policy* VanillaCFR< cfr_config, Game, Policy >::iterate(
   Player player_to_update,
   size_t n_iters)
{
   if constexpr(cfr_config.alternating_updates) {
      // we assert here that the chosen player to update is not the chance player as is defined by
      // default. Seeing the chance player here inidcates that the player forgot to set the plaeyr
      // parameter with this rm config.
      if(player_to_update == Player::chance) {
         std::stringstream ssout;
         ssout << "Given combination of '";
         ssout << Player::chance;
         ssout << "' and '";
         ssout << "alternating updates'";
         ssout << "is incompatible. Did you forget to pass the correct player parameter?";
         throw std::invalid_argument(ssout.str());
      }
   }
   for(auto iteration : ranges::views::iota(size_t(0), n_iters)) {
      // the tree needs to be traversed. To do so, every node (starting from the root node aka the
      // current game state) will emplace its child states - as generated from its possible actions
      // - into the queue. This queue is a First-In-First-Out (FIFO) stack which will guarantee that
      // we perform a depth-first-traversal of the game tree. This is necessary since any
      // state-value of a given node are computed via the probability of each action multiplied by
      // their successor state-values, i.e. v(s) = \sum_a \pi(s,a) * v(s')
      std::stack< std::tuple< cfr_node_type*, std::vector< action_type >, bool > > visit_stack;

      // emplace the root node in the queue to start the traversal from
      auto root_node = cfr_node_type(m_game, std::array< double, Game::player_count >(1.));
      m_game_tree.emplace(root_node.info_state(), root_node);
      // the root node is NOT a trigger for value propagation (=false), since only the children of a
      // state can trigger value collection of their parent.
      visit_stack.emplace(root_node, m_game.actions(Player::emily), false);

      while(not visit_stack.empty()) {
         // get the next tree node from the queue:
         // curr_node: stands for the popped node from the queue
         // actions: all the possible actions the active player has in this node
         // trigger_value_propagation: a boolean indicating if this is the 'rightmost' child node of
         // the parent and thus should trigger the collection of state values
         auto& [curr_node, actions, trigger_value_propagation] = visit_stack.top();
         // the last action index decides which child of curr_node will be the one to trigger
         // the curr_node's value collection from its child states
         size_t last_action_idx = actions.size() - 1;

         Player curr_player = curr_node->player();

         // we have to reverse the index range, in order to guarantee that the last_action_idx child
         // node is going to be emplace FIRST and thus popped as the LAST child of the current node
         // (which will then correctly trigger the value propagation)
         for(auto&& [action_idx, action] : iter::enumerate(actions) | ranges::views::reverse) {
            // copy the current nodes world state
            auto new_wstate = curr_node->world_state();
            // move the new world state forward by the current action
            m_game.transition(new_wstate, action);
            // extract the active player's private state from the new world state. This will serve
            // e.g. as the key for the new node to emplace
            auto new_infostate = m_game.private_state(new_wstate, m_game.active_player());
            // copy the current reach probabilities as well. These will now be updated and then
            // emplaced in the new node
            std::map< Player, double > node_reach_probs = curr_node->reach_probability();
            // multiply the current reach probabilities of this state by the policy of choosing the
            // action by the active player.
            node_reach_probs[curr_player] *= m_curr_policy[curr_player]
                                                          [{curr_node->info_state(), action}];
            // add this new world state into the tree with its active player's info state as key
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
            curr_node->children(action) = child_node;
            if(m_game.is_terminal(new_wstate)) {
               // we have reached a terminal state and can save the reward as the value of this
               // node within the node. This value will later in the algorithm be propagated up in
               // the tree.
               if constexpr(nor::concepts::has::method::reward_multi< Game >) {
                  child_node.value() = m_game.reward(m_game.players(), new_wstate);

               } else {
                  for(auto player : m_game.players()) {
                     child_node.value(player) = m_game.reward(player, new_wstate);
                  }
               }
            } else {
               // if the newly reached world state is not a terminal state, then we merely append
               // the new child node to the queue. This way we further explore its child states as
               // reachable from the next possible actions.
               visit_stack.emplace(
                  &child_node,
                  m_game.actions(curr_player),
                  action_idx == last_action_idx  // reaching the last action index means that this
                                                 // node should trigger value propagation
               );
            }
         }
         if(trigger_value_propagation) {
            // the value propagation works by specifying the last action in the list to be the
            // trigger. Upon trigger, this node's parent calls upon all its children to sent back
            // the their respective values which will be included in the parent's value by weighting
            // them by the current policy.
            // This entire idea relies upon a POST-ORDER DEPTH-FIRST GAME TREE traversal!
            auto collector = [&](cfr_node_type& node, Player player) {
               auto action_value_view = child_collector(
                  node, player, [&](cfr_node_type* child) { return child->value(player); });
               // v <- v + pi_{player}(s, a) * v(s'(a))
               node.value(player) += std::transform_reduce(
                  action_value_view.begin(),
                  action_value_view.end(),
                  0.,
                  std::plus{},
                  [&](const auto& action_value_pair) {
                     // applied action is the action that lead to the child node!
                     const auto& [applied_action, action_value] = action_value_pair;
                     // pi_{player}(s, a) * v(s'(a))
                     return m_curr_policy[player][{node.info_state(), applied_action}]
                            * action_value;
                  });
            };
            // TODO: time the parallel execution. It might be too much overhead.
            // the unparallelized version is a simple for loop over the players.
            // auto exec_policy = std::execution::seq;
            // the parallelized version uses std for_each with parallel execution policy
            auto exec_policy = std::execution::par_unseq;
            std::for_each(
               exec_policy,
               m_game.players().begin(),
               m_game.players().end(),
               [node = curr_node->parent(), this](Player player) { collector(*node, player); });
         }

         if constexpr(cfr_config.alternating_updates) {
            // in alternating updates, we only update the regret and strategy if the current player
            // is the chosen player to update.
            if(player_to_update == curr_player) {
               update_regret_and_policy(*curr_node, curr_player);
            }
         } else {
            // if we do simultaenous updates, then we always update the regret and strategy values
            // of the current player.
            update_regret_and_policy(*curr_node, curr_player);
         }
         // now the first element in the queue can be removed.
         visit_stack.pop();
      }
      // lastly add a completed iteration to the counter
      m_iteration++;
   }
   return &m_curr_policy;
}

template <
   CFRConfig cfr_config,
   concepts::fosg Game,
   concepts::state_policy< typename Game::info_state_type, typename Game::action_type > Policy >
template <
   typename ResultType,
   std::invocable< typename VanillaCFR< cfr_config, Game, Policy >::cfr_node_type*, ResultType >
      Functor >
auto VanillaCFR< cfr_config, Game, Policy >::child_collector(
   VanillaCFR::cfr_node_type& node,
   Player player,
   Functor f)
{
   return ranges::views::zip(
      node.children() | ranges::views::keys,
      node.children() | ranges::views::values | ranges::views::transform(f));
}

template <
   CFRConfig cfr_config,
   concepts::fosg Game,
   concepts::state_policy< typename Game::info_state_type, typename Game::action_type > Policy >
void VanillaCFR< cfr_config, Game, Policy >::update_regret_and_policy(
   VanillaCFR::cfr_node_type& node,
   Player player)
{
   double reach_prob = reach_probability(node);
   ranges::for_each(
      child_collector(node, player, [&](cfr_node_type* child) { return child->value(player); }),
      [&](const auto& action_value_pair) {
         const auto& [action, action_value] = action_value_pair;
         node.regret(player) += cf_reach_probability(node, reach_prob, player)
                                * (action_value - node.value(player));
         m_avg_policy[node.info_state()][action] += reach_prob
                                                    * m_curr_policy[node.info_state()][action];
      });
   regret_matching(
      m_curr_policy[node.info_state()],
      child_collector(node, player, [&](cfr_node_type* child) { return child->regret(player); }));
}

template <
   CFRConfig cfr_config,
   concepts::fosg Game,
   concepts::state_policy< typename Game::info_state_type, typename Game::action_type > Policy >
double VanillaCFR< cfr_config, Game, Policy >::reach_probability(const cfr_node_type& node) const
{
   auto values_view = node.reach_probability() | ranges::views::values;
   return std::reduce(values_view.begin(), values_view.end(), 1, std::multiplies{});
}

template <
   CFRConfig cfr_config,
   concepts::fosg Game,
   concepts::state_policy< typename Game::info_state_type, typename Game::action_type > Policy >
double VanillaCFR< cfr_config, Game, Policy >::cf_reach_probability(
   const VanillaCFR::cfr_node_type& node,
   const Player& player) const
{
   auto reach_prob = reach_probability(node);
   return cf_reach_probability(node, reach_prob, player);
}

template <
   CFRConfig cfr_config,
   concepts::fosg Game,
   concepts::state_policy< typename Game::info_state_type, typename Game::action_type > Policy >
double VanillaCFR< cfr_config, Game, Policy >::cf_reach_probability(
   const VanillaCFR::cfr_node_type& node,
   double reach_prob,
   const Player& player) const
{
   // remove the players contribution to the likelihood
   auto cf_reach_prob = reach_prob / node.reach_probability(player);
   return cf_reach_prob;
}

}  // namespace nor::rm

#endif  // NOR_VANILLA_HPP
