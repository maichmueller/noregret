
#ifndef NOR_VANILLA_HPP
#define NOR_VANILLA_HPP

#include <cppitertools/enumerate.hpp>
#include <cppitertools/reversed.hpp>
#include <execution>
#include <list>
#include <map>
#include <range/v3/all.hpp>
#include <stack>
#include <unordered_map>
#include <utility>
#include <vector>

#include "node.hpp"
#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "nor/policy.hpp"
#include "nor/type_defs.hpp"
#include "nor/utils/logging_macros.h"
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
 * terminology of the Factored-Observation Stochastic Games (FOSG) formulation.
 * @tparam Env, the environment type to run VanillaCFR on.
 *
 */
template <
   CFRConfig cfr_config,
   concepts::fosg Env,
   typename FOSGTraitType,
   concepts::state_policy Policy >
requires concepts::fosg_trait_match< fosg_traits< Policy >, FOSGTraitType >
class VanillaCFR {
   /// define all aliases to be used in this class from the game type.
   using env_type = Env;
   using action_type = typename Env::action_type;
   using world_state_type = typename Env::world_state_type;
   using info_state_type = typename Env::info_state_type;
   using public_state_type = typename Env::public_state_type;
   using observation_type = typename Env::observation_type;
   // the CFR Node type to store in the tree
   using cfr_node_type = CFRNode<
      action_type,
      std::conditional_t< cfr_config.store_public_states, public_state_type, NEW_EMPTY_TYPE >,
      info_state_type >;
   using game_tree_type = std::unordered_map< info_state_type, sptr< cfr_node_type > >;

   explicit VanillaCFR(Env&& game, Policy&& policy = Policy())
       : m_env(std::forward< Env >(game)),
         m_curr_policy(std::forward< Policy >(policy)),
         m_avg_policy()
   {
      static_assert(
         Env::turn_dynamic == TurnDynamic::sequential,
         "VanillaCFR can only be performed on a sequential turn-based game.");
   }

   /**
    * @brief Executes n iterations of the VanillaCFR algorithm in unrolled form (no recursion).
    *
    * The decision for doing alternating updates or simultaneous updates happens at compile time via
    * the cfr config. This optimizes some unncessary repeated if-branching away at the cost of
    * higher maintenance. The user can also decide whether to store the public state at each node
    *
    * @param n_iterations, the number of iterations to perform.
    * @return the updated state_policy
    */
   const Policy* iterate(size_t n_iterations = 1) requires(
      cfr_config.alternating_updates and concepts::has::method::initial_world_state< Env >)
   {
      return iterate(Player::chance, n_iterations);
   }
   const Policy* iterate([[maybe_unused]] Player player_to_update, size_t n_iters = 1) requires
      concepts::has::method::initial_world_state< Env >
   {
      return iterate(m_env.initial_world_state(), player_to_update, n_iters);
   }
   const Policy* iterate(
      world_state_type&& initial_wstate,
      [[maybe_unused]] Player player_to_update,
      size_t n_iters = 1);

   void update_regret_and_policy(cfr_node_type& node, Player player);
   double reach_probability(const cfr_node_type& node) const;
   double cf_reach_probability(const cfr_node_type& node, const Player& player) const;
   double cf_reach_probability(const cfr_node_type& node, double reach_prob, const Player& player)
      const;

  private:
   /// the environment object to maneuver the states with.
   Env m_env;
   /// the game tree mapping information states to the associated game nodes.
   std::map< Player, game_tree_type > m_game_trees;
   /// the current policy $\pi^t$ that each player is following in this iteration.
   std::map< Player, Policy > m_curr_policy;
   /// the average policy table.
   std::map< Player, Policy > m_avg_policy;
   /// the number of iterations we have run so far.
   size_t m_iteration = 0;

   template < typename ResultType, std::invocable< cfr_node_type*, ResultType > Functor >
   auto child_collector(VanillaCFR::cfr_node_type& node, Functor f)
   {
      return ranges::views::zip(
         node.children() | ranges::views::keys,
         node.children() | ranges::views::values | ranges::views::transform(f));
   }
};

template < CFRConfig cfg, typename Env, typename Policy, typename FOSGTraitType = fosg_traits< Env > >
VanillaCFR< cfg, Env, FOSGTraitType, Policy >
make_cfr(Env&& env, Policy&& policy, FOSGTraitType = {})
{
   return {std::forward< Env >(env), std::forward< Policy >(policy)};
}

template <
   CFRConfig cfr_config,
   concepts::fosg Env,
   typename FOSGTraitType,
   concepts::state_policy Policy >
requires concepts::fosg_trait_match< fosg_traits< Policy >, FOSGTraitType >
const Policy* VanillaCFR< cfr_config, Env, FOSGTraitType, Policy >::iterate(
   world_state_type&& initial_wstate,
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
      LOGD2("Iteration number: ", iteration);
      // the tree needs to be traversed. To do so, every node (starting from the root node aka the
      // current game state) will emplace its child states - as generated from its possible actions
      // - into the queue. This queue is Last-In-First-Out (LIFO), hence referred to as 'stack',
      // which will guarantee that we perform a depth-first-traversal of the game tree (FIFO would
      // lead to breadth-first). This is necessary since any state-value of a given node is
      // computed via the probability of each action multiplied by their successor state-values,
      // i.e. v(s) = \sum_a \pi(s,a) * v(s').
      // The stack uses raw pointers to nodes, since nodes are first emplaced in the tree(s) and
      // then put on the stack for later visitation. Their lifetime management is thus handled by
      // shared pointers stored in the trees.
      std::stack< std::tuple< world_state_type, cfr_node_type*, std::vector< action_type >, bool > >
         visit_stack;

      Player curr_player = m_env->active_player();
      auto root_node = std::make_shared< cfr_node_type >(
         []< concepts::iterable T >(T&& players) {
            std::map< Player, info_state_type > is_map;
            for(auto player : players) {
               is_map.emplace(player, info_state_type{});
            }
            return is_map;
         }(m_env.players()),
         typename cfr_node_type::public_state_type{},
         curr_player,
         []< concepts::iterable T >(T&& players) {
            std::map< Player, double > rp;
            for(auto player : players) {
               rp.emplace(player, 1.);
            }
            return rp;
         }(m_env.players()));
      // emplace the root node in the queue to start the traversal from
      for(const auto& info_state : root_node.info_states()) {
         m_game_trees.emplace(info_state, root_node.get());
      }
      // the root node is NOT a trigger for value propagation (=false), since only the children of a
      // state can trigger value collection of their parent.
      visit_stack.emplace(
         initial_wstate, root_node, m_env.actions(curr_player, initial_wstate), false);

      while(not visit_stack.empty()) {
         // get the next tree node from the queue:
         // curr_node: stands for the popped node from the queue
         // actions: all the possible actions the active player has in this node
         // trigger_value_propagation: a boolean indicating if this is the 'rightmost' child node of
         // the parent and thus should trigger the collection of state values
         auto& [curr_wstate, curr_node, actions, trigger_value_propagation] = visit_stack.top();
         // the last action index decides which child of curr_node will be the one to trigger
         // the curr_node's value collection from its child states
         size_t last_action_idx = actions.size() - 1;

         curr_player = curr_node->player();

         // we have to reverse the index range, in order to guarantee that the last_action_idx child
         // node is going to be emplaced FIRST and thus popped as the LAST child of the current node
         // (and as such trigger the value propagation)
         for(auto&& [action_idx, action] : iter::enumerate(iter::reversed(actions))) {
            // the index starts enumerating at 0, but we are stepping through the actions in
            // reverse, so we need to adapt the index acoordingly
            action_idx = last_action_idx - action_idx;
            // copy the current nodes world state
            auto next_wstate = utils::clone_any_way(curr_wstate);
            // move the new world state forward by the current action
            m_env.transition(next_wstate, action);
            // copy the current information states of all players. Each state will be updated with
            // the new private observation the respective player receives from the environment.
            auto next_infostates = utils::clone_any_way(curr_node->info_states());
            // copy the current reach probabilities as well. These will now be updated by the active
            // player's likelihood of playing towards this node.
            std::map< Player, double > node_reach_probs = utils::clone_any_way(
               curr_node->reach_probability());
            // append each players action observation and world state observation to the current
            // stream of information states.
            for(auto player : m_env.players()) {
               next_infostates[player].append(
                  {m_env.private_observation(player, action),
                   m_env.private_observation(player, next_wstate)});
            }
            // multiply the current reach probabilities of this state by the policy of choosing the
            // action by the active player.
            node_reach_probs[curr_player] *= m_curr_policy[curr_player][{
               curr_node->info_states(curr_player), action}];
            // the child node has shared ownership by each player's game tree
            auto child_node_sptr = std::make_shared< cfr_node_type >(
               next_infostates, m_env.active_player(next_wstate), node_reach_probs, curr_node);
            // add this new world state into each player's tree with their respective info state as
            // key
            for(auto player : m_env.players()) {
               m_game_trees[player].emplace(next_infostates[player], child_node_sptr);
            }
            // append the new tree node as a child of the currently visited node. We are using a raw
            // pointer here to avoid the unnecessary sptr counter increase cost
            curr_node->children(action) = child_node_sptr.get();
            if(m_env.is_terminal(next_wstate)) {
               // we have reached a terminal state and can save the reward as the value of this
               // node within the node. This value will later in the algorithm be propagated up in
               // the tree.
               if constexpr(nor::concepts::has::method::reward_multi< Env >) {
                  // if the environment has a method for returning all rewards for given players at
                  // once, then we will assume this is a more performant alternative and use it
                  // instead (e.g. when it is costly to compute the reward of each player
                  // individually).
                  auto& value_map = child_node_sptr->value();
                  auto rewards = m_env.reward(m_env.players(), next_wstate);
                  ranges::views::for_each(
                     ranges::views::zip(m_env.player(), rewards), [&](const auto& p_r_pair) {
                        value_map.emplace(std::get< 0 >(p_r_pair), std::get< 1 >(p_r_pair));
                     });
               } else {
                  // otherwise we just loop over the per player reward method
                  for(auto player : m_env.players()) {
                     child_node_sptr->value(player) = m_env.reward(player, next_wstate);
                  }
               }
            } else {
               // if the newly reached world state is not a terminal state, then we merely append
               // the new child node to the queue. This way we further explore its child states as
               // reachable from the next possible actions.
               auto next_actions = m_env.actions(curr_player, next_wstate);
               visit_stack.emplace(
                  move_if< std::is_move_constructible_v< world_state_type > >(next_wstate),
                  child_node_sptr.get(),
                  next_actions,
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
            auto value_collector = [&](cfr_node_type& node, Player player) {
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
                     return m_curr_policy[player][{node.info_state(player), applied_action}]
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
               m_env.players().begin(),
               m_env.players().end(),
               [&, node = curr_node->parent(), this](Player player) {
                  value_collector(*node, player);
               });
         }

         if constexpr(cfr_config.alternating_updates) {
            // in alternating updates, we only update the regret and strategy if the current player
            // is the chosen player to update.
            if(curr_player == player_to_update) {
               update_regret_and_policy(*curr_node, curr_player);
            }
         } else {
            // if we do simultaenous updates, then we always update the regret and strategy values
            // of the current player.
            update_regret_and_policy(*curr_node, curr_player);
         }
         // now the first element in the stack can be removed.
         visit_stack.pop();
      }
      // lastly add a completed iteration to the counter
      m_iteration++;
   }
   return &m_curr_policy;
}

template <
   CFRConfig cfr_config,
   concepts::fosg Env,
   typename FOSGTraitType,
   concepts::state_policy Policy >
requires concepts::fosg_trait_match< fosg_traits< Policy >, FOSGTraitType >
void VanillaCFR< cfr_config, Env, FOSGTraitType, Policy >::update_regret_and_policy(
   VanillaCFR::cfr_node_type& node,
   Player player)
{
   double reach_prob = reach_probability(node);
   ranges::for_each(
      child_collector(node, [&](cfr_node_type* child) { return child->value(player); }),
      [&](const auto& action_value_pair) {
         const auto& [action, action_value] = action_value_pair;
         node.regret(player) += cf_reach_probability(node, reach_prob, player)
                                * (action_value - node.value(player));
         m_avg_policy[node.info_state()][action] += reach_prob
                                                    * m_curr_policy[node.info_state()][action];
      });
   regret_matching(
      m_curr_policy[node.info_state()],
      child_collector(node, [&](cfr_node_type* child) { return child->regret(player); }));
}

template <
   CFRConfig cfr_config,
   concepts::fosg Env,
   typename FOSGTraitType,
   concepts::state_policy Policy >
requires concepts::fosg_trait_match< fosg_traits< Policy >, FOSGTraitType >
double VanillaCFR< cfr_config, Env, FOSGTraitType, Policy >::reach_probability(
   const cfr_node_type& node) const
{
   auto values_view = node.reach_probability() | ranges::views::values;
   return std::reduce(values_view.begin(), values_view.end(), 1, std::multiplies{});
}

template <
   CFRConfig cfr_config,
   concepts::fosg Env,
   typename FOSGTraitType,
   concepts::state_policy Policy >
requires concepts::fosg_trait_match< fosg_traits< Policy >, FOSGTraitType >
double VanillaCFR< cfr_config, Env, FOSGTraitType, Policy >::cf_reach_probability(
   const VanillaCFR::cfr_node_type& node,
   const Player& player) const
{
   auto reach_prob = reach_probability(node);
   return cf_reach_probability(node, reach_prob, player);
}

template <
   CFRConfig cfr_config,
   concepts::fosg Env,
   typename FOSGTraitType,
   concepts::state_policy Policy >
requires concepts::fosg_trait_match< fosg_traits< Policy >, FOSGTraitType >
double VanillaCFR< cfr_config, Env, FOSGTraitType, Policy >::cf_reach_probability(
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
