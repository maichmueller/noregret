
#ifndef NOR_VANILLA_HPP
#define NOR_VANILLA_HPP

//#include <cppitertools/enumerate.hpp>
//#include <cppitertools/reversed.hpp>
#include <execution>
#include <iostream>
#include <list>
#include <map>
#include <queue>
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

struct CFRConfig {
   bool alternating_updates = true;
   bool store_public_states = false;
};

/**
 * A (Vanilla) Counterfactual Regret Minimization algorithm class following the
 * terminology of the Factored-Observation Stochastic Games (FOSG) formulation.
 * @tparam Env, the environment/game type to run VanillaCFR on.
 *
 */
template <
   CFRConfig cfr_config,
   typename Env,
   typename Policy,
   typename DefaultPolicy,
   typename AveragePolicy = Policy >
// clang-format off
 requires
    concepts::fosg< Env >
    && fosg_traits_partial_match< Policy, Env >::value
    && concepts::state_policy<
          Policy,
          typename fosg_auto_traits< Env >::info_state_type,
          typename fosg_auto_traits< Env >::observation_type
       >
   && concepts::state_policy<
         AveragePolicy,
         typename fosg_auto_traits< Env >::info_state_type,
         typename fosg_auto_traits< Env >::observation_type
      >
    && concepts::default_state_policy<
          DefaultPolicy,
          typename fosg_auto_traits< Env >::info_state_type,
          typename fosg_auto_traits< Env >::observation_type
       >
class VanillaCFR {
   // clang-format on
  public:
   /// aliases for the template types
   using env_type = Env;
   using policy_type = Policy;
   using default_policy_type = DefaultPolicy;
   /// import all fosg aliases to be used in this class from the env type.
   using action_type = typename fosg_auto_traits< Env >::action_type;
   using world_state_type = typename fosg_auto_traits< Env >::world_state_type;
   using info_state_type = typename fosg_auto_traits< Env >::info_state_type;
   using public_state_type = typename fosg_auto_traits< Env >::public_state_type;
   using observation_type = typename fosg_auto_traits< Env >::observation_type;
   // the CFR Node type to store in the tree
#if(COMPILER_NAME == gcc)
   // gcc has a bug here in which using the lambda macro for declaring a unique new empty type
   // will cause static assertion fails of the allocator within std::map and unordered_map. See
   // https://stackoverflow.com/questions/71382976
   using cfr_node_type = CFRNode<
      action_type,
      info_state_type,
      std::conditional_t< cfr_config.store_public_states, public_state_type, utils::empty > >;
#else
   using cfr_node_type = CFRNode<
      action_type,
      info_state_type,
      std::conditional_t< cfr_config.store_public_states, public_state_type, NEW_EMPTY_TYPE > >;
#endif
   using game_tree_type = std::unordered_map< info_state_type, sptr< cfr_node_type > >;

   VanillaCFR(
      sptr< world_state_type > root_state,
      Env game,
      Policy policy = Policy(),
      DefaultPolicy default_policy = DefaultPolicy(),
      AveragePolicy avg_policy = AveragePolicy())
      // clang-format off
      requires
         all_predicate_v<
            std::is_default_constructible,
            Policy,
            AveragePolicy>
         and all_predicate_v<
            std::is_copy_constructible,
            Policy,
            AveragePolicy >
       // clang-format on
       :
       m_env(std::move(game)),
       m_curr_policy(),
       m_avg_policy(),
       m_default_policy(std::move(default_policy)),
       m_root_state(std::move(root_state)),
       m_root_node(_make_root_node())
   {
      _assert_sequential_game();
      for(auto player : game.players()) {
         m_curr_policy[player] = policy;
         m_avg_policy[player] = avg_policy;
      }
      _init_player_update_schedule();
   }

   VanillaCFR(
      Env env,
      Policy policy = Policy(),
      DefaultPolicy default_policy = DefaultPolicy(),
      AveragePolicy avg_policy = AveragePolicy())
      // clang-format off
      requires
         concepts::has::method::initial_world_state< Env >
       // clang-format on
       :
       VanillaCFR(
          std::make_shared< world_state_type >(env.initial_world_state()),
          std::move(env),
          std::move(policy),
          std::move(default_policy),
          std::move(avg_policy))
   {
   }

   VanillaCFR(
      sptr< world_state_type > root_state,
      Env env,
      Policy policy,
      AveragePolicy avg_policy,
      DefaultPolicy default_policy = DefaultPolicy())
      // clang-format off
   requires
      all_predicate_v<
         std::is_copy_constructible,
         Policy,
         AveragePolicy >
       // clang-format on
       :
       VanillaCFR(
          std::move(env),
          std::move(root_state),
          std::move(policy),
          std::move(avg_policy),
          std::move(default_policy))
   {
   }

   VanillaCFR(
      sptr< world_state_type > root_state,
      Env game,
      std::map< Player, Policy > policy,
      std::map< Player, AveragePolicy > avg_policy,
      DefaultPolicy default_policy = DefaultPolicy())
      // clang-format off
      requires std::is_copy_constructible_v< Policy >
       // clang-format on
       :
       m_env(std::move(game)),
       m_root_state(std::move(root_state)),
       m_curr_policy(std::move(policy)),
       m_avg_policy(std::move(avg_policy)),
       m_default_policy(std::move(default_policy)),
       m_root_node(_make_root_node())
   {
      _assert_sequential_game();
   }

  public:
   /// API

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
   auto iterate(size_t n_iterations = 1);
   auto iterate(Player player_to_update) requires(cfr_config.alternating_updates);

   template < bool current_policy >
   auto& fetch_policy(Player player, const cfr_node_type& node)
   {
      const auto& info_state = node.info_states(player);
      if constexpr(current_policy) {
         auto& player_policy = m_curr_policy[player];
         auto found_action_policy = player_policy.find(info_state);
         if(found_action_policy == player_policy.end()) {
            auto legal_actions = node.actions();
            auto debug_val = m_default_policy[{info_state, legal_actions}];
            auto v = m_default_policy[{info_state, legal_actions}][legal_actions[0]];
            return player_policy.emplace(info_state, m_default_policy[{info_state, legal_actions}])
               .first->second;
         }
         return found_action_policy->second;
      } else {
         auto& player_policy = m_avg_policy[player];
         auto found_action_policy = player_policy.find(info_state);
         if(found_action_policy == player_policy.end()) {
            typename AveragePolicy::action_policy_type default_avg_policy;
            for(const auto& action : node.actions()) {
               default_avg_policy[action] = 0.;
            }
            return player_policy.emplace(info_state, default_avg_policy).first->second;
         }
         return found_action_policy->second;
      }
   }

   void update_regret_and_policy(cfr_node_type& node, Player player);
   double reach_probability(const cfr_node_type& node) const;
   double cf_reach_probability(const cfr_node_type& node, const Player& player) const;
   double cf_reach_probability(const cfr_node_type& node, double reach_prob, const Player& player)
      const;

   auto iteration() { return m_iteration; }
   [[nodiscard]] auto& root_state() const { return m_root_state; }
   [[nodiscard]] auto& game_tree() const { return m_game_trees; }
   [[nodiscard]] auto& game_tree(Player player) const { return m_game_trees.at(player); }
   [[nodiscard]] auto& policy() const { return m_curr_policy; }
   [[nodiscard]] auto& average_policy() const { return m_avg_policy; }

  private:
   /// the environment object to maneuver the states with.
   env_type m_env;

   /// the game tree mapping information states to the associated game nodes.
   std::map< Player, game_tree_type > m_game_trees;
   /// the current policy $\pi^t$ that each player is following in this iteration (t).
   std::map< Player, Policy > m_curr_policy;
   /// the average policy table.
   std::map< Player, AveragePolicy > m_avg_policy;
   /// the fallback policy to use when the encountered infostate has not been obseved before
   default_policy_type m_default_policy;

   /// the root game state to solve
   const sptr< world_state_type > m_root_state;
   /// the root game state to solve
   const sptr< cfr_node_type > m_root_node;
   /// The update queue to use once the nodes have been filled from a tree traversal.
   /// We need to arrange a delayed update of tree nodes, since any node's values depend on the
   /// child values and the reach probabilities. Those values are found once the full traversal
   /// encounters terminal states (leaf nodes) whose values are then propagated back up with the
   /// relevant reach probability. The order in which this propragation occurs is crucial and thus
   /// defined by the update_queue. This queue needs to be a First-In-Last-Out structure, as it
   /// ensures we are updating the values from the leaves, which are inserted last, to the root of
   /// the tree, which is inserted first.
   std::stack< cfr_node_type* > m_update_stack;

   /// the next player to update when doing alternative updates. Otherwise this member will be
   /// unused.
   std::deque< Player > m_player_update_schedule{};
   /// the number of iterations we have run so far.
   size_t m_iteration = 0;

   void _iterate(std::optional< Player > player_to_update);

   template < std::invocable< cfr_node_type* > Functor >
   auto _child_collector(VanillaCFR::cfr_node_type& node, Functor f)
   {
      return ranges::views::zip(
         node.children() | ranges::views::keys,
         node.children() | ranges::views::values | ranges::views::transform(f));
   }

   inline void _assert_sequential_game()
   {
      if(m_env.turn_dynamic() != TurnDynamic::sequential) {
         throw std::invalid_argument(
            "VanillaCFR can only be performed on a sequential turn-based game.");
      }
   }

   inline void _init_player_update_schedule()
   {
      if constexpr(cfr_config.alternating_updates) {
         for(auto player : m_env.players()) {
            m_player_update_schedule.push_back(player);
         }
      }
   }

   sptr< cfr_node_type > _make_root_node()
   {
      auto curr_player = m_env.active_player(*m_root_state);
      auto root_node = std::make_shared< cfr_node_type >(
         curr_player,  // the root's active_player
         m_env.actions(curr_player, *m_root_state),  // the root actions
         m_env.is_terminal(*m_root_state),
         /*infostates=*/
         []< concepts::iterable T >(T&& players) {
            std::map< Player, info_state_type > istate_map;
            for(auto player : players) {
               istate_map.emplace(player, info_state_type{});
            }
            return istate_map;
         }(m_env.players()),
         typename cfr_node_type::public_state_type{},
         /*reach_probs=*/
         []< concepts::iterable T >(T&& players) {
            std::map< Player, double > reach_p;
            for(auto player : players) {
               reach_p.emplace(player, 1.);
            }
            return reach_p;
         }(m_env.players()));
      // add the root node to each player's game tree
      _emplace_node(root_node->info_states(), root_node);
      return root_node;
   }

   void _first_traversal()
   {
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
      std::stack< std::tuple< sptr< world_state_type >, cfr_node_type* > > visit_stack;
      // copy the root state into the visitation stack
      visit_stack.emplace(m_root_state, m_root_node.get());

      while(not visit_stack.empty()) {
         // get the top node and world state from the stack and move them into our values.
         auto [curr_wstate, curr_node] = std::move(visit_stack.top());
         // remove those elements from the stack (there is no unified pop-and-return method for
         // stack)
         visit_stack.pop();

         // we allocate the current player here to not ask for its retrieval for every action anew
         Player curr_player = curr_node->player();
         for(const auto& action : curr_node->actions()) {
            // copy the current nodes world state
            auto next_wstate_uptr = utils::static_unique_ptr_cast< world_state_type >(
               utils::clone_any_way(curr_wstate));
            // move the new world state forward by the current action
            m_env.transition(*next_wstate_uptr, action);
            // copy the current information states of all players. Each state will be updated with
            // the new private observation the respective player receives from the environment.
            static_assert(
               std::is_copy_constructible_v< info_state_type >,
               "Infostate type needs to be copy constructible.");
            auto next_infostates = curr_node->info_states();
            // append each players action observation and world state observation to the current
            // stream of information states.
            for(auto player : m_env.players()) {
               next_infostates[player].append(
                  m_env.private_observation(player, action),
                  m_env.private_observation(player, *next_wstate_uptr));
            }
            auto next_public_state = curr_node->public_state();
            if constexpr(not concepts::is::empty< typename cfr_node_type::public_state_type >) {
               next_public_state.append(m_env.public_observation(*next_wstate_uptr));
            }
            // the child node has shared ownership by each player's game tree
            auto child_node_sptr = std::make_shared< cfr_node_type >(
               m_env.active_player(*next_wstate_uptr),
               m_env.actions(m_env.active_player(*next_wstate_uptr), *next_wstate_uptr),
               m_env.is_terminal(*next_wstate_uptr),
               next_infostates,  // the node copies these infostate maps in its constructor
               std::move(next_public_state),
               curr_node->reach_probability(),  // we for now only copy the parent's reach
                                                // probs. they will be updated later
               curr_node);
            // add this new world state into each player's tree with their respective info state key
            _emplace_node(std::move(next_infostates), child_node_sptr);
            // append the new tree node as a child of the currently visited node. We are using a raw
            // pointer here to avoid the unnecessary sptr counter increase cost
            curr_node->children(action) = child_node_sptr.get();
            // Update the active player's reach probability of this node by its current policy of
            // playing the action that lead to it.
            child_node_sptr->reach_probability(curr_player) *= fetch_policy< true >(
               curr_player, *curr_node)[action];

            if(m_env.is_terminal(*next_wstate_uptr)) {
               // we have reached a terminal state and can save the reward as the value of this
               // node within the node. This value will later in the algorithm be propagated up in
               // the tree.
               collect_rewards(*next_wstate_uptr, *child_node_sptr);
            } else {
               // if the newly reached world state is not a terminal state, then we merely append
               // the new child node to the queue. This way we further explore its child states as
               // reachable from the next possible actions.
               visit_stack.emplace(std::move(next_wstate_uptr), child_node_sptr.get());
            }
         }
         // enqueue the current node for delayed regret & strategy updates
         m_update_stack.push(curr_node);
      }
   }
   void _emplace_node(
      std::map< Player, info_state_type > infostates,
      const std::shared_ptr< cfr_node_type >& node_sptr)
   {
      for(auto player : m_env.players()) {
         auto [it, emplaced] = m_game_trees[player].emplace(
            infostates[player], node_sptr);
         if(not emplaced) {
            std::stringstream ss;
            ss << "Could not emplace child node into game tree of player ";
            ss << player
               << ", since another value was already found at its place. This could indicate "
                  "a bug in "
               << std::quoted("operator==") << "of the provided info_state_type.";
            throw std::logic_error(ss.str());
         }
      }
   }

   void collect_rewards(world_state_type& next_wstate, cfr_node_type& node) const
   {
      if constexpr(nor::concepts::has::method::reward_multi< Env >) {
         // if the environment has a method for returning all rewards for given players at
         // once, then we will assume this is a more performant alternative and use it
         // instead (e.g. when it is costly to compute the reward of each player
         // individually).
         ranges::views::for_each(
            ranges::views::zip(m_env.players(), m_env.reward(m_env.players(), next_wstate)),
            [&value_map = node.value()](const auto& p_r_pair) {
               value_map.emplace(std::get< 0 >(p_r_pair), std::get< 1 >(p_r_pair));
            });
      } else {
         // otherwise we just loop over the per player reward method
         for(auto player : m_env.players()) {
            node.value(player) = m_env.reward(player, next_wstate);
         }
      }
   }

   void _traversal()
   {
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
      std::stack< cfr_node_type* > visit_stack;
      // copy the root state into the visitation stack
      visit_stack.emplace(m_root_node.get());

      while(not visit_stack.empty()) {
         // move the next tree node from the queue into a local variable
         auto curr_node = std::move(visit_stack.top());
         // remove this node from the visit stack
         visit_stack.pop();

         Player curr_player = curr_node->player();

         for(const auto& [action, child_node] : curr_node->children()) {
            // Update the active player's reach probability of this node by its current policy of
            // playing the action that lead to it.
            child_node->reach_probability(curr_player) *= fetch_policy< true >(
               curr_player, *curr_node)[action];

            if(not child_node->terminal()) {
               // append the child node to the queue. This way we further explore its child states
               // as reachable from the next possible actions.
               visit_stack.emplace(child_node);
            }
         }

         // enqueue this node in the delayed update queue to ensure its regret and policy values are
         // being updated
         m_update_stack.push(curr_node);
      }
   }

   void _update_queued_nodes(std::optional< Player > player_to_update)
   {
      while(not m_update_stack.empty()) {
         // the value propagation works by specifying the last action in the list to be the
         // trigger. Upon trigger, this node's parent calls upon all its children to sent back
         // their respective values which will be factored into in the parent's value by
         // weighting them by the current policy. This entire idea relies upon a POST-ORDER
         // DEPTH-FIRST GAME TREE traversal!
         auto value_collector = [&](cfr_node_type& node, Player player) {
            auto action_value_view = _child_collector(
               node, [&](cfr_node_type* child) { return child->value(player); });
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
                  return fetch_policy< true >(player, node)[applied_action] * action_value;
               });
         };

         auto node = std::move(m_update_stack.top());
         m_update_stack.pop();
         ranges::for_each(m_env.players(), [&](Player player) { value_collector(*node, player); });

         if constexpr(cfr_config.alternating_updates) {
            // in alternating updates, we only update the regret and strategy if the current
            // player is the chosen player to update.
            if(node->player() == player_to_update.value_or(Player::chance)) {
               update_regret_and_policy(*node, node->player());
            }
         } else {
            // if we do simultaenous updates, then we always update the regret and strategy
            // values of the node's active player.
            update_regret_and_policy(*node, node->player());
         }
      }
   }

   Player _cycle_player_update_list()
   {
      Player next_to_update = m_player_update_schedule.front();
      m_player_update_schedule.pop_front();
      m_player_update_schedule.push_back(next_to_update);
      return next_to_update;
   }
};

template <
   CFRConfig cfr_config,
   typename Env,
   typename Policy,
   typename DefaultPolicy,
   typename AveragePolicy >
// clang-format off
 requires
    concepts::fosg< Env >
    && fosg_traits_partial_match< Policy, Env >::value
    && concepts::state_policy<
          Policy,
          typename fosg_auto_traits< Env >::info_state_type,
          typename fosg_auto_traits< Env >::observation_type
       >
   && concepts::state_policy<
         AveragePolicy,
         typename fosg_auto_traits< Env >::info_state_type,
         typename fosg_auto_traits< Env >::observation_type
      >
    && concepts::default_state_policy<
          DefaultPolicy,
          typename fosg_auto_traits< Env >::info_state_type,
          typename fosg_auto_traits< Env >::observation_type
       >
     // clang-format on
     auto VanillaCFR< cfr_config, Env, Policy, DefaultPolicy, AveragePolicy >::iterate(
        size_t n_iters)
{
   for(auto iteration : ranges::views::iota(size_t(0), n_iters)) {
      LOGD2("Iteration number: ", iteration);
      Player next_to_update = _cycle_player_update_list();
      _iterate(next_to_update);

      m_iteration++;
   }
   return &m_curr_policy;
}

template <
   CFRConfig cfr_config,
   typename Env,
   typename Policy,
   typename DefaultPolicy,
   typename AveragePolicy >
// clang-format off
 requires
    concepts::fosg< Env >
    && fosg_traits_partial_match< Policy, Env >::value
    && concepts::state_policy<
          Policy,
          typename fosg_auto_traits< Env >::info_state_type,
          typename fosg_auto_traits< Env >::observation_type
       >
   && concepts::state_policy<
         AveragePolicy,
         typename fosg_auto_traits< Env >::info_state_type,
         typename fosg_auto_traits< Env >::observation_type
      >
    && concepts::default_state_policy<
          DefaultPolicy,
          typename fosg_auto_traits< Env >::info_state_type,
          typename fosg_auto_traits< Env >::observation_type
       >

     // clang-format on
     auto VanillaCFR< cfr_config, Env, Policy, DefaultPolicy, AveragePolicy >::iterate(
        Player player_to_update) requires(cfr_config.alternating_updates)
{
   // we assert here that the chosen player to update is not the chance player as is defined
   // by default. Seeing the chance player here inidcates that the player forgot to set the
   // plaeyr parameter with this rm config.
   if(player_to_update == Player::chance) {
      std::stringstream ssout;
      ssout << "Given combination of '";
      ssout << Player::chance;
      ssout << "' and '";
      ssout << "alternating updates'";
      ssout << "is incompatible. Did you forget to pass the correct player parameter?";
      throw std::invalid_argument(ssout.str());
   } else if(auto env_players = m_env.players();
             ranges::find(env_players, player_to_update) == env_players.end()) {
      std::stringstream ssout;
      ssout << "Given player to update ";
      ssout << player_to_update;
      ssout << " is not a member of the game's player list ";
      ssout << ranges::views::all(env_players) << ".";
      throw std::invalid_argument(ssout.str());
   }

   LOGD2("Iteration number: ", m_iteration);
   // retrieve the next player to update from the update schedule and reattach it to the back of the
   // queue.
   auto player_q_iter = std::find(
      m_player_update_schedule.begin(), m_player_update_schedule.end(), player_to_update);
   m_player_update_schedule.push_back(*player_q_iter);
   m_player_update_schedule.erase(player_q_iter);
   // run the iteration
   _iterate(std::move(player_to_update));
   // and increment our iteration counter
   m_iteration++;
   // we always return the current policy. This way the user can decide whether to store the n-th
   // iteraton's policies user-side or not.
   return &m_curr_policy;
}

template <
   CFRConfig cfr_config,
   typename Env,
   typename Policy,
   typename DefaultPolicy,
   typename AveragePolicy >
// clang-format off
 requires
    concepts::fosg< Env >
    && fosg_traits_partial_match< Policy, Env >::value
    && concepts::state_policy<
          Policy,
          typename fosg_auto_traits< Env >::info_state_type,
          typename fosg_auto_traits< Env >::observation_type
       >
   && concepts::state_policy<
         AveragePolicy,
         typename fosg_auto_traits< Env >::info_state_type,
         typename fosg_auto_traits< Env >::observation_type
      >
    && concepts::default_state_policy<
          DefaultPolicy,
          typename fosg_auto_traits< Env >::info_state_type,
          typename fosg_auto_traits< Env >::observation_type
       >
     // clang-format on
     void VanillaCFR< cfr_config, Env, Policy, DefaultPolicy, AveragePolicy >::_iterate(
        std::optional< Player > player_to_update)
{
   if(m_iteration == 0) {
      _make_root_node();
      _first_traversal();
   } else {
      _traversal();
   }
   if(m_update_stack.empty()) {
      throw std::logic_error(
         "The update queue is empty. It should have been filled by previous tree traversals.");
   }
   _update_queued_nodes(player_to_update);
}

template <
   CFRConfig cfr_config,
   typename Env,
   typename Policy,
   typename DefaultPolicy,
   typename AveragePolicy >
// clang-format off
 requires
    concepts::fosg< Env >
    && fosg_traits_partial_match< Policy, Env >::value
    && concepts::state_policy<
          Policy,
          typename fosg_auto_traits< Env >::info_state_type,
          typename fosg_auto_traits< Env >::observation_type
       >
   && concepts::state_policy<
         AveragePolicy,
         typename fosg_auto_traits< Env >::info_state_type,
         typename fosg_auto_traits< Env >::observation_type
      >
    && concepts::default_state_policy<
          DefaultPolicy,
          typename fosg_auto_traits< Env >::info_state_type,
          typename fosg_auto_traits< Env >::observation_type
       >
     // clang-format on
     void
     VanillaCFR< cfr_config, Env, Policy, DefaultPolicy, AveragePolicy >::update_regret_and_policy(
        VanillaCFR::cfr_node_type& node,
        Player player)
{
   double reach_prob = reach_probability(node);
   auto& curr_state_policy = fetch_policy< true >(player, node);
   auto& avg_state_policy = fetch_policy< false >(player, node);
   ranges::for_each(
      _child_collector(node, [&](cfr_node_type* child) { return child->value(player); }),
      [&](const auto& action_value_pair) {
         const auto& [action, action_value] = action_value_pair;
         node.regret(action) += cf_reach_probability(node, reach_prob, player)
                                * (action_value - node.value(player));
         avg_state_policy[action] += reach_prob * curr_state_policy[action];
      });
   regret_matching(curr_state_policy, node.regret());
}

template <
   CFRConfig cfr_config,
   typename Env,
   typename Policy,
   typename DefaultPolicy,
   typename AveragePolicy >
// clang-format off
 requires
    concepts::fosg< Env >
    && fosg_traits_partial_match< Policy, Env >::value
    && concepts::state_policy<
          Policy,
          typename fosg_auto_traits< Env >::info_state_type,
          typename fosg_auto_traits< Env >::observation_type
       >
   && concepts::state_policy<
         AveragePolicy,
         typename fosg_auto_traits< Env >::info_state_type,
         typename fosg_auto_traits< Env >::observation_type
      >
    && concepts::default_state_policy<
          DefaultPolicy,
          typename fosg_auto_traits< Env >::info_state_type,
          typename fosg_auto_traits< Env >::observation_type
       >
     // clang-format on
     double VanillaCFR< cfr_config, Env, Policy, DefaultPolicy, AveragePolicy >::reach_probability(
        const cfr_node_type& node) const
{
   auto values_view = node.reach_probability() | ranges::views::values;
   return std::reduce(values_view.begin(), values_view.end(), 1, std::multiplies{});
}

template <
   CFRConfig cfr_config,
   typename Env,
   typename Policy,
   typename DefaultPolicy,
   typename AveragePolicy >
// clang-format off
 requires
    concepts::fosg< Env >
    && fosg_traits_partial_match< Policy, Env >::value
    && concepts::state_policy<
          Policy,
          typename fosg_auto_traits< Env >::info_state_type,
          typename fosg_auto_traits< Env >::observation_type
       >
   && concepts::state_policy<
         AveragePolicy,
         typename fosg_auto_traits< Env >::info_state_type,
         typename fosg_auto_traits< Env >::observation_type
      >
    && concepts::default_state_policy<
          DefaultPolicy,
          typename fosg_auto_traits< Env >::info_state_type,
          typename fosg_auto_traits< Env >::observation_type
       >
     // clang-format on
     double
     VanillaCFR< cfr_config, Env, Policy, DefaultPolicy, AveragePolicy >::cf_reach_probability(
        const VanillaCFR::cfr_node_type& node,
        const Player& player) const
{
   return cf_reach_probability(node, reach_probability(node), player);
}

template <
   CFRConfig cfr_config,
   typename Env,
   typename Policy,
   typename DefaultPolicy,
   typename AveragePolicy >
// clang-format off
 requires
    concepts::fosg< Env >
    && fosg_traits_partial_match< Policy, Env >::value
    && concepts::state_policy<
          Policy,
          typename fosg_auto_traits< Env >::info_state_type,
          typename fosg_auto_traits< Env >::observation_type
       >
   && concepts::state_policy<
         AveragePolicy,
         typename fosg_auto_traits< Env >::info_state_type,
         typename fosg_auto_traits< Env >::observation_type
      >
    && concepts::default_state_policy<
          DefaultPolicy,
          typename fosg_auto_traits< Env >::info_state_type,
          typename fosg_auto_traits< Env >::observation_type
       >
     // clang-format on
     double
     VanillaCFR< cfr_config, Env, Policy, DefaultPolicy, AveragePolicy >::cf_reach_probability(
        const VanillaCFR::cfr_node_type& node,
        double reach_prob,
        const Player& player) const
{
   // remove the players contribution to the likelihood
   return reach_prob / node.reach_probability(player);

}

}  // namespace nor::rm

#endif  // NOR_VANILLA_HPP
