
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
#include "rm.hpp"

namespace nor::rm {

struct CFRConfig {
   bool alternating_updates = true;
   bool store_public_states = false;
};

/**
 * A (Vanilla) Counterfactual Regret Minimization algorithm class following the
 * terminology of the Factored-Observation Stochastic Games (FOSG) formulation.
 *
 * The implementation follows the algorithm detail of @cite Neller2013
 *
 *
 * @tparam Env, the environment/game type to run VanillaCFR on.
 *
 */
template <
   CFRConfig cfr_config,
   typename Env,
   typename Policy,
   typename DefaultPolicy,
   typename AveragePolicy = Policy >
requires concepts::vanilla_cfr_requirements< Env, Policy, DefaultPolicy, AveragePolicy >
class VanillaCFR {
   ////////////////////////////
   /// API: public typedefs ///
   ////////////////////////////
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
   using node_data_type = CFRNodeData<
      action_type,
      info_state_type,
      std::conditional_t< cfr_config.store_public_states, public_state_type, utils::empty > >;
#else
   using node_data_type = CFRNodeData<
      action_type,
      info_state_type,
      std::conditional_t< cfr_config.store_public_states, public_state_type, NEW_EMPTY_TYPE > >;
#endif
   using game_tree_type = forest::GameTree< env_type, node_data_type >;
   using node_type = typename game_tree_type::node_type;

   ////////////////////
   /// Constructors ///
   ////////////////////

   VanillaCFR(
      Env game,
      uptr< world_state_type > root_state,
      Policy policy = Policy(),
      DefaultPolicy default_policy = DefaultPolicy(),
      AveragePolicy avg_policy = AveragePolicy())
      // clang-format off
      requires
         all_predicate_v<
            std::is_copy_constructible,
            Policy,
            AveragePolicy >
       // clang-format on
       :
       m_env(std::move(game)),
       m_game_tree(m_env, std::move(root_state)),
       m_curr_policy(),
       m_avg_policy(),
       m_default_policy(std::move(default_policy))
   {
      _assert_sequential_game();
      for(auto player : game.players()) {
         m_curr_policy[player] = policy;
         m_avg_policy[player] = avg_policy;
      }
      _init_player_update_schedule();
      m_game_tree.build([&]< typename... Args >(Args && ... args) {
         return data_extractor(std::forward< Args >(args)...);
      });
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
          std::move(env),
          std::make_unique< world_state_type >(env.initial_world_state()),
          std::move(policy),
          std::move(default_policy),
          std::move(avg_policy))
   {
   }

   VanillaCFR(
      Env game,
      uptr< world_state_type > root_state,
      std::map< Player, Policy > policy,
      std::map< Player, AveragePolicy > avg_policy,
      DefaultPolicy default_policy = DefaultPolicy())
       : m_env(std::move(game)),
         m_game_tree(m_env, std::move(root_state)),
         m_curr_policy(std::move(policy)),
         m_avg_policy(std::move(avg_policy)),
         m_default_policy(std::move(default_policy))
   {
      _assert_sequential_game();
      m_game_tree.build([&]< typename... Args >(Args && ... args) {
         return data_extractor(std::forward< Args >(args)...);
      });
   }

   ////////////////////////////////////
   /// API: public member functions ///
   ////////////////////////////////////

  public:
   /**
    * @brief executes n iterations of the VanillaCFR algorithm in unrolled form (no recursion).
    *
    * The decision for doing alternating updates or simultaneous updates happens at compile time via
    * the cfr config. This optimizes some unncessary repeated if-branching away at the cost of
    * higher maintenance. The user can also decide whether to store the public state at each node
    * within the CFR config! This can save some memory, since the public states are not needed,
    * unless one wants to e.g. perform analysis.
    *
    * By returning a pointer to the constant current policy after the n iterations, the user can
    * select to store a copy of the policy at each step themselves.
    *
    * @param n_iterations the number of iterations to perform.
    * @return a pointer to the constant current policy after the update
    */
   auto const* iterate(size_t n_iterations);
   /**
    * @brief executes one iteration of alternating updates vanilla cfr.
    *
    * This overload only participates if the config defined alternating updates to be made.
    *
    * By returning a pointer to the constant current policy after the n iterations, the user can
    * select to store a copy of the policy at each step themselves.
    *
    * @param player_to_update the optional player to update this iteration. If not provided, the
    * function will continue with the regular update cycle. By providing this parameter the user can
    * expressly modify the update cycle to even update individual players multiple times in a row.
    * @return a pointer to the constant current policy after the update
    */
   auto const* iterate(std::optional< Player > player_to_update = std::nullopt) requires(
      cfr_config.alternating_updates);

   /**
    * @brief gets the current or average state policy of a node
    *
    * Depending on the template parameter either the current policy (true) or the average policy
    * (false) over the last iterations is queried. If the current node has not been emplaced in the
    * policy yet, then the default policy will be asked to provide an initial entry.
    * @tparam current_policy switch for querying either the current (true) or the average (false)
    * policy.
    * @param player the player whose policy is to be fetched
    * @param node the game node at which the policy is required
    * @return action_policy_type the player's state policy (distribution) over all actions at this
    * node
    */
   template < bool current_policy >
   auto& fetch_policy(Player player, const node_type& node);

   /**
    * The const overload of this function (@see fetch_policy) will not insert or return a default
    * policy if one wasn't found, but throw an error instead.
    */
   template < bool current_policy >
   auto& fetch_policy(Player player, const node_type& node) const;

   /**
    *
    * @param action the action to select at this node
    * @return
    */
   template < bool current_policy >
   inline auto& fetch_policy(Player player, const node_type& node, const action_type& action)
   {
      return fetch_policy< current_policy >(player, node)[action];
   }
   /**
    * Const overload for action selection at the node.
    */
   template < bool current_policy >
   inline auto& fetch_policy(Player player, const node_type& node, const action_type& action) const
   {
      return fetch_policy< current_policy >(player, node)[action];
   }

   /**
    * @brief updates the regret and policy tables of the node with the state-values. Then performs
    * regret-matching.
    *
    * The method implements lines 21-25 of @cite Neller2013.
    *
    * @param node the node to update the values at
    * @param player the player whose values are to be updated
    */
   void update_regret_and_policy(const VanillaCFR::node_type& node);

   /**
    * @brief computes the reach probability of the node.
    *
    * Since each player's compounding likelihood contribution is stored in the nodes themselves, the
    * actual computation is nothing more than merely multiplying all player's individual
    * contributions.
    * @param node the node whose reach probability to provide
    * @return the reach probability of the nde
    */
   inline double reach_probability(const node_data_type& node_data) const
   {
      auto values_view = node_data.reach_probability_contrib() | ranges::views::all;
      return std::reduce(values_view.begin(), values_view.end(), 1, std::multiplies{});
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
   inline double cf_reach_probability(const node_data_type& node_data, const Player& player) const
   {
      return cf_reach_probability(node_data, reach_probability(node_data), player);
   }
   /**
    * @brief computes the counterfactual reach probability of the player for this node.
    *
    * The method overload is useful if the reach probability has already been computed and needs to
    * be simply scaled by removing the player's contribution.
    * @param node the node at which the counterfactual reach probability is to be computed
    * @param player the player for which the value is computed
    * @return the counterfactual reach probability
    */
   inline double cf_reach_probability(
      const node_data_type& node_data,
      double reach_prob,
      const Player& player) const
   {  // remove the players contribution to the likelihood
      return reach_prob / node_data.reach_probability_contrib(player);
   }

   /// getter methods for stored data

   auto iteration() const { return m_iteration; }
   [[nodiscard]] const auto& game_tree() const { return m_game_tree; }
   [[nodiscard]] const auto& policy() const { return m_curr_policy; }
   [[nodiscard]] const auto& average_policy() const { return m_avg_policy; }

   ////////////////////////////////
   /// private member functions ///
   ////////////////////////////////

  private:
   /**
    * @brief The internal vanilla cfr iteration routine.
    *
    * This function sets the algorithm scheme for vanilla cfr by delegating to the right functions.
    *
    * @param player_to_update the player to update (optionally). If not given either the next player
    * to update from the schedule is taken (alternating updates) or every player is updated at the
    * same time (simultaneous updates).
    */
   void _iterate(std::optional< Player > player_to_update);

   /**
    * @brief Convenience method to iterate over the child as map with transformed values.
    *
    * The function can be used to e.g. extract the regret values from each child or their state
    * values. It provides a zip iterator that can be decomposed via structured binding into the
    * relevant pieces.
    *
    * @tparam Functor the callable type that can be invoked with a node pointer. Inferred from
    * passed functor object.
    * @param node a reference to the node whose children are to be iterated over.
    * @param f the functor object to apply on the child node.
    * @return
    */
   template < std::invocable< node_type* > Functor >
   auto _child_collector(const node_type& node, Functor f)
   {
      return ranges::views::zip(
         node.children | ranges::views::keys,
         node.children | ranges::views::values | ranges::views::transform(f));
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

   uptr< node_data_type > data_extractor(
      const node_type& node,
      [[maybe_unused]] world_state_type* /*unused*/,
      world_state_type& curr_worldstate,
      const node_data_type* parent_node_data,
      const std::map< Player, info_state_type >& curr_infostates,
      const typename node_data_type::public_state_type& curr_publicstate,
      const std::optional< action_type >& action_from_parent)
   {
      if(node.category == forest::NodeCategory::chance) {
         // if this is a chance node we don't store any data for it. This data is only needed for
         // when actual regret and value computations should be possible
         return nullptr;
      }
      auto curr_player = m_env.active_player(curr_worldstate);
      auto parent_node = node.parent;
      if(parent_node_data == nullptr) {
         // in this position we are either at the root or the parent was a chance node

         auto init_root = [&]() {
            return std::make_unique< node_data_type >(
               m_env.player_count(),
               curr_player,
               curr_infostates.at(curr_player),
               curr_publicstate,
               /*reach_probs=*/std::vector< double >(size_t(m_env.players().size()), 1.));
         };
         if(node.parent == nullptr) {
            // we are currently at the root and need to initialize the root node's data
            return init_root();
         } else {
            parent_node = m_game_tree.nonchance_parent(&node);
            if(parent_node == nullptr) {
               // there is no nonchance parent of the current node all the way up to the root
               // so we initialize this as if it was a root node, i.e. reach probabilities are 1
               // (only with the observed chance actions)
               return init_root();
            }
            // reassign the correct node data instead then
            parent_node_data = m_game_tree.data(*parent_node);
         }
      }
      // from here on out we know that the curr node is not a CHANCE node and that the parent node
      // data ptr is defnitely not a nullptr. We al

      // Update the parent player's reach probability of this node by its current policy of
      // playing the action that lead to it.
      auto parent_player = parent_node_data->player();
      auto reach_probability_contrib = parent_node_data->reach_probability_contrib();
      reach_probability_contrib[static_cast< uint8_t >(parent_player)] *= fetch_policy< true >(
         parent_player, *parent_node, action_from_parent.value());

      // return a new unique node data pointer
      return std::make_unique< node_data_type >(
         m_env.player_count(),
         curr_player,
         curr_infostates.at(curr_player),
         curr_publicstate,
         std::move(reach_probability_contrib));
   }

   /**
    * @brief traverses the game tree and fills the nodes with current policy weighted value updates.
    *
    * This function is the regular traversal function to call on iteration i > 0, after the nodes
    * have been emplaced by @see _first_traversal.
    */
   void _traversal();

   /**
    * @brief updates the value, regret, and strategy of each node
    *
    * @param player_to_update
    */
   void _update_queued_nodes(std::optional< Player > player_to_update);

   /**
    * @brief emplaces the environment rewards for a terminal state and stores them in the node.
    *
    * No terminality checking is done within this method! Hence only call this method if you are
    * already certain that the node is a terminal one. Whether the environment rewards for
    * non-terminal states would be problematic is dependant on the environment.
    * @param[in] terminal_wstate the terminal state to collect rewards for.
    * @param[out] node the terminal node in which to store the reward values.
    */
   void collect_rewards(
      std::conditional_t<  // the fosg concept asserts a reward function taking world_state_type.
                           // But if it can be passed a const world state then do so instead
         concepts::has::method::reward< env_type, const world_state_type& >,
         const world_state_type&,
         world_state_type& > terminal_wstate,
      node_data_type& node) const;

   /**
    * @brief Cycles the update schedule by popping the next player to update and requeueing them as
    * last.
    *
    * The update schedule for alternative updates is a cycle P1-P2-...-PN. After each update the
    * schedule moves by returning the first player and reattaching it to the back of the queue. This
    * way every other player is pushed up by one position in the update queue.
    * @param player_to_update the optional player to update next. If not given, the next in line
    * will be chosen.
    * @return the player to update next.
    */
   Player _cycle_player_to_update(std::optional< Player > player_to_update = std::nullopt);

   ///////////////////////////////////////////
   /// private member variable definitions ///
   ///////////////////////////////////////////

  private:
   /// the environment object to maneuver the states with.
   env_type m_env;

   /// the game tree mapping information states to the associated game nodes.
   game_tree_type m_game_tree;
   /// the current policy $\pi^t$ that each player is following in this iteration (t).
   std::map< Player, Policy > m_curr_policy;
   /// the average policy table.
   std::map< Player, AveragePolicy > m_avg_policy;
   /// the fallback policy to use when the encountered infostate has not been obseved before
   default_policy_type m_default_policy;

   /// The update queue to use once the nodes have been filled from a tree traversal.
   /// We need to arrange a delayed update of tree nodes, since any node's values depend on the
   /// child values and the reach probabilities. Those values are found once the full traversal
   /// encounters terminal states (leaf nodes) whose values are then propagated back up with the
   /// relevant reach probability. The order in which this propragation occurs is crucial and thus
   /// defined by the update_queue. This queue needs to be a First-In-Last-Out structure, as it
   /// ensures we are updating the values from the leaves, which are inserted last, to the root of
   /// the tree, which is inserted first.
   std::stack< const node_type* > m_update_stack;

   /// the next player to update when doing alternative updates. Otherwise this member will be
   /// unused.
   std::deque< Player > m_player_update_schedule{};
   /// the number of iterations we have run so far.
   size_t m_iteration = 0;
};


///////////////////////////////////
/// member function definitions ///
///////////////////////////////////

template <
   CFRConfig cfr_config,
   typename Env,
   typename Policy,
   typename DefaultPolicy,
   typename AveragePolicy >
requires concepts::vanilla_cfr_requirements< Env, Policy, DefaultPolicy, AveragePolicy >
auto const* VanillaCFR< cfr_config, Env, Policy, DefaultPolicy, AveragePolicy >::iterate(
   size_t n_iters)
{
   for(auto iteration : ranges::views::iota(size_t(0), n_iters)) {
      LOGD2("Iteration number: ", iteration);
      if constexpr(cfr_config.alternating_updates) {
         _iterate(_cycle_player_to_update());
      } else {
         _iterate(std::nullopt);
      }
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
requires concepts::vanilla_cfr_requirements< Env, Policy, DefaultPolicy, AveragePolicy >
auto const* VanillaCFR< cfr_config, Env, Policy, DefaultPolicy, AveragePolicy >::iterate(
   std::optional< Player > player_to_update) requires(cfr_config.alternating_updates)
{
   // we assert here that the chosen player to update is not the chance player as is defined
   // by default. Seeing the chance player here inidcates that the player forgot to set the
   // plaeyr parameter with this rm config.
   if(player_to_update.value_or(Player::alex) == Player::chance) {
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
      ssout << player_to_update.value();
      ssout << " is not a member of the game's player list ";
      ssout << ranges::views::all(env_players) << ".";
      throw std::invalid_argument(ssout.str());
   }

   LOGD2("Iteration number: ", m_iteration);
   // run the iteration
   _iterate(_cycle_player_to_update(player_to_update));
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
requires concepts::vanilla_cfr_requirements< Env, Policy, DefaultPolicy, AveragePolicy >
void VanillaCFR< cfr_config, Env, Policy, DefaultPolicy, AveragePolicy >::_iterate(
   std::optional< Player > player_to_update)
{
   _traversal();
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
requires concepts::vanilla_cfr_requirements< Env, Policy, DefaultPolicy, AveragePolicy >
void VanillaCFR< cfr_config, Env, Policy, DefaultPolicy, AveragePolicy >::update_regret_and_policy(
   const VanillaCFR::node_type& node)
{
   auto& node_data = *(m_game_tree.data(node));
   Player player = node_data.player();
   double reach_prob = reach_probability(node_data);
   auto& curr_state_policy = fetch_policy< true >(player, node);
   auto& avg_state_policy = fetch_policy< false >(player, node);
   ranges::for_each(
      _child_collector(
         node, [&](node_type* child) { return m_game_tree.data(*child)->value(player); }),
      [&](const auto& action_value_pair) {
         const auto& [action, action_value] = action_value_pair;
         node_data.regret(action) += cf_reach_probability(node_data, reach_prob, player)
                                     * (action_value - node_data.value(player));
         avg_state_policy[action] += reach_prob * curr_state_policy[action];
      });
   regret_matching(curr_state_policy, node_data.regret());
}

template <
   CFRConfig cfr_config,
   typename Env,
   typename Policy,
   typename DefaultPolicy,
   typename AveragePolicy >
requires concepts::vanilla_cfr_requirements< Env, Policy, DefaultPolicy, AveragePolicy >
void VanillaCFR< cfr_config, Env, Policy, DefaultPolicy, AveragePolicy >::_traversal()
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
   std::stack< const node_type* > visit_stack;
   // copy the root state into the visitation stack
   visit_stack.emplace(&(std::as_const(m_game_tree).root_node()));

   while(not visit_stack.empty()) {
      // move the next tree node from the queue into a local variable
      auto curr_node = visit_stack.top();
      // remove this node from the visit stack
      visit_stack.pop();
      // fetch the node's data
      auto& curr_node_data = *(m_game_tree.data(*curr_node));
      // allocate the current player here to avoid reallocation for every action.
      Player curr_player = curr_node_data.player();

      for(const auto& [action, child_node] : curr_node->children) {
         // based on the type of node we encountered we will proceed differently
         if(child_node->category == forest::NodeCategory::terminal) {
            // at a terminal node we don't do anything further.
            continue;
         } else if(child_node->category == forest::NodeCategory::choice) {
            // Update the active player's reach probability of this node by its current policy of
            // playing the action that lead to it.
            m_game_tree.data(*child_node)
               ->reach_probability_contrib(curr_player) *= fetch_policy< true >(
               curr_player, *curr_node, action);
         }
         // We only reach here for a non-terminal child node!
         // append the child node to the queue. This way we further explore its child states
         // as reachable from the next possible actions.
         visit_stack.emplace(child_node);
      }
      // enqueue this node in the delayed update queue to ensure its regret and policy values are
      // being updated
      m_update_stack.push(curr_node);
   }
}

template <
   CFRConfig cfr_config,
   typename Env,
   typename Policy,
   typename DefaultPolicy,
   typename AveragePolicy >
requires concepts::vanilla_cfr_requirements< Env, Policy, DefaultPolicy, AveragePolicy >
void VanillaCFR< cfr_config, Env, Policy, DefaultPolicy, AveragePolicy >::_update_queued_nodes(
   std::optional< Player > player_to_update)
{
   while(not m_update_stack.empty()) {
      // pop the next node to update from the stack
      auto& node = *m_update_stack.top();
      m_update_stack.pop();
      node_data_type& node_data = *(m_game_tree.data(node));
      ranges::for_each(m_env.players(), [&](Player player) {
         // the value propagation works by specifying the last action in the list to be the
         // trigger. Upon trigger, this node calls upon all its children to send back
         // their respective values which will be factored into in the node's value by
         // weighting them by the current policy. This entire idea relies upon a POST-ORDER
         // DEPTH-FIRST GAME TREE traversal, i.e. if the nodes popped from the update stack are
         // not emplaced in the wrong order, then these values will be gibberish/an error might
         // occur.
         auto action_value_view = _child_collector(
            node, [&](node_type* child) { return m_game_tree.data(*child)->value(player); });
         // v <- v + pi_{player}(s, a) * v(s'(a))
         node_data.value(player) += std::transform_reduce(
            action_value_view.begin(),
            action_value_view.end(),
            0.,
            std::plus{},
            [&](const auto& action_value_pair) {
               // applied action is the action that lead to the child node!
               const auto& [applied_action, action_value] = action_value_pair;
               // pi_{player}(s, a) * v(s'(a))
               return fetch_policy< true >(player, node, applied_action) * action_value;
            });
      });

      if constexpr(cfr_config.alternating_updates) {
         // in alternating updates, we only update the regret and strategy if the current
         // player is the chosen player to update.
         if(node_data.player() == player_to_update.value_or(Player::chance)) {
            update_regret_and_policy(node);
         }
      } else {
         // if we do simultaenous updates, then we always update the regret and strategy
         // values of the node's active player.
         update_regret_and_policy(node);
      }
   }
}

template <
   CFRConfig cfr_config,
   typename Env,
   typename Policy,
   typename DefaultPolicy,
   typename AveragePolicy >
requires concepts::vanilla_cfr_requirements< Env, Policy, DefaultPolicy, AveragePolicy >
void VanillaCFR< cfr_config, Env, Policy, DefaultPolicy, AveragePolicy >::collect_rewards(
   std::conditional_t<
      concepts::has::method::reward< env_type, const world_state_type& >,
      const world_state_type&,
      world_state_type& > terminal_wstate,
   VanillaCFR::node_data_type& node_data) const
{
   if constexpr(nor::concepts::has::method::reward_multi< Env >) {
      // if the environment has a method for returning all rewards for given players at
      // once, then we will assume this is a more performant alternative and use it
      // instead (e.g. when it is costly to compute the reward of each player
      // individually).
      ranges::views::for_each(
         ranges::views::zip(m_env.players(), m_env.reward(m_env.players(), terminal_wstate)),
         [&value_map = node_data.value()](const auto& p_r_pair) {
            value_map.emplace(std::get< 0 >(p_r_pair), std::get< 1 >(p_r_pair));
         });
   } else {
      // otherwise we just loop over the per player reward method
      for(auto player : m_env.players()) {
         node_data.value(player) = m_env.reward(player, terminal_wstate);
      }
   }
}

template <
   CFRConfig cfr_config,
   typename Env,
   typename Policy,
   typename DefaultPolicy,
   typename AveragePolicy >
requires concepts::vanilla_cfr_requirements< Env, Policy, DefaultPolicy, AveragePolicy > Player
VanillaCFR< cfr_config, Env, Policy, DefaultPolicy, AveragePolicy >::_cycle_player_to_update(
   std::optional< Player > player_to_update)
{
   auto player_q_iter = std::find(
      m_player_update_schedule.begin(),
      m_player_update_schedule.end(),
      player_to_update.value_or(m_player_update_schedule.front()));
   m_player_update_schedule.push_back(*player_q_iter);
   Player next_to_update = *player_q_iter;
   m_player_update_schedule.erase(player_q_iter);
   return next_to_update;
}

template <
   CFRConfig cfr_config,
   typename Env,
   typename Policy,
   typename DefaultPolicy,
   typename AveragePolicy >
requires concepts::vanilla_cfr_requirements< Env, Policy, DefaultPolicy, AveragePolicy >
template < bool current_policy >
auto& VanillaCFR< cfr_config, Env, Policy, DefaultPolicy, AveragePolicy >::fetch_policy(
   Player player,
   const VanillaCFR::node_type& node)
{
   const auto& node_data = *(m_game_tree.data(node));
   const auto& infostate = node_data.infostate();
   if constexpr(current_policy) {
      auto& player_policy = m_curr_policy[player];
      auto found_action_policy = player_policy.find(infostate);
      if(found_action_policy == player_policy.end()) {
         return player_policy
            .emplace(
               infostate,
               m_default_policy[{
                  infostate, ranges::to< std::vector >(node.children | ranges::views::keys)}])
            .first->second;
      }
      return found_action_policy->second;
   } else {
      auto& player_policy = m_avg_policy[player];
      auto found_action_policy = player_policy.find(infostate);
      if(found_action_policy == player_policy.end()) {
         // the average policy is necessary to be 0 initialized on all unseen entries, since
         // these entries are to be updated cumulatively.
         typename AveragePolicy::action_policy_type default_avg_policy;
         for(const auto& action : node.children | ranges::views::keys) {
            default_avg_policy[action] = 0.;
         }
         return player_policy.emplace(infostate, default_avg_policy).first->second;
      }
      return found_action_policy->second;
   }
}

}  // namespace nor::rm

#endif  // NOR_VANILLA_HPP
