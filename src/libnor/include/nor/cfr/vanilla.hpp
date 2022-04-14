
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

#include "common/common.hpp"
#include "node.hpp"
#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "nor/policy.hpp"
#include "nor/type_defs.hpp"
#include "nor/utils/utils.hpp"
#include "rm.hpp"

namespace nor::rm {

struct CFRConfig {
   bool alternating_updates = true;
   bool store_public_states = false;
   bool store_world_states = false;
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
   using chance_policy_type = typename fosg_auto_traits< Env >::chance_distribution_type;
   /// the CFR Node type to store in the tree
   using node_data_type = CFRNodeData<
      action_type,
      info_state_type,
      std::conditional_t< cfr_config.store_world_states, world_state_type, utils::empty >,
      std::conditional_t< cfr_config.store_public_states, public_state_type, utils::empty > >;
   /// the game tree type specific to our environment
   using game_tree_type = forest::GameTree< env_type >;
   /// the node type used within the game tree
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
         m_curr_policy.emplace(player, policy);
         m_avg_policy.emplace(player, avg_policy);
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
          std::move(env),
          std::make_unique< world_state_type >(env.initial_world_state()),
          std::move(policy),
          std::move(default_policy),
          std::move(avg_policy))
   {
      _init_player_update_schedule();
   }

   VanillaCFR(
      Env game,
      uptr< world_state_type > root_state,
      std::unordered_map< Player, Policy > policy,
      std::unordered_map< Player, AveragePolicy > avg_policy,
      DefaultPolicy default_policy = DefaultPolicy())
       : m_env(std::move(game)),
         m_game_tree(m_env, std::move(root_state)),
         m_curr_policy(std::move(policy)),
         m_avg_policy(std::move(avg_policy)),
         m_default_policy(std::move(default_policy))
   {
      _assert_sequential_game();
      _init_player_update_schedule();
   }

   ////////////////////////////////////
   /// API: public member functions ///
   ////////////////////////////////////

  public:
   /**
    * @brief initializes the game tree.
    *
    * This method does an immediate full traversal of the entire game tree, may thus be very time
    * consuming.
    */
   inline void initialize()
   {
      auto child_hook = [&]< typename... Args >(Args && ... args)
      {
         return _child_node_hook(std::forward< Args >(args)...);
      };
      auto root_hook = [&]< typename... Args >(Args && ... args)
      {
         return _root_node_hook(std::forward< Args >(args)...);
      };

      m_game_tree.traverse(
         [&]< typename... Args >(Args && ... args) {
            return m_game_tree.traverse_all_actions(std::forward< Args >(args)...);
         },
         forest::TraversalHooks{
            .root_hook = std::move(root_hook), .child_hook = std::move(child_hook)});
   }

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
    * @param n_iters the number of iterations to perform.
    * @return a pointer to the constant current policy after the update
    */
   auto const* iterate(size_t n_iters);
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
    * @param current_policy switch for querying either the current (true) or the average (false)
    * policy.
    * @param player the player whose policy is to be fetched
    * @param node the game node at which the policy is required
    * @return action_policy_type the player's state policy (distribution) over all actions at this
    * node
    */
   auto& fetch_policy(bool current_policy, const node_type& node);

   /**
    * The const overload of this function (@see fetch_policy) will not insert or return a default
    * policy if one wasn't found, but throw an error instead.
    */
   auto& fetch_policy(bool current_policy, const node_type& node) const;

   /**
    *
    * @param action the action to select at this node
    * @return
    */
   inline auto& fetch_policy(bool current_policy, const node_type& node, const action_type& action)
   {
      return fetch_policy(current_policy, node)[action];
   }
   /**
    * Const overload for action selection at the node.
    */
   inline auto& fetch_policy(bool current_policy, const node_type& node, const action_type& action)
      const
   {
      return fetch_policy(current_policy, node)[action];
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
    * @brief Computes the value of either the current or average policy for a player
    * @param current_policy whether to use the current iteration's policy or the average policy
    * @param player the player whose value is to be computed
    * @return the floating point value of the game
    */
   double policy_value(bool current_policy, Player player);

   /// getter methods for stored data

   auto& data(const node_type& node) { return m_node_data.at(node.id); }
   [[nodiscard]] const auto& data(const node_type& node) const { return m_node_data.at(node.id); }
   [[nodiscard]] auto iteration() const { return m_iteration; }
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
    * @brief Convenience method to iterate over each child and extract data as per functor.
    *
    * The function can be used to e.g. extract the regret values from each child or their state
    * values. It provides a zip iterator that can be decomposed via structured binding into the
    * relevant pieces.
    *
    * @tparam Functor the callable type that can be invoked with a node pointer. Inferred from
    * passed functor object.
    * @param node a reference to the node whose children are to be iterated over.
    * @param f the functor object to apply on each child node.
    * @return view over (child, data) pairs
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
            if(player != Player::chance) {
               m_player_update_schedule.push_back(player);
            }
         }
      }
   }

   /**
    * @brief computes the reach probability of the node's data.
    *
    * Ease of access overload for the node data itself
    * @param node the node whose reach probability to provide
    * @return the reach probability of the nde
    */
   inline double _reach_probability(const node_data_type& node_data) const
   {
      return rm::reach_probability(node_data.reach_probability());
   }
   /**
    * @brief computes the counterfactual reach probability of the player for this node's data.
    *
    * The method delegates the actual computation to the overload with an already provided reach
    * probability.
    * @param node the node at which the counterfactual reach probability is to be computed
    * @param player the player for which the value is computed
    * @return the counterfactual reach probability
    */
   inline double _cf_reach_probability(const node_data_type& node_data, const Player& player) const
   {
      return rm::cf_reach_probability(node_data.reach_probability(), player);
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
   inline double _cf_reach_probability(
      const node_data_type& node_data,
      double reach_prob,
      const Player& player) const
   {  // remove the players contribution to the likelihood
      return rm::cf_reach_probability(node_data) / node_data.reach_probability(player);
   }

   node_type* _upstream_player_parent(const node_type& node, Player player)
   {
      if(node.parent == nullptr) {
         // we have reached the root
         return nullptr;
      }
      if(data(*(node.parent)).player() == player) {
         return node.parent;
      }
      return _upstream_player_parent(*(node.parent), player);
   }

   void _root_node_hook(const node_type& root_node, world_state_type* root_worldstate);

   /**
    *
    * @param node
    * @param parent_worldstate
    * @param curr_worldstate
    * @param curr_infostates
    * @param curr_publicstate
    * @param action_from_parent
    */
   void _child_node_hook(
      const node_type& node,
      const auto* action_from_parent_ptr,
      world_state_type* parent_worldstate,
      world_state_type* curr_worldstate);

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
   void _collect_rewards(
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
   /// the paired node data vector holds at entry i the data for node i
   std::unordered_map< size_t, node_data_type > m_node_data;
   /// the current policy $\pi^t$ that each player is following in this iteration (t).
   std::unordered_map< Player, Policy > m_curr_policy;
   /// the average policy table. The values stored in this table are the UNNORMALIZED average state
   /// policies. This means that the state policy p(s, . ) for a given info state s needs to
   /// normalize its probabilities p(s, . ) by \sum_a p(s,a) when used for evaluation.
   std::unordered_map< Player, AveragePolicy > m_avg_policy;
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
///
//////
/////////
////////////
///////////////
//////////////////
/////////////////////
////////////////////////
///////////////////////////
//////////////////////////////
/////////////////////////////////
///////////////////////////////////
///                              ///
/// member function definitions  ///
///                              ///
///////////////////////////////////
/////////////////////////////////
//////////////////////////////
///////////////////////////
////////////////////////
/////////////////////
//////////////////
///////////////
////////////
/////////
//////
///

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
   for([[maybe_unused]] auto _ : ranges::views::iota(size_t(0), n_iters)) {
      LOGD2("Iteration number: ", m_iteration);
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
   // player parameter with this rm config.
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
   auto& node_data = data(node);
   Player player = node_data.player();
   double player_reach_prob = node_data.reach_probability(player);
   auto& curr_state_policy = fetch_policy(true, node);
   auto& avg_state_policy = fetch_policy(false, node);

   for(const auto& [action_variant, action_value] :
       _child_collector(node, [&](node_type* child) { return data(*child).value(player); })) {
      const auto& action = std::get< action_type >(action_variant);
      //      if constexpr(requires { m_env.tiny_repr(*node_data.worldstate()); }) {
      //         //         LOGD2("State rep", m_env.tiny_repr(*node_data.worldstate()));
      //         if(m_env.tiny_repr(*node_data.worldstate()) == "queen-king-check") {
      //            //         if(player == Player::bob) {
      //            LOGD2("State rep", m_env.tiny_repr(*node_data.worldstate()));
      //            LOGD2(
      //               "Player - Action",
      //               std::string(common::to_string(player)) + " - "
      //                  + std::string(common::to_string(action)));
      //            LOGD2("CF Reach probablity", _cf_reach_probability(node_data, player));
      //            LOGD2("Reach probablity keys",
      //            ranges::views::keys(node_data.reach_probability())); LOGD2("Reach probablity
      //            vals", ranges::views::values(node_data.reach_probability())); LOGD2("Node
      //            value", node_data.value(player)); LOGD2("Action value", action_value);
      //            LOGD2("Action Policy", ranges::views::keys(curr_state_policy));
      //            LOGD2("Action Policy", ranges::views::values(curr_state_policy));
      //            LOGD2(
      //               "Regret increment",
      //               _cf_reach_probability(node_data, player) * (action_value -
      //               node_data.value(player)));
      //            LOGD2("AVG policy update", player_reach_prob * curr_state_policy[action]);
      //            LOGD2(
      //               "New AVG policy",
      //               avg_state_policy[action] + player_reach_prob * curr_state_policy[action]);
      //         }
      //      }

      node_data.regret(action) += _cf_reach_probability(node_data, player)
                                  * (action_value - node_data.value(player));
      //      LOGD2("New regret", node_data.regret(action));
      avg_state_policy[action] += player_reach_prob * curr_state_policy[action];
   }
   regret_matching(curr_state_policy, node_data.regret());

   //   LOGD2("Action Policy after", ranges::views::keys(curr_state_policy));
   //   LOGD2("Action Policy after", ranges::views::values(curr_state_policy));
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
   auto child_hook = [&](
                        const node_type& child_node,
                        const typename node_type::action_variant_type* action_from_parent,
                        world_state_type*,
                        world_state_type*) {
      // allocate the current player here to avoid reallocation for every action.
      Player parent_player = data(*child_node.parent).player();
      if(parent_player != Player::chance) {
         // the chance distribution is assumed to be fixed, hence never needs to be updated

         //         if(m_env.tiny_repr(*data(child_node).worldstate()) == "queen-king-check") {
         //            LOGD2("State rep", m_env.tiny_repr(*data(child_node).worldstate()));
         //            LOGD2("RP pre update",
         //            ranges::views::keys(data(child_node).reach_probability())); LOGD2("RP pre
         //            update", ranges::views::values(data(child_node).reach_probability()));
         //         }

         auto parent_rp = data(*child_node.parent).reach_probability(parent_player);
         auto new_policy_prob = fetch_policy(
            true, *child_node.parent, std::get< action_type >(*action_from_parent));
         data(child_node).reach_probability(parent_player) = parent_rp * new_policy_prob;

         //         if(m_env.tiny_repr(*data(child_node).worldstate()) == "queen-king-check") {
         //            LOGD2("RP post update",
         //            ranges::views::keys(data(child_node).reach_probability())); LOGD2("RP post
         //            update", ranges::views::values(data(child_node).reach_probability()));
         //         }
      }
   };
   auto post_child_hook = [&](const node_type& node, auto&&...) {
      // enqueue this node in the delayed update queue to ensure its regret and policy values are
      // being updated. We only need choice nodes here, since the values of terminal nodes are never
      // updated and neither those of chance nodes
      if(node.category == forest::NodeCategory::choice) {
         m_update_stack.push(&node);
      }
   };
   m_game_tree.traverse(
      [&](auto&&... args) {
         return m_game_tree.traverse_all_actions(std::forward< decltype(args) >(args)...);
      },
      forest::TraversalHooks{
         .child_hook = std::move(child_hook), .post_child_hook = std::move(post_child_hook)},
      /*traverse_via_worldstates=*/false);
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
#ifndef NDEBUG
      // any node popped here has to be a choice node, otherwise: logical error!
      if(node.category != forest::NodeCategory::choice) {
         throw std::logic_error("Non-choice node emplaced in the update-stack.");
      }
#endif
      node_data_type& node_data = data(node);
      for(auto player : m_env.players()) {
         if(player == Player::chance) {
            continue;
         }
         // the value propagation works by specifying the last action in the list to be the
         // trigger. Upon trigger, this node calls upon all its children to send back
         // their respective values which will be factored into in the node's value by
         // weighting them by the current policy. This entire idea relies upon a POST-ORDER
         // DEPTH-FIRST GAME TREE traversal, i.e. if the nodes popped from the update stack are
         // not emplaced in the wrong order, then these values will be gibberish/an error might
         // occur.
         auto action_value_view = _child_collector(
            node, [&](node_type* child) { return data(*child).value(player); });
         // v <- \sum_a pi_{player}(s, a) * v_{player}(s' | s, a)
         node_data.value(player) = std::transform_reduce(
            action_value_view.begin(),
            action_value_view.end(),
            0.,
            std::plus{},
            [&](const auto& action_value_pair) {
               // applied action is the action that lead to the child node! The value is the one we
               // queryied for in _child_collector
               const auto& [applied_action, action_value] = action_value_pair;
               // the update rule for collecting the value from child nodes is:
               //    pi_{active_player}(s, a) * v_{player}(s'(a))
               // Note, that we always have to take the policy of the active player at this node,
               // but we multiply it onto the value associated with the current player we iterate
               // for! The value herein reflects the player's perspective, while the policy
               // probabilities simply decide how likely it is that the active player chooses this
               // node. If the active player is also the current player we iterate for, then this is
               // obviously the choice that player is making. For the other players, this likelihood
               // might as well be considered random decisions by the environment.
               auto action_prob = fetch_policy(true, node, std::get< action_type >(applied_action));
               return action_prob * action_value;
            });
      }

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
void VanillaCFR< cfr_config, Env, Policy, DefaultPolicy, AveragePolicy >::_collect_rewards(
   std::conditional_t<
      concepts::has::method::reward< env_type, const world_state_type& >,
      const world_state_type&,
      world_state_type& > terminal_wstate,
   node_data_type& node_data) const
{
   if constexpr(nor::concepts::has::method::reward_multi< Env >) {
      // if the environment has a method for returning all rewards for given players at
      // once, then we will assume this is a more performant alternative and use it
      // instead (e.g. when it is costly to compute the reward of each player
      // individually).
      auto non_chance_filter = [](Player player) { return player != Player::chance; };
      ranges::views::for_each(
         m_env.players() | ranges::views::filter(non_chance_filter),
         [&, &value_map = node_data.value()](Player player) {
            value_map.emplace(player, m_env.reward(m_env.players(), terminal_wstate));
         });
   } else {
      // otherwise we just loop over the per player reward method
      for(auto player : m_env.players()) {
         if(player != Player::chance)
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
   Player next_to_update = *player_q_iter;
   m_player_update_schedule.erase(player_q_iter);
   m_player_update_schedule.push_back(next_to_update);
   return next_to_update;
}

template <
   CFRConfig cfr_config,
   typename Env,
   typename Policy,
   typename DefaultPolicy,
   typename AveragePolicy >
requires concepts::vanilla_cfr_requirements< Env, Policy, DefaultPolicy, AveragePolicy >
void VanillaCFR< cfr_config, Env, Policy, DefaultPolicy, AveragePolicy >::_child_node_hook(
   const node_type& node,
   const auto* action_from_parent_ptr,
   world_state_type* parent_worldstate,
   world_state_type* curr_worldstate)
{
   auto child_player = m_env.active_player(*curr_worldstate);
   auto parent_node = node.parent;
   auto& parent_node_data = data(*parent_node);

   auto parent_player = parent_node_data.player();
   auto reach_probability = parent_node_data.reach_probability();

   if(parent_player == Player::chance) {
      // for the chance player we need to query the environment itself for the likelihood
      // of the action. This is a willful design to later down the line enable the usage of
      // sample based chance actions whose probability may not easily accessible and costly
      // to generate and store.
      if constexpr(concepts::has::method::chance_probability<
                      env_type,
                      world_state_type,
                      typename fosg_auto_traits< env_type >::chance_outcome_type >) {
         reach_probability[Player::chance] *= m_env.chance_probability(std::pair{
            *parent_worldstate,
            std::get< typename fosg_auto_traits< env_type >::chance_outcome_type >(
               *action_from_parent_ptr)});
      } else {
         // if the environment is not safely deterministic then it will need to provide at
         // least a dummy implemenation (e.g. deterministic game, but derived from common
         // interface for all kinds of games as in open_spiel)
         throw std::logic_error(
            "A chance player's node was encountered, but the environment has no "
            "chance_probability member function.");
      }
   } else {
      // for a non chance player we simply ask for its policy to have played this action and
      // multiply it onto the existing contribution stream of the player. We can be sure here that
      // the action is of action_type
      reach_probability[parent_player] *= fetch_policy(
         true, *parent_node, std::get< action_type >(*action_from_parent_ptr));
   }
   auto& node_data = m_node_data
                        .emplace(
                           node.id,
                           node_data_type{
                              child_player,
                              /*the next infostates are added via an immediately executed lambda.
                                 This allows to see directly how the next infostate is computed and
                                 passed to the emplace method*/
                              [&] {
                                 std::unordered_map< Player, info_state_type > next_infostates;
                                 for(auto [player, next_infostate] :  // we have to copy the istate
                                     data(*(node.parent)).infostates()) {
                                    next_infostate.append([&, player = player] {
                                       if constexpr(concepts::deterministic_fosg< Env >) {
                                          return m_env.private_observation(
                                             player,
                                             std::get< action_type >(*action_from_parent_ptr));
                                       } else {
                                          return std::visit(
                                             common::Overload{[&](const auto& any_action) {
                                                return m_env.private_observation(
                                                   player, any_action);
                                             }},
                                             *action_from_parent_ptr);
                                       }
                                    }());
                                    next_infostate.append(
                                       m_env.private_observation(player, *curr_worldstate));
                                    next_infostates.emplace(player, std::move(next_infostate));
                                 }
                                 return next_infostates;
                              }(),
                              curr_worldstate,
                              /*the next public state is computed via an immediately executed lambda
                                 function. This allows us to differentiate between storing public
                                 state or not storing it, and executing the right path on the fly*/
                              [&] {
                                 if constexpr(cfr_config.store_public_states) {
                                    auto next_publicstate = parent_node_data.publicstate();
                                    next_publicstate.append([&] {
                                       if constexpr(concepts::deterministic_fosg< Env >) {
                                          return m_env.public_observation(
                                             std::get< action_type >(*action_from_parent_ptr));
                                       } else {
                                          return std::visit(
                                             common::Overload{[&](const auto& any_action) {
                                                return m_env.public_observation(any_action);
                                             }},
                                             *action_from_parent_ptr);
                                       }
                                    }());
                                    next_publicstate.append(
                                       m_env->public_observation(*curr_worldstate));
                                    return next_publicstate;
                                 } else {
                                    return typename node_data_type::public_state_type{};
                                 }
                              }(),
                              std::move(reach_probability)})
                        .first->second;
   // terminal states need their rewards set so that the upstream nodes can update their
   // values later on
   if(node.category == forest::NodeCategory::terminal) {
      _collect_rewards(*curr_worldstate, node_data);
   }
}
template <
   CFRConfig cfr_config,
   typename Env,
   typename Policy,
   typename DefaultPolicy,
   typename AveragePolicy >
requires concepts::vanilla_cfr_requirements< Env, Policy, DefaultPolicy, AveragePolicy >
void VanillaCFR< cfr_config, Env, Policy, DefaultPolicy, AveragePolicy >::_root_node_hook(
   const node_type& root_node,
   world_state_type* root_worldstate)
{
   auto curr_player = m_env.active_player(*root_worldstate);
   // we are currently at the root and need to initialize the root node's data
   m_node_data.emplace(
      root_node.id,
      node_data_type{
         curr_player,
         [&] {
            std::unordered_map< Player, info_state_type > init_infostates;
            for(auto player : m_env.players()) {
               if(player != Player::chance) {
                  auto& player_infostate = init_infostates.emplace(player, info_state_type{player})
                                              .first->second;
                  // the world state is filled with each player's private observation of the initial
                  // world state
                  player_infostate.append(m_env.private_observation(player, *root_worldstate));
               }
            }
            return init_infostates;
         }(),
         root_worldstate,
         [&] {
            auto init_publicstate = typename node_data_type::public_state_type{};
            if constexpr(cfr_config.store_public_states) {
               init_publicstate.append(m_env.public_observation(*root_worldstate));
            }
            return init_publicstate;
         }(),
         [&] {
            std::unordered_map< Player, double > rp_map;
            for(auto player : m_env.players()) {
               rp_map[player] = 1.;
            }
            return rp_map;
         }()});
}

template <
   CFRConfig cfr_config,
   typename Env,
   typename Policy,
   typename DefaultPolicy,
   typename AveragePolicy >
requires concepts::vanilla_cfr_requirements< Env, Policy, DefaultPolicy, AveragePolicy >
auto& VanillaCFR< cfr_config, Env, Policy, DefaultPolicy, AveragePolicy >::fetch_policy(
   bool current_policy,
   const node_type& node)
{
   const auto& infostate = data(node).infostate();
   if(current_policy) {
      auto& player_policy = m_curr_policy[data(node).player()];
      auto found_action_policy = player_policy.find(infostate);
      if(found_action_policy == player_policy.end()) {
         auto actions = ranges::to< std::vector< action_type > >(
            // the actual rangev3 equivalents cannot compile here for some obscure reason
            node.children | ranges::cpp20::views::keys
            | ranges::cpp20::views::transform([](const auto& action_or_chance_outcome_variant) {
                 // we know that no chance outcome could be demanded here so we ask for the
                 // action type
                 return std::get< action_type >(action_or_chance_outcome_variant);
              }));
         return player_policy.emplace(infostate, m_default_policy[{infostate, actions}])
            .first->second;
      }
      return found_action_policy->second;
   } else {
      auto& player_policy = m_avg_policy[data(node).player()];
      auto found_action_policy = player_policy.find(infostate);
      if(found_action_policy == player_policy.end()) {
         // the average policy is necessary to be 0 initialized on all unseen entries, since
         // these entries are to be updated cumulatively.
         typename AveragePolicy::action_policy_type default_avg_policy;
         for(const auto& action : node.children | ranges::views::keys) {
            // we know that no chance outcome could be demanded here
            // so we ask for the action type to be returned
            default_avg_policy[std::get< action_type >(action)] = 0.;
         }
         return player_policy.emplace(infostate, default_avg_policy).first->second;
      }
      return found_action_policy->second;
   }
}

template <
   CFRConfig cfr_config,
   typename Env,
   typename Policy,
   typename DefaultPolicy,
   typename AveragePolicy >
requires concepts::vanilla_cfr_requirements< Env, Policy, DefaultPolicy, AveragePolicy >
double VanillaCFR< cfr_config, Env, Policy, DefaultPolicy, AveragePolicy >::policy_value(
   bool current_policy,
   Player player)
{
   double value = 0.;
   if(current_policy) {
      auto child_hook = [&](const node_type& child_node, auto&&...) {
         if(child_node.category == forest::NodeCategory::terminal) {
            value += _reach_probability(data(child_node)) * data(child_node).value(player);
         }
      };

      m_game_tree.traverse(
         [&]< typename... Args >(Args && ... args) {
            return m_game_tree.traverse_all_actions(std::forward< Args >(args)...);
         },
         forest::TraversalHooks{.child_hook = std::move(child_hook)},
         /*traverse_via_worldstate=*/false);

   } else {
      std::unordered_map< size_t, std::unordered_map< Player, double > > reach_probs;
      auto root_hook = [&](const node_type& root_node, auto&&...) {
         for(auto p : m_env.players()) {
            reach_probs[root_node.id][p] = 1.;
         }
      };
      auto child_hook = [&](const node_type& child_node, auto&&...) {
         auto* parent_node = child_node.parent;
         auto next_reach_prob = reach_probs[parent_node->id];
         if constexpr(concepts::deterministic_fosg< env_type >) {
            auto normalized_policy = rm::normalize_action_policy(fetch_policy(false, *parent_node));
            next_reach_prob[data(*parent_node).player()] *= normalized_policy
               [std::get< action_type >(child_node.action_from_parent.value())];
         } else {
            std::visit(
               common::Overload{
                  [&](const action_type& action) {
                     const auto& action_policy = fetch_policy(false, *parent_node);
                     next_reach_prob[data(*parent_node).player()] *= rm::normalize_action_policy(
                        action_policy)[action];
                  },
                  [&](const auto&) {
            // a chance action's likelihood must have been already stored in the
            // relevant child node during a traversal of the tree, so we don't need to
            // requery this.
#ifndef NDEBUG
                     if(data(*(child_node.parent)).player() != Player::chance) {
                        throw std::logic_error("Was expecting an action from the chance player.");
                     }
#endif
                     // copy over the already existing rp
                     next_reach_prob[Player::chance] = data(child_node)
                                                          .reach_probability(Player::chance);
                  }},
               child_node.action_from_parent.value());
         }
         auto& child_node_rp = reach_probs.emplace(child_node.id, next_reach_prob).first->second;
         if(child_node.category == forest::NodeCategory::terminal) {
            value += rm::reach_probability(child_node_rp) * data(child_node).value(player);
         }
      };

      m_game_tree.traverse(
         [&]< typename... Args >(Args && ... args) {
            return m_game_tree.traverse_all_actions(std::forward< Args >(args)...);
         },
         forest::TraversalHooks{
            .root_hook = std::move(root_hook), .child_hook = std::move(child_hook)},
         /*traverse_via_worldstate=*/false);
   }
   return value;
}

}  // namespace nor::rm

#endif  // NOR_VANILLA_HPP
