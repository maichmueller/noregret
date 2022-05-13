
#ifndef NOR_MCCFR_HPP
#define NOR_MCCFR_HPP

#include <execution>
#include <iostream>
#include <list>
#include <map>
#include <named_type.hpp>
#include <queue>
#include <range/v3/all.hpp>
#include <stack>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cfr_utils.hpp"
#include "common/common.hpp"
#include "forest.hpp"
#include "node.hpp"
#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "nor/policy.hpp"
#include "nor/type_defs.hpp"
#include "nor/utils/utils.hpp"
#include "tabular_cfr_base.hpp"

namespace nor::rm {

enum class MCCFRAlgorithmMode { outcome_sampling = 0, external_sampling = 1 };

enum class MCCFRWeightingMode { lazy = 0, optimistic = 1, stochastic = 2 };

enum class MCCFRExplorationMode {
   epsilon_on_policy = 0,
};

struct MCCFRConfig {
   bool alternating_updates = true;
   MCCFRAlgorithmMode algorithm = MCCFRAlgorithmMode::outcome_sampling;
   MCCFRExplorationMode exploration = MCCFRExplorationMode::epsilon_on_policy;
   MCCFRWeightingMode weighting = MCCFRWeightingMode::lazy;
};

namespace detail {

template < MCCFRConfig config, typename Env >
struct MCCFRNodeDataSelector {
  private:
   using action_type = typename fosg_auto_traits< Env >::action_type;
   /// for lazy weighting we need a weight for each individual action at each infostate
   using lazy_oos_data_type = InfostateNodeData<
      action_type,
      std::unordered_map<
         std::reference_wrapper< const action_type >,
         double,
         common::default_ref_hasher< const action_type >,
         common::default_ref_comparator< const action_type > > >;
   /// for optimistic weghting we merely need a counter for each infostate
   using optimistic_oos_data_type = InfostateNodeData< action_type, size_t >;

  public:
   using type = std::conditional_t<
      config.algorithm == MCCFRAlgorithmMode::outcome_sampling,
      std::conditional_t<
         config.weighting == MCCFRWeightingMode::lazy,
         lazy_oos_data_type,
         std::conditional_t<
            config.weighting == MCCFRWeightingMode::optimistic,
            optimistic_oos_data_type,
            void > >,
      void >;
};
}  // namespace detail

/**
 * The Monte-Carlo Counterfactual Regret Minimization algorithm class following the
 * terminology of the Factored-Observation Stochastic Games (FOSG) formulation.
 *
 * @tparam Env, the environment/game type to run VanillaCFR on.
 *
 */
template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
class MCCFR: public TabularCFRBase< config.alternating_updates, Env, Policy, AveragePolicy > {
   ////////////////////////////
   /// API: public typedefs ///
   ////////////////////////////
  public:
   /// aliases for the template types
   using base = TabularCFRBase< config.alternating_updates, Env, Policy, AveragePolicy >;
   using env_type = Env;
   using policy_type = Policy;
   /// import all fosg aliases to be used in this class from the env type.
   using typename base::action_type;
   using typename base::world_state_type;
   using typename base::info_state_type;
   using typename base::public_state_type;
   using typename base::observation_type;
   using typename base::chance_outcome_type;
   using typename base::chance_distribution_type;
   using action_variant_type = std::variant<
      action_type,
      std::conditional_t<
         std::is_same_v< chance_outcome_type, void >,
         std::monostate,
         chance_outcome_type > >;
   /// the data to store per infostate entry
   using infostate_data_type = typename detail::MCCFRNodeDataSelector< config, env_type >::type;

   /// assert that the node data type did not turn out to be void
   static_assert(
      not std::is_same_v< infostate_data_type, void >,
      "Infostate node data is void type.");
   static_assert(
      not infostate_data_type::cond_weight_base::empty_optimization,
      "Infostate node data is using an empty weights base, but would need to store the values.");
   /// strong-types for player based maps
   using WeightMap = fluent::
      NamedType< std::unordered_map< Player, double >, struct weight_map_tag >;
   using InfostateMap = typename base::InfostateMap;
   using ObservationbufferMap = typename base::ObservationbufferMap;

   ////////////////////
   /// Constructors ///
   ////////////////////

   /// inherit all constructors from base
   template < typename... BaseArgs >
   MCCFR(BaseArgs&&... base_args, double epsilon, size_t seed = std::random_device{}())
       : base(std::forward< BaseArgs >(base_args)...), m_epsilon(epsilon), m_rng(seed)
   {
      static_assert(_sanity_check_config(), "Config did not pass the check for correctiveness.");
   }

   ////////////////////////////////////
   /// API: public member functions ///
   ////////////////////////////////////

   /// import public getters

   using base::env;
   using base::policy;
   using base::average_policy;
   using base::iteration;
   using base::root_state;

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
    * @param n_iters the number of iterations to perform.
    * @return a pointer to the constant current policy after the update
    */
   auto iterate(size_t n_iters);
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
   auto iterate(std::optional< Player > player_to_update = std::nullopt)
      requires(config.alternating_updates)
   ;

   StateValue game_value(Player player) { return _iterate< false >(player); }
   //   StateValueMap game_value() { return _iterate< false >(std::nullopt); }

   ////////////////////////////////
   /// private member functions ///
   ////////////////////////////////

   [[nodiscard]] inline auto& _infonodes() { return m_infonode; }
   [[nodiscard]] inline auto& infonode(const sptr< info_state_type >& infostate) const
   {
      return m_infonode.at(infostate);
   }
   [[nodiscard]] inline auto& _infonode(const sptr< info_state_type >& infostate)
   {
      return m_infonode.at(infostate);
   }

  private:
   /// import the parent's member variable accessors
   using base::_env;
   using base::_iteration;
   using base::_root_state_uptr;
   using base::_policy;
   using base::_average_policy;
   using base::_player_update_schedule;
   using base::_cycle_player_to_update;
   using base::_child_state;
   using base::_fill_infostate_and_obs_buffers_inplace;

   /// the relevant data stored at each infostate
   std::unordered_map<
      sptr< info_state_type >,
      infostate_data_type,
      decltype(
         [](const sptr< info_state_type >& ptr) { return std::hash< info_state_type >{}(*ptr); }),
      decltype([](const sptr< info_state_type >& ptr1, const sptr< info_state_type >& ptr2) {
         return *ptr1 == *ptr2;
      }) >
      m_infonode{};
   /// the parameter to control the epsilon-on-policy epxloration
   double m_epsilon;
   /// the rng state to produce random numbers with
   common::random::RNG m_rng;
   /// the standard 0 to 1. floating point uniform distribution
   std::uniform_real_distribution< double > m_uniform_01_dist{0., 1.};

   /// define the implementation details of the

   /**
    * @brief The internal vanilla cfr iteration routine.
    *
    * This function sets the algorithm scheme for vanilla cfr by delegating to the right functions.
    *
    * @param player_to_update the player to update (optionally). If not given either the next player
    * to update from the schedule is taken (alternating updates) or every player is updated at the
    * same time (simultaneous updates).
    */
   template < bool use_current_policy = true >
   auto _iterate(std::optional< Player > player_to_update);

   /**
    * @brief traverses the game tree and fills the nodes with current policy weighted value updates.
    *
    * This function is the regular traversal function to call on iteration i > 0, after the nodes
    * have been emplaced by @see _first_traversal.
    */
   template < bool use_current_policy = true >
      requires(
         config.algorithm == MCCFRAlgorithmMode::outcome_sampling
         and config.weighting == MCCFRWeightingMode::optimistic)
   std::pair< StateValue, Probability > _traverse(
      std::optional< Player > player_to_update,
      uptr< world_state_type > curr_worldstate,
      ReachProbabilityMap reach_probability,
      ObservationbufferMap observation_buffer,
      InfostateMap infostates,
      Probability sample_probability);

   template < bool use_current_policy = true >
      requires(
         config.algorithm == MCCFRAlgorithmMode::outcome_sampling
         and config.weighting == MCCFRWeightingMode::lazy)
   std::pair< StateValue, Probability > _traverse(
      std::optional< Player > player_to_update,
      uptr< world_state_type > state,
      ReachProbabilityMap reach_probability,
      ObservationbufferMap observation_buffer,
      InfostateMap infostates,
      Probability sample_probability,
      WeightMap weights);

   template < bool use_current_policy = true >
      requires(config.algorithm == MCCFRAlgorithmMode::external_sampling)
   StateValueMap _traverse(
      std::optional< Player > player_to_update,
      uptr< world_state_type > curr_worldstate,
      ReachProbabilityMap reach_probability,
      ObservationbufferMap observation_buffer,
      InfostateMap infostates,
      Probability sample_probability,
      WeightMap sampling_weights);

   void apply_regret_matching();

   constexpr bool _sanity_check_config();
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

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
auto MCCFR< config, Env, Policy, AveragePolicy >::iterate(size_t n_iters)
{
   std::vector< std::unordered_map< Player, double > > root_values_per_iteration;
   root_values_per_iteration.reserve(n_iters);
   for([[maybe_unused]] auto _ : ranges::views::iota(size_t(0), n_iters)) {
      LOGD2("Iteration number: ", _iteration());
      root_values_per_iteration.emplace_back(std::get< 0 >(_iterate(_cycle_player_to_update())));
      _iteration()++;
   }
   return root_values_per_iteration;
}

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
auto MCCFR< config, Env, Policy, AveragePolicy >::iterate(std::optional< Player > player_to_update)
   requires(config.alternating_updates)
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
   } else if(auto env_players = _env().players();
             ranges::find(env_players, player_to_update) == env_players.end()) {
      std::stringstream ssout;
      ssout << "Given player to update ";
      ssout << player_to_update.value();
      ssout << " is not a member of the game's player list ";
      ssout << ranges::views::all(env_players) << ".";
      throw std::invalid_argument(ssout.str());
   }
   LOGD2("Iteration number: ", _iteration());
   // run the iteration
   auto value = std::vector{
      std::get< 0 >(_iterate(_cycle_player_to_update(player_to_update))).get()};
   // and increment our iteration counter
   _iteration()++;
   return value;
}

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
template < bool use_current_policy >
auto MCCFR< config, Env, Policy, AveragePolicy >::_iterate(std::optional< Player > player_to_update)
{
   auto players = _env().players();
   auto init_infostates = [&] {
      std::unordered_map< Player, sptr< info_state_type > > infostates;
      for(auto player : players | utils::is_nonchance_player_filter) {
         auto& infostate = infostates.emplace(player, std::make_shared< info_state_type >(player))
                              .first->second;
         infostate->append(_env().private_observation(player, root_state()));
      }
      return InfostateMap{std::move(infostates)};
   };
   auto init_reach_probs = [&] {
      std::unordered_map< Player, double > rp_map;
      for(auto player : players) {
         rp_map.emplace(player, 1.);
      }
      return ReachProbabilityMap{std::move(rp_map)};
   };
   auto init_obs_buffer = [&] {
      std::unordered_map< Player, std::vector< observation_type > > obs_map;
      for(auto player : players | utils::is_nonchance_player_filter) {
         obs_map.emplace(player, std::vector< observation_type >{});
      }
      return ObservationbufferMap{std::move(obs_map)};
   };
   if constexpr(config.algorithm == MCCFRAlgorithmMode::outcome_sampling) {
      if constexpr(config.weighting == MCCFRWeightingMode::lazy) {
         return _traverse< use_current_policy >(
            player_to_update,
            utils::static_unique_ptr_downcast< world_state_type >(
               utils::clone_any_way(_root_state_uptr())),
            init_reach_probs(),
            init_obs_buffer(),
            init_infostates(),
            Probability{1.},
            [&] {
               std::unordered_map< Player, double > weights;
               for(auto player : players | utils::is_nonchance_player_filter) {
                  weights.emplace(player, 0.);
               }
               return WeightMap{std::move(weights)};
            }());
      }
      if constexpr(config.weighting == MCCFRWeightingMode::optimistic) {
         return _traverse< use_current_policy >(
            player_to_update,
            utils::static_unique_ptr_downcast< world_state_type >(
               utils::clone_any_way(_root_state_uptr())),
            init_reach_probs(),
            init_obs_buffer(),
            init_infostates(),
            Probability{1.});
      }
   }
   if constexpr(config.algorithm == MCCFRAlgorithmMode::external_sampling) {
      if constexpr(config.weighting == MCCFRWeightingMode::lazy) {
         return _traverse< use_current_policy >(
            player_to_update,
            utils::static_unique_ptr_downcast< world_state_type >(
               utils::clone_any_way(_root_state_uptr())),
            init_obs_buffer(),
            init_infostates());
      }
   }
}

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
void MCCFR< config, Env, Policy, AveragePolicy >::apply_regret_matching()
{
   for(auto& [infostate_ptr, data] : _infonodes()) {
      rm::regret_matching(
         fetch_policy< true >(infostate_ptr, data.actions()),
         data.regret(),
         // we provide the accessor to get the underlying referenced action, as the infodata
         // stores only reference wrappers to the actions
         [](const action_type& action) { return std::ref(action); });
   }
}

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
template < bool use_current_policy >
   requires(
      config.algorithm == MCCFRAlgorithmMode::outcome_sampling
      and config.weighting == MCCFRWeightingMode::lazy)
std::pair< StateValue, Probability > MCCFR< config, Env, Policy, AveragePolicy >::_traverse(
   std::optional< Player > player_to_update,
   uptr< world_state_type > state,
   ReachProbabilityMap reach_probability,
   ObservationbufferMap observation_buffer,
   InfostateMap infostates,
   Probability sample_probability,
   WeightMap weights)
{
   Player active_player = _env().active_player(*state);

   if(_env().is_terminal(*state)) {
      return std::pair{
         StateValue{_env().reward(active_player, *state) / sample_probability.get()},
         Probability{1.}};
   }

   // now we check first if we even need to consider a chance player, as the env could be
   // simply deterministic. In that case we might need to traverse the chance player's actions
   // or an active player's actions
   if constexpr(not concepts::deterministic_fosg< env_type >) {
      if(active_player == Player::chance) {
         auto chance_actions = _env().chance_actions(*state);
         auto& chosen_action = common::random::choose(chance_actions, m_rng);
         double chance_prob = 1. / static_cast< double >(chance_actions.size());
         reach_probability.get()[active_player] *= chance_prob;
         return _traverse(
            player_to_update,
            _child_state(state, chosen_action),
            std::move(reach_probability),
            std::move(observation_buffer),
            std::move(infostates),
            Probability{sample_probability.get() * chance_prob},
            std::move(weights));
      }
   }

   auto infostate = infostates.get().at(active_player);
   auto [infostate_and_data_iter, success] = _infonodes().try_emplace(
      infostate, infostate_data_type{});
   auto& infonode_data = infostate_and_data_iter->second;
   if(success) {
      infonode_data.emplace(_env().actions(active_player, *state));
   }

   auto& player_policy = fetch_policy< use_current_policy >(infostate, infonode_data.actions());
   // apply one round of regret matching on the current policy before using it. MCCFR only updates
   // the policy once you revisit it, as it is a lazy update schedule. As such, one would need to
   // update all infostates after the last iteration to ensure that the policy is fully up-to-date
   if constexpr(use_current_policy) {
      // we only do regret-matching if we are actually traversing with the current policy and not
      // with the average policy.
      rm::regret_matching(
         player_policy,
         infonode_data.regret(),
         // we provide the accessor to get the underlying referenced action, as the infodata
         // stores only reference wrappers to the actions
         [](const action_type& action) { return std::cref(action); });
   }

   auto choose_by_policy = [&](auto action_policy_functor) {
      auto& chosen_action = common::random::choose(
         infonode_data.actions(), action_policy_functor, m_rng);
      return std::tuple{chosen_action, action_policy_functor(chosen_action)};
   };

   auto [sampled_action, action_sample_prob, action_player_prob] = [&]() {
      if(active_player == player_to_update and m_uniform_01_dist(m_rng) < m_epsilon) {
         auto [action, sample_prob] = choose_by_policy(
            [count = double(infonode_data.actions().size())](const auto&) { return 1. / count; });
         return std::tuple{action, sample_prob, player_policy[action]};
      } else {
         // in the non-epsilon case we simply use the player's policy to sample the next move
         // from. Thus in this case, the action's sample probability and action's policy
         // probability are the same, i.e. action_sample_prob = action_policy_prob
         auto [action, action_prob] = choose_by_policy(
            [&](const auto& act) { return player_policy[act]; });
         return std::tuple{action, action_prob, action_prob};
      }
   }();

   auto policy_incrementer = [&](const Weight& weight, const Probability& reach_prob) {
      // we only update the average policy when we traverse with the current policy, as we then
      // get estimates to update with. When traversing with the average policy we are only
      // interseted in the estimated game value
      auto& avg_policy = fetch_policy< false >(infostate, infonode_data.actions());
      for(const action_type& action : infonode_data.actions()) {
         auto policy_incr = (weight.get() + reach_prob.get()) * action_player_prob;
         avg_policy[action] += policy_incr;
         if(action == sampled_action) [[unlikely]] {
            infonode_data.weight()[std::cref(action)] = 0.;
         } else [[likely]] {
            infonode_data.weight()[std::cref(action)] += policy_incr;
         }
      }
   };

   auto next_reach_prob = reach_probability.get();
   auto next_weights = weights.get();
   next_reach_prob[active_player] *= action_player_prob;
   next_weights[active_player] = next_weights[active_player] * action_player_prob
                                 + infonode_data.weight()[std::cref(sampled_action)];

   auto next_state = _child_state(state, sampled_action);

   _fill_infostate_and_obs_buffers_inplace(
      observation_buffer, infostates, sampled_action, *next_state);

   auto [action_value, tail_prob] = _traverse(
      player_to_update,
      std::move(next_state),
      ReachProbabilityMap{std::move(next_reach_prob)},
      std::move(observation_buffer),
      std::move(infostates),
      Probability{sample_probability.get() * action_sample_prob},
      WeightMap{std::move(next_weights)});

   if constexpr(use_current_policy) {
      if(not config.alternating_updates or active_player == player_to_update.value()) {
         for(const auto& action : infonode_data.actions()) {
            auto cf_value_weight = action_value.get()
                                   * rm::cf_reach_probability(
                                      active_player, reach_probability.get());
            // compute the esimtated counterfactual regret and add it to the cumulative regret table
            infonode_data.regret(action) += [&] {
               if(action == sampled_action) {
                  // note that tail_prob = pi^sigma(z[I]a, z)
                  // the probability pi^sigma(z[I]a, z) - pi^sigma(z[I], z) can also be expressed as
                  // pi^sigma(z[I]a, z) * (1 - sigma(I, a)), since pi(h, z) = pi(z) / pi(h) and
                  // pi(ha) = pi(h) * sigma(I, a)
                  // --> pi(ha, z) - pi(h, z) = pi(ha, z) * ( 1 - sigma(I, a))
                  return cf_value_weight * tail_prob.get() * (1. - action_sample_prob);
               } else {
                  // we are returning here the formula: -W * pi(z[I], z)
                  return -cf_value_weight * tail_prob.get() * action_sample_prob;
               }
            }();
         }
         policy_incrementer(
            Weight{weights.get()[active_player]},
            Probability{reach_probability.get()[active_player]});
      } else {
         policy_incrementer(
            Weight{[&] {
               // we are computing the 'counterfactual weight' here (all, but active_player's weight
               // contribution)
               auto value_view = ranges::views::values(weights.get());
               return std::reduce(
                         value_view.begin(), value_view.end(), double(1.), std::multiplies{})
                      / weights.get()[active_player];
            }()},
            Probability{rm::cf_reach_probability(active_player, reach_probability.get())});
      }
   }
   return std::pair{action_value, Probability{tail_prob.get() * action_player_prob}};
}

template < MCCFRConfig config, typename Env, typename Policy, typename AveragePolicy >
constexpr bool MCCFR< config, Env, Policy, AveragePolicy >::_sanity_check_config()
{
   if constexpr(config.algorithm == MCCFRAlgorithmMode::outcome_sampling) {
      if constexpr(not config.alternating_updates) {
         return false;
      }
   }
   return true;
};

}  // namespace nor::rm

#endif  // NOR_MCCFR_HPP
