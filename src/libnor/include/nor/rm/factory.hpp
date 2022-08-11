
#ifndef NOR_FACTORY_HPP
#define NOR_FACTORY_HPP

#include "cfr_discounted.hpp"
#include "cfr_exponential.hpp"
#include "cfr_monte_carlo.hpp"
#include "cfr_plus.hpp"
#include "cfr_vanilla.hpp"

namespace nor::rm {

struct factory {
  private:
   template < typename ValueType >
   static std::unordered_map< Player, ValueType > to_map(
      ranges::range auto players,
      const ValueType& value)
   {
      std::unordered_map< Player, ValueType > map;
      for(auto player : players | utils::is_actual_player_filter) {
         map.emplace(player, value);
      }
      return map;
   }

   template < CFRConfig cfg >
   static consteval CFRConfig to_discounted_cfg()
   {
      return CFRConfig{
         .update_mode = cfg.update_mode,
         .regret_minimizing_mode = cfg.regret_minimizing_mode,
         .weighting_mode = CFRWeightingMode::discounted};
   }

  public:
   /////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////// Vanilla Counterfactual Regret Minimizer Factory ////////////////////////
   /////////////////////////////////////////////////////////////////////////////////////////////

   template < CFRConfig cfg, bool as_map, typename Env, typename Policy, typename AveragePolicy >
   static VanillaCFR<
      cfg,
      std::remove_cvref_t< Env >,  // remove_cvref_t necessary to avoid Env captured as const Env&
      std::remove_cvref_t< Policy >,
      std::remove_cvref_t< AveragePolicy > >
   make_cfr_vanilla(
      Env&& env,
      uptr< typename fosg_auto_traits< Env >::world_state_type > root_state,
      Policy&& policy,
      AveragePolicy&& avg_policy)
   {
      if constexpr(as_map) {
         auto players = env.players(*root_state);
         return {
            std::forward< Env >(env),
            std::move(root_state),
            to_map(players, std::forward< Policy >(policy)),
            to_map(players, std::forward< AveragePolicy >(avg_policy))};
      } else {
         return {
            std::forward< Env >(env),
            std::move(root_state),
            std::forward< Policy >(policy),
            std::forward< AveragePolicy >(avg_policy)};
      }
   }

   template < CFRConfig cfg, typename Env, typename Policy, typename AveragePolicy >
   static VanillaCFR< cfg, Env, Policy, AveragePolicy > make_cfr_vanilla(
      Env&& env,
      uptr< typename fosg_auto_traits< Env >::world_state_type > root_state,
      std::unordered_map< Player, Policy > policy_map,
      std::unordered_map< Player, AveragePolicy > avg_policy_map)
   {
      return {
         std::forward< Env >(env),
         std::move(root_state),
         std::move(policy_map),
         std::move(avg_policy_map)};
   }

   template < CFRConfig cfg, bool as_map, typename Env, typename Policy >
   static VanillaCFR< cfg, Env, Policy, Policy > make_cfr_vanilla(
      Env&& env,
      uptr< typename fosg_auto_traits< Env >::world_state_type > root_state,
      const Policy& policy)
   {
      return make_cfr_vanilla< cfg, as_map >(
         std::forward< Env >(env), std::move(root_state), policy, policy);
   }

   /////////////////////////////////////////////////////////////////////////////////////////////
   ///////////////////// Counterfactual Regret PLUS Minimizer Factory //////////////////////////
   /////////////////////////////////////////////////////////////////////////////////////////////

   template <
      bool as_map,
      typename Env,
      typename Policy,
      typename AveragePolicy >
   static CFRPlus<
      std::remove_cvref_t< Env >,  // remove_cvref_t necessary to avoid Env captured as const Env&
      std::remove_cvref_t< Policy >,
      std::remove_cvref_t< AveragePolicy > >
   make_cfr_plus(
      Env&& env,
      uptr< typename fosg_auto_traits< Env >::world_state_type > root_state,
      Policy&& policy,
      AveragePolicy&& avg_policy)
   {
      if constexpr(as_map) {
         auto players = env.players(*root_state);
         return {
            std::forward< Env >(env),
            std::move(root_state),
            to_map(players, std::forward< Policy >(policy)),
            to_map(players, std::forward< AveragePolicy >(avg_policy))};
      } else {
         return {
            std::forward< Env >(env),
            std::move(root_state),
            std::forward< Policy >(policy),
            std::forward< AveragePolicy >(avg_policy)};
      }
   }

   template < typename Env, typename Policy, typename AveragePolicy >
   static CFRPlus< Env, Policy, AveragePolicy > make_cfr_plus(
      Env&& env,
      uptr< typename fosg_auto_traits< Env >::world_state_type > root_state,
      std::unordered_map< Player, Policy > policy_map,
      std::unordered_map< Player, AveragePolicy > avg_policy_map)
   {
      return {
         std::forward< Env >(env),
         std::move(root_state),
         std::move(policy_map),
         std::move(avg_policy_map)};
   }

   template < bool as_map, typename Env, typename Policy >
   static CFRPlus< Env, Policy, Policy > make_cfr_plus(
      Env&& env,
      uptr< typename fosg_auto_traits< Env >::world_state_type > root_state,
      const Policy& policy)
   {
      return make_cfr_plus< as_map >(
         std::forward< Env >(env), std::move(root_state), policy, policy);
   }

   /////////////////////////////////////////////////////////////////////////////////////////////
   /////////////////// DISCOUNTED Counterfactual Regret Minimizer Factory //////////////////////
   /////////////////////////////////////////////////////////////////////////////////////////////

   template <
      CFRDiscountedConfig cfg,
      bool as_map,
      typename Env,
      typename Policy,
      typename AveragePolicy >
   static CFRDiscounted<
      cfg,
      std::remove_cvref_t< Env >,  // remove_cvref_t necessary to avoid Env captured as const Env&
      std::remove_cvref_t< Policy >,
      std::remove_cvref_t< AveragePolicy > >
   make_cfr_discounted(
      Env&& env,
      uptr< typename fosg_auto_traits< Env >::world_state_type > root_state,
      Policy&& policy,
      AveragePolicy&& avg_policy,
      CFRDiscountedParameters params = {})
   {
      if constexpr(as_map) {
         auto players = env.players(*root_state);
         return {
            std::move(params),
            std::forward< Env >(env),
            std::move(root_state),
            to_map(players, std::forward< Policy >(policy)),
            to_map(players, std::forward< AveragePolicy >(avg_policy))};
      } else {
         return {
            std::move(params),
            std::forward< Env >(env),
            std::move(root_state),
            std::forward< Policy >(policy),
            std::forward< AveragePolicy >(avg_policy)};
      }
   }

   template < CFRDiscountedConfig cfg, typename Env, typename Policy, typename AveragePolicy >
   static CFRDiscounted< cfg, Env, Policy, AveragePolicy > make_cfr_discounted(
      Env&& env,
      uptr< typename fosg_auto_traits< Env >::world_state_type > root_state,
      std::unordered_map< Player, Policy > policy_map,
      std::unordered_map< Player, AveragePolicy > avg_policy_map,
      CFRDiscountedParameters params = {})
   {
      return {
         std::move(params),
         std::forward< Env >(env),
         std::move(root_state),
         std::move(policy_map),
         std::move(avg_policy_map)};
   }

   template < CFRDiscountedConfig cfg, bool as_map, typename Env, typename Policy >
   static CFRDiscounted< cfg, Env, Policy, Policy > make_cfr_discounted(
      Env&& env,
      uptr< typename fosg_auto_traits< Env >::world_state_type > root_state,
      const Policy& policy,
      CFRDiscountedParameters params = {})
   {
      return make_cfr_discounted< cfg, as_map >(
         std::move(params), std::forward< Env >(env), std::move(root_state), policy, policy);
   }

   /////////////////////////////////////////////////////////////////////////////////////////////
   ///////////////////// LINEAR Counterfactual Regret Minimizer Factory ////////////////////////
   /////////////////////////////////////////////////////////////////////////////////////////////

   template <
      CFRDiscountedConfig cfg,
      bool as_map,
      typename Env,
      typename Policy,
      typename AveragePolicy >
   static CFRDiscounted<
      cfg,
      std::remove_cvref_t< Env >,  // remove_cvref_t necessary to avoid Env captured as const Env&
      std::remove_cvref_t< Policy >,
      std::remove_cvref_t< AveragePolicy > >
   make_cfr_linear(
      Env&& env,
      uptr< typename fosg_auto_traits< Env >::world_state_type > root_state,
      Policy&& policy,
      AveragePolicy&& avg_policy)
   {
      if constexpr(as_map) {
         auto players = env.players(*root_state);
         return {
            CFRDiscountedParameters{.alpha = 1, .beta = 1, .gamma = 1},
            std::forward< Env >(env),
            std::move(root_state),
            to_map(players, std::forward< Policy >(policy)),
            to_map(players, std::forward< AveragePolicy >(avg_policy))};
      } else {
         return {
            CFRDiscountedParameters{.alpha = 1, .beta = 1, .gamma = 1},
            std::forward< Env >(env),
            std::move(root_state),
            std::forward< Policy >(policy),
            std::forward< AveragePolicy >(avg_policy)};
      }
   }

   template < CFRDiscountedConfig cfg, typename Env, typename Policy, typename AveragePolicy >
   static CFRDiscounted< cfg, Env, Policy, AveragePolicy > make_cfr_linear(
      Env&& env,
      uptr< typename fosg_auto_traits< Env >::world_state_type > root_state,
      std::unordered_map< Player, Policy > policy_map,
      std::unordered_map< Player, AveragePolicy > avg_policy_map)
   {
      return {
         CFRDiscountedParameters{.alpha = 1, .beta = 1, .gamma = 1},
         std::forward< Env >(env),
         std::move(root_state),
         std::move(policy_map),
         std::move(avg_policy_map)};
   }

   template < CFRDiscountedConfig cfg, bool as_map, typename Env, typename Policy >
   static CFRDiscounted< cfg, Env, Policy, Policy > make_cfr_linear(
      Env&& env,
      uptr< typename fosg_auto_traits< Env >::world_state_type > root_state,
      const Policy& policy)
   {
      return make_cfr_linear< cfg, as_map >(
         CFRDiscountedParameters{.alpha = 1, .beta = 1, .gamma = 1},
         std::forward< Env >(env),
         std::move(root_state),
         policy,
         policy);
   }

   /////////////////////////////////////////////////////////////////////////////////////////////
   /////////////////// EXPONENTIAL Counterfactual Regret Minimizer Factory //////////////////////
   /////////////////////////////////////////////////////////////////////////////////////////////

   template <
      CFRExponentialConfig cfg,
      bool as_map,
      typename Env,
      typename Policy,
      typename AveragePolicy >
   static CFRExponential<
      cfg,
      std::remove_cvref_t< Env >,  // remove_cvref_t necessary to avoid Env captured as const Env&
      std::remove_cvref_t< Policy >,
      std::remove_cvref_t< AveragePolicy > >
   make_cfr_exponential(
      Env&& env,
      uptr< typename fosg_auto_traits< Env >::world_state_type > root_state,
      Policy&& policy,
      AveragePolicy&& avg_policy,
      CFRExponentialParameters params = {})
   {
      if constexpr(as_map) {
         auto players = env.players(*root_state);
         return {
            std::move(params),
            std::forward< Env >(env),
            std::move(root_state),
            to_map(players, std::forward< Policy >(policy)),
            to_map(players, std::forward< AveragePolicy >(avg_policy))};
      } else {
         return {
            std::move(params),
            std::forward< Env >(env),
            std::move(root_state),
            std::forward< Policy >(policy),
            std::forward< AveragePolicy >(avg_policy)};
      }
   }

   template < CFRExponentialConfig cfg, typename Env, typename Policy, typename AveragePolicy >
   static CFRExponential< cfg, Env, Policy, AveragePolicy > make_cfr_exponential(
      Env&& env,
      uptr< typename fosg_auto_traits< Env >::world_state_type > root_state,
      std::unordered_map< Player, Policy > policy_map,
      std::unordered_map< Player, AveragePolicy > avg_policy_map,
      CFRExponentialParameters params = {})
   {
      return {
         std::move(params),
         std::forward< Env >(env),
         std::move(root_state),
         std::move(policy_map),
         std::move(avg_policy_map)};
   }

   template < CFRExponentialConfig cfg, bool as_map, typename Env, typename Policy >
   static CFRExponential< cfg, Env, Policy, Policy > make_cfr_exponential(
      Env&& env,
      uptr< typename fosg_auto_traits< Env >::world_state_type > root_state,
      const Policy& policy,
      CFRExponentialParameters params = {})
   {
      return make_cfr_exponential< cfg, as_map >(
         std::move(params), std::forward< Env >(env), std::move(root_state), policy, policy);
   }

   /////////////////////////////////////////////////////////////////////////////////////////////
   ////////////////// Monte-Carlo Counterfactual Regret Minimizer Factory //////////////////////
   /////////////////////////////////////////////////////////////////////////////////////////////

   template < MCCFRConfig cfg, bool as_map, typename Env, typename Policy, typename AvgPolicy >
   static MCCFR<
      cfg,
      // remove_cvref_t necessary to avoid e.g. Env captured as const Env&
      std::remove_cvref_t< Env >,
      std::remove_cvref_t< Policy >,
      std::remove_cvref_t< AvgPolicy > >
   make_mccfr(
      Env&& env,
      uptr< typename fosg_auto_traits< Env >::world_state_type > root_state,
      Policy&& policy,
      AvgPolicy&& avg_policy,
      double epsilon,
      size_t seed = 0)
   {
      if constexpr(as_map) {
         auto players = env.players(*root_state);
         auto each_player_current_policy_map = to_map(players, std::forward< Policy >(policy));
         auto each_player_avg_policy_map = to_map(players, std::forward< AvgPolicy >(avg_policy));
         return MCCFR<
            cfg,
            // remove_cvref_t necessary to avoid e.g. Env captured as const Env&
            std::remove_cvref_t< Env >,
            std::remove_cvref_t< Policy >,
            std::remove_cvref_t< AvgPolicy > >{
            std::forward< Env >(env),
            std::move(root_state),
            std::move(each_player_current_policy_map),
            std::move(each_player_avg_policy_map),
            epsilon,
            seed};
      } else {
         return {
            std::forward< Env >(env),
            std::move(root_state),
            std::forward< Policy >(policy),
            std::forward< AvgPolicy >(avg_policy),
            epsilon,
            seed};
      }
   }

   template < MCCFRConfig cfg, typename Env, typename Policy, typename AveragePolicy >
   static MCCFR< cfg, Env, Policy, AveragePolicy > make_mccfr(
      Env&& env,
      uptr< typename fosg_auto_traits< Env >::world_state_type > root_state,
      std::unordered_map< Player, Policy > policy_map,
      std::unordered_map< Player, AveragePolicy > avg_policy_map,
      double epsilon,
      size_t seed = 0)
   {
      return {
         std::forward< Env >(env),
         std::move(root_state),
         std::move(policy_map),
         std::move(avg_policy_map),
         epsilon,
         seed};
   }

   template < MCCFRConfig cfg, bool as_map, typename Env, typename Policy >
   static MCCFR< cfg, Env, Policy, Policy > make_mccfr(
      Env&& env,
      uptr< typename fosg_auto_traits< Env >::world_state_type > root_state,
      const Policy& policy,
      double epsilon,
      size_t seed = 0)
   {
      return make_mccfr< cfg, as_map >(
         std::forward< Env >(env), std::move(root_state), policy, policy, epsilon, seed);
   }

   /////////////////////////////////////////////////////////////////////////////////////////////
   ////////////////////////////////// Policy Table Factory /////////////////////////////////////
   /////////////////////////////////////////////////////////////////////////////////////////////

   template <
      typename Infostate,
      typename ActionPolicy,
      typename Table,
      typename DefaultPolicy = UniformPolicy< Infostate, ActionPolicy > >
   static TabularPolicy< Infostate, ActionPolicy, DefaultPolicy, Table > make_tabular_policy(
      Table&& table = Table(),
      DefaultPolicy&& def_policy = DefaultPolicy())
   {
      return {std::forward< Table >(table), std::forward< DefaultPolicy >(def_policy)};
   }

   template <
      typename Table,
      typename DefaultPolicy =
         UniformPolicy< typename Table::key_type, typename Table::mapped_type > >
   static TabularPolicy<
      typename Table::key_type,
      typename Table::mapped_type,
      DefaultPolicy,
      Table >
   make_tabular_policy(Table&& table = Table(), DefaultPolicy&& def_policy = DefaultPolicy())
   {
      return {std::forward< Table >(table), std::forward< DefaultPolicy >(def_policy)};
   }

   template < typename Infostate, typename ActionPolicy, size_t extent = std::dynamic_extent >
   static UniformPolicy< Infostate, ActionPolicy, extent > make_uniform_policy()
   {
      return {};
   }

   template < typename Infostate, typename ActionPolicy, size_t extent = std::dynamic_extent >
   static ZeroDefaultPolicy< Infostate, ActionPolicy, extent > make_zero_policy()
   {
      return {};
   }
};

}  // namespace nor::rm

#endif  // NOR_FACTORY_HPP
