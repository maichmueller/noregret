
#ifndef NOR_FACTORY_HPP
#define NOR_FACTORY_HPP

#include "mccfr.hpp"
#include "vcfr.hpp"

namespace nor::rm {

struct factory {
  private:
   template < typename ValueType >
   static std::unordered_map< Player, ValueType > to_map(
      ranges::range auto players,
      const ValueType& value)
   {
      std::unordered_map< Player, ValueType > map;
      for(auto player : players) {
         map[player] = value;
      }
      return map;
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
   make_vanilla(
      Env&& env,
      uptr< typename fosg_auto_traits< Env >::world_state_type > root_state,
      Policy&& policy,
      AveragePolicy&& avg_policy)
   {
      if constexpr(as_map) {
         auto players = env.players();
         return {
            std::forward< Env >(env),
            std::move(root_state),
            to_map(players | utils::is_nonchance_player_filter, std::forward< Policy >(policy)),
            to_map(
               players | utils::is_nonchance_player_filter,
               std::forward< AveragePolicy >(avg_policy))};
      } else {
         return {
            std::forward< Env >(env),
            std::move(root_state),
            std::forward< Policy >(policy),
            std::forward< AveragePolicy >(avg_policy)};
      }
   }

   template < CFRConfig cfg, typename Env, typename Policy, typename AveragePolicy >
   static VanillaCFR< cfg, Env, Policy, AveragePolicy > make_vanilla(
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
   static VanillaCFR< cfg, Env, Policy, Policy > make_vanilla(
      Env&& env,
      uptr< typename fosg_auto_traits< Env >::world_state_type > root_state,
      const Policy& policy)
   {
      return make_vanilla< cfg, as_map >(
         std::forward< Env >(env), std::move(root_state), policy, policy);
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
         auto players = env.players();
         auto each_player_current_policy_map = to_map(utils::is_nonchance_player_filter(players), std::forward< Policy >(policy));
         auto each_player_avg_policy_map = to_map(
            utils::is_nonchance_player_filter(players), std::forward< AvgPolicy >(avg_policy));
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
