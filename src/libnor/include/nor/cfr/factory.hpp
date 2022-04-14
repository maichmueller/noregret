
#ifndef NOR_FACTORY_HPP
#define NOR_FACTORY_HPP

#include "vanilla.hpp"

namespace nor::rm {

struct factory {
  private:
   template < typename ValueType >
   static std::unordered_map< Player, ValueType > to_map(
      std::vector< Player > players,
      const ValueType& value)
   {
      std::unordered_map< Player, ValueType > map;
      for(auto player : players) {
         map[player] = value;
      }
      return map;
   }

  public:
   template <
      CFRConfig cfg,
      bool as_map,
      typename Env,
      typename Policy,
      typename DefaultPolicy,
      typename AveragePolicy >
   static VanillaCFR<
      cfg,
      std::remove_cvref_t< Env >,  // remove_cvref_t necessary to avoid Env captured as const Env&
      std::remove_cvref_t< Policy >,
      std::remove_cvref_t< DefaultPolicy >,
      std::remove_cvref_t< AveragePolicy > >
   make_vanilla(
      Env&& env,
      uptr< typename fosg_auto_traits< Env >::world_state_type > root_state,
      Policy&& policy,
      AveragePolicy&& avg_policy,
      DefaultPolicy&& def_policy)
   {
      if constexpr(as_map) {
         return {
            std::forward< Env >(env),
            std::move(root_state),
            to_map(env.players(), std::forward< Policy >(policy)),
            to_map(env.players(), std::forward< AveragePolicy >(avg_policy)),
            std::forward< DefaultPolicy >(def_policy)};
      } else {
         return {
            std::forward< Env >(env),
            std::move(root_state),
            std::forward< Policy >(policy),
            std::forward< AveragePolicy >(avg_policy),
            std::forward< DefaultPolicy >(def_policy)};
      }
   }

   template <
      CFRConfig cfg,
      typename Env,
      typename Policy,
      typename AveragePolicy,
      typename DefaultPolicy >
   static VanillaCFR< cfg, Env, Policy, DefaultPolicy, AveragePolicy > make_vanilla(
      Env&& env,
      uptr< typename fosg_auto_traits< Env >::world_state_type > root_state,
      std::unordered_map< Player, Policy > policy_map,
      std::unordered_map< Player, AveragePolicy > avg_policy_map,
      DefaultPolicy&& def_policy = DefaultPolicy())
   {
      return {
         std::forward< Env >(env),
         std::move(root_state),
         std::move(policy_map),
         std::move(avg_policy_map),
         std::forward< DefaultPolicy >(def_policy)};
   }

   template < CFRConfig cfg, bool as_map, typename Env, typename Policy, typename DefaultPolicy >
   static VanillaCFR< cfg, Env, Policy, DefaultPolicy, Policy > make_vanilla(
      Env&& env,
      uptr< typename fosg_auto_traits< Env >::world_state_type > root_state,
      const Policy& policy,
      DefaultPolicy&& def_policy = DefaultPolicy())
   {
      return make_vanilla< cfg, as_map >(
         std::forward< Env >(env),
         std::move(root_state),
         policy,
         policy,
         std::forward< DefaultPolicy >(def_policy));
   }

   template < typename Infostate, typename ActionPolicy, typename Table >
   static TabularPolicy< Infostate, ActionPolicy, Table > make_tabular_policy(Table&& table)
   {
      return {std::forward< Table >(table)};
   }

   template < typename Table >
   static TabularPolicy< typename Table::key_type, typename Table::mapped_type, Table >
   make_tabular_policy(Table&& table)
   {
      return {std::forward< Table >(table)};
   }

   template < typename Infostate, typename ActionPolicy, size_t extent = std::dynamic_extent >
   static UniformPolicy< Infostate, ActionPolicy, extent > make_uniform_policy()
   {
      return {};
   }
};

}  // namespace nor::rm

#endif  // NOR_FACTORY_HPP
