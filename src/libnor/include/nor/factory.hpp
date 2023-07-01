
#ifndef NOR_FACTORY_HPP
#define NOR_FACTORY_HPP

#include "nor/policy/policy.hpp"
#include "nor/rm/cfr_tabular/cfr.hpp"
#include "nor/rm/cfr_tabular/mccfr.hpp"

namespace nor {

struct factory {
  private:
   template < typename ValueType >
   static std::unordered_map< Player, ValueType >
   to_map(ranges::range auto players, const ValueType& value)
   {
      std::unordered_map< Player, ValueType > map;
      for(auto player : players | is_actual_player_filter) {
         map.emplace(player, value);
      }
      return map;
   }

  public:
   /////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////// Vanilla Counterfactual Regret Minimizer Factory ////////////////////////
   /////////////////////////////////////////////////////////////////////////////////////////////

   template <
      rm::CFRConfig cfg,
      bool as_map,
      typename Env,
      typename Policy,
      typename AveragePolicy >
   static rm::VanillaCFR<
      cfg,
      std::remove_cvref_t< Env >,  // remove_cvref_t necessary to avoid Env captured as const Env&
      std::remove_cvref_t< Policy >,
      std::remove_cvref_t< AveragePolicy > >
   make_cfr_vanilla(
      Env&& env,
      uptr< auto_world_state_type< std::remove_cvref_t< Env > > > root_state,
      Policy&& policy,
      AveragePolicy&& avg_policy
   )
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

   template < rm::CFRConfig cfg, typename Env, typename Policy, typename AveragePolicy >
   static rm::VanillaCFR< cfg, Env, Policy, AveragePolicy > make_cfr_vanilla(
      Env&& env,
      uptr< auto_world_state_type< std::remove_cvref_t< Env > > > root_state,
      std::unordered_map< Player, Policy > policy_map,
      std::unordered_map< Player, AveragePolicy > avg_policy_map
   )
   {
      return {
         std::forward< Env >(env),
         std::move(root_state),
         std::move(policy_map),
         std::move(avg_policy_map)};
   }

   template < rm::CFRConfig cfg, bool as_map, typename Env, typename Policy >
   static rm::VanillaCFR< cfg, Env, Policy, Policy > make_cfr_vanilla(
      Env&& env,
      uptr< auto_world_state_type< std::remove_cvref_t< Env > > > root_state,
      const Policy& policy
   )
   {
      return make_cfr_vanilla< cfg, as_map >(
         std::forward< Env >(env), std::move(root_state), policy, policy
      );
   }

   /////////////////////////////////////////////////////////////////////////////////////////////
   ///////////////////// Counterfactual Regret PLUS Minimizer Factory //////////////////////////
   /////////////////////////////////////////////////////////////////////////////////////////////

   template <
      bool as_map,
      typename Env,
      typename Policy,
      typename AveragePolicy >
   static rm::CFRPlus<
      std::remove_cvref_t< Env >,  // remove_cvref_t necessary to avoid Env captured as const Env&
      std::remove_cvref_t< Policy >,
      std::remove_cvref_t< AveragePolicy > >
   make_cfr_plus(
      Env&& env,
      uptr< auto_world_state_type< std::remove_cvref_t< Env > > > root_state,
      Policy&& policy,
      AveragePolicy&& avg_policy
   )
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
   static rm::CFRPlus< Env, Policy, AveragePolicy > make_cfr_plus(
      Env&& env,
      uptr< auto_world_state_type< std::remove_cvref_t< Env > > > root_state,
      std::unordered_map< Player, Policy > policy_map,
      std::unordered_map< Player, AveragePolicy > avg_policy_map
   )
   {
      return {
         std::forward< Env >(env),
         std::move(root_state),
         std::move(policy_map),
         std::move(avg_policy_map)};
   }

   template < bool as_map, typename Env, typename Policy >
   static rm::CFRPlus< Env, Policy, Policy > make_cfr_plus(
      Env&& env,
      uptr< auto_world_state_type< std::remove_cvref_t< Env > > > root_state,
      const Policy& policy
   )
   {
      return make_cfr_plus< as_map >(
         std::forward< Env >(env), std::move(root_state), policy, policy
      );
   }

   /////////////////////////////////////////////////////////////////////////////////////////////
   /////////////////// DISCOUNTED Counterfactual Regret Minimizer Factory //////////////////////
   /////////////////////////////////////////////////////////////////////////////////////////////

   template <
      rm::CFRDiscountedConfig cfg,
      bool as_map,
      typename Env,
      typename Policy,
      typename AveragePolicy >
   static rm::CFRDiscounted<
      cfg,
      std::remove_cvref_t< Env >,  // remove_cvref_t necessary to avoid Env captured as const Env&
      std::remove_cvref_t< Policy >,
      std::remove_cvref_t< AveragePolicy > >
   make_cfr_discounted(
      Env&& env,
      uptr< auto_world_state_type< std::remove_cvref_t< Env > > > root_state,
      Policy&& policy,
      AveragePolicy&& avg_policy,
      rm::CFRDiscountedParameters params = {}
   )
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

   template < rm::CFRDiscountedConfig cfg, typename Env, typename Policy, typename AveragePolicy >
   static rm::CFRDiscounted< cfg, Env, Policy, AveragePolicy > make_cfr_discounted(
      Env&& env,
      uptr< auto_world_state_type< std::remove_cvref_t< Env > > > root_state,
      std::unordered_map< Player, Policy > policy_map,
      std::unordered_map< Player, AveragePolicy > avg_policy_map,
      rm::CFRDiscountedParameters params = {}
   )
   {
      return {
         std::move(params),
         std::forward< Env >(env),
         std::move(root_state),
         std::move(policy_map),
         std::move(avg_policy_map)};
   }

   template < rm::CFRDiscountedConfig cfg, bool as_map, typename Env, typename Policy >
   static rm::CFRDiscounted< cfg, Env, Policy, Policy > make_cfr_discounted(
      Env&& env,
      uptr< auto_world_state_type< std::remove_cvref_t< Env > > > root_state,
      const Policy& policy,
      rm::CFRDiscountedParameters params = {}
   )
   {
      return make_cfr_discounted< cfg, as_map >(
         std::move(params), std::forward< Env >(env), std::move(root_state), policy, policy
      );
   }

   /////////////////////////////////////////////////////////////////////////////////////////////
   ///////////////////// LINEAR Counterfactual Regret Minimizer Factory ////////////////////////
   /////////////////////////////////////////////////////////////////////////////////////////////

   template <
      rm::CFRLinearConfig cfg,
      bool as_map,
      typename Env,
      typename Policy,
      typename AveragePolicy >
   static rm::CFRLinear<
      cfg,
      std::remove_cvref_t< Env >,  // remove_cvref_t necessary to avoid Env captured as const Env&
      std::remove_cvref_t< Policy >,
      std::remove_cvref_t< AveragePolicy > >
   make_cfr_linear(
      Env&& env,
      uptr< auto_world_state_type< std::remove_cvref_t< Env > > > root_state,
      Policy&& policy,
      AveragePolicy&& avg_policy
   )
   {
      if constexpr(as_map) {
         auto players = env.players(*root_state);
         return {
            rm::CFRDiscountedParameters{.alpha = 1, .beta = 1, .gamma = 1},
            std::forward< Env >(env),
            std::move(root_state),
            to_map(players, std::forward< Policy >(policy)),
            to_map(players, std::forward< AveragePolicy >(avg_policy))};
      } else {
         return {
            rm::CFRDiscountedParameters{.alpha = 1, .beta = 1, .gamma = 1},
            std::forward< Env >(env),
            std::move(root_state),
            std::forward< Policy >(policy),
            std::forward< AveragePolicy >(avg_policy)};
      }
   }

   template < rm::CFRLinearConfig cfg, typename Env, typename Policy, typename AveragePolicy >
   static rm::CFRLinear< cfg, std::remove_cvref_t< Env >, Policy, AveragePolicy > make_cfr_linear(
      Env&& env,
      uptr< auto_world_state_type< std::remove_cvref_t< Env > > > root_state,
      std::unordered_map< Player, Policy > policy_map,
      std::unordered_map< Player, AveragePolicy > avg_policy_map
   )
   {
      return {
         rm::CFRDiscountedParameters{.alpha = 1, .beta = 1, .gamma = 1},
         std::forward< Env >(env),
         std::move(root_state),
         std::move(policy_map),
         std::move(avg_policy_map)};
   }

   template < rm::CFRLinearConfig cfg, bool as_map, typename Env, typename Policy >
   static rm::CFRLinear< cfg, std::remove_cvref_t< Env >, Policy, Policy > make_cfr_linear(
      Env&& env,
      uptr< auto_world_state_type< std::remove_cvref_t< Env > > > root_state,
      const Policy& policy
   )
   {
      return make_cfr_linear< cfg, as_map >(
         rm::CFRDiscountedParameters{.alpha = 1, .beta = 1, .gamma = 1},
         std::forward< Env >(env),
         std::move(root_state),
         policy,
         policy
      );
   }

   /////////////////////////////////////////////////////////////////////////////////////////////
   /////////////////// EXPONENTIAL Counterfactual Regret Minimizer Factory //////////////////////
   /////////////////////////////////////////////////////////////////////////////////////////////

   template <
      rm::CFRExponentialConfig cfg,
      bool as_map,
      typename Env,
      typename Policy,
      typename AveragePolicy >
   static rm::CFRExponential<
      cfg,
      std::remove_cvref_t< Env >,  // remove_cvref_t necessary to avoid Env captured as const Env&
      std::remove_cvref_t< Policy >,
      std::remove_cvref_t< AveragePolicy > >
   make_cfr_exponential(
      Env&& env,
      uptr< auto_world_state_type< std::remove_cvref_t< Env > > > root_state,
      Policy&& policy,
      AveragePolicy&& avg_policy,
      rm::CFRExponentialParameters params = {}
   )
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

   template < rm::CFRExponentialConfig cfg, typename Env, typename Policy, typename AveragePolicy >
   static rm::CFRExponential< cfg, Env, Policy, AveragePolicy > make_cfr_exponential(
      Env&& env,
      uptr< auto_world_state_type< std::remove_cvref_t< Env > > > root_state,
      std::unordered_map< Player, Policy > policy_map,
      std::unordered_map< Player, AveragePolicy > avg_policy_map,
      rm::CFRExponentialParameters params = {}
   )
   {
      return {
         std::move(params),
         std::forward< Env >(env),
         std::move(root_state),
         std::move(policy_map),
         std::move(avg_policy_map)};
   }

   template < rm::CFRExponentialConfig cfg, bool as_map, typename Env, typename Policy >
   static rm::CFRExponential< cfg, Env, Policy, Policy > make_cfr_exponential(
      Env&& env,
      uptr< auto_world_state_type< std::remove_cvref_t< Env > > > root_state,
      const Policy& policy,
      rm::CFRExponentialParameters params = {}
   )
   {
      return make_cfr_exponential< cfg, as_map >(
         std::move(params), std::forward< Env >(env), std::move(root_state), policy, policy
      );
   }

   /////////////////////////////////////////////////////////////////////////////////////////////
   ////////////////// Monte-Carlo Counterfactual Regret Minimizer Factory //////////////////////
   /////////////////////////////////////////////////////////////////////////////////////////////

   template < rm::MCCFRConfig cfg, bool as_map, typename Env, typename Policy, typename AvgPolicy >
   static rm::MCCFR<
      cfg,
      // remove_cvref_t necessary to avoid e.g. Env captured as const Env&
      std::remove_cvref_t< Env >,
      std::remove_cvref_t< Policy >,
      std::remove_cvref_t< AvgPolicy > >
   make_mccfr(
      Env&& env,
      uptr< auto_world_state_type< std::remove_cvref_t< Env > > > root_state,
      Policy&& policy,
      AvgPolicy&& avg_policy,
      double epsilon,
      size_t seed = 0
   )
   {
      if constexpr(as_map) {
         auto players = env.players(*root_state);
         auto each_player_current_policy_map = to_map(players, std::forward< Policy >(policy));
         auto each_player_avg_policy_map = to_map(players, std::forward< AvgPolicy >(avg_policy));
         return rm::MCCFR<
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

   template < rm::MCCFRConfig cfg, typename Env, typename Policy, typename AveragePolicy >
   static rm::MCCFR< cfg, Env, Policy, AveragePolicy > make_mccfr(
      Env&& env,
      uptr< auto_world_state_type< std::remove_cvref_t< Env > > > root_state,
      std::unordered_map< Player, Policy > policy_map,
      std::unordered_map< Player, AveragePolicy > avg_policy_map,
      double epsilon,
      size_t seed = 0
   )
   {
      return {
         std::forward< Env >(env),
         std::move(root_state),
         std::move(policy_map),
         std::move(avg_policy_map),
         epsilon,
         seed};
   }

   template < rm::MCCFRConfig cfg, bool as_map, typename Env, typename Policy >
   static rm::MCCFR< cfg, Env, Policy, Policy > make_mccfr(
      Env&& env,
      uptr< auto_world_state_type< std::remove_cvref_t< Env > > > root_state,
      const Policy& policy,
      double epsilon,
      size_t seed = 0
   )
   {
      return make_mccfr< cfg, as_map >(
         std::forward< Env >(env), std::move(root_state), policy, policy, epsilon, seed
      );
   }

   /////////////////////////////////////////////////////////////////////////////////////////////
   ////////////////////////////////// Generic CFR Factory /////////////////////////////////////
   /////////////////////////////////////////////////////////////////////////////////////////////

   // build the correct CFR solver depending on the config type passed in. This is merely a
   // forwarding station to choose the specific algorithm via the config type.
   template < auto config, bool as_map, typename... Args >
   static decltype(auto) make_cfr(Args&&... args)
   {
      using namespace nor;
      using ConfigType = std::decay_t< decltype(config) >;
      if constexpr(std::same_as< ConfigType, rm::CFRConfig >) {
         return factory::make_cfr_vanilla< config, as_map >(std::forward< Args >(args)...);
      } else if constexpr(std::same_as< ConfigType, rm::CFRDiscountedConfig >) {
         return factory::make_cfr_discounted< config, as_map >(std::forward< Args >(args)...);
      } else if constexpr(std::same_as< ConfigType, rm::CFRLinearConfig >) {
         return factory::make_cfr_linear< config, as_map >(std::forward< Args >(args)...);
      } else if constexpr(std::same_as< ConfigType, rm::CFRPlusConfig >) {
         return factory::make_cfr_plus< as_map >(std::forward< Args >(args)...);
      } else if constexpr(std::same_as< ConfigType, rm::CFRExponentialConfig >) {
         return factory::make_cfr_exponential< config, as_map >(std::forward< Args >(args)...);
      } else if constexpr(std::same_as< ConfigType, rm::CFRPlusConfig >) {
         return factory::make_cfr_plus< config, as_map >(std::forward< Args >(args)...);
      } else if constexpr(std::same_as< ConfigType, rm::MCCFRConfig >) {
         return factory::make_mccfr< config, as_map >(std::forward< Args >(args)...);
      }
   }

   template < auto config, typename... Args >
   static decltype(auto) make_cfr(Args&&... args)
   {
      using namespace nor;
      using ConfigType = std::decay_t< decltype(config) >;
      if constexpr(std::same_as< ConfigType, rm::CFRConfig >) {
         return factory::make_cfr_vanilla< config >(std::forward< Args >(args)...);
      } else if constexpr(std::same_as< ConfigType, rm::CFRDiscountedConfig >) {
         return factory::make_cfr_discounted< config >(std::forward< Args >(args)...);
      } else if constexpr(std::same_as< ConfigType, rm::CFRLinearConfig >) {
         return factory::make_cfr_linear< config >(std::forward< Args >(args)...);
      } else if constexpr(std::same_as< ConfigType, rm::CFRPlusConfig >) {
         return factory::make_cfr_plus(std::forward< Args >(args)...);
      } else if constexpr(std::same_as< ConfigType, rm::CFRExponentialConfig >) {
         return factory::make_cfr_exponential< config >(std::forward< Args >(args)...);
      } else if constexpr(std::same_as< ConfigType, rm::CFRPlusConfig >) {
         return factory::make_cfr_plus< config >(std::forward< Args >(args)...);
      } else if constexpr(std::same_as< ConfigType, rm::MCCFRConfig >) {
         return factory::make_mccfr< config >(std::forward< Args >(args)...);
      }
   }

   /////////////////////////////////////////////////////////////////////////////////////////////
   ////////////////////////////////// Policy Table Factory /////////////////////////////////////
   /////////////////////////////////////////////////////////////////////////////////////////////

   template < typename Infostate, typename ActionPolicy >
   static TabularPolicy< Infostate, ActionPolicy > make_tabular_policy()
   {
      return {};
   }

   template < typename Infostate, typename ActionPolicy, typename Table >
   static TabularPolicy< Infostate, ActionPolicy, Table > make_tabular_policy(
      Table&& table = Table()
   )
   {
      return {std::forward< Table >(table)};
   }

   template < typename Table >
   static auto make_tabular_policy(Table&& table = Table())
   {
      return TabularPolicy{std::forward< Table >(table)};
   }

   template < typename Infostate, typename ActionPolicy, size_t extent = std::dynamic_extent >
   static auto make_uniform_policy()
   {
      return UniformPolicy< Infostate, ActionPolicy, extent >{};
   }

   template < typename Infostate, typename ActionPolicy, size_t extent = std::dynamic_extent >
   static auto make_zero_policy()
   {
      return ZeroDefaultPolicy< Infostate, ActionPolicy, extent >{};
   }

   template <
      typename Infostate,
      typename Action,
      BRConfig config = BRConfig{.store_infostate_values = false} >
   static BestResponsePolicy< Infostate, Action, config > make_best_response_policy(
      std::vector< Player > best_response_players,
      std::unordered_map<
         InfostateHolder< Infostate >,
         detail::mapped_br_type< config, ActionHolder< Action > > > cached_br_map = {}
   )
   {
      return {best_response_players, std::move(cached_br_map)};
   }
   template <
      typename Infostate,
      typename Action,
      BRConfig config = BRConfig{.store_infostate_values = false} >
   static BestResponsePolicy< Infostate, Action, config > make_best_response_policy(
      Player best_response_player,
      std::unordered_map<
         InfostateHolder< Infostate >,
         detail::mapped_br_type< config, ActionHolder< Action > > > cached_br_map = {}
   )
   {
      return {{best_response_player}, std::move(cached_br_map)};
   }
};

}  // namespace nor

#endif  // NOR_FACTORY_HPP
