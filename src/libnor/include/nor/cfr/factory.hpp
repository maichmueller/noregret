
#ifndef NOR_FACTORY_HPP
#define NOR_FACTORY_HPP

#include "vanilla.hpp"

namespace nor::rm {

struct factory {
   template < CFRConfig cfg, typename Env, typename Policy, typename DefaultPolicy >
   static VanillaCFR< cfg, Env, Policy, DefaultPolicy > make_vanilla(
      Env&& env,
      Policy&& policy,
      DefaultPolicy&& def_policy)
   {
      return {
         std::forward< Env >(env),
         std::forward< Policy >(policy),
         std::forward< DefaultPolicy >(def_policy)};
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

   template <
      typename Infostate,
      typename ActionPolicy,
      size_t extent,
      typename LegalActionsGetter >
   static UniformPolicy< Infostate, ActionPolicy, extent, LegalActionsGetter > make_uniform_policy(
      LegalActionsGetter&& lag)
   {
      return {std::forward< LegalActionsGetter >(lag)};
   }

   template < typename Infostate, typename ActionPolicy, typename LegalActionsGetter >
   static UniformPolicy< Infostate, ActionPolicy, std::dynamic_extent, LegalActionsGetter >
   make_uniform_policy(LegalActionsGetter&& lag)
   {
      return {std::forward< LegalActionsGetter >(lag)};
   }
};

}  // namespace nor::rm

#endif  // NOR_FACTORY_HPP
