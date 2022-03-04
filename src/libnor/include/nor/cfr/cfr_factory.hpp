
#ifndef NOR_CFR_FACTORY_HPP
#define NOR_CFR_FACTORY_HPP

#include "vanilla.hpp"

namespace nor::rm {

struct cfr_factory {
   template <
      CFRConfig cfg,
      typename Env,
      typename Policy,
      typename FOSGTraitType = fosg_traits< Env > >
   static VanillaCFR< cfg, Env, FOSGTraitType, Policy >
   make_vanilla(Env&& env, Policy&& policy, FOSGTraitType = {})
   {
      return {std::forward< Env >(env), std::forward< Policy >(policy)};
   }
};

}  // namespace nor::rm

#endif  // NOR_CFR_FACTORY_HPP
