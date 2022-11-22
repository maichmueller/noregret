
#ifndef NOR_POLICY_VALUE_HPP
#define NOR_POLICY_VALUE_HPP

#include "cfr_config.hpp"
#include "cfr_vanilla.hpp"
#include "nor/concepts.hpp"

namespace nor {

template <
   typename Env,
   typename Policy,
   typename ActionPolicy = typename fosg_auto_traits< Env >::action_policy_type >
   requires concepts::state_policy_no_default<
      Policy,
      typename fosg_auto_traits< Env >::info_state_type,
      typename fosg_auto_traits< Env >::action_type,
      ActionPolicy >
double policy_value(
   Env& env,
   const typename fosg_auto_traits< Env >::world_state_type& root_state,
   const player_hash_map< Policy >& policy_map
)
{
   using world_state_type = typename fosg_auto_traits< Env >::world_state_type;
   using info_state_type = typename fosg_auto_traits< Env >::info_state_type;
   using observation_type = typename fosg_auto_traits< Env >::observation_type;
   using action_type = typename fosg_auto_traits< Env >::action_type;
   using action_variant_type = typename fosg_auto_traits< Env >::action_variant_type;

   auto cfr_solver = rm::VanillaCFR<
      rm::CFRConfig{},
      std::remove_cvref_t< Env >,
      nor::TabularPolicy< info_state_type, HashmapActionPolicy< action_type > >,
      Policy >{
      env,
      utils::static_unique_ptr_downcast< world_state_type >(utils::clone_any_way(root_state)),
      player_hash_map< nor::TabularPolicy<
         info_state_type,
         HashmapActionPolicy< action_type > > >{},  // this argument should be ignored in the
                                                    // internal computations of the solver
      policy_map};
   return cfr_solver.game_value();
}

}  // namespace nor

#endif  // NOR_POLICY_VALUE_HPP