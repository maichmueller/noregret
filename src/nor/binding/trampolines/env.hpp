
#ifndef NOR_PY_TRAMPOLINES_ENV_HPP
#define NOR_PY_TRAMPOLINES_ENV_HPP

#include "nor/env/polymorphic.hpp"

namespace nor::binding {

/* Trampoline Class */
struct PyEnvironment: public nor::games::polymorph::Environment {
   using Environment::action_type;
   using Environment::chance_outcome_type;
   using Environment::observation_type;
   using Environment::info_state_type;
   using Environment::public_state_type;
   using Environment::world_state_type;

   /* Inherit the constructors */
   using Environment::Environment;

   std::vector< nor::Player > players(const world_state_type& wstate) const override
   {
      PYBIND11_OVERRIDE_NAME(
         std::vector< nor::Player >, /* Return type */
         Environment, /* Parent class */
         "players", /* Name of function in Python */
         players, /* Name of function in C++ */
         wstate /* Argument(s) */
      );
   }

   nor::Player active_player(const world_state_type& wstate) const override
   {
      PYBIND11_OVERRIDE_NAME(
         nor::Player, /* Return type */
         Environment, /* Parent class */
         "active_player", /* Name of function in Python */
         active_player, /* Name of function in C++ */
         wstate /* Argument(s) */
      );
   }

   bool is_terminal(const world_state_type& wstate) const override
   {
      PYBIND11_OVERRIDE_NAME(
         bool, /* Return type */
         Environment, /* Parent class */
         "is_terminal", /* Name of function in Python */
         is_terminal, /* Name of function in C++ */
         wstate /* Argument(s) */
      );
   }

   double reward(nor::Player player, const world_state_type& wstate) const override
   {
      PYBIND11_OVERRIDE_NAME(
         double, /* Return type */
         Environment, /* Parent class */
         "reward", /* Name of function in Python */
         reward, /* Name of function in C++ */
         player, /* Argument(s) */
         wstate
      );
   }

   double rewards(std::vector< nor::Player > players, const world_state_type& wstate) const override
   {
      PYBIND11_OVERRIDE_NAME(
         double, /* Return type */
         Environment, /* Parent class */
         "rewards", /* Name of function in Python */
         rewards, /* Name of function in C++ */
         players, /* Argument(s) */
         wstate
      );
   }

   void transition(world_state_type& wstate, const action_type& action) const override
   {
      PYBIND11_OVERRIDE_NAME(
         void, /* Return type */
         Environment, /* Parent class */
         "transition", /* Name of function in Python */
         transition, /* Name of function in C++ */
         wstate, /* Argument(s) */
         action
      );
   }

   void transition(world_state_type& wstate, const chance_outcome_type& action) const override
   {
      PYBIND11_OVERRIDE_NAME(
         void, /* Return type */
         Environment, /* Parent class */
         "transition", /* Name of function in Python */
         transition, /* Name of function in C++ */
         wstate, /* Argument(s) */
         action
      );
   }

   ObservationHolder< observation_type > private_observation(
      nor::Player player,
      const world_state_type& wstate,
      const action_type& action,
      const world_state_type& next_wstate
   ) const override
   {
      PYBIND11_OVERRIDE_NAME(
         ObservationHolder< observation_type >, /* Return type */
         Environment, /* Parent class */
         "private_observation", /* Name of function in Python */
         private_observation, /* Name of function in C++ */
         player, /* Argument(s) */
         wstate,
         action,
         next_wstate
      );
   }

   std::vector< ActionHolder< action_type > >
   actions(nor::Player player, const world_state_type& wstate) const override
   {
      PYBIND11_OVERRIDE_NAME(
         std::vector< ActionHolder< action_type > >, /* Return type */
         Environment, /* Parent class */
         "actions", /* Name of function in Python */
         actions, /* Name of function in C++ */
         player,
         wstate /* Argument(s) */
      );
   }

   std::vector< ChanceOutcomeHolder< chance_outcome_type > > chance_actions(
      const world_state_type& wstate
   ) const override
   {
      PYBIND11_OVERRIDE_NAME(
         std::vector< ChanceOutcomeHolder< chance_outcome_type > >, /* Return type */
         Environment, /* Parent class */
         "chance_actions", /* Name of function in Python */
         chance_actions, /* Name of function in C++ */
         wstate /* Argument(s) */
      );
   }

   double chance_probability(const world_state_type& wstate, const chance_outcome_type& outcome)
      const override
   {
      PYBIND11_OVERRIDE_NAME(
         double, /* Return type */
         Environment, /* Parent class */
         "chance_probability", /* Name of function in Python */
         chance_probability, /* Name of function in C++ */
         wstate,
         outcome /* Argument(s) */
      );
   }

   std::vector< nor::PlayerInformedType< std::optional<
      std::variant< ChanceOutcomeHolder< chance_outcome_type >, ActionHolder< action_type > > > > >
   private_history(nor::Player player, const world_state_type& wstate) const override
   {
      PYBIND11_OVERRIDE_NAME(
         NOR_SINGLE_ARG(std::vector< nor::PlayerInformedType< std::optional< std::variant<
                           ChanceOutcomeHolder< chance_outcome_type >,
                           ActionHolder< action_type > > > > >), /* Return type */
         Environment, /* Parent class */
         "private_history", /* Name of function in Python */
         private_history, /* Name of function in C++ */
         player, /* Argument(s) */
         wstate
      );
   }

   std::vector< nor::PlayerInformedType<
      std::variant< ChanceOutcomeHolder< chance_outcome_type >, ActionHolder< action_type > > > >
   open_history(const world_state_type& wstate) const override
   {
      PYBIND11_OVERRIDE_NAME(
         NOR_SINGLE_ARG(std::vector< nor::PlayerInformedType< std::variant<
                           ChanceOutcomeHolder< chance_outcome_type >,
                           ActionHolder< action_type > > > >), /* Return type */
         Environment, /* Parent class */
         "open_history", /* Name of function in Python */
         open_history, /* Name of function in C++ */
         wstate /* Argument(s) */
      );
   }

   std::vector< nor::PlayerInformedType<
      std::variant< ChanceOutcomeHolder< chance_outcome_type >, ActionHolder< action_type > > > >
   public_history(const world_state_type& wstate) const override
   {
      PYBIND11_OVERRIDE_NAME(
         NOR_SINGLE_ARG(std::vector< nor::PlayerInformedType< std::variant<
                           ChanceOutcomeHolder< chance_outcome_type >,
                           ActionHolder< action_type > > > >), /* Return type */
         Environment, /* Parent class */
         "public_history", /* Name of function in Python */
         public_history, /* Name of function in C++ */
         wstate /* Argument(s) */
      );
   }
};

}  // namespace nor::binding

#endif  // NOR_PY_TRAMPOLINES_ENV_HPP
