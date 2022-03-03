#include <gtest/gtest.h>

#include "nor/nor.hpp"

class DummEnv {
   // nor fosg typedefs
   using world_state_type = struct {
      int i;
   };
   using info_state_type = std::vector< std::string >;
   using public_state_type = std::vector< std::string >;
   using action_type = int;
   using observation_type = std::string;
   // nor fosg traits
   static constexpr size_t max_player_count = 10;
   static constexpr nor::TurnDynamic turn_dynamic = nor::TurnDynamic::simultaneous;

   std::vector< action_type > actions(nor::Player player, const world_state_type& wstate);
   std::vector< action_type > actions(const info_state_type& istate);
   std::vector< nor::Player > players();
   auto reset(world_state_type& wstate);
   bool is_terminal(world_state_type& wstate);
   double reward(nor::Player player, world_state_type& wstate);
   void transition(const action_type& action, world_state_type& worldstate);
   observation_type private_observation(nor::Player player, const world_state_type& wstate);
   observation_type private_observation(nor::Player player, const action_type& action);
   observation_type public_observation(nor::Player player, const world_state_type& wstate);
   observation_type public_observation(nor::Player player, const action_type& action);
};

class DummConstEnv {
   // nor fosg typedefs
   using world_state_type = struct {
      int i;
   };
   using info_state_type = std::vector< std::string >;
   using public_state_type = std::vector< std::string >;
   using action_type = int;
   using observation_type = std::string;
   // nor fosg traits
   static constexpr size_t max_player_count = 10;
   static constexpr nor::TurnDynamic turn_dynamic = nor::TurnDynamic::simultaneous;

   std::vector< action_type > actions(nor::Player player, const world_state_type& wstate) const;
   std::vector< action_type > actions(const info_state_type& istate) const;
   std::vector< nor::Player > players() const;
   auto reset(world_state_type& wstate) const;
   bool is_terminal(world_state_type& wstate) const;
   double reward(nor::Player player, world_state_type& wstate) const;
   void transition(const action_type& action, world_state_type& worldstate) const;
   observation_type private_observation(nor::Player player, const world_state_type& wstate) const;
   observation_type private_observation(nor::Player player, const action_type& action) const;
   observation_type public_observation(nor::Player player, const world_state_type& wstate) const;
   observation_type public_observation(nor::Player player, const action_type& action) const;
};

TEST(DummyEnvironment, fosg_concept_agreement)
{

}
