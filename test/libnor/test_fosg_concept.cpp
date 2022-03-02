#include <gtest/gtest.h>

#include "../games/stratego/fixtures.hpp"
#include "nor/nor.hpp"
#include "stratego/stratego.hpp"

class DummEnv {
   // nor fosg typedefs
   using world_state_type = int;
   using info_state_type = std::vector<int>;
   using public_state_type = PublicState;
   using action_type = Action;
   using observation_type = InfoState::observation_type;
   // nor fosg traits
   static constexpr size_t max_player_count = 2;
   static constexpr TurnDynamic turn_dynamic = TurnDynamic::sequential;

   std::vector< action_type > actions(Player player, const world_state_type& wstate);
   std::vector< action_type > actions(const info_state_type& istate);
   std::vector< Player > players();
   auto reset(world_state_type& wstate);
   bool is_terminal(world_state_type& wstate);
   double reward(Player player, world_state_type& wstate);
   void transition(const action_type& action, world_state_type& worldstate);
   observation_type private_observation(Player player, const world_state_type& wstate);
   observation_type private_observation(Player player, const action_type& action);
   observation_type public_observation(Player player, const world_state_type& wstate);
   observation_type public_observation(Player player, const action_type& action);
};

class DummConstEnv {
   // nor fosg typedefs
   using world_state_type = int;
   using info_state_type = std::vector<int>;
   using public_state_type = PublicState;
   using action_type = Action;
   using observation_type = InfoState::observation_type;
   // nor fosg traits
   static constexpr size_t max_player_count = 2;
   static constexpr TurnDynamic turn_dynamic = TurnDynamic::sequential;

   std::vector< action_type > actions(Player player, const world_state_type& wstate) const;
   std::vector< action_type > actions(const info_state_type& istate) const;
   std::vector< Player > players() const;
   auto reset(world_state_type& wstate) const;
   bool is_terminal(world_state_type& wstate) const;
   double reward(Player player, world_state_type& wstate) const;
   void transition(const action_type& action, world_state_type& worldstate) const;
   observation_type private_observation(Player player, const world_state_type& wstate) const;
   observation_type private_observation(Player player, const action_type& action) const;
   observation_type public_observation(Player player, const world_state_type& wstate) const;
   observation_type public_observation(Player player, const action_type& action) const;
};

TEST_F(MinimalState, vanilla_cfr_usage_stratego)
{
   using namespace games::stratego;

   //      auto env = std::make_shared< Environment >(std::make_unique< Logic >());
   Environment env{std::make_unique< Logic >()};

   constexpr rm::CFRConfig cfr_config{.alternating_updates = false, .store_public_states = false};
   rm::VanillaCFR cfr_runner = rm::make_cfr< cfr_config >(
      env,
      TabularPolicy{UniformPolicy{
         [env = Environment{std::make_unique< Logic >()}](const InfoState& istate) {
            return env.actions(istate);
         },
         Hint< InfoState, Action, HashMapActionPolicy< Action > >{}}});
   cfr_runner.iterate();
}
