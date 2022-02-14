#include <gtest/gtest.h>

#include "nor/nor.hpp"
#include "stratego/stratego.hpp"

namespace nor {

TEST(vanilla_cfr, basic_usage)
{
   CFR cfr(
      starting_state_func,
      action_func,
      transition_func,
      reward_func,
      observation_func);
   cfr.run();
}


TEST(vanilla_cfr, usage_with_stratego)
{

   CFR cfr(
      starting_state_func,
      action_func,
      transition_func,
      private_observation_func);
   cfr.run();
}

}  // namespace nor