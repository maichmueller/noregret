#include <gtest/gtest.h>

#include "../games/stratego/fixtures.hpp"
#include "nor/nor.hpp"
#include "stratego/stratego.hpp"

using namespace nor;

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
