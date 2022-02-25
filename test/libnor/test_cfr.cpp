#include <gtest/gtest.h>

#include "../games/stratego/fixtures.hpp"
#include "nor/nor.hpp"
#include "stratego/stratego.hpp"

using namespace nor;

TEST_F(MinimalState, vanilla_cfr_basic_usage)
{
   auto env = std::make_shared< games::stratego::Environment >(
      std::make_unique< stratego::Logic >());
   rm::VanillaCFR cfr(
      rm::CFRConfig{.alternating_updates = false, .store_public_states = false},
      env,
      TabularPolicy<
         std::unordered_map< games::stratego::Infostate, std::vector< double > >,
         UniformPolicy< games::stratego::Infostate, std::dynamic_extent > >{
         [env = env](Player player, const games::stratego::Infostate& state) {
            return env->actions(player, state);
         }});
   cfr.iterate();
}
