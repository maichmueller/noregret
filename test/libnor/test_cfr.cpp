#include <gtest/gtest.h>

#include "../games/stratego/fixtures.hpp"
#include "nor/nor.hpp"

using namespace nor;
using namespace games::stratego;

TEST_F(StrategoState5x5, vanilla_cfr_usage_stratego)
{
   //      auto env = std::make_shared< Environment >(std::make_unique< Logic >());
   Environment env{std::make_unique< Logic >()};
   UniformPolicy uniform_policy = rm::factory::
      make_uniform_policy< InfoState, HashmapActionPolicy< Action > >();

   auto tabular_policy = rm::factory::make_tabular_policy(
      std::unordered_map< InfoState, HashmapActionPolicy< Action > >{});

   constexpr rm::CFRConfig cfr_config{.alternating_updates = false, .store_public_states = false};

   auto cfr_runner = rm::factory::make_vanilla< cfr_config, true >(
      std::move(env),
      std::make_shared< State >(std::move(state)),
      tabular_policy,
      std::move(uniform_policy));

   cfr_runner.iterate(1);
}
