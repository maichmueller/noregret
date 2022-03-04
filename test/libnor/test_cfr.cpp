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
   UniformPolicy u{[env = std::move(Environment{std::make_unique< Logic >()})](
                      const InfoState& istate) { return env.actions(istate); }};

   auto tabular_policy = TabularPolicy{
      std::unordered_map< InfoState, HashMapActionPolicy >{}, std::move(u)};

   constexpr rm::CFRConfig cfr_config{.alternating_updates = false, .store_public_states = false};

   auto cfr_runner = rm::cfr_factory::make_vanilla< cfr_config >(
      std::move(env), std::move(tabular_policy));
   //   cfr_runner.iterate();
}
