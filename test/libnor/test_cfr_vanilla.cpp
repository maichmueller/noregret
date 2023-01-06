#include <gtest/gtest.h>

#include <unordered_map>

#include "../games/stratego/fixtures.hpp"
#include "cfr_run_funcs.hpp"
#include "nor/env.hpp"
#include "nor/exploitability.hpp"
#include "nor/nor.hpp"
#include "rm_specific_testing_utils.hpp"

using namespace nor;

TEST(KuhnPoker, CFR_VANILLA_alternating)
{
   run_cfr_on_kuhn_poker< rm::CFRDiscountedConfig{.update_mode = rm::UpdateMode::alternating} >();
}

TEST(KuhnPoker, CFR_VANILLA_simultaneous)
{
   run_cfr_on_kuhn_poker< rm::CFRDiscountedConfig{.update_mode = rm::UpdateMode::simultaneous} >();
}

TEST(RockPaperScissors, CFR_VANILLA_alternating)
{
   run_cfr_on_rps< rm::CFRDiscountedConfig{.update_mode = rm::UpdateMode::alternating} >();
}

TEST(RockPaperScissors, CFR_VANILLA_simultaneous)
{
   run_cfr_on_rps< rm::CFRDiscountedConfig{.update_mode = rm::UpdateMode::simultaneous} >();
}
