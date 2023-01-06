#include <gtest/gtest.h>

#include <unordered_map>

#include "../games/stratego/fixtures.hpp"
#include "cfr_run_funcs.hpp"
#include "nor/env.hpp"
#include "nor/nor.hpp"
#include "rm_specific_testing_utils.hpp"

using namespace nor;

TEST(KuhnPoker, CFR_PLUS)
{
   run_cfr_on_kuhn_poker< rm::CFRPlusConfig{} >();
}

TEST(RockPaperScissors, CFR_PLUS)
{
   run_cfr_on_rps< rm::CFRPlusConfig{} >();
}
