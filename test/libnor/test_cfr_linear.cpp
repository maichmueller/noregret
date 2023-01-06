#include <gtest/gtest.h>

#include <unordered_map>

#include "../games/stratego/fixtures.hpp"
#include "nor/env.hpp"
#include "nor/nor.hpp"
#include "rm_specific_testing_utils.hpp"
#include "cfr_run_funcs.hpp"

using namespace nor;


TEST(KuhnPoker, CFR_LINEAR_alternating)
{
   constexpr rm::CFRLinearConfig cfr_config{.update_mode = rm::UpdateMode::alternating};
   run_cfr_on_kuhn_poker< cfr_config >();
}

TEST(KuhnPoker, CFR_LINEAR_simultaneous)
{
   constexpr rm::CFRLinearConfig cfr_config{.update_mode = rm::UpdateMode::simultaneous};
   run_cfr_on_kuhn_poker< cfr_config >();
}

TEST(RockPaperScissors, CFR_LINEAR_alternating)
{
   constexpr rm::CFRLinearConfig cfr_config{.update_mode = rm::UpdateMode::alternating};
   run_cfr_on_rps< cfr_config >();
}

TEST(RockPaperScissors, CFR_LINEAR_simultaneous)
{
   constexpr rm::CFRLinearConfig cfr_config{.update_mode = rm::UpdateMode::simultaneous};
   run_cfr_on_rps< cfr_config >();
}
