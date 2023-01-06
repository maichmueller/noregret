#include <gtest/gtest.h>

#include <unordered_map>

#include "../games/stratego/fixtures.hpp"
#include "cfr_run_funcs.hpp"
#include "nor/env.hpp"
#include "nor/nor.hpp"
#include "rm_specific_testing_utils.hpp"

using namespace nor;

TEST(KuhnPoker, CFR_EXPONENTIAL_alternating)
{
   constexpr rm::CFRExponentialConfig cfr_config{.update_mode = rm::UpdateMode::alternating};
   run_cfr_on_kuhn_poker< cfr_config >();
}

TEST(KuhnPoker, CFR_EXPONENTIAL_simultaneous)
{
   constexpr rm::CFRExponentialConfig cfr_config{.update_mode = rm::UpdateMode::simultaneous};
   run_cfr_on_kuhn_poker< cfr_config >();
}

TEST(RockPaperScissors, CFR_EXPONENTIAL_alternating)
{
   constexpr rm::CFRExponentialConfig cfr_config{.update_mode = rm::UpdateMode::alternating};
   run_cfr_on_rps< cfr_config >();
}

TEST(RockPaperScissors, CFR_EXPONENTIAL_simultaneous)
{
   constexpr rm::CFRExponentialConfig cfr_config{.update_mode = rm::UpdateMode::simultaneous};
   run_cfr_on_rps< cfr_config >();
}
