#include <gtest/gtest.h>

#include <unordered_map>

#include "../games/stratego/fixtures.hpp"
#include "cfr_run_funcs.hpp"
#include "nor/env.hpp"
#include "nor/nor.hpp"
#include "rm_specific_testing_utils.hpp"

using namespace nor;

template < auto config >
void run_mccfr_on_kuhn_poker(
   size_t max_iters = 2e5,
   size_t update_freq = 500,
   double epsilon = 0.6,
   size_t seed = 0
)
{
   run_cfr_on_kuhn_poker< config >(max_iters, update_freq, epsilon, seed);
}
template < auto config >
void run_mccfr_on_rockpaperscissors(
   size_t max_iters = 1e5,
   size_t update_freq = 10,
   double epsilon = 0.6,
   size_t seed = 0
)
{
   run_cfr_on_rps< config >(max_iters, update_freq, epsilon, seed);
}
TEST(KuhnPoker, MCCFR_OS_optimistic_alternating)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::optimistic};
   run_mccfr_on_kuhn_poker< config >();
}

TEST(KuhnPoker, MCCFR_OS_optimistic_simultaneous)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::simultaneous,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::optimistic};
   run_mccfr_on_kuhn_poker< config >();
}

TEST(KuhnPoker, MCCFR_OS_lazy_alternating)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::lazy};
   run_mccfr_on_kuhn_poker< config >();
}

TEST(KuhnPoker, MCCFR_OS_lazy_simultaneous)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::simultaneous,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::lazy};
   run_mccfr_on_kuhn_poker< config >();
}

TEST(KuhnPoker, MCCFR_OS_stochastic_alternating)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::stochastic};
   run_mccfr_on_kuhn_poker< config >();
}

TEST(KuhnPoker, MCCFR_OS_stochastic_simultaneous)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::simultaneous,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::stochastic};
   run_mccfr_on_kuhn_poker< config >();
}

TEST(KuhnPoker, MCCFR_ES_stochastic)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::external_sampling,
      .weighting = rm::MCCFRWeightingMode::stochastic};
   run_mccfr_on_kuhn_poker< config >();
}

TEST(KuhnPoker, MCCFR_CS_alternating)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::chance_sampling,
      .weighting = rm::MCCFRWeightingMode::none};
   run_mccfr_on_kuhn_poker< config >();
}

TEST(KuhnPoker, MCCFR_CS_simultaneous)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::simultaneous,
      .algorithm = rm::MCCFRAlgorithmMode::chance_sampling,
      .weighting = rm::MCCFRWeightingMode::none};
   run_mccfr_on_kuhn_poker< config >();
}

TEST(KuhnPoker, CFR_PURE_alternating)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::pure_cfr,
      .weighting = rm::MCCFRWeightingMode::none};
   run_mccfr_on_kuhn_poker< config >();
}

TEST(KuhnPoker, CFR_PURE_simultaneous)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::simultaneous,
      .algorithm = rm::MCCFRAlgorithmMode::pure_cfr,
      .weighting = rm::MCCFRWeightingMode::none};
   run_mccfr_on_kuhn_poker< config >();
}

TEST(RockPaperScissors, MCCFR_OS_optimistic_alternating)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::optimistic};
   run_mccfr_on_rockpaperscissors< config >();
}

TEST(RockPaperScissors, MCCFR_OS_optimistic_simultaneous)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::simultaneous,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::optimistic};
   run_mccfr_on_rockpaperscissors< config >();
}

TEST(RockPaperScissors, MCCFR_OS_lazy_alternating)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::lazy};
   run_mccfr_on_rockpaperscissors< config >();
}

TEST(RockPaperScissors, MCCFR_OS_lazy_simultaneous)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::simultaneous,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::lazy};
   run_mccfr_on_rockpaperscissors< config >();
}

TEST(RockPaperScissors, MCCFR_OS_stochastic_alternating)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::stochastic};
   run_mccfr_on_rockpaperscissors< config >();
}

TEST(RockPaperScissors, MCCFR_OS_stochastic_simultaneous)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::simultaneous,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::stochastic};
   run_mccfr_on_rockpaperscissors< config >();
}

TEST(RockPaperScissors, MCCFR_ES_stochastic)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::external_sampling,
      .weighting = rm::MCCFRWeightingMode::stochastic};
   run_mccfr_on_rockpaperscissors< config >();
}

TEST(RockPaperScissors, MCCFR_CS_alternating)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::chance_sampling,
      .weighting = rm::MCCFRWeightingMode::none};
   run_mccfr_on_rockpaperscissors< config >();
}

TEST(RockPaperScissors, MCCFR_CS_simultaneous)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::simultaneous,
      .algorithm = rm::MCCFRAlgorithmMode::chance_sampling,
      .weighting = rm::MCCFRWeightingMode::none};
   run_mccfr_on_rockpaperscissors< config >();
}

TEST(RockPaperScissors, CFR_PURE_alternating)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::pure_cfr,
      .weighting = rm::MCCFRWeightingMode::none};
   run_mccfr_on_rockpaperscissors< config >();
}

TEST(RockPaperScissors, CFR_PURE_simultaneous)
{
   constexpr rm::MCCFRConfig config{
      .update_mode = rm::UpdateMode::simultaneous,
      .algorithm = rm::MCCFRAlgorithmMode::pure_cfr,
      .weighting = rm::MCCFRWeightingMode::none};
   run_mccfr_on_rockpaperscissors< config >();
}
