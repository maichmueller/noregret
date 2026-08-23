

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <random>

#include "fixtures.hpp"
#include "leduc_poker/leduc_poker.hpp"
#include "testing_utils.hpp"

using namespace leduc;

//////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// Zero-sum over random playouts ////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

TEST(BigLeducRandomPlayouts, zero_sum_invariant)
{
   std::mt19937 rng{42};
   auto draw = [&rng](size_t n) { return std::uniform_int_distribution< size_t >(0, n - 1)(rng); };

   for(int trial = 0; trial < 200; ++trial) {
      auto local_state = leduc::State{big_leduc_config()};
      int guard = 0;
      while(not local_state.is_terminal()) {
         ASSERT_LT(guard++, 64) << "trial " << trial;
         if(local_state.active_player() == Player::chance) {
            auto outcomes = local_state.chance_actions();
            ASSERT_FALSE(outcomes.empty());
            local_state.apply_action(outcomes[draw(outcomes.size())]);
         } else {
            auto legal = local_state.actions();
            ASSERT_FALSE(legal.empty());
            local_state.apply_action(legal[draw(legal.size())]);
         }
      }
      auto payoffs = local_state.payoff();
      double sum = 0.;
      for(double p : payoffs) {
         sum += p;
      }
      EXPECT_NEAR(sum, 0., 1e-9) << "trial " << trial
                                 << fmt::format(" payoffs={}", fmt::join(payoffs, ", "));
   }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////// Full-tree size vs. HS paper Table 1 report ////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

/// The HS-schedules paper (arXiv:2404.09097, Table 1 / v1 Table 2) reports for Big Leduc poker:
/// #Histories ~= 6.2e6 and #Leaves ~= 4.0e6. Their reference implementation suppresses fold
/// when no bet is outstanding while this engine always offers it, so we can only expect
/// agreement up to a constant factor; assert the order of magnitude instead.
TEST(BigLeducTreeSize, full_tree_scale_matches_hs_reported_scale)
{
   size_t histories = 0;
   size_t leaves = 0;

   std::function< void(leduc::State) > dfs = [&](leduc::State current) {
      ++histories;
      if(current.is_terminal()) {
         ++leaves;
         return;
      }
      if(current.active_player() == Player::chance) {
         for(auto card : current.chance_actions()) {
            auto child = current;
            child.apply_action(card);
            dfs(std::move(child));
         }
      } else {
         for(auto action : current.actions()) {
            auto child = current;
            child.apply_action(action);
            dfs(std::move(child));
         }
      }
   };
   dfs(leduc::State{big_leduc_config()});

   std::cout << "Big Leduc full tree: " << histories << " histories, " << leaves
             << " leaves (HS reports ~6.2e6 / ~4.0e6)\n";
   EXPECT_NEAR(std::log10(static_cast< double >(histories)), std::log10(6.2e6), 0.75);
   EXPECT_NEAR(std::log10(static_cast< double >(leaves)), std::log10(4.0e6), 0.75);
}
