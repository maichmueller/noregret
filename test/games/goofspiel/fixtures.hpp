
#ifndef NOR_GOOFSPIEL_FIXTURES_HPP
#define NOR_GOOFSPIEL_FIXTURES_HPP

#include <gtest/gtest.h>

#include <random>
#include <variant>

#include "goofspiel/goofspiel.hpp"

struct GoofspielState: public ::testing::Test {
   goofspiel::State state{goofspiel::GoofspielConfig{}};
};

/// one scripted transition of a goofspiel game: either a chance outcome or a bid
using GoofspielStep = std::variant< goofspiel::PrizeCard, goofspiel::Bid >;

/// applies one full round (prize reveal, both commits, resolve confirmation)
inline void play_round(goofspiel::State& state, uint8_t prize, uint8_t bid_one, uint8_t bid_two)
{
   state.apply_action(goofspiel::PrizeCard{prize});
   state.apply_action(goofspiel::Bid{bid_one});
   state.apply_action(goofspiel::Bid{bid_two});
   state.apply_action(goofspiel::PrizeCard{0});
}

/// applies a scripted sequence of steps
inline void apply_steps(goofspiel::State& state, const std::vector< GoofspielStep >& steps)
{
   for(const auto& step : steps) {
      if(std::holds_alternative< goofspiel::PrizeCard >(step)) {
         state.apply_action(std::get< goofspiel::PrizeCard >(step));
      } else {
         state.apply_action(std::get< goofspiel::Bid >(step));
      }
   }
}

/// plays a whole game by drawing uniformly among legal moves; returns false on any internal
/// inconsistency (checked softly since gtest fatal assertions may not leave a value-returning
/// function)
template < typename Rng >
bool random_playout(
   goofspiel::State& state,
   Rng&& rng,
   std::array< int32_t, 2 >* final_scores = nullptr,
   size_t* n_reveals = nullptr
)
{
   size_t reveals = 0;
   while(not state.is_terminal()) {
      switch(state.phase()) {
         case goofspiel::Phase::prize_reveal: {
            auto outcomes = state.chance_actions();
            if(outcomes.empty()) {
               ADD_FAILURE() << "no chance outcomes during the reveal phase";
               return false;
            }
            std::uniform_int_distribution< size_t > dist(0, outcomes.size() - 1);
            state.apply_action(outcomes[dist(rng)]);
            ++reveals;
            break;
         }
         case goofspiel::Phase::commit_p1:
         case goofspiel::Phase::commit_p2: {
            auto acts = state.actions();
            if(acts.empty()) {
               ADD_FAILURE() << "no legal bids during a commit phase";
               return false;
            }
            std::uniform_int_distribution< size_t > dist(0, acts.size() - 1);
            state.apply_action(acts[dist(rng)]);
            break;
         }
         case goofspiel::Phase::resolve: {
            state.apply_action(goofspiel::PrizeCard{0});
            break;
         }
      }
   }
   if(n_reveals != nullptr) {
      *n_reveals = reveals;
   }
   if(final_scores != nullptr) {
      *final_scores = {state.score(goofspiel::Player::one), state.score(goofspiel::Player::two)};
   }
   return true;
}

class GoofspielTerminalParamsF:
    public ::testing::TestWithParam< std::tuple<
       goofspiel::GoofspielConfig,  // the state config
       std::vector< GoofspielStep >,  // the full scripted outcome/action sequence
       bool  // whether the state is terminal afterwards
       > > {};

class GoofspielPayoffParamsF:
    public ::testing::TestWithParam< std::tuple<
       goofspiel::GoofspielConfig,  // the state config
       std::vector< std::tuple< uint8_t, uint8_t, uint8_t > >,  // rounds of (prize, bid1, bid2)
       double,  // expected payoff of player one
       double  // expected payoff of player two
       > > {};

#endif  // NOR_GOOFSPIEL_FIXTURES_HPP
