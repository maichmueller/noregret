
#ifndef NOR_BATTLESHIP_FIXTURES_HPP
#define NOR_BATTLESHIP_FIXTURES_HPP

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <vector>

#include "battleship/battleship.hpp"

struct BattleshipState: public ::testing::Test {
   battleship::State state{};
};

/// a fully scripted battleship game
struct GameScript {
   battleship::Config config{};
   /// fleets as lists of (cell_a, cell_b) pairs
   std::vector< std::array< battleship::Cell, 2 > > fleet_one;
   std::vector< std::array< battleship::Cell, 2 > > fleet_two;
   /// the shots of every fire round (round i: player one fires shots[i], then player two)
   std::vector< battleship::Cell > shots_one;
   std::vector< battleship::Cell > shots_two;
};

/// plays out a scripted game: alternating secret placements ship-by-ship followed by
/// alternating fire rounds, stopping early once the state becomes terminal or the actor
/// whose turn it is has no scripted shot left
inline battleship::State play_script(const GameScript& script)
{
   using namespace battleship;
   State state{script.config};
   const auto n_ships = std::max(script.fleet_one.size(), script.fleet_two.size());
   for(size_t ship = 0; ship < n_ships; ++ship) {
      if(ship < script.fleet_one.size()) {
         state.apply_action(Place{script.fleet_one[ship][0], script.fleet_one[ship][1]});
      }
      if(ship < script.fleet_two.size()) {
         state.apply_action(Place{script.fleet_two[ship][0], script.fleet_two[ship][1]});
      }
   }
   size_t next_shot_one = 0;
   size_t next_shot_two = 0;
   while(not state.terminal()) {
      if(state.phase() == Phase::one_fire) {
         if(next_shot_one >= script.shots_one.size()) {
            break;
         }
         state.apply_action(Fire{script.shots_one[next_shot_one++]});
      } else if(state.phase() == Phase::two_fire) {
         if(next_shot_two >= script.shots_two.size()) {
            break;
         }
         state.apply_action(Fire{script.shots_two[next_shot_two++]});
      } else {
         break;
      }
   }
   return state;
}

class BattleshipPayoffParamsF:
    public ::testing::TestWithParam< std::tuple< GameScript, double, double > > {};

#endif  // NOR_BATTLESHIP_FIXTURES_HPP
