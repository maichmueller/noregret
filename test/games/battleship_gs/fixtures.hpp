
#ifndef NOR_BATTLESHIP_GS_FIXTURES_HPP
#define NOR_BATTLESHIP_GS_FIXTURES_HPP

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <vector>

#include "battleship_gs/battleship_gs.hpp"

namespace bgs_test {

/// canonical 'Place' action covering `cells` (which must form a straight contiguous line in
/// canonical orientation; single cells use the unique horizontal encoding)
inline battleship_gs::Place place_of(const std::vector< battleship_gs::Cell >& cells)
{
   using namespace battleship_gs;
   EXPECT_GE(cells.size(), 1u);
   EXPECT_LE(cells.size(), max_ship_length);
   if(cells.size() == 1 or cells.front().row == cells.at(1).row) {
      return Place{cells.front(), int8_t{0}, int8_t{1}};
   }
   return Place{cells.front(), int8_t{1}, int8_t{0}};
}

inline battleship_gs::Place place_of(
   battleship_gs::Cell a,
   battleship_gs::Cell b,
   std::initializer_list< battleship_gs::Cell > rest = {}
)
{
   std::vector< battleship_gs::Cell > cells{a, b};
   cells.insert(cells.end(), rest);
   return place_of(cells);
}

}  // namespace bgs_test

struct BattleshipGsState: public ::testing::Test {
   battleship_gs::State state{};
};

/// a fully scripted general-sum battleship game
struct GameScript {
   battleship_gs::Config config{};
   /// fleets as lists of ships, each ship an ordered list of its cells
   std::vector< std::vector< battleship_gs::Cell > > fleet_one;
   std::vector< std::vector< battleship_gs::Cell > > fleet_two;
   /// the shots of every fire round (round i: player one fires shots[i], then player two)
   std::vector< battleship_gs::Cell > shots_one;
   std::vector< battleship_gs::Cell > shots_two;
};

/// plays out a scripted game: alternating secret placements ship-by-ship followed by alternating
/// fire rounds, stopping early once the state becomes terminal or the actor whose turn it is has
/// no scripted shot left
inline battleship_gs::State play_script(const GameScript& script)
{
   using namespace battleship_gs;
   State state{script.config};
   const auto n_ships = std::max(script.fleet_one.size(), script.fleet_two.size());
   for(size_t ship = 0; ship < n_ships; ++ship) {
      if(ship < script.fleet_one.size()) {
         state.apply_action(Action{bgs_test::place_of(script.fleet_one.at(ship))});
      }
      if(ship < script.fleet_two.size()) {
         state.apply_action(Action{bgs_test::place_of(script.fleet_two.at(ship))});
      }
   }
   size_t next_shot_one = 0;
   size_t next_shot_two = 0;
   while(not state.terminal()) {
      if(state.phase() == Phase::one_fire) {
         if(next_shot_one >= script.shots_one.size()) {
            break;
         }
         state.apply_action(Action{Fire{script.shots_one[next_shot_one++]}});
      } else if(state.phase() == Phase::two_fire) {
         if(next_shot_two >= script.shots_two.size()) {
            break;
         }
         state.apply_action(Action{Fire{script.shots_two[next_shot_two++]}});
      } else {
         break;
      }
   }
   return state;
}

class BattleshipGsPayoffParamsF:
    public ::testing::TestWithParam< std::tuple< GameScript, double, double > > {};

#endif  // NOR_BATTLESHIP_GS_FIXTURES_HPP
