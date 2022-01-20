#pragma once

#include <xtensor/xtensor.hpp>

#include <aze/aze.h>
#include "aze/game/Position.h"
#include "aze/game/Piece.h"

//#include "aze/game/Piece.h"


namespace stratego {

enum class Token {
   flag = 0,
   spy = 1,
   scout = 2,
   miner = 3,
   sergeant = 4,
   lieutenant = 5,
   captain = 6,
   major = 7,
   colonel = 8,
   general = 9,
   marshall = 10,
   bomb = 11,
   obstacle = 99
};

enum class FightOutcome {
   death = -1,
   kill = 1,
   stalemate = 0
};

using Position = aze::Position<int, 2>;
using Piece = aze::Piece< Position, Token >;
using Board = xt::xtensor< std::optional< Piece >, 2 >;

}  // namespace stratego

namespace aze {
// specify exactly what statuses and teams are present in this game
enum class Status {
   ONGOING = 404,
   TIE = 0,
   WIN_BLUE = 1,
   WIN_RED = -1,
};
enum class Team { BLUE = 0, RED = 1 };


}  // namespace aze