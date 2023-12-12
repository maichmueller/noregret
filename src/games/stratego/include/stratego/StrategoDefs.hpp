#pragma once

#include <optional>
#include <string_view>
#include <xtensor/xtensor.hpp>

#include "Position.h"

namespace stratego {

// specify exactly what statuses and teams are present in this game
enum class Status : int16_t {
   ONGOING = 404,
   TIE = 0,
   WIN_BLUE = 1,
   WIN_RED = -1,
};
enum class Team : int8_t { BLUE = 0, RED = 1, NEUTRAL = 2 };

enum class Token : int8_t {
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
   hole = 99
};

enum DefinedBoardSizes : int8_t { small = 5, medium = 7, large = 10 };

class Piece;

using Position2D = Position< int, 2 >;
using Board = xt::xtensor< std::optional< Piece >, 2 >;

enum class FightOutcome : int8_t { death = -1, kill = 1, tie = 0 };

}  // namespace stratego
