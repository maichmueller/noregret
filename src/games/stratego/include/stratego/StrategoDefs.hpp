#pragma once

#include <string_view>
#include <xtensor/xtensor.hpp>

#include "aze/aze.h"

namespace aze {

// specify exactly what statuses and teams are present in this game
enum class Status {
   ONGOING = 404,
   TIE = 0,
   WIN_BLUE = 1,
   WIN_RED = -1,
};
enum class Team { BLUE = 0, RED = 1, NEUTRAL = 2 };

}  // namespace aze

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
   hole = 99
};

enum DefinedBoardSizes { small = 5, medium = 7, large = 10 };

using Position = aze::Position< int, 2 >;
using Piece = aze::Piece< Position, Token >;
using Board = xt::xtensor< std::optional< Piece >, 2 >;
using Team = aze::Team;
using Status = aze::Status;

enum class FightOutcome { death = -1, kill = 1, tie = 0 };

}  // namespace stratego
