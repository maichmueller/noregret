#pragma once

#include <aze/aze.h>

#include <string_view>
#include <xtensor/xtensor.hpp>


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

using Position = aze::Position< int, 2 >;
using Piece = aze::Piece< Position, Token >;
using Board = xt::xtensor< std::optional< Piece >, 2 >;
using Team = aze::Team;
using Status = aze::Status;

enum class FightOutcome { death = -1, kill = 1, stalemate = 0 };


inline auto team_name(Team t)
{
   constexpr aze::utils::CEMap< Team, std::string_view, 2 > team_name_lookup = {
      std::pair{Team::BLUE, std::string_view("BLUE")},
      std::pair{Team::RED, std::string_view("RED")},
   };
   return std::string(team_name_lookup.at(t));
}

inline auto token_name(Token t)
{
   constexpr aze::utils::CEMap< Token, std::string_view, 13 > token_name_lookup = {
      std::pair{Token::flag, std::string_view("Flag")},
      std::pair{Token::spy, std::string_view("spy")},
      std::pair{Token::scout, std::string_view("scout")},
      std::pair{Token::miner, std::string_view("miner")},
      std::pair{Token::sergeant, std::string_view("sergeant")},
      std::pair{Token::lieutenant, std::string_view("lieutenant")},
      std::pair{Token::captain, std::string_view("captain")},
      std::pair{Token::major, std::string_view("major")},
      std::pair{Token::colonel, std::string_view("colonel")},
      std::pair{Token::general, std::string_view("general")},
      std::pair{Token::marshall, std::string_view("marshall")},
      std::pair{Token::bomb, std::string_view("bomb")},
      std::pair{Token::hole, std::string_view("hole")},
   };
   return std::string(token_name_lookup.at(t));
}

inline auto& operator<<(std::ostream& os, Token t)
{
   os << token_name(t);
   return os;
}



}  // namespace stratego