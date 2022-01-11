#pragma once

#include <aze/aze.h>

#include <array>

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

class PieceStratego: public Piece< Position< int, 2 >, Token > {
  public:
   using base_type = Piece< Position< int, 2 >, Token >;

   // inheriting the base constructors with the next command!
   using base_type::base_type;
};