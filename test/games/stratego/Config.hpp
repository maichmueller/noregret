#pragma once

#include <algorithm>
#include <map>

#include "Piece.h"
#include "aze/aze.h"

namespace stratego {

auto _default_mr()
{
   std::map< Token, int > mr;
   for(auto token_value : std::array{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 99}) {
      Token token = Token(token_value);
      if(token == Token::scout) {
         mr[token] = std::numeric_limits< int >::infinity();
      } else if(token == Token::flag or token == Token::bomb) {
         mr[token] = 0;
      } else {
         mr[token] = 1;
      }
   }
   return mr;
}

auto _default_bm()
{
   std::map< std::array< Token, 2 >, int > bm;
   for(auto i : std::array{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 99}) {
      for(auto j : std::array{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 99}) {
         if(i < j) {
            bm[{Token(i), Token(j)}] = -1;
            bm[{Token(j), Token(i)}] = 1;
         } else if(i == j) {
            bm[{Token(i), Token(i)}] = 0;
         }
      }
      bm[{Token(i), Token::flag}] = 1;
      if(Token(i) == Token::miner) {
         bm[{Token(i), Token::bomb}] = 1;
      } else {
         bm[{Token(i), Token::bomb}] = -1;
      }
   }
   return bm;
}

auto _default_setups(size_t game_size) {
   std::map< Team, std::map< Position< int, 2 >, Token > > setups;



   return setups;
}

struct Config {
   Team starting_team;
   size_t game_size = 5;
   std::map< Team, std::map< Position< int, 2 >, Token > > setups = _default_setups(game_size);
   size_t max_turn_count = 500;
   std::map< std::array< Token, 2 >, int > battle_matrix = _default_bm();
   std::map< Token, int > move_ranges = _default_mr();
};

}