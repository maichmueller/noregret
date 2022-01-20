
#include "Config.hpp"

namespace stratego {

std::map< Token, int > _default_mr()
{
   std::map< Token, int > mr;
   for(auto token_value : std::array{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 99}) {
      auto token = Token(token_value);
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

std::map< std::array< Token, 2 >, FightOutcome > _default_bm()
{
   std::map< std::array< Token, 2 >, FightOutcome > bm;
   for(auto i : std::array{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 99}) {
      for(auto j : std::array{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 99}) {
         if(i < j) {
            bm[{Token(i), Token(j)}] = FightOutcome::death;
            bm[{Token(j), Token(i)}] = FightOutcome::kill;
         } else if(i == j) {
            bm[{Token(i), Token(i)}] = FightOutcome::stalemate;
         }
      }
      bm[{Token(i), Token::flag}] = FightOutcome::kill;
      if(Token(i) == Token::miner) {
         bm[{Token(i), Token::bomb}] = FightOutcome::kill;
      } else {
         bm[{Token(i), Token::bomb}] = FightOutcome::death;
      }
   }
   return bm;
}

std::map< aze::Team, std::map< Position, Token > > _default_setups(size_t game_dims)
{
   using Team = aze::Team;
   std::map< Team, std::map< Position, Token > > setups;

   // TODO: Fill

   return setups;
}
std::map< aze::Team, std::map< Position, Token > > _default_setups(std::array< size_t, 2 > game_dims)
{
   if(game_dims[0] == game_dims[1] and std::set{5, 7, 10}.contains(game_dims[0])) {
      return _default_setups(game_dims[0]);
   } else {
      throw std::invalid_argument("Cannot provide default setups for non-default game dimensions.");
   }
}

std::vector< Position > _default_obs(size_t game_dims)
{
   if(game_dims == 5)
      return {{2, 2}};
   else if(game_dims == 7)
      return {{3, 1}, {3, 5}};
   else if(game_dims == 10)
      return {{4, 2}, {5, 2}, {4, 3}, {5, 3}, {4, 6}, {5, 6}, {4, 7}, {5, 7}};
   else
      throw std::invalid_argument(
         "'dimension' not in {5, 7, 10}. User has to provide custom obstacle positions.");
}
std::vector< Position > _default_obs(std::array< size_t, 2 > game_dims)
{
   if(game_dims[0] == game_dims[1] and std::set< size_t >{5, 7, 10}.contains(game_dims[0])) {
      return _default_obs(game_dims[0]);
   } else {
      throw std::invalid_argument(
         "Cannot provide default obstacle positions for non-default game dimensions.");
   }
}

std::map< aze::Team, std::vector< Token > > _token_set(size_t game_dim)
{
   switch(game_dim) {
      case 5: {
         using seq = std::index_sequence< 0, 1, 2, 2, 2, 3, 3, 10, 11, 11 >;
         return {
            std::pair{aze::Team::BLUE, make_tokens(seq())},
            std::pair{aze::Team::RED, make_tokens(seq())}};
      }
      case 7: {
         using seq = std::
            index_sequence< 0, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 10, 11, 11, 11, 11 >;
         return {
            std::pair{aze::Team::BLUE, make_tokens(seq())},
            std::pair{aze::Team::RED, make_tokens(seq())}};
      }
      case 10: {
         using seq = std::index_sequence<
            0,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            3,
            4,
            4,
            4,
            4,
            5,
            5,
            5,
            5,
            6,
            6,
            6,
            6,
            7,
            7,
            7,
            8,
            8,
            9,
            10,
            11,
            11,
            11,
            11,
            11,
            11 >;
         return {
            std::pair{aze::Team::BLUE, make_tokens(seq())},
            std::pair{aze::Team::RED, make_tokens(seq())}};
      }
      default:
         throw std::invalid_argument("Cannot provide tokenset for non-default game dimensions.");
   }
}

std::map< aze::Team, std::vector< Token > > _gen_tokensets(const std::map< aze::Team, std::map< Position, Token > >& setups)
{
   std::map< aze::Team, std::vector< Token > > tokens;
   for(const auto& [team, setup] : setups) {
      for(const auto& [_, token] : setup) {
         tokens[team].emplace_back(token);
      }
   }
   return tokens;
}

std::map< aze::Team, std::map< Token, unsigned int > > to_tokencounters(const std::map< aze::Team, std::vector< Token > >& token_vecs)
{
   std::map< aze::Team, std::map< Token, unsigned int > > m;
   for(const auto& [team, vec] : token_vecs) {
      m[team] = aze::utils::counter(vec);
   }
   return m;
}

std::vector< Position > _default_start_positions(size_t game_dim, aze::Team team)
{
   using aze::Team;
   if(std::set{Team::RED, Team::BLUE}.contains(team))
      throw std::invalid_argument("'team' not in {0, 1}.");

   switch(game_dim) {
      case 5: {
         if(team == Team::BLUE)
            return {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 0}, {1, 1}, {1, 2}, {1, 3}, {1, 4}};
         else
            return {{4, 0}, {4, 1}, {4, 2}, {4, 3}, {4, 4}, {3, 0}, {3, 1}, {3, 2}, {3, 3}, {3, 4}};
      }
      case 7: {
         if(team == Team::BLUE)
            return {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {0, 6},
                    {1, 0}, {1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6},
                    {2, 0}, {2, 1}, {2, 2}, {2, 3}, {2, 4}, {2, 5}, {2, 6}};
         else
            return {{4, 0}, {4, 1}, {4, 2}, {4, 3}, {4, 4}, {4, 5}, {4, 6},
                    {5, 0}, {5, 1}, {5, 2}, {5, 3}, {5, 4}, {5, 5}, {5, 6},
                    {6, 0}, {6, 1}, {6, 2}, {6, 3}, {6, 4}, {6, 5}, {6, 6}};
      }
      case 10: {
         if(team == Team::BLUE)
            return {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {0, 6}, {0, 7}, {0, 8}, {0, 9},
                    {1, 0}, {1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {1, 7}, {1, 8}, {1, 9},
                    {2, 0}, {2, 1}, {2, 2}, {2, 3}, {2, 4}, {2, 5}, {2, 6}, {2, 7}, {2, 8}, {2, 9},
                    {3, 0}, {3, 1}, {3, 2}, {3, 3}, {3, 4}, {3, 5}, {3, 6}, {3, 7}, {3, 8}, {3, 9}};

         else
            return {{6, 0}, {6, 1}, {6, 2}, {6, 3}, {6, 4}, {6, 5}, {6, 6}, {6, 7}, {6, 8}, {6, 9},
                    {7, 0}, {7, 1}, {7, 2}, {7, 3}, {7, 4}, {7, 5}, {7, 6}, {7, 7}, {7, 8}, {7, 9},
                    {8, 0}, {8, 1}, {8, 2}, {8, 3}, {8, 4}, {8, 5}, {8, 6}, {8, 7}, {8, 8}, {8, 9},
                    {9, 0}, {9, 1}, {9, 2}, {9, 3}, {9, 4}, {9, 5}, {9, 6}, {9, 7}, {9, 8}, {9, 9}};
      }
      default: {
         throw std::invalid_argument("'shape' not in {5, 7, 10}.");
      }
   }
}

std::map< aze::Team, std::vector< Position > > _check_alignment(
   const std::map< aze::Team, std::vector< Position > >& positions,
   const std::map< aze::Team, std::map< Position, Token > >& setups)
{
   for(const auto& [team, setup] : setups) {
      const auto& pos_vec = positions.at(team);
      if(pos_vec.size() != setup.size()
         or std::any_of(pos_vec.begin(), pos_vec.end(), [&setup = setup](const Position& pos) {
               return not setup.contains(pos);
            })) {
         throw std::invalid_argument(
            "Passed starting positions parameter and setup parameter do not match for team "
            + std::to_string(static_cast< int >(team)) + " .");
      }
   }
   return positions;
}

std::map< aze::Team, std::vector< Position > > _positions_from_setups(
   std::map< aze::Team, std::map< Position, Token > >& setups)
{
   std::map< aze::Team, std::vector< Position > > positions;
   for(const auto& [team, setup] : setups) {
      std::vector< Position > pos;
      pos.reserve(setup.size());
      std::transform(setup.begin(), setup.end(), std::back_inserter(pos), [](const auto& pair) {
         return pair.first;
      });
   }
   return positions;
}


}