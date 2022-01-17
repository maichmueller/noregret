#pragma once

#include <algorithm>
#include <map>
#include <set>
#include <utility>
#include <variant>

#include "StrategoDefs.hpp"
#include "aze/aze.h"

namespace stratego {

auto _default_mr()
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

auto _default_setups(size_t game_dims)
{
   using Team = aze::Team;
   std::map< Team, std::map< Position, Token > > setups;

   return setups;
}
auto _default_setups(std::array< size_t, 2 > game_dims)
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
auto _default_obs(std::array< size_t, 2 > game_dims)
{
   if(game_dims[0] == game_dims[1] and std::set{5, 7, 10}.contains(game_dims[0])) {
      return _default_obs(game_dims[0]);
   } else {
      throw std::invalid_argument(
         "Cannot provide default obstacle positions for non-default game dimensions.");
   }
}
template < size_t... Is >
constexpr auto make_tokens(std::index_sequence< Is... >)
{
   return std::vector{Token(Is)...};
}

inline std::map< aze::Team, std::vector< Token > > _token_set(size_t game_dim)
{
   switch(game_dim) {
      case 5: {
         using seq = std::index_sequence< 0, 1, 2, 2, 2, 3, 3, 10, 11, 11 >;
         return {make_tokens(seq()), make_tokens(seq())};
      }
      case 7: {
         using seq = std::
            index_sequence< 0, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 10, 11, 11, 11, 11 >;
         return {make_tokens(seq()), make_tokens(seq())};
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
         return {make_tokens(seq()), make_tokens(seq())};
      }
      default:
         throw std::invalid_argument("Cannot provide tokenset for non-default game dimensions.");
   }
}

auto _gen_tokensets(const std::map< aze::Team, std::map< Position, Token > >& setups)
{
   std::map< aze::Team, std::vector< Token > > tokens;
   for(const auto& [team, setup] : setups) {
      for(const auto& [_, token] : setup) {
         tokens[team].emplace_back(token);
      }
   }
   return tokens;
}

auto to_tokencounters(const std::map< aze::Team, std::vector< Token > >& token_vecs)
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

struct Config {
   using setup_type = std::map< Position, Token >;
   using token_counter = std::map< Token, unsigned int >;

   aze::Team starting_team;
   std::array< size_t, 2 > game_dims;
   size_t max_turn_count;
   bool fixed_setups;
   std::optional< std::map< aze::Team, setup_type > > setups;
   std::optional< std::map< aze::Team, token_counter > > token_counters;
   std::optional< std::map< aze::Team, std::vector< Position > > > start_positions;
   std::map< std::array< Token, 2 >, int > battle_matrix;
   std::vector< Position > obstacle_positions;
   std::map< Token, int > move_ranges;

   Config(
      aze::Team starting_team_,
      std::variant< size_t, std::array< size_t, 2 > > game_dims_ = size_t(5),
      size_t max_turn_count_ = 500,
      bool fixed_setups_ = false,
      std::optional< std::map< aze::Team, setup_type > > setups_ = std::nullopt,
      std::optional< std::map< aze::Team, std::vector< Token > > > token_set_ = std::nullopt,
      std::optional< std::map< aze::Team, std::vector< Position > > > start_positions_ =
         std::nullopt,
      std::map< std::array< Token, 2 >, int > battle_matrix_ = _default_bm(),
      std::optional< std::vector< Position > > obstacle_positions_ = std::nullopt,
      std::map< Token, int > move_ranges_ = _default_mr())
       : starting_team(starting_team_),
         game_dims(std::visit(
            aze::utils::Overload{
               [](size_t d) {
                  return std::array{d, d};
               },
               [](std::array< size_t, 2 > d) { return d; }},
            game_dims_)),
         max_turn_count(max_turn_count_),
         fixed_setups(fixed_setups_),
         setups(
            setups_.has_value() ? setups_.value()
                                : std::visit(
                                   aze::utils::Overload{
                                      [](size_t d) { return _default_setups(d); },
                                      [](std::array< size_t, 2 > a) { return _default_setups(a); }},
                                   game_dims_)),
         token_counters(to_tokencounters(
            setups_.has_value()
               ? _gen_tokensets(setups_.value())
               : (token_set_.has_value()
                     ? std::move(token_set_.value())
                     : throw std::invalid_argument("No setup passed and no tokenset passed. Either "
                                                   "of these need to be set.")))),
         start_positions(
            start_positions_.has_value()
               ? setups_.has_value() ? _check_alignment(start_positions_.value(), setups_.value())
                                     : start_positions_.value()
               : _positions_from_setups(setups_.value())),
         battle_matrix(std::move(battle_matrix_)),
         obstacle_positions(
            obstacle_positions_.has_value()
               ? std::move(obstacle_positions_.value())
               : std::visit(
                  aze::utils::Overload{
                     [](size_t d) { return _default_obs(d); },
                     [](std::array< size_t, 2 > a) { return _default_obs(a); }},
                  game_dims_)),
         move_ranges(std::move(move_ranges_))
   {
   }
};

}  // namespace stratego