#pragma once

#include <algorithm>
#include <map>
#include <range/v3/all.hpp>
#include <set>
#include <utility>
#include <variant>

#include "StrategoDefs.hpp"
#include "aze/aze.h"
#include "Utils.hpp"

namespace stratego {

auto default_move_ranges() -> std::map< Token, std::function< bool(size_t) > >;

auto default_battlematrix() -> std::map< std::array< Token, 2 >, FightOutcome >;

auto _default_setup(size_t game_dims) -> std::map< aze::Team, std::map< Position, Token > >;
auto _default_setups(std::array< size_t, 2 > game_dims)
   -> std::map< aze::Team, std::map< Position, Token > >;

auto default_holes(size_t game_dims) -> std::vector< Position >;

auto default_holes(ranges::span< size_t, 2 > game_dims) -> std::vector< Position >;

auto token_vector(size_t game_dim) -> std::map< aze::Team, std::vector< Token > >;

auto default_start_positions(size_t game_dim, aze::Team team) -> std::vector< Position >;

auto _check_alignment(
   const std::vector< Position >& positions,
   const std::map< Position, Token >& setup) -> const std::vector< Position >&;


struct Config {
   using setup_t = std::map< Position, Token >;
   using token_counter_t = std::map< Token, unsigned int >;
   using token_vector_t = std::vector< Token >;
   using position_vector_t = std::vector< Position >;

   /// the team that starts the game with the first turn
   aze::Team starting_team;
   /// whether the starting team is always the same
   bool fixed_starting_team;
   /// the board dimensions in (x,y)
   std::array< size_t, 2 > game_dims;
   /// the maximum number of turns to play before the game is counted as a draw
   size_t max_turn_count;
   /// whether a given setup in the config is to be seen as fixed (no resampling on reset)
   std::array< bool, 2 > fixed_setups;
   /// an optional setup for each team
   std::map< aze::Team, std::optional< setup_t > > setups;
   /// the tokens that each player gets to place on the board
   std::map< aze::Team, token_counter_t > token_counters;
   /// the start positions that each team can use to place tokens
   std::map< aze::Team, std::vector< Position > > start_positions;
   /// the battle matrix determining outcomes of token fights
   std::map< std::array< Token, 2 >, FightOutcome > battle_matrix;
   /// the positions of the holes for the gane
   std::vector< Position > hole_positions;
   /// holds a predicate for each token to check if a given distance is within move range
   std::map< Token, std::function< bool(size_t) > > move_ranges;

  private:
   static std::map< aze::Team, std::optional< setup_t > > _init_setups(
      const std::map< aze::Team, std::optional< setup_t > >& setups_,
      const std::map< aze::Team, std::optional< std::vector< Token > > >& tokenset_,
      const std::map< aze::Team, std::optional< std::vector< Position > > >& start_positions_,
      std::variant< size_t, ranges::span< size_t, 2 > > game_dims_);

   static std::map< aze::Team, token_counter_t > _init_tokencounters(
      const std::map< aze::Team, std::optional< std::vector< Token > > >& token_sets,
      const std::map< aze::Team, std::optional< Config::setup_t > >& setups_);

   static std::map< aze::Team, std::vector< Position > > _init_start_positions(
      const std::map< aze::Team, std::optional< std::vector< Position > > >& start_pos,
      const std::map< aze::Team, std::optional< Config::setup_t > >& setups_);

   static std::vector< Position > _init_hole_positions(
      const std::optional< std::vector< Position > >& hole_pos,
      std::variant< size_t, ranges::span< size_t, 2 > > game_dims_);

  public:
   template < typename T >
   static constexpr std::map< aze::Team, std::optional< T > > null_arg()
   {
      return {std::pair{aze::Team::BLUE, std::nullopt}, std::pair{aze::Team::RED, std::nullopt}};
   }

   explicit Config(
      aze::Team starting_team_,
      bool fixed_starting_team_ = true,
      std::variant< size_t, ranges::span< size_t, 2 > > game_dims_ = size_t(5),
      size_t max_turn_count_ = 500,
      std::variant< bool, ranges::span< bool, 2 > > fixed_setups_ = false,
      const std::map< aze::Team, std::optional< setup_t > >& setups_ = null_arg< setup_t >(),
      const std::optional< std::vector< Position > >& hole_positions_ = std::nullopt,
      const std::map< aze::Team, std::optional< std::vector< Token > > >& token_set_ =
         null_arg< token_vector_t >(),
      const std::map< aze::Team, std::optional< std::vector< Position > > >& start_positions_ =
         null_arg< position_vector_t >(),
      std::map< std::array< Token, 2 >, FightOutcome > battle_matrix_ = default_battlematrix(),
      std::map< Token, std::function< bool(size_t) > > move_ranges_ = default_move_ranges())
       : starting_team(starting_team_),
         fixed_starting_team(fixed_starting_team_),
         game_dims(std::visit(
            aze::utils::Overload{
               [](size_t d) {
                  return std::array{d, d};
               },
               [](ranges::span< size_t, 2 > d) {
                  return std::array{d[0], d[1]};
                  ;
               }},
            game_dims_)),
         max_turn_count(max_turn_count_),
         fixed_setups(std::visit(
            aze::utils::Overload{
               [](bool value) {
                  return std::array{value, value};
               },
               [](ranges::span< bool, 2 > d) {
                  return std::array{d[0], d[1]};
                  ;
               }},
            fixed_setups_)),
         setups(_init_setups(setups_, token_set_, start_positions_, game_dims_)),
         token_counters(_init_tokencounters(token_set_, setups_)),
         start_positions(_init_start_positions(start_positions_, setups_)),
         battle_matrix(std::move(battle_matrix_)),
         hole_positions(_init_hole_positions(hole_positions_, game_dims_)),
         move_ranges(std::move(move_ranges_))
   {
      for(int i = 0; i < 2; ++i) {
         if(utils::flatten_counter(token_counters[aze::Team(i)]).size()
            != start_positions[aze::Team(i)].size()) {
            throw std::invalid_argument(
               "Token counters and start position vectors do not match in size");
         }
      }
   }
};

}  // namespace stratego