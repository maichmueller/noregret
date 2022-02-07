#pragma once

#include <algorithm>
#include <map>
#include <range/v3/all.hpp>
#include <set>
#include <utility>
#include <variant>

#include "StrategoDefs.hpp"
#include "Utils.hpp"
#include "aze/aze.h"

namespace stratego {

auto default_move_ranges() -> std::map< Token, std::function< bool(size_t) > >;

auto default_battlematrix() -> std::map< std::array< Token, 2 >, FightOutcome >;

struct Config {
   using setup_t = std::map< Position, Token >;
   using token_counter_t = std::map< Token, unsigned int >;
   using token_vector_t = std::vector< Token >;
   using position_vector_t = std::vector< Position >;
   using game_dim_variant_t = std::variant< size_t, DefinedBoardSizes, ranges::span< size_t, 2 > >;
   using token_variant_t = std::variant< std::vector< Token >, token_counter_t >;

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
   std::map< aze::Team, std::vector< Position > > start_fields;
   /// the battle matrix determining outcomes of token fights
   std::map< std::array< Token, 2 >, FightOutcome > battle_matrix;
   /// the positions of the holes for the gane
   std::vector< Position > hole_positions;
   /// holds a predicate for each token to check if a given distance is within move range
   std::map< Token, std::function< bool(size_t) > > move_ranges;

  private:
   template < typename T, typename U >
   static std::map< aze::Team, std::optional< setup_t > > _init_setups(
      const std::map< aze::Team, std::optional< setup_t > >& setups_,
      const std::map< Team, std::optional< T > >& tokenset_,
      const std::map< aze::Team, std::optional< U > >& start_fields_,
      game_dim_variant_t game_dims_);

   static std::map< aze::Team, token_counter_t > _init_tokencounters(
      const std::map< aze::Team, std::optional< token_variant_t > >& token_sets,
      const std::map< aze::Team, std::optional< Config::setup_t > >& setups_);

   static std::map< aze::Team, std::vector< Position > > _init_start_fields(
      const std::map< aze::Team, std::optional< std::vector< Position > > >& start_pos,
      const std::map< aze::Team, std::optional< Config::setup_t > >& setups_);

   static std::vector< Position > _init_hole_positions(
      const std::optional< std::vector< Position > >& hole_pos,
      game_dim_variant_t game_dims_);

  public:
   template < aze::utils::StringLiteral arg >
   static auto nullarg();

   /// the default game constructor for default sizes
   explicit Config(
      aze::Team starting_team_,
      DefinedBoardSizes game_dims = DefinedBoardSizes::small,
      bool fixed_starting_team_ = true,
      std::variant< bool, ranges::span< bool, 2 > > fixed_setups_ = false,
      size_t max_turn_count_ = 5000,
      std::map< std::array< Token, 2 >, FightOutcome > battle_matrix_ = default_battlematrix(),
      std::map< Token, std::function< bool(size_t) > > move_ranges_ = default_move_ranges());

   Config(
      aze::Team starting_team_,
      game_dim_variant_t game_dims_,
      const std::map< aze::Team, std::optional< setup_t > >& setups_,
      const std::optional< std::vector< Position > >& hole_positions_ = std::nullopt,
      bool fixed_starting_team_ = true,
      std::variant< bool, ranges::span< bool, 2 > > fixed_setups_ = false,
      size_t max_turn_count_ = 500,
      std::map< std::array< Token, 2 >, FightOutcome > battle_matrix_ = default_battlematrix(),
      std::map< Token, std::function< bool(size_t) > > move_ranges_ = default_move_ranges());

   Config(
      aze::Team starting_team_,
      game_dim_variant_t game_dims_,
      const std::optional< std::vector< Position > >& hole_positions_,
      const std::map< aze::Team, std::optional< token_variant_t > >& token_set_,
      const std::map< aze::Team, std::optional< std::vector< Position > > >& start_fields_,
      bool fixed_starting_team_ = true,
      std::variant< bool, ranges::span< bool, 2 > > fixed_setups_ = false,
      size_t max_turn_count_ = 5000,
      std::map< std::array< Token, 2 >, FightOutcome > battle_matrix_ = default_battlematrix(),
      std::map< Token, std::function< bool(size_t) > > move_ranges_ = default_move_ranges());

   Config(
      aze::Team starting_team_,
      game_dim_variant_t game_dims_,
      const std::map< aze::Team, std::optional< setup_t > >& setups_,
      const std::optional< std::vector< Position > >& hole_positions_,
      const std::map< aze::Team, std::optional< token_variant_t > >& token_set_,
      const std::map< aze::Team, std::optional< std::vector< Position > > >& start_fields_,
      bool fixed_starting_team_ = true,
      std::variant< bool, ranges::span< bool, 2 > > fixed_setups_ = false,
      size_t max_turn_count_ = 5000,
      std::map< std::array< Token, 2 >, FightOutcome > battle_matrix_ = default_battlematrix(),
      std::map< Token, std::function< bool(size_t) > > move_ranges_ = default_move_ranges());
};

auto default_setup(size_t game_dims, aze::Team team) -> std::map< Position, Token >;
auto default_setup(ranges::span< size_t, 2 > game_dims, aze::Team team)
   -> std::map< Position, Token >;

auto default_setup(size_t game_dims) -> std::map< Team, std::map< Position, Token > >;
auto default_setup(ranges::span< size_t, 2 > game_dims)
   -> std::map< Team, std::optional< std::map< Position, Token > > >;

auto default_holes(size_t game_dims) -> std::vector< Position >;

auto default_holes(ranges::span< size_t, 2 > game_dims) -> std::vector< Position >;

auto token_vector(size_t game_dim) -> std::map< aze::Team, std::vector< Token > >;

auto default_start_fields(size_t game_dim, aze::Team team) -> std::vector< Position >;
auto default_start_fields(size_t game_dim)
   -> std::map< Team, std::optional< std::vector< Position > > >;

auto default_token_sets(size_t game_dim) -> Config::token_counter_t;

auto tokens_from_setup(const std::map< aze::Team, std::optional< Config::setup_t > >& setups)
   -> std::map< aze::Team, std::optional< Config::token_counter_t > >;

auto tokens_from_setup(const Config::setup_t& setups) -> Config::token_counter_t;

auto _check_alignment(
   const std::vector< Position >& positions,
   const std::map< Position, Token >& setup) -> const std::vector< Position >&;

template < typename T, typename U >
std::map< aze::Team, std::optional< Config::setup_t > > Config::_init_setups(
   const std::map< aze::Team, std::optional< Config::setup_t > >& setups_,
   const std::map< Team, std::optional< T > >& tokenset_,
   const std::map< aze::Team, std::optional< U > >& start_fields_,
   game_dim_variant_t game_dims_)
{
   std::map< aze::Team, std::optional< setup_t > > sets;
   for(auto team : std::set{aze::Team::BLUE, aze::Team::RED}) {
      if(setups_.at(team).has_value()) {
         sets[team] = setups_.at(team).value();
      } else if(start_fields_.at(team).has_value() && tokenset_.at(team).has_value()) {
         sets[team] = std::nullopt;
      } else {
         sets[team] = std::visit(
            aze::utils::Overload{
               [&](size_t d) { return default_setup(d, team); },
               [&](DefinedBoardSizes d) { return default_setup(static_cast< size_t >(d), team); },
               [&](ranges::span< size_t, 2 > a) { return default_setup(a, team); }},
            game_dims_);
      }
   }
   return sets;
}

template <>
auto Config::nullarg< aze::utils::StringLiteral{"setups"} >()
{
   return std::map< aze::Team, std::optional< setup_t > >{
      std::pair{aze::Team::BLUE, std::nullopt}, std::pair{aze::Team::RED, std::nullopt}};
}
template <>
auto Config::nullarg< aze::utils::StringLiteral{"tokens"} >()
{
   return std::map< aze::Team, std::optional< token_variant_t > >{
      std::pair{aze::Team::BLUE, std::nullopt}, std::pair{aze::Team::RED, std::nullopt}};
}
template <>
auto Config::nullarg< aze::utils::StringLiteral{"fields"} >()
{
   return std::map< aze::Team, std::optional< std::vector< Position > > >{
      std::pair{aze::Team::BLUE, std::nullopt}, std::pair{aze::Team::RED, std::nullopt}};
}
template <>
auto Config::nullarg< aze::utils::StringLiteral{"holes"} >()
{
   return std::nullopt;
}

}  // namespace stratego