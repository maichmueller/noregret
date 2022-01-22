#pragma once

#include <aze/aze.h>

#include <algorithm>
#include <map>
#include <set>
#include <utility>
#include <variant>

#include "StrategoDefs.hpp"
#include "utils.hpp"

namespace stratego {

template < size_t... Is >
inline constexpr auto make_tokens(std::index_sequence< Is... >)
{
   return std::vector{Token(Is)...};
}

auto _default_mr() -> std::map< Token, int >;

auto _default_bm() -> std::map< std::array< Token, 2 >, FightOutcome >;

auto _default_setups(size_t game_dims) -> std::map< aze::Team, std::map< Position, Token > >;
auto _default_setups(std::array< size_t, 2 > game_dims)
   -> std::map< aze::Team, std::map< Position, Token > >;

auto _default_obs(size_t game_dims) -> std::vector< Position >;

auto _default_obs(std::array< size_t, 2 > game_dims) -> std::vector< Position >;

auto _token_set(size_t game_dim) -> std::map< aze::Team, std::vector< Token > >;

auto _gen_tokensets(const std::map< aze::Team, std::map< Position, Token > >& setups)
   -> std::map< aze::Team, std::vector< Token > >;

auto to_tokencounters(const std::map< aze::Team, std::vector< Token > >& token_vecs)
   -> std::map< aze::Team, std::map< Token, unsigned int > >;

auto _default_start_positions(size_t game_dim, aze::Team team) -> std::vector< Position >;

auto _check_alignment(
   const std::map< aze::Team, std::vector< Position > >& positions,
   const std::map< aze::Team, std::map< Position, Token > >& setups)
   -> std::map< aze::Team, std::vector< Position > >;

auto _positions_from_setups(std::map< aze::Team, std::map< Position, Token > >& setups)
   -> std::map< aze::Team, std::vector< Position > >;

// template < class StateType, class LogicType, class Derived, size_t n_teams >
// std::vector< typename Game< StateType, LogicType, Derived, n_teams >::sptr_piece_type >
// Game< StateType, LogicType, Derived, n_teams >::extract_pieces_from_setup(
//    const std::map< position_type, token_type > &setup,
//    Team team)
//{
//    using val_type = typename std::map< position_type, token_type >::value_type;
//    std::vector< sptr_piece_type > pc_vec;
//    pc_vec.reserve(setup.size());
//    std::transform(
//       setup.begin(),
//       setup.end(),
//       std::back_inserter(pc_vec),
//       [&](const val_type &pos_token) -> piece_type {
//          return std::make_shared< piece_type >(pos_token.first, pos_token.second, team);
//       });
//    return pc_vec;
// }
//
// template < class StateType, class LogicType, class Derived, size_t n_teams >
// std::vector< typename Game< StateType, LogicType, Derived, n_teams >::sptr_piece_type >
// Game< StateType, LogicType, Derived, n_teams >::extract_pieces_from_setup(
//    const std::map< position_type, sptr_piece_type > &setup,
//    Team team)
//{
//    using val_type = typename std::map< position_type, sptr_piece_type >::value_type;
//    std::vector< sptr_piece_type > pc_vec;
//    pc_vec.reserve(setup.size());
//    std::transform(
//       setup.begin(),
//       setup.end(),
//       std::back_inserter(pc_vec),
//       [&](const val_type &pos_piecesptr) -> sptr_piece_type {
//          auto piece_sptr = pos_piecesptr.second;
//          if(piece_sptr->team() != team)
//             throw std::logic_error(
//                "Pieces of team " + std::to_string(static_cast< int >(team))
//                + " were expected, but received piece of team "
//                + std::to_string(piece_sptr->team()));
//          return piece_sptr;
//       });
//    return pc_vec;
// }

struct Config {
   using setup_type = std::map< Position, Token >;
   using token_counter = std::map< Token, unsigned int >;

   aze::Team starting_team;
   std::array< size_t, 2 > game_dims;
   size_t max_turn_count;
   bool fixed_setups;
   std::optional< std::map< aze::Team, setup_type > > setups;
   std::map< aze::Team, token_counter > token_counters;
   std::map< aze::Team, std::vector< Position > > start_positions;
   std::map< std::array< Token, 2 >, FightOutcome > battle_matrix;
   std::vector< Position > hole_positions;
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
      std::map< std::array< Token, 2 >, FightOutcome > battle_matrix_ = _default_bm(),
      std::optional< std::vector< Position > > hole_positions_ = std::nullopt,
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
         hole_positions(
            hole_positions_.has_value()
               ? std::move(hole_positions_.value())
               : std::visit(
                  aze::utils::Overload{
                     [](size_t d) { return _default_obs(d); },
                     [](std::array< size_t, 2 > a) { return _default_obs(a); }},
                  game_dims_)),
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