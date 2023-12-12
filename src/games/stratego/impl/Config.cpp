
#include "stratego/Config.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <range/v3/all.hpp>
#include <utility>

#include "stratego/StrategoDefs.hpp"

namespace stratego {

std::map< Token, std::function< bool(size_t) > > default_move_ranges()
{
   std::map< Token, std::function< bool(size_t) > > moverange;
   for(auto token_value : std::array{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 99}) {
      auto token = Token(token_value);
      if(token == Token::scout) {
         moverange[token] = [](size_t) { return true; };
      } else if(token == Token::flag or token == Token::bomb) {
         moverange[token] = [](size_t dist) { return 0 == dist; };
      } else {
         moverange[token] = [](size_t dist) { return 1 == dist; };
      }
   }
   return moverange;
}

std::map< std::pair< Token, Token >, FightOutcome > default_battlematrix()
{
   std::map< std::pair< Token, Token >, FightOutcome > bm;
   for(auto i : std::array{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 99}) {
      for(auto j : std::array{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 99}) {
         if(i < j) {
            bm[{Token(i), Token(j)}] = FightOutcome::death;
         } else if(i == j) {
            bm[{Token(i), Token(i)}] = FightOutcome::tie;
         } else {
            bm[{Token(i), Token(j)}] = FightOutcome::kill;
         }
      }
      if(ranges::contains(std::array{0, 11, 99}, i)) {
         continue;
      }
      bm[{Token(i), Token::flag}] = FightOutcome::kill;
      if(Token(i) == Token::miner) {
         bm[{Token(i), Token::bomb}] = FightOutcome::kill;
      } else {
         bm[{Token(i), Token::bomb}] = FightOutcome::death;
      }
   }
   // the spy kills the marshall on attack
   bm[{Token::spy, Token::marshall}] = FightOutcome::kill;
   return bm;
}

std::map< Position2D, Token > default_setup(size_t /*game_dims*/, Team /*team*/)
{
   std::map< Position2D, Token > setup;

   // TODO: Fill

   return setup;
}
std::map< Position2D, Token > default_setup(ranges::span< size_t, 2 > game_dims, Team team)
{
   if(game_dims[0] == game_dims[1]
      and ranges::contains(
         std::array{DefinedBoardSizes::small, DefinedBoardSizes::medium, DefinedBoardSizes::large},
         static_cast< DefinedBoardSizes >(game_dims[0])
      )) {
      return default_setup(game_dims[0], team);
   }
   throw std::invalid_argument("Cannot provide default setups for non-default game dimensions.");
}

std::map< Team, std::map< Position2D, Token > > default_setup(size_t game_dims)
{
   return std::map{
      std::pair{Team::BLUE, default_setup(game_dims, Team::BLUE)},
      std::pair{Team::RED, default_setup(game_dims, Team::RED)}
   };
}
std::map< Team, std::optional< std::map< Position2D, Token > > > default_setup(
   ranges::span< size_t, 2 > game_dims
)
{
   return std::map{
      std::pair{Team::BLUE, std::optional{default_setup(game_dims, Team::BLUE)}},
      std::pair{Team::RED, std::optional{default_setup(game_dims, Team::RED)}}
   };
}

std::vector< Position2D > default_holes(size_t game_dims)
{
   if(game_dims == DefinedBoardSizes::small) {
      return {{2, 2}};
   }
   if(game_dims == DefinedBoardSizes::medium) {
      return {{3, 1}, {3, 5}};
   }
   if(game_dims == DefinedBoardSizes::large) {
      return {{4, 2}, {5, 2}, {4, 3}, {5, 3}, {4, 6}, {5, 6}, {4, 7}, {5, 7}};
   }

   throw std::invalid_argument(fmt::format(
      "'dimension' not in {}. User has to provide custom hole positions.",
      fmt::join(
         std::array{DefinedBoardSizes::small, DefinedBoardSizes::medium, DefinedBoardSizes::large},
         ", "
      )
   ));
}
std::vector< Position2D > default_holes(ranges::span< size_t, 2 > game_dims)
{
   if(game_dims[0] == game_dims[1]
      and std::set<
             size_t >{DefinedBoardSizes::small, DefinedBoardSizes::medium, DefinedBoardSizes::large}
             .contains(game_dims[0])) {
      return default_holes(game_dims[0]);
   }
   throw std::invalid_argument(
      "Cannot provide default hole positions for non-default game dimensions."
   );
}

std::map< Team, std::vector< Token > > token_vector(size_t game_dim)
{
   switch(game_dim) {
      case DefinedBoardSizes::small: {
         using seq = std::index_sequence< 0, 1, 2, 2, 2, 3, 3, 10, 11, 11 >;
         return {
            std::pair{Team::BLUE, common::make_enum_vec< Token >(seq())},
            std::pair{Team::RED, common::make_enum_vec< Token >(seq())}
         };
      }
      case DefinedBoardSizes::medium: {
         using seq = std::
            index_sequence< 0, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 10, 11, 11, 11, 11 >;
         return {
            std::pair{Team::BLUE, common::make_enum_vec< Token >(seq())},
            std::pair{Team::RED, common::make_enum_vec< Token >(seq())}
         };
      }
      case DefinedBoardSizes::large: {
         using seq = std::index_sequence<
            // clang-format off
            0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6,
            7, 7, 7, 8, 8, 9, 10, 11, 11, 11, 11, 11, 11 >;
         // clang-format on
         return {
            std::pair{Team::BLUE, common::make_enum_vec< Token >(seq())},
            std::pair{Team::RED, common::make_enum_vec< Token >(seq())}
         };
      }
      default:
         throw std::invalid_argument("Cannot provide tokenset for non-default game dimensions.");
   }
}

std::map< Team, std::optional< std::vector< Position2D > > > default_start_fields(size_t game_dim)
{
   return std::map{
      std::pair{Team::BLUE, std::optional{default_start_fields(game_dim, Team::BLUE)}},
      std::pair{Team::RED, std::optional{default_start_fields(game_dim, Team::RED)}}
   };
}

std::vector< Position2D > default_start_fields(size_t game_dim, Team team)
{
   if(std::set{Team::RED, Team::BLUE}.contains(team)) {
      throw std::invalid_argument("'team' not in {0, 1}.");
   }

   switch(game_dim) {
      case 5: {
         if(team == Team::BLUE) {
            return {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 0}, {1, 1}, {1, 2}, {1, 3}, {1, 4}};
         }
         return {{4, 0}, {4, 1}, {4, 2}, {4, 3}, {4, 4}, {3, 0}, {3, 1}, {3, 2}, {3, 3}, {3, 4}};
      }
      case 7: {
         if(team == Team::BLUE) {
            return {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {0, 6},
                    {1, 0}, {1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6},
                    {2, 0}, {2, 1}, {2, 2}, {2, 3}, {2, 4}, {2, 5}, {2, 6}};
         }
         return {{4, 0}, {4, 1}, {4, 2}, {4, 3}, {4, 4}, {4, 5}, {4, 6},
                 {5, 0}, {5, 1}, {5, 2}, {5, 3}, {5, 4}, {5, 5}, {5, 6},
                 {6, 0}, {6, 1}, {6, 2}, {6, 3}, {6, 4}, {6, 5}, {6, 6}};
      }
      case 10: {
         if(team == Team::BLUE) {
            return {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {0, 6}, {0, 7}, {0, 8}, {0, 9},
                    {1, 0}, {1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {1, 7}, {1, 8}, {1, 9},
                    {2, 0}, {2, 1}, {2, 2}, {2, 3}, {2, 4}, {2, 5}, {2, 6}, {2, 7}, {2, 8}, {2, 9},
                    {3, 0}, {3, 1}, {3, 2}, {3, 3}, {3, 4}, {3, 5}, {3, 6}, {3, 7}, {3, 8}, {3, 9}};
         }
         return {{6, 0}, {6, 1}, {6, 2}, {6, 3}, {6, 4}, {6, 5}, {6, 6}, {6, 7}, {6, 8}, {6, 9},
                 {7, 0}, {7, 1}, {7, 2}, {7, 3}, {7, 4}, {7, 5}, {7, 6}, {7, 7}, {7, 8}, {7, 9},
                 {8, 0}, {8, 1}, {8, 2}, {8, 3}, {8, 4}, {8, 5}, {8, 6}, {8, 7}, {8, 8}, {8, 9},
                 {9, 0}, {9, 1}, {9, 2}, {9, 3}, {9, 4}, {9, 5}, {9, 6}, {9, 7}, {9, 8}, {9, 9}};
      }
      default: {
         throw std::invalid_argument(fmt::format(
            "'shape' not one of {}.",
            fmt::join(std::array< DefinedBoardSizes, 3 >{small, medium, large}, ", ")
         ));
      }
   }
}

const std::vector< Position2D >& _check_alignment(
   const std::vector< Position2D >& positions,
   const std::map< Position2D, Token >& setup
)
{
   if(ranges::any_of(setup | ranges::views::keys, [&](const Position2D& pos) {
         return ranges::find(positions, pos) == positions.end();
      })) {
      throw std::invalid_argument(
         "Passed starting positions parameter and setup parameter do not match."
      );
   }
   return positions;
}

std::map< Team, Config::token_counter_t > Config::_init_tokencounters(
   const std::map< Team, std::optional< std::variant< std::vector< Token >, token_counter_t > > >&
      token_sets,
   const std::map< Team, std::optional< Config::setup_t > >& setups_
)
{
   auto token_visitor = common::Overload{
      [](const std::vector< Token >& tv) { return common::counter(tv); },
      [](const token_counter_t& tc) { return tc; }
   };
   std::map< Team, token_counter_t > counters;
   for(auto team : std::set{Team::BLUE, Team::RED}) {
      if(setups_.at(team).has_value()) {
         if(token_sets.at(team).has_value()) {
            // both setup and tokensets are given, so we take the superset of both as the truth
            auto token_counter = std::visit(token_visitor, token_sets.at(team).value());
            auto token_counter_from_setup = tokens_from_setup(setups_.at(team).value());
            for(auto token : ranges::views::concat(
                                token_counter | ranges::views::keys,
                                token_counter_from_setup | ranges::views::keys
                             ) | ranges::views::unique) {
               counters[team][token] = std::max(
                  token_counter[token], token_counter_from_setup[token]
               );
            }
         } else {
            counters[team] = tokens_from_setup(setups_.at(team).value());
         }
      } else {
         if(token_sets.at(team).has_value()) {
            counters[team] = std::visit(token_visitor, token_sets.at(team).value());
         } else {
            throw std::invalid_argument(
               "No setup passed and no tokenset passed. Either "
               "of these need to be set."
            );
         }
      }
   }
   return counters;
}

std::map< Team, std::vector< Position2D > > Config::_init_start_fields(
   const std::map< Team, std::optional< std::vector< Position2D > > >& start_pos,
   const std::map< Team, std::optional< Config::setup_t > >& setups_
)
{
   std::map< Team, std::vector< Position2D > > positions;
   for(auto team : std::set{Team::BLUE, Team::RED}) {
      if(start_pos.at(team).has_value()) {
         if(setups_.at(team).has_value()) {
            positions[team] = _check_alignment(
               start_pos.at(team).value(), setups_.at(team).value()
            );
         } else {
            positions[team] = start_pos.at(team).value();
         }
      } else {
         positions[team] = [](const std::map< Position2D, Token >& setup) {
            std::vector< Position2D > pos;
            pos.reserve(setup.size());
            std::transform(
               setup.begin(),
               setup.end(),
               std::back_inserter(pos),
               [](const auto& pair) { return pair.first; }
            );
            return pos;
         }(setups_.at(team).value());
      }
   }
   return positions;
}

std::vector< Position2D > Config::_init_hole_positions(
   const std::optional< std::vector< Position2D > >& hole_pos,
   game_dim_variant_t game_dims_
)
{
   return hole_pos.has_value()
             ? hole_pos.value()
             : std::visit(
                  common::Overload{
                     [](size_t d) { return default_holes(d); },
                     [](DefinedBoardSizes d) { return default_holes(static_cast< size_t >(d)); },
                     [](ranges::span< size_t, 2 > a) { return default_holes(a); }
                  },
                  game_dims_
               );
}

Config::token_counter_t tokens_from_setup(const Config::setup_t& setup)
{
   auto values = setup | ranges::views::values;
   return common::counter(std::vector< Token >{values.begin(), values.end()});
}

std::map< Team, std::optional< Config::token_counter_t > > tokens_from_setup(
   const std::map< Team, std::optional< Config::setup_t > >& setups
)
{
   std::map< Team, std::optional< Config::token_counter_t > > counters;
   for(auto team : std::set{Team::BLUE, Team::RED}) {
      if(setups.at(team).has_value()) {
         auto values = setups.at(team).value() | ranges::views::values;
         counters[team] = common::counter(std::vector< Token >{values.begin(), values.end()});
      }
   }
   return counters;
}

auto default_token_sets(size_t game_dim) -> Config::token_counter_t
{
   switch(game_dim) {
      case DefinedBoardSizes::small: {
         return Config::token_counter_t{
            std::pair{Token::flag, 1},
            std::pair{Token::spy, 1},
            std::pair{Token::scout, 3},
            std::pair{Token::miner, 2},
            std::pair{Token::marshall, 1},
            std::pair{Token::bomb, 2}
         };
      }
      case DefinedBoardSizes::medium: {
         return Config::token_counter_t{
            std::pair{Token::flag, 1},
            std::pair{Token::spy, 1},
            std::pair{Token::scout, 5},
            std::pair{Token::miner, 3},
            std::pair{Token::sergeant, 3},
            std::pair{Token::lieutenant, 2},
            std::pair{Token::captain, 1},
            std::pair{Token::marshall, 1},
            std::pair{Token::bomb, 4}
         };
      }
      case DefinedBoardSizes::large: {
         return Config::token_counter_t{
            std::pair{Token::flag, 1},
            std::pair{Token::spy, 1},
            std::pair{Token::scout, 8},
            std::pair{Token::miner, 5},
            std::pair{Token::sergeant, 4},
            std::pair{Token::lieutenant, 4},
            std::pair{Token::captain, 4},
            std::pair{Token::major, 3},
            std::pair{Token::colonel, 2},
            std::pair{Token::general, 1},
            std::pair{Token::marshall, 1},
            std::pair{Token::bomb, 6}
         };
      }
      default: {
         throw std::invalid_argument(
            "Cannot provide default token sets for non-default game dimensions."
         );
      }
   }
}

Config::Config(
   Team starting_team_,
   Config::game_dim_variant_t game_dims_,
   const std::map< Team, std::optional< setup_t > >& setups_,
   const std::optional< std::vector< Position2D > >& hole_positions_,
   const std::map< Team, std::optional< std::variant< std::vector< Token >, token_counter_t > > >&
      token_set_,
   const std::map< Team, std::optional< std::vector< Position2D > > >& start_fields_,
   bool fixed_starting_team_,
   std::variant< bool, ranges::span< bool, 2 > > fixed_setups_,
   size_t max_turn_count_,
   std::map< std::pair< Token, Token >, FightOutcome > battle_matrix_,
   std::map< Token, std::function< bool(size_t) > > move_ranges_
)
    : starting_team(starting_team_),
      fixed_starting_team(fixed_starting_team_),
      game_dims(std::visit(
         common::Overload{
            [](size_t d) {
               return std::array{d, d};
            },
            [](ranges::span< size_t, 2 > d) {
               return std::array{d[0], d[1]};
            }
         },
         game_dims_
      )),
      max_turn_count(max_turn_count_),
      fixed_setups(std::visit(
         common::Overload{
            [](bool value) {
               return std::array{value, value};
            },
            [](ranges::span< bool, 2 > d) {
               return std::array{d[0], d[1]};
               ;
            }
         },
         fixed_setups_
      )),
      setups(_init_setups(setups_, token_set_, start_fields_, game_dims_)),
      token_counters(_init_tokencounters(token_set_, setups_)),
      start_fields(_init_start_fields(start_fields_, setups_)),
      battle_matrix(std::move(battle_matrix_)),
      hole_positions(_init_hole_positions(hole_positions_, game_dims_)),
      move_ranges(std::move(move_ranges_))
{
   for(int i = 0; i < 2; ++i) {
      if(utils::flatten_counter(token_counters[Team(i)]).size() != start_fields[Team(i)].size()) {
         SPDLOG_DEBUG(
            "Token vector size: {}", utils::flatten_counter(token_counters[Team(i)]).size()
         );
         SPDLOG_DEBUG("Field vector size: {}", start_fields[Team(i)].size());
         throw std::invalid_argument(
            "Token counters and start position vectors do not match in size"
         );
      }
   }
}

Config::Config(
   Team starting_team_,
   Config::game_dim_variant_t game_dims_,
   const std::map< Team, std::optional< setup_t > >& setups_,
   const std::optional< std::vector< Position2D > >& hole_positions_,
   bool fixed_starting_team_,
   std::variant< bool, ranges::span< bool, 2 > > fixed_setups_,
   size_t max_turn_count_,
   std::map< std::pair< Token, Token >, FightOutcome > battle_matrix_,
   std::map< Token, std::function< bool(size_t) > > move_ranges_
)
    : Config(
         starting_team_,
         game_dims_,
         setups_,
         hole_positions_,
         nullarg< "tokens" >(),
         nullarg< "fields" >(),
         fixed_starting_team_,
         fixed_setups_,
         max_turn_count_,
         std::move(battle_matrix_),
         std::move(move_ranges_)
      )
{
}
Config::Config(
   Team starting_team_,
   Config::game_dim_variant_t game_dims_,
   const std::optional< std::vector< Position2D > >& hole_positions_,
   const std::map< Team, std::optional< std::variant< std::vector< Token >, token_counter_t > > >&
      token_set_,
   const std::map< Team, std::optional< std::vector< Position2D > > >& start_fields_,
   bool fixed_starting_team_,
   std::variant< bool, ranges::span< bool, 2 > > fixed_setups_,
   size_t max_turn_count_,
   std::map< std::pair< Token, Token >, FightOutcome > battle_matrix_,
   std::map< Token, std::function< bool(size_t) > > move_ranges_
)
    : Config(
         starting_team_,
         game_dims_,
         nullarg< "setups" >(),
         hole_positions_,
         token_set_,
         start_fields_,
         fixed_starting_team_,
         fixed_setups_,
         max_turn_count_,
         std::move(battle_matrix_),
         std::move(move_ranges_)
      )
{
}

Config::Config(
   Team starting_team_,
   DefinedBoardSizes game_dims_,
   bool fixed_starting_team_,
   std::variant< bool, ranges::span< bool, 2 > > fixed_setups_,
   size_t max_turn_count_,
   std::map< std::pair< Token, Token >, FightOutcome > battle_matrix_,
   std::map< Token, std::function< bool(size_t) > > move_ranges_
)
    : Config(
         starting_team_,
         game_dims_,
         default_setup(game_dims),
         default_holes(static_cast< size_t >(game_dims_)),
         std::map{
            std::pair{Team::BLUE, std::optional< token_variant_t >{default_token_sets(game_dims_)}},
            std::pair{Team::BLUE, std::optional< token_variant_t >{default_token_sets(game_dims_)}}
         },
         default_start_fields(game_dims_),
         fixed_starting_team_,
         fixed_setups_,
         max_turn_count_,
         std::move(battle_matrix_),
         std::move(move_ranges_)
      )
{
}

}  // namespace stratego
