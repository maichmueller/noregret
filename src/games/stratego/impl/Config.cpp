
#include "stratego/Config.hpp"

#include <range/v3/all.hpp>

namespace stratego {

std::map< Token, std::function< bool(size_t) > > default_move_ranges()
{
   std::map< Token, std::function< bool(size_t) > > mr;
   for(auto token_value : std::array{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 99}) {
      auto token = Token(token_value);
      if(token == Token::scout) {
         mr[token] = [](size_t dist) { return true; };
      } else if(token == Token::flag or token == Token::bomb) {
         mr[token] = [](size_t dist) { return 0 == dist; };
      } else {
         mr[token] = [](size_t dist) { return 1 == dist; };
      }
   }
   return mr;
}

std::map< std::array< Token, 2 >, FightOutcome > default_battlematrix()
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
      if(std::set{0, 11, 99}.contains(i)) {
         continue;
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

std::map< Position, Token > default_setup(size_t game_dims, aze::Team team)
{
   using Team = aze::Team;
   std::map< Position, Token > setup;

   // TODO: Fill

   return setup;
}
std::map< Position, Token > default_setup(ranges::span< size_t, 2 > game_dims, aze::Team team)
{
   if(game_dims[0] == game_dims[1] and std::set{5, 7, 10}.contains(game_dims[0])) {
      return default_setup(game_dims[0], team);
   } else {
      throw std::invalid_argument("Cannot provide default setups for non-default game dimensions.");
   }
}

std::vector< Position > default_holes(size_t game_dims)
{
   if(game_dims == 5)
      return {{2, 2}};
   else if(game_dims == 7)
      return {{3, 1}, {3, 5}};
   else if(game_dims == 10)
      return {{4, 2}, {5, 2}, {4, 3}, {5, 3}, {4, 6}, {5, 6}, {4, 7}, {5, 7}};
   else
      throw std::invalid_argument(
         "'dimension' not in {5, 7, 10}. User has to provide custom hole positions.");
}
std::vector< Position > default_holes(ranges::span< size_t, 2 > game_dims)
{
   if(game_dims[0] == game_dims[1] and std::set< size_t >{5, 7, 10}.contains(game_dims[0])) {
      return default_holes(game_dims[0]);
   } else {
      throw std::invalid_argument(
         "Cannot provide default hole positions for non-default game dimensions.");
   }
}

std::map< aze::Team, std::vector< Token > > token_vector(size_t game_dim)
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

std::vector< Position > default_start_positions(size_t game_dim, aze::Team team)
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

const std::vector< Position >& _check_alignment(
   const std::vector< Position >& positions,
   const std::map< Position, Token >& setup)
{
   if(positions.size() != setup.size()
      or ranges::any_of(positions, [&](const Position& pos) { return not setup.contains(pos); })) {
      throw std::invalid_argument(
         "Passed starting positions parameter and setup parameter do not match.");
   }
   return positions;
}

std::map< aze::Team, std::optional< Config::setup_t > > Config::_init_setups(
   const std::map< aze::Team, std::optional< Config::setup_t > >& setups_,
   const std::map< aze::Team, std::optional< std::vector< Token > > >& tokenset_,
   const std::map< aze::Team, std::optional< std::vector< Position > > >& start_positions_,
   std::variant< size_t, ranges::span< size_t, 2 > > game_dims_)
{
   std::map< aze::Team, std::optional< setup_t > > sets;
   for(auto team : std::set{aze::Team::BLUE, aze::Team::RED}) {
      if(setups_.at(team).has_value()) {
         sets[team] = setups_.at(team).value();
      } else if(start_positions_.at(team).has_value() && tokenset_.at(team).has_value()) {
         sets[team] = std::nullopt;
      } else {
         sets[team] = std::visit(
            aze::utils::Overload{
               [&](size_t d) { return default_setup(d, team); },
               [&](ranges::span< size_t, 2 > a) { return default_setup(a, team); }},
            game_dims_);
      }
   }
   return sets;
}

std::map< aze::Team, Config::token_counter_t > Config::_init_tokencounters(
   const std::map< aze::Team, std::optional< std::vector< Token > > >& token_sets,
   const std::map< aze::Team, std::optional< Config::setup_t > >& setups_)
{
   std::map< aze::Team, token_counter_t > counters;
   for(auto team : std::set{aze::Team::BLUE, aze::Team::RED}) {
      if(setups_.at(team).has_value()) {
         auto values = setups_.at(team).value() | ranges::views::values;
         counters[team] = aze::utils::counter(std::vector< Token >{values.begin(), values.end()});
      } else {
         if(token_sets.at(team).has_value()) {
            counters[team] = aze::utils::counter(token_sets.at(team).value());
         } else {
            throw std::invalid_argument(
               "No setup passed and no tokenset passed. Either "
               "of these need to be set.");
         }
      }
   }
   return counters;
}

std::map< aze::Team, std::vector< Position > > Config::_init_start_positions(
   const std::map< aze::Team, std::optional< std::vector< Position > > >& start_pos,
   const std::map< aze::Team, std::optional< Config::setup_t > >& setups_)
{
   std::map< aze::Team, std::vector< Position > > positions;
   for(auto team : std::set{aze::Team::BLUE, aze::Team::RED}) {
      if(start_pos.at(team).has_value()) {
         if(setups_.at(team).has_value()) {
            positions[team] = _check_alignment(
               start_pos.at(team).value(), setups_.at(team).value());
         } else {
            positions[team] = start_pos.at(team).value();
         }
      } else {
         positions[team] = [](const std::map< Position, Token >& setup) {
            std::vector< Position > pos;
            pos.reserve(setup.size());
            std::transform(
               setup.begin(), setup.end(), std::back_inserter(pos), [](const auto& pair) {
                  return pair.first;
               });
            return pos;
         }(setups_.at(team).value());
      }
   }
   return positions;
}

std::vector< Position > Config::_init_hole_positions(
   const std::optional< std::vector< Position > >& hole_pos,
   std::variant< size_t, ranges::span< size_t, 2 > > game_dims_)
{
   return hole_pos.has_value() ? hole_pos.value()
                               : std::visit(
                                  aze::utils::Overload{
                                     [](size_t d) { return default_holes(d); },
                                     [](ranges::span< size_t, 2 > a) { return default_holes(a); }},
                                  game_dims_);
}

}  // namespace stratego
