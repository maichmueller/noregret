
#pragma once

#include <aze/aze.h>

#include <functional>
#include <range/v3/all.hpp>

#include "Action.h"
#include "Config.hpp"
#include "State.h"

namespace stratego {

class Logic {
   using Team = aze::Team;

  public:
   virtual ~Logic() = default;

   inline FightOutcome fight(const Config &config, const Piece &attacker, const Piece &defender)
   {
      return fight(config, std::array{attacker.token(), defender.token()});
   }

   inline FightOutcome fight(const Config &config, const std::array< Token, 2 > &att_def)
   {
      return config.battle_matrix.at(att_def);
   }

   template < FightOutcome Outcome >
   void handle(State &state, Piece &attacker, Piece &defender)
   {
   }

   template <>
   virtual void handle< FightOutcome::death >(State &state, Piece &attacker, Piece &defender)
   {
      state.to_graveyard(attacker);
   }
   template <>
   virtual void handle< FightOutcome::kill >(State &state, Piece &attacker, Piece &defender)
   {
      state.to_graveyard(defender);
   }
   template <>
   virtual void handle< FightOutcome::stalemate >(State &state, Piece &attacker, Piece &defender)
   {
      state.to_graveyard(attacker);
      state.to_graveyard(defender);
   }

   template < std::integral IntType >
   bool check_bounds(const Board &board, IntType value)
   {
      auto shape = board.shape();
      for(const auto &limit : shape) {
         if(std::cmp_less(value, 0) or std::cmp_greater_equal(value, limit)) {
            return false;
         }
      }
      return true;
   }

   /**
    * Checks boundaries of given values with respect to the board shape in every dimension.
    *
    * If the value sequence is shorter than the board dimension, then 0s are padded from the front
    * to make up for the difference.
    * @tparam Range
    * @param board
    * @param values
    * @return
    */
   template < typename Range >
   requires ranges::sized_range< Range > && ranges::bidirectional_range< Range > && std::integral<
      typename Range::value_type >
   bool check_bounds(const Board &board, Range values)
   {
      auto shape = board.shape();
      if(values.size() > shape.size()) {
         return false;
      }
      constexpr bool val = ranges::bidirectional_range< Position >;
      auto size_diff = shape.size() - values.size();
      for(auto value : ranges::views::reverse(values)) {
         auto i = shape[value + size_diff];
         if(std::cmp_less(value, 0) or std::cmp_greater_equal(value, shape[i])) {
            return false;
         }
      }
      return true;
   }

   void place_setup(const std::map< Position, Token > setup, Board &board, aze::Team team);

   aze::Status check_terminal(State &state)
   {
      if(auto dead_pieces = state.graveyard(0);
         dead_pieces.find(Token::flag) != dead_pieces.end()) {
         // flag of team 0 has been captured (killed), therefore team 0 lost
         return state.status(aze::Status::WIN_RED);
      } else if(dead_pieces = state.graveyard(1);
                dead_pieces.find(Token::flag) != dead_pieces.end()) {
         // flag of team 1 has been captured (killed), therefore team 1 lost
         return state.status(aze::Status::WIN_BLUE);
      }

      // committing draw rules here

      // Rule 1: If either team has no moves left.
      if(not has_valid_actions(state, aze::Team::BLUE)
         or not has_valid_actions(state, aze::Team::RED)) {
         return state.status(aze::Status::TIE);
      }

      // Rule 1: The maximum turn count has been reached
      if(state.turn_count() >= state.config().max_turn_count) {
         return state.status(aze::Status::TIE);
      }
      return state.status();
   }

   bool is_valid(const State &state, const Action &action)
   {
      const auto &[pos_before, pos_after] = action.positions();
      const auto &board = state.board();
      if(not check_bounds(board, pos_before) or not check_bounds(board, pos_after))
         return false;

      auto p_b_opt = board[pos_before];
      auto p_a_opt = board[pos_after];

      if(not p_b_opt.has_value())
         return false;
      if(std::set{Token::flag, Token::bomb}.contains(p_b_opt.value().token())) {
         return false;
      }
      if(not p_a_opt.has_value()) {
         const auto &p_a = p_a_opt.value();
         if(p_a.team() == p_b_opt.value().team())
            return false;  // cant fight
                           // pieces of
                           // own team
         if(p_a.token() == Token::hole) {
            return false;  // cant fight hole
         }
      }

      int move_dist = abs(pos_after[1] - pos_before[1]) + abs(pos_after[0] - pos_before[0]);
      if(move_dist > 1) {
         if(p_b_opt.value().token() != Token::scout)
            return false;  // not of type 2 , but is supposed to go far

         if(pos_after[0] == pos_before[0]) {
            int dist = pos_after[1] - pos_before[1];
            int sign = (dist >= 0) ? 1 : -1;
            for(int i = 1; i < std::abs(dist); ++i) {
               Position pos{pos_before[0], pos_before[1] + sign * i};
               if(board[pos].has_value())
                  return false;
            }
         } else if(pos_after[1] == pos_before[1]) {
            int dist = pos_after[0] - pos_before[0];
            int sign = (dist >= 0) ? 1 : -1;
            for(int i = 1; i < std::abs(dist); ++i) {
               Position pos = {pos_before[0] + sign * i, pos_before[1]};
               if(board[pos].has_value())
                  return false;
            }
         } else
            return false;  // diagonal moves not allowed
      }
      return true;
   }

   template < typename T, size_t N >
   auto _valid_vectors(const Position &pos, const std::span< T, N > &shape)
   {
      // all possible positions to the left until board ends
      auto left_view = ranges::views::iota(1, pos[0] + 1) | ranges::views::transform([&](auto i) {
                          return Position{-i, 0};
                       });
      // all possible positions to the right until board ends
      auto right_view = ranges::views::iota(size_t(1), shape[0] - pos[0])
                        | ranges::views::transform([&](auto i) {
                             return Position{i, 0};
                          });
      // all possible positions to the bottom until board ends
      auto down_view = ranges::views::iota(1, pos[1] + 1) | ranges::views::transform([&](auto i) {
                          return Position{0, -i};
                       });
      // all possible positions to the top until board ends
      auto top_view = ranges::views::iota(size_t(1), shape[1] - pos[1])
                      | ranges::views::transform([&](auto i) {
                           return Position{0, i};
                        });
      return ranges::views::concat(right_view, top_view, left_view, down_view);
   }

   std::vector< Action > valid_actions(const State &state, Team team)
   {
      const auto &board = state.board();
      std::vector< Action > actions_possible;
      for(const auto &elem : board) {
         if(elem.has_value()) {
            const auto &piece = elem.value();
            if(piece.team() == team) {
               // the position we are dealing with
               auto pos = piece.position();
               auto append_action = [&](const Position &pos_to) {
                  Action action{pos, pos + pos_to};
                  if(is_valid(state, action)) {
                     actions_possible.emplace_back(action);
                  }
               };
               if(piece.token() == Token::scout) {
                  ranges::for_each(_valid_vectors(pos, std::span{board.shape()}), append_action);
               } else {
                  // all actions are 1 step to left, right, top, or bottom
                  ranges::for_each(
                     std::vector< Position >{{1, 0}, {0, 1}, {-1, 0}, {0, -1}}, append_action);
               }
            }
         }
      }

      return actions_possible;
   }

   bool has_valid_actions(const State &state, Team team)
   {
      const auto &board = state.board();
      for(const auto &piece_opt : board) {
         if(piece_opt.has_value()) {
            const auto &piece = piece_opt.value();
            if(Token token = piece.token();
               piece.team() == team && token != Token::flag && token != Token::bomb) {
               // the position we are dealing with
               auto pos = piece.position();
               auto check_legal = [&](const Position &pos_to) -> bool {
                  Action action{pos, pos_to};
                  if(is_valid(state, action)) {
                     return true;
                  }
                  return false;
               };
               if(piece.token() == Token::scout) {
                  if(ranges::any_of(_valid_vectors(pos, std::span{board.shape()}), check_legal)) {
                     return true;
                  }
               } else {
                  // all actions are 1 step to left, right, top, or bottom
                  ranges::any_of(
                     std::vector< Position >{{1, 0}, {0, 1}, {-1, 0}, {0, -1}}, check_legal);
               }
            }
         }
      }
      return false;
   }

   static std::map< Position, Token >
   uniform_setup_draw(const Config &config, aze::Team team, aze::utils::RNG &rng)
   {
      std::map< Position, Token > setup_out;

      auto start_positions = config.start_positions.at(team);
      auto token_counter = config.token_counters.at(team);
      auto uniq_token_view = token_counter | ranges::views::keys;
      std::vector< Token > tokenvec = {uniq_token_view.begin(), uniq_token_view.end()};

      ranges::shuffle(start_positions, rng);
      auto int_dist = std::uniform_int_distribution< size_t >(0, tokenvec.size());

      while(not start_positions.empty()) {
         auto &pos = start_positions.back();
         auto choice = int_dist(rng);
         if(auto token = tokenvec[choice]; token_counter[token] > 0) {
            setup_out[pos] = token;
            token_counter[token]--;
            start_positions.pop_back();
         } else {
            tokenvec.erase(tokenvec.begin() + choice);
         }
      }
      return setup_out;
   }

   static Board create_empty_board(const Config &config)
   {
      Board b;
      for(auto x : ranges::views::iota(size_t(0), config.game_dims[0])) {
         for(auto y : ranges::views::iota(size_t(0), config.game_dims[1])) {
            b[{x, y}] = std::nullopt;
         }
      }
      return b;
   }

   template < std::invocable< const Config &, aze::Team, aze::utils::RNG & > SampleStrategyType >
   Board draw_board(
      const Config &config,
      aze::utils::RNG &rng,
      SampleStrategyType setup_sampling_strategy)
   {
      Board board;
      for(int i = 0; i < 2; i++) {
         auto team = aze::Team(i);
         if(config.fixed_setups[i]) {
            const auto &setup = config.setups.at(team).value();
            for(const auto &[position, token] : setup) {
               board[position] = Piece(position, token, team);
            }
         } else {
            for(const auto &[position, token] : setup_sampling_strategy(config, team, rng)) {
               board[position] = Piece(position, token, team);
            }
         }
         return board;
      }
   }

   template < std::invocable< const Config &, aze::Team, aze::utils::RNG & > SampleStrategyType >
   Board
   draw_setup(const Config &config, const Board &curr_board, SampleStrategyType sampling_strategy)
   {
      Board board;
      for(int i = 0; i < 2; i++) {
         auto team = aze::Team(i);
         if(config.fixed_setups[i]) {
            const auto &setup = config.setups.at(team).value();
            for(const auto &[position, token] : setup) {
               board[position] = Piece(position, token, team);
            }
         } else {
            for(const auto &[position, token] : sampling_strategy(config, team)) {
               board[position] = Piece(position, token, team);
            }
         }
         return board;
      }
   }
};
}  // namespace stratego