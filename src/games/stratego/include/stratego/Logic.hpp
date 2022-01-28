
#pragma once

#include <functional>
#include <range/v3/all.hpp>

#include "Action.hpp"
#include "Config.hpp"
#include "State.hpp"
#include "aze/aze.h"

namespace stratego {

class Logic {
   using Team = aze::Team;

  public:
   virtual ~Logic() = default;

   static inline FightOutcome
   fight(const Config &config, const Piece &attacker, const Piece &defender)
   {
      return fight(config, std::array{attacker.token(), defender.token()});
   }

   static inline FightOutcome fight(const Config &config, const std::array< Token, 2 > &att_def)
   {
      return config.battle_matrix.at(att_def);
   }

   static void update_board(Board &board, const Position &new_pos, Piece &piece)
   {
      board[new_pos] = piece;
      piece.position(new_pos);
   }
   static void update_board(Board &board, const Position &new_pos)
   {
      board[new_pos] = std::nullopt;
   }

   FightOutcome handle_fight(State &state, Piece &attacker, Piece &defender)
   {
      auto &board = state.board();
      // uncover participant pieces
      attacker.flag_unhidden(true);
      defender.flag_unhidden(true);
      auto outcome = fight(state.config(), attacker, defender);
      switch(outcome) {
         case FightOutcome::kill: {
            state.to_graveyard(defender);
            update_board(board, attacker.position());
            update_board(board, defender.position(), attacker);
         }
         case FightOutcome::death: {
            state.to_graveyard(attacker);
            update_board(board, attacker.position());
         }
         case FightOutcome::stalemate: {
            state.to_graveyard(attacker);
            state.to_graveyard(defender);
            update_board(board, attacker.position());
            update_board(board, defender.position());
         }
      }
      return outcome;
   }

   void apply_action(State &state, const Action &action)
   {
      // preliminaries
      const Position &from = action[0];
      const Position &to = action[1];

      // save the access to the pieces in question
      // (removes redundant searching in board later)
      Board &board = state.board();
      auto piece_from = board[from].value();
      auto piece_to_opt = board[to];

      piece_from.flag_has_moved(true);

      // enact the move
      if(piece_to_opt.has_value()) {
         auto &piece_to = piece_to_opt.value();
         // engage in fight, since piece_to is not a null piece
         handle_fight(state, piece_from, piece_to);
      } else {
         // no fight happened, simply move piece_from onto new position
         update_board(board, to, piece_from);
         update_board(board, from);
      }
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
      return ranges::all_of(ranges::views::zip(values, shape), [&](auto v_s) {
         auto [v, s] = v_s;
         return std::cmp_greater_equal(v, 0) and std::cmp_less(v, s);
      });
   }

   static void place_setup(const std::map< Position, Token > &setup, Board &board, aze::Team team)
   {
      for(const auto &[pos, token] : setup) {
         board[pos] = Piece(pos, token, team);
      }
   }

   static void place_holes(const Config &cfg, Board &board)
   {
      for(const auto &pos : cfg.hole_positions) {
         board[pos] = Piece(pos, Token::hole, Team::BLUE);
      }
   }

   aze::Status check_terminal(State &state)
   {
      if(auto dead_pieces = state.graveyard(Team::BLUE);
         dead_pieces.find(Token::flag) != dead_pieces.end()) {
         // flag of team 0 has been captured (killed), therefore team 0 lost
         return state.status(aze::Status::WIN_RED);
      } else if(dead_pieces = state.graveyard(Team::RED);
                dead_pieces.find(Token::flag) != dead_pieces.end()) {
         // flag of team 1 has been captured (killed), therefore team 1 lost
         return state.status(aze::Status::WIN_BLUE);
      }

      // committing draw rules here

      // Rule 1: If either team has no moves left.
      if(not has_valid_actions(state, Team::BLUE) or not has_valid_actions(state, Team::RED)) {
         LOGD2("Valid actions BLUE:", aze::utils::VectorPrinter(valid_actions(state, Team::BLUE)));
         LOGD2("Valid actions RED:", aze::utils::VectorPrinter(valid_actions(state, Team::RED)));
         return state.status(Status::TIE);
      }

      // Rule 1: The maximum turn count has been reached
      if(std::cmp_greater_equal(state.turn_count(), state.config().max_turn_count)) {
         LOGD2("Turn count on finish: ", state.turn_count());
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

      const auto &p_b = p_b_opt.value();

      // check if the target position holds a piece and whose team it belongs to
      if(p_a_opt.has_value()) {
         const auto &p_a = p_a_opt.value();
         if(p_a.team() == p_b.team())
            return false;  // cant fight
                           // pieces of
                           // own team
         if(p_a.token() == Token::hole) {
            return false;  // cant fight hole
         }
      }

      int move_dist = abs(pos_after[1] - pos_before[1]) + abs(pos_after[0] - pos_before[0]);

      // check if the move distance is within the move range of the token
      if(not state.config().move_ranges.at(p_b.token())(static_cast< unsigned long >(move_dist))) {
         return false;
      }

      // check for any "diagonal" moves (move not in a straight line) and, if not, for any pieces
      // blocking the way
      if(move_dist > 1) {
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

   template < ranges::contiguous_range Range >
   auto _valid_vectors(Position pos, Range shape, int distance = 1)
   {
      int right_end = static_cast< int >(shape[0]) - pos[0];
      int top_end = static_cast< int >(shape[1]) - pos[1];
      return ranges::views::concat(
                // all possible positions to the left until board ends
                ranges::views::zip(
                   ranges::views::iota(std::max(-distance, -pos[0]), 0), ranges::views::repeat(0)),
                // all possible positions to the right until board ends
                ranges::views::zip(
                   ranges::views::iota(1, std::min(right_end, distance + 1)),
                   ranges::views::repeat(0)),
                // all possible positions to the bottom until board ends
                ranges::views::zip(
                   ranges::views::repeat(0), ranges::views::iota(std::max(-distance, -pos[1]), 0)),
                // all possible positions to the top until board ends
                ranges::views::zip(
                   ranges::views::repeat(0),
                   ranges::views::iota(1, std::min(top_end, distance + 1))))
             | ranges::views::transform([](auto x_y) {
                  //                  LOGD2("In _valid_vectors: ", Position(std::get< 0 >(x_y),
                  //                  std::get< 1 >(x_y)));
                  return Position(std::get< 0 >(x_y), std::get< 1 >(x_y));
               });
   }

   std::vector< Action > valid_actions(const State &state, Team team)
   {
      LOGD("Checking for valid actions.")
      const auto &board = state.board();
      std::vector< Action > actions_possible;
      for(const auto &elem : board) {
         if(elem.has_value()) {
            const auto &piece = elem.value();
            if(piece.team() == team) {
               // the position we are dealing with
               auto pos = piece.position();
               int token_move_range = 0;
               auto mr_tester = state.config().move_ranges.at(piece.token());
               for(int distance : ranges::views::iota(1, int(ranges::max(board.shape())))
                                     | ranges::views::reverse) {
                  if(mr_tester(static_cast< size_t >(distance))) {
                     token_move_range = distance;
                     break;
                  }
               }
               ranges::for_each(
                  _valid_vectors(pos, board.shape(), token_move_range),
                  [&](const Position &pos_to) {
                     Action action{pos, pos + pos_to};
                     if(is_valid(state, action)) {
                        actions_possible.emplace_back(action);
                     }
                  });
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
               piece.team() == team && not std::set{Token::flag, Token::bomb}.contains(token)) {
               // the position we are dealing with
               auto pos = piece.position();

               //               LOGD2("check for piece", token_name(piece.token()) + " " +
               //               team_name(piece.team()));
               int token_move_range = 0;
               auto mr_tester = state.config().move_ranges.at(piece.token());
               for(int distance : ranges::views::iota(1, int(ranges::max(board.shape())))
                                     | ranges::views::reverse) {
                  if(mr_tester(static_cast< size_t >(distance))) {
                     token_move_range = distance;
                     break;
                  }
               }

               //               auto val = ranges::any_of(
               //                  _valid_vectors(pos, board.shape(), token_move_range),
               //                  [&](const Position &vector) -> bool {
               //                     return is_valid(state, Action{pos, pos + vector});
               //                  });
               //               LOGD("We're here again");
               if(ranges::any_of(
                     _valid_vectors(pos, board.shape(), token_move_range),
                     [&](const Position &vector) -> bool {
                        return is_valid(state, Action{pos, pos + vector});
                     })) {
                  return true;
               }
            }
         }
      }
      return false;
   }

   static std::map< Position, Token >
   draw_setup_uniform(const Config &config, Board &curr_board, Team team, aze::utils::RNG &rng)
   {
      std::map< Position, Token > setup_out;

      auto start_positions = config.start_positions.at(team);
      auto token_counter = config.token_counters.at(team);
      auto uniq_token_view = token_counter | ranges::views::keys;
      std::vector< Token > tokenvec = {uniq_token_view.begin(), uniq_token_view.end()};

      ranges::shuffle(start_positions, rng);
      auto int_dist = std::uniform_int_distribution< size_t >(0, tokenvec.size() - 1);

      while(not start_positions.empty()) {
         auto &pos = start_positions.back();
         if(curr_board[pos].has_value()) {
            // if the current board already has a piece at this location, then remove the
            // position from the possible ones.
            start_positions.pop_back();
            continue;
         }
         auto choice = int_dist(rng);
         auto token = tokenvec[choice];
         // needs to be refernce since it is decremented inplace
         auto &count = token_counter[token];
         if(count > 0) {
            setup_out[pos] = token;
            count--;
            start_positions.pop_back();
         }
         if(count == 0) {
            tokenvec.erase(tokenvec.begin() + static_cast< long >(choice));
            int_dist = std::uniform_int_distribution< size_t >(0, tokenvec.size() - 1);
         }
      }
      if(not tokenvec.empty()) {
         throw std::invalid_argument(
            "Current board setup and config could not be made to agree with number of tokens to "
            "place on it.");
      }
      return setup_out;
   }

   static Board create_empty_board(const Config &config)
   {
      Board b(config.game_dims);
      for(auto x : ranges::views::iota(size_t(0), config.game_dims[0])) {
         for(auto y : ranges::views::iota(size_t(0), config.game_dims[1])) {
            b[{x, y}] = std::nullopt;
         }
      }
      return b;
   }

   template <
      std::invocable< const Config &, Board &, Team, aze::utils::RNG & > SampleStrategyType >
   Board draw_board(
      const Config &config,
      Board curr_board,
      aze::utils::RNG &rng,
      SampleStrategyType setup_sampler = [](...) { return; })
   {
      for(size_t i = 0; i < 2; i++) {
         auto team = Team(i);
         if(config.fixed_setups[i]) {
            place_setup(config.setups.at(team).value(), curr_board, team);
         } else {
            place_setup(setup_sampler(config, curr_board, team, rng), curr_board, team);
         }
      }
      return curr_board;
   }
};

}  // namespace stratego
