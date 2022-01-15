
#pragma once

#include <aze/aze.h>

#include <functional>

#include "Action.h"
#include "State.h"

namespace stratego::logic {

using Team = aze::Team;

inline int fight_outcome(const State& state, const Piece &attacker, const Piece &defender)
{
   return fight_outcome(std::array{attacker.token(), defender.token()});
}

inline int fight_outcome(const State& state, std::array< Token, 2 > att_def)
{
   return BattleMatrix::fight_outcome(att_def);
}

static inline std::vector< Position > start_positions(int shape, aze::Team team)
{
   int t = static_cast< int >(team);
   if(t != 0 && t != 1)
      throw std::invalid_argument("'team' not in {0, 1}.");

   if(shape == 5) {
      if(t == 0)
         return {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 0}, {1, 1}, {1, 2}, {1, 3}, {1, 4}};
      else
         return {{4, 0}, {4, 1}, {4, 2}, {4, 3}, {4, 4}, {3, 0}, {3, 1}, {3, 2}, {3, 3}, {3, 4}};
   } else if(shape == 7) {
      if(t == 0)
         return {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {0, 6},
                 {1, 0}, {1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6},
                 {2, 0}, {2, 1}, {2, 2}, {2, 3}, {2, 4}, {2, 5}, {2, 6}};
      else
         return {{4, 0}, {4, 1}, {4, 2}, {4, 3}, {4, 4}, {4, 5}, {4, 6},
                 {5, 0}, {5, 1}, {5, 2}, {5, 3}, {5, 4}, {5, 5}, {5, 6},
                 {6, 0}, {6, 1}, {6, 2}, {6, 3}, {6, 4}, {6, 5}, {6, 6}};

   } else if(shape == 10) {
      if(t == 0)
         return {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {0, 6}, {0, 7}, {0, 8}, {0, 9},
                 {1, 0}, {1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {1, 7}, {1, 8}, {1, 9},
                 {2, 0}, {2, 1}, {2, 2}, {2, 3}, {2, 4}, {2, 5}, {2, 6}, {2, 7}, {2, 8}, {2, 9},
                 {3, 0}, {3, 1}, {3, 2}, {3, 3}, {3, 4}, {3, 5}, {3, 6}, {3, 7}, {3, 8}, {3, 9}};

      else
         return {{6, 0}, {6, 1}, {6, 2}, {6, 3}, {6, 4}, {6, 5}, {6, 6}, {6, 7}, {6, 8}, {6, 9},
                 {7, 0}, {7, 1}, {7, 2}, {7, 3}, {7, 4}, {7, 5}, {7, 6}, {7, 7}, {7, 8}, {7, 9},
                 {8, 0}, {8, 1}, {8, 2}, {8, 3}, {8, 4}, {8, 5}, {8, 6}, {8, 7}, {8, 8}, {8, 9},
                 {9, 0}, {9, 1}, {9, 2}, {9, 3}, {9, 4}, {9, 5}, {9, 6}, {9, 7}, {9, 8}, {9, 9}};

   } else
      throw std::invalid_argument("'shape' not in {5, 7, 10}.");
}
};

bool is_legal_action(const State &state, const Action &action)
{
   const auto &[pos_before, pos_after] = action.get_positions();
   const auto &board = state.board();
   if(! (std::get< 0 >(board.check_bounds(pos_before)))
      || ! (std::get< 0 >(board.check_bounds(pos_after))))
      return false;

   sptr< Piece > p_b = board[pos_before];
   sptr< Piece > p_a = board[pos_after];

   if(p_b->is_null())
      return false;
   if(int type = p_b->token()[0]; type == 0 || type == 11) {
      return false;
   }
   if(! p_a->is_null()) {
      if(p_a->team() == p_b->team())
         return false;  // cant fight pieces of own team
      if(p_a->token()[0] == 99) {
         return false;  // cant fight obstacle
      }
   }

   int move_dist = abs(pos_after[1] - pos_before[1]) + abs(pos_after[0] - pos_before[0]);
   if(move_dist > 1) {
      if(p_b->token()[0] != 2)
         return false;  // not of type 2 , but is supposed to go far

      if(pos_after[0] == pos_before[0]) {
         int dist = pos_after[1] - pos_before[1];
         int sign = (dist >= 0) ? 1 : -1;
         for(int i = 1; i < std::abs(dist); ++i) {
            Position pos{pos_before[0], pos_before[1] + sign * i};
            if(! board[pos]->is_null())
               return false;
         }
      } else if(pos_after[1] == pos_before[1]) {
         int dist = pos_after[0] - pos_before[0];
         int sign = (dist >= 0) ? 1 : -1;
         for(int i = 1; i < std::abs(dist); ++i) {
            Position pos = {pos_before[0] + sign * i, pos_before[1]};
            if(! board[pos]->is_null())
               return false;
         }
      } else
         return false;  // diagonal moves not allowed
   }
   return true;
}

template < class StateType >
std::vector< typename Logic< StateType >::Action > Logic< StateType >::get_legal_actions_(
   const State &state,
   Team team)
{
   const auto &board = state.board();
   int shape_x = board.get_shape()[0];
   int shape_y = board.get_shape()[1];
   int starts_x = board.get_starts()[0];
   int starts_y = board.get_starts()[1];
   std::vector< Action > actions_possible;
   for(auto elem = board.begin(); elem != board.end(); ++elem) {
      sptr< Piece > piece = elem->second;
      if(! piece->is_null() && piece->team() == team) {
         // the position we are dealing with
         auto pos = piece->position();

         if(piece->token()[0] == 2) {
            // all possible actions to the right until board ends
            for(int i = 1; i < starts_x + shape_x - pos[0]; ++i) {
               Position pos_to{pos[0] + i, pos[1]};
               Action action{pos, pos_to};
               if(is_legal_action_(board, action)) {
                  actions_possible.push_back(action);
               }
            }
            // all possible actions to the top until board ends
            for(int i = 1; i < starts_y + shape_y - pos[1]; ++i) {
               Position pos_to{pos[0], pos[1] + i};
               Action action{pos, pos_to};
               if(is_legal_action_(board, action)) {
                  actions_possible.push_back(action);
               }
            }
            // all possible actions to the left until board ends
            for(int i = 1; i < starts_x + pos[0] + 1; ++i) {
               Position pos_to{pos[0] - i, pos[1]};
               Action action{pos, pos_to};
               if(is_legal_action_(board, action)) {
                  actions_possible.push_back(action);
               }
            }
            // all possible actions to the bottom until board ends
            for(int i = 1; i < starts_y + pos[1] + 1; ++i) {
               Position pos_to{pos[0], pos[1] - i};
               Action action{pos, pos_to};
               if(is_legal_action_(board, action)) {
                  actions_possible.push_back(action);
               }
            }
         } else {
            // all actions are 1 step to left, right, top, or bottom
            std::vector< Position > pos_tos = {
               {pos[0] + 1, pos[1]},
               {pos[0], pos[1] + 1},
               {pos[0] - 1, pos[1]},
               {pos[0], pos[1] - 1}};
            for(auto &pos_to : pos_tos) {
               Action action{pos, pos_to};
               if(is_legal_action_(board, action)) {
                  actions_possible.push_back(action);
               }
            }
         }
      }
   }
   return actions_possible;
}

bool Logic< StateType >::has_legal_actions_(const State &state, Team team)
{
   const auto &board = state.board();
   int shape_x = board.get_shape()[0];
   int shape_y = board.get_shape()[1];
   int starts_x = board.get_starts()[0];
   int starts_y = board.get_starts()[1];
   for(auto elem = board.begin(); elem != board.end(); ++elem) {
      sptr< Piece > piece = elem->second;
      if(token_type essential_token = piece->token(); ! piece->is_null() && piece->team() == team
                                                      && essential_token != Token::flag
                                                      && essential_token != Token::bomb) {
         // the position we are dealing with
         auto pos = piece->position();

         if(piece->token() == Token::scout) {
            // all possible actions to the right until board ends
            for(int i = 1; i < starts_x + shape_x - pos[0]; ++i) {
               Position pos_to{pos[0] + i, pos[1]};
               Action action{pos, pos_to};
               if(is_legal_action_(board, action)) {
                  return true;
               }
            }
            // all possible actions to the top until board ends
            for(int i = 1; i < starts_y + shape_y - pos[1]; ++i) {
               Position pos_to{pos[0], pos[1] + i};
               Action action{pos, pos_to};
               if(is_legal_action_(board, action)) {
                  return true;
               }
            }
            // all possible actions to the left until board ends
            for(int i = 1; i < starts_x + pos[0] + 1; ++i) {
               Position pos_to{pos[0] - i, pos[1]};
               Action action{pos, pos_to};
               if(is_legal_action_(board, action)) {
                  return true;
               }
            }
            // all possible actions to the bottom until board ends
            for(int i = 1; i < starts_y + pos[1] + 1; ++i) {
               Position pos_to{pos[0], pos[1] - i};
               Action action{pos, pos_to};
               if(is_legal_action_(board, action)) {
                  return true;
               }
            }
         } else {
            // all actions are 1 step to left, right, top, or bottom
            std::vector< Position > pos_tos = {
               {pos[0] + 1, pos[1]},
               {pos[0], pos[1] + 1},
               {pos[0] - 1, pos[1]},
               {pos[0], pos[1] - 1}};
            for(const auto &pos_to : pos_tos) {
               Action action{pos, pos_to};
               if(is_legal_action_(board, action)) {
                  return true;
               }
            }
         }
      }
   }
   return false;
}

}  // namespace stratego