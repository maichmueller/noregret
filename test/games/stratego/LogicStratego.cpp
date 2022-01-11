
#include "LogicStratego.h"

std::map< std::array< int, 2 >, int > BattleMatrix::initialize_battle_matrix()
{
   std::map< std::array< int, 2 >, int > bm;
   for(int i = 1; i < 11; ++i) {
      bm[{i, i}] = 0;
      for(int j = i + 1; j < 11; ++j) {
         bm[{i, j}] = -1;
         bm[{j, i}] = 1;
      }
      bm[{i, 0}] = 1;
      if(i == 3)
         bm[{i, 11}] = 1;
      else
         bm[{i, 11}] = -1;
   }
   bm[{1, 10}] = 1;
   return bm;
}

const std::map< std::array< int, 2 >, int >
   BattleMatrix::battle_matrix = initialize_battle_matrix();

// int LogicStratego::_find_action_idx(std::vector<strat_move_base_t>
// &vec_to_search, strat_move_base_t &action_to_find) {
//
//    // func to find a specific vector in another (assuming types are all
//    known)
//
//    int idx = 0;
//    std::vector<strat_move_base_t >::iterator entry;
//    for(entry = vec_to_search.begin(); entry != vec_to_search.end(); ++entry)
//    {
//        if((*entry)[0] == action_to_find[0] && (*entry)[1] ==
//        action_to_find[1]) {
//            // element has been found, exit the for-loop
//            break;
//        }
//        // increase idx, as this idx wasnt the right one
//        ++idx;
//    }
//    if(entry == vec_to_search.end())
//        return -1;  // vector has not been found, -1 as error index
//    else
//        return idx;
//}
//
//
//
///**
// * Method to check a move (as given by @pos and @pos_to) for validity on the
// @board. If the move
// * is legal, its associated action will be enabled (vector position set to 1)
// in the provided @action_mask.
// * The user needs to also provide the vector of the action representation and
// the vector of the action range
// * that this action belongs to, in order to search for the correct index of
// the action.
// * If @flip_board is true, the move change is inverted (multiplied comp. wise
// by -1). This becomes necessary,
// * since the action representation will need to be consistent for the neural
// net. As such, the action_mask for
// * team 1 needs to have the correct action associated for a flipped board.
// The example further demonstrates
// * this need.
// *
// * Example:
// * Imagine a board (table) in which the top row is index 0 and the leftmost
// column is index 0
// *      For team 0: The move m := (0,4) -> (2,4) means 'move 2 rows DOWN' ->
// action effect (2,0)
// *      For team 1: The move n := (4,0) -> (2,0) means 'move 2 rows UP'   ->
// action effect (-2,0)
// * However as the NN needs its representation always from the perspective of
// team 0, the move n of team 1
// * is translated to move m, and as such the action effect needs to be flipped.
// *
// * @param action_mask binary vector of action validity. Position i says action
// i is valid or invalid.
// * @param board the game board depicting the current scenario to which the
// action mask is sought.
// * @param act_range_start the start index of the action range that this
// potentially valid move falls into.
// * @param action_arr the vector of the action representation.
// * @param act_range the range of actions in which to search for the
// represented move.
// * @param pos the starting positon.
// * @param pos_to the target position.
// * @param flip_board switch to flip the for team 1.
// */
// void LogicStratego::enable_action_if_legal(std::vector<int>& action_mask,
// const Board& board,
//                                           int act_range_start,
//                                           const std::vector<strat_move_base_t
//                                           >& action_arr, const
//                                           std::vector<int>& act_range, const
//                                           Position& pos, const Position&
//                                           pos_to, const bool flip_board) {
//
//    strat_move_t move = {pos, pos_to};
//
//    if(is_legal_move(board, move)) {
//        strat_move_base_t action_effect;
//
//        if(flip_board)
//            action_effect = {pos[0] - pos_to[0], pos[1] - pos_to[1]};
//        else
//            action_effect = {pos_to[0] - pos[0], pos_to[1] - pos[1]};
//        std::vector<strat_move_base_t > slice(act_range.size());
//        for(unsigned long idx = 0; idx < slice.size(); ++idx) {
//            slice[idx] = action_arr[act_range[idx]];
//        }
//        int idx = LogicStratego::_find_action_idx(slice, action_effect);
//        action_mask[act_range_start + idx] = 1;
//    }
//};
//
//
// std::vector<int> LogicStratego::get_action_mask(
//        const Board &board, const std::vector<strat_move_base_t >& action_arr,
//        const std::map<std::array<int, 2>, std::tuple<int, std::vector<int>>>&
//        piece_act_map, Team team) {
//
//    std::vector<int> action_mask(action_arr.size(), 0);
//    int board_len = board.get_shape();
//
//    for( auto elem = board.begin(); elem != board.end(); ++elem) {
//        sptr<Piece> piece = elem->second;
//        if(!piece->is_null() && piece->get_team() == team &&
//        piece->get_flag_can_move()) {
//
//            std::array<int, 2> type_ver = {piece->get_token(),
//            piece->get_version()}; const auto& [start_idx, act_range] =
//            piece_act_map.at(type_ver);
//
//            // the position we are dealing with
//            Position pos = piece->get_position();
//
//            std::vector<Position> all_pos_targets(4);
//            if(piece->get_token() == 2) {
//                all_pos_targets.resize(board_len - pos[0] - 1 + board_len -
//                pos[1] - 1 + pos[0] + pos[1]); auto all_pos_targets_it =
//                all_pos_targets.begin();
//                // all possible moves to the top until board ends
//                for(int i = 1; i < board_len - pos[0]; ++i,
//                ++all_pos_targets_it) {
//                    *all_pos_targets_it = {pos[0] + i, pos[1] + 0};
//                }
//                // all possible moves to the right until board ends
//                for(int i = 1; i < board_len - pos[1]; ++i,
//                ++all_pos_targets_it) {
//                    *all_pos_targets_it = {pos[0] + 0, pos[1] + i};
//                }
//                // all possible moves to the bottom until board ends
//                for(int i = 1; i < pos[0] + 1; ++i, ++all_pos_targets_it) {
//                    *all_pos_targets_it = {pos[0] - i, pos[1] + 0};
//                }
//                // all possible moves to the left until board ends
//                for(int i = 1; i < pos[1] + 1; ++i, ++all_pos_targets_it) {
//                    *all_pos_targets_it = {pos[0] + 0, pos[1] -i};
//                }
//            }
//            else {
//                // all moves are 1 step to left, right, top, or bottom
//                all_pos_targets[0] = {pos[0] + 1, pos[1]};
//                all_pos_targets[1] = {pos[0], pos[1] + 1};
//                all_pos_targets[2] = {pos[0] - 1, pos[1]};
//                all_pos_targets[3] = {pos[0], pos[1] - 1};
//
//            }
//            for(auto& pos_to : all_pos_targets) {
//                LogicStratego::_enable_action_if_legal(action_mask, board,
//                                                       start_idx, action_arr,
//                                                       act_range, pos, pos_to,
//                                                       team);
//            }
//        }
//    }
//    return action_mask;
//}