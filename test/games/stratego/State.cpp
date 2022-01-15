#include "State.h"

#include <functional>

namespace stratego {

int State::fight(piece_type &attacker, piece_type &defender)
{
   return Logic< board_type >::fight_outcome(attacker, defender);
}

int State::do_move(const move_type &move)
{
   // preliminaries
   const position_type &from = move[0];
   const position_type &to = move[1];
   int fight_outcome = 404;

   // save the access to the pieces in question
   // (removes redundant searching in board later)
   sptr< piece_type > piece_from = (*board())[from];
   sptr< piece_type > piece_to = (*board())[to];
   piece_from->flag_has_moved();

   // enact the move
   if(! piece_to->is_null()) {
      // uncover participant pieces
      piece_from->flag_unhidden();
      piece_to->flag_unhidden();

      // engage in fight, since piece_to is not a null piece
      fight_outcome = fight(*piece_from, *piece_to);
      if(fight_outcome == 1) {
         // 1 means attacker won, defender died
         auto null_piece = std::make_shared< piece_type >(from);
         board()->update_board(from, null_piece);
         board()->update_board(to, piece_from);

         _to_graveyard(piece_to);
      } else if(fight_outcome == 0) {
         // 0 means stalemate, both die
         auto null_piece_from = std::make_shared< piece_type >(from);
         board()->update_board(from, null_piece_from);
         auto null_piece_to = std::make_shared< piece_type >(to);
         board()->update_board(to, null_piece_to);

         _to_graveyard(piece_from);
         _to_graveyard(piece_to);
      } else {
         // -1 means defender won, attacker died
         auto null_piece = std::make_shared< piece_type >(from);
         board()->update_board(from, null_piece);

         _to_graveyard(piece_from);
      }
   } else {
      // no fight happened, simply move piece_from onto new position
      auto null_piece = std::make_shared< piece_type >(from);
      board()->update_board(from, null_piece);
      board()->update_board(to, piece_from);
   }
   return fight_outcome;
}
State *State::clone_impl() const
{
   const auto &hist = history();
   History hist_copy;
   // copy the contents of each map
   for(const auto &turn : hist.turns()) {
      std::apply(
         &History::commit_move,
         std::tuple_cat(std::forward_as_tuple(hist_copy, turn), hist.get_by_turn(turn)));
   }
   return new State(
      std::static_pointer_cast< board_type >(board()->clone()),
      status(),
      false,
      turn_count(),
      hist,
      graveyard());
}
State::State(Config config) : base_type(nullptr), m_config(config), m_graveyard() {

}

}  // namespace stratego