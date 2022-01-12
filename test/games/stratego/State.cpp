#include "State.h"

#include <functional>


namespace stratego {

State::State(size_t shape_x, size_t shape_y)
    : State(std::array< size_t, 2 >{shape_x, shape_y}, std::array< int, 2 >{0, 0})
{
}

State::State(size_t shape) : State(shape, shape) {}

State::State(
   std::array< size_t, 2 > shape,
   const std::map< position_type, token_type > &setup_0,
   const std::map< position_type, token_type > &setup_1)
    : State(shape, std::array< int, 2 >{0, 0}, setup_0, setup_1)
{
}

State::State(
   size_t shape,
   const std::map< position_type, token_type > &setup_0,
   const std::map< position_type, token_type > &setup_1)
    : State(std::array< size_t, 2 >{shape, shape}, setup_0, setup_1)
{
}

State::State(
   std::array< size_t, 2 > shape,
   const std::map< position_type, int > &setup_0,
   const std::map< position_type, int > &setup_1)
    : State(std::make_shared< board_type >(shape, setup_0, setup_1))
{
}

State::State(
   size_t shape,
   const std::map< position_type, int > &setup_0,
   const std::map< position_type, int > &setup_1)
    : State({shape, shape}, setup_0, setup_1)
{
}

int State::fight(piece_type &attacker, piece_type &defender)
{
   return Logic< board_type >::fight_outcome(attacker, defender);
}

int State::_do_move(const move_type &move)
{
   // preliminaries
   const position_type &from = move[0];
   const position_type &to = move[1];
   int fight_outcome = 404;

   // save the access to the pieces in question
   // (removes redundant searching in board later)
   sptr< piece_type > piece_from = (*board())[from];
   sptr< piece_type > piece_to = (*board())[to];
   piece_from->set_flag_has_moved();

   // enact the move
   if(! piece_to->is_null()) {
      // uncover participant pieces
      piece_from->set_flag_unhidden();
      piece_to->set_flag_unhidden();

      nr_rounds_without_fight() = 0;

      // engage in fight, since piece_to is not a null piece
      fight_outcome = fight(*piece_from, *piece_to);
      if(fight_outcome == 1) {
         // 1 means attacker won, defender died
         auto null_piece = std::make_shared< piece_type >(from);
         board()->update_board(from, null_piece);
         board()->update_board(to, piece_from);

         _update_dead_pieces(piece_to);
      } else if(fight_outcome == 0) {
         // 0 means stalemate, both die
         auto null_piece_from = std::make_shared< piece_type >(from);
         board()->update_board(from, null_piece_from);
         auto null_piece_to = std::make_shared< piece_type >(to);
         board()->update_board(to, null_piece_to);

         _update_dead_pieces(piece_from);
         _update_dead_pieces(piece_to);
      } else {
         // -1 means defender won, attacker died
         auto null_piece = std::make_shared< piece_type >(from);
         board()->update_board(from, null_piece);

         _update_dead_pieces(piece_from);
      }
   } else {
      // no fight happened, simply move piece_from onto new position
      auto null_piece = std::make_shared< piece_type >(from);
      board()->update_board(from, null_piece);
      board()->update_board(to, piece_from);

      nr_rounds_without_fight() += 1;
   }
   return fight_outcome;
}
State *State::clone_impl() const
{
   const auto &hist = history();
   HistoryStratego hist_copy;
   // copy the contents of each map
   for(const auto &turn : hist.turns()) {
      std::apply(
         &HistoryStratego::commit_move,
         std::tuple_cat(std::forward_as_tuple(hist_copy, turn), hist.get_by_turn(turn)));
   }
   return new State(
      std::static_pointer_cast< board_type >(board()->clone()),
      status(),
      false,
      turn_count(),
      hist,
      nr_rounds_without_fight(),
      graveyard());
}

}