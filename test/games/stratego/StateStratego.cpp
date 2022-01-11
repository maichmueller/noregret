#include "StateStratego.h"

#include <functional>

StateStratego::StateStratego(size_t shape_x, size_t shape_y)
    : StateStratego(std::array< size_t, 2 >{shape_x, shape_y}, std::array< int, 2 >{0, 0})
{
}

StateStratego::StateStratego(size_t shape) : StateStratego(shape, shape) {}

StateStratego::StateStratego(
   std::array< size_t, 2 > shape,
   const std::map< position_type, token_type > &setup_0,
   const std::map< position_type, token_type > &setup_1)
    : StateStratego(shape, std::array< int, 2 >{0, 0}, setup_0, setup_1)
{
}

StateStratego::StateStratego(
   size_t shape,
   const std::map< position_type, token_type > &setup_0,
   const std::map< position_type, token_type > &setup_1)
    : StateStratego(std::array< size_t, 2 >{shape, shape}, setup_0, setup_1)
{
}

StateStratego::StateStratego(
   std::array< size_t, 2 > shape,
   const std::map< position_type, int > &setup_0,
   const std::map< position_type, int > &setup_1)
    : StateStratego(std::make_shared< board_type >(shape, setup_0, setup_1))
{
}

StateStratego::StateStratego(
   size_t shape,
   const std::map< position_type, int > &setup_0,
   const std::map< position_type, int > &setup_1)
    : StateStratego({shape, shape}, setup_0, setup_1)
{
}

int StateStratego::fight(piece_type &attacker, piece_type &defender)
{
   return LogicStratego< board_type >::fight_outcome(attacker, defender);
}

int StateStratego::_do_move(const move_type &move)
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
StateStratego *StateStratego::clone_impl() const
{
   const auto &hist = history();
   HistoryStratego hist_copy;
   // copy the contents of each map
   for(const auto &turn : hist.turns()) {
      std::apply(
         &HistoryStratego::commit_move,
         std::tuple_cat(std::forward_as_tuple(hist_copy, turn), hist.get_by_turn(turn)));
   }
   return new StateStratego(
      std::static_pointer_cast< board_type >(board()->clone()),
      status(),
      false,
      turn_count(),
      hist,
      nr_rounds_without_fight(),
      graveyard());
}
