#include "State.h"

#include <functional>
#include <utility>

#include "Logic.h"

namespace stratego {

int State::apply_action(const action_type &action)
{
   // preliminaries
   const position_type &from = action[0];
   const position_type &to = action[1];
   int fight_outcome = 404;

   // save the access to the pieces in question
   // (removes redundant searching in board later)
   auto piece_from = board()[from].value();
   auto piece_to_opt = board()[to];

   piece_from.flag_has_moved();

   // enact the move
   if(piece_to_opt.has_value()) {
      auto &piece_to = piece_to_opt.value();
      // uncover participant pieces
      piece_from.flag_unhidden(true);
      piece_to.flag_unhidden(true);

      // engage in fight, since piece_to is not a null piece
      fight_outcome = m_logic->fight(m_config, piece_from, piece_to);
      if(fight_outcome == 1) {
         // 1 means attacker won, defender died
         board()[from] = std::nullopt;
         board()[to] = piece_from;

         _to_graveyard(piece_to);
      } else if(fight_outcome == 0) {
         // 0 means stalemate, both die
         board()[from] = std::nullopt;
         board()[to] = std::nullopt;

         _to_graveyard(piece_from);
         _to_graveyard(piece_to);
      } else {
         // -1 means defender won, attacker died
         board()[from] = std::nullopt;

         _to_graveyard(piece_from);
      }
   } else {
      // no fight happened, simply move piece_from onto new position
      board()[from] = std::nullopt;
      board()[to] = piece_from;
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
   return new State(board(), status(), false, turn_count(), hist, graveyard());
}
State::State(Config config) : base_type(nullptr), m_config(std::move(config)), m_graveyard() {}

}  // namespace stratego