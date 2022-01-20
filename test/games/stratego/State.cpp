#include "State.h"

#include <functional>
#include <utility>

#include "Logic.h"
#include "utils.hpp"

namespace stratego {

void State::apply_action(const action_type &action)
{
   // preliminaries
   const position_type &from = action[0];
   const position_type &to = action[1];

   // save the access to the pieces in question
   // (removes redundant searching in board later)
   auto piece_from = board()[from].value();
   auto piece_to_opt = board()[to];

   piece_from.flag_has_moved(true);

   // enact the move
   if(piece_to_opt.has_value()) {
      auto &piece_to = piece_to_opt.value();
      // uncover participant pieces
      piece_from.flag_unhidden(true);
      piece_to.flag_unhidden(true);

      // engage in fight, since piece_to is not a null piece
      FightOutcome fight_outcome = m_logic->fight(m_config, piece_from, piece_to);
      if(fight_outcome == FightOutcome::kill) {
         // means attacker won, defender died
         board()[from] = std::nullopt;
         board()[to] = piece_from;
         logic()->handle< FightOutcome::kill >(*this, piece_from, piece_to);
      } else if(fight_outcome == FightOutcome::stalemate) {
         // means stalemate, both die
         board()[from] = std::nullopt;
         board()[to] = std::nullopt;
         logic()->handle< FightOutcome::stalemate >(*this, piece_from, piece_to);
      } else if(fight_outcome == FightOutcome::death) {
         // means defender won, attacker died
         board()[from] = std::nullopt;
         logic()->handle< FightOutcome::death >(*this, piece_from, piece_to);
      } else {
         // handle unknown enum value?
      }
   } else {
      // no fight happened, simply move piece_from onto new position
      board()[from] = std::nullopt;
      board()[to] = piece_from;
   }
}

State *State::clone_impl() const
{
   const auto &hist = history();
   History hist_copy;
   // copy the contents of each map
   for(const size_t &turn : hist.turns()) {
      const auto &[team, action, pieces] = hist.get_by_turn(turn);
      hist_copy.commit_action(turn, team, action, pieces);
   }
   return new State(m_config, graveyard(), board(), turn_count(), hist_copy, rng());
}
State::State(Config config) : base_type(board()), m_config(std::move(config)), m_graveyard() {}

std::string State::string_representation() const
{
   return string_representation(aze::Team::BLUE, false);
}

std::string State::string_representation(aze::Team team, bool hide_unknowns) const
{
   return utils::print_board(board(), team, hide_unknowns);
}

}  // namespace stratego