
#include "nor/wrappers/stratego_game.hpp"

#include "stratego/Logic.hpp"

namespace nor::games {

double NORStrategoGame::_status_to_reward(stratego::Status status, Player player)
{
   switch(status) {
      case stratego::Status::ONGOING:
      case stratego::Status::TIE: {
         return 0.;
      }
      case stratego::Status::WIN_BLUE: {
         return player == Player::alex ? 1. : -1.;
      }
      case stratego::Status::WIN_RED: {
         return player == Player::bob ? 1. : -1.;
      }
      default: {
         throw std::logic_error("Unknown Status enum returned.");
      }
   }
}
double NORStrategoGame::reward(Player player, world_state_type& wstate)
{
   return _status_to_reward(wstate.logic()->check_terminal(wstate), player);
}
bool NORStrategoGame::is_terminal(NORStrategoGame::world_state_type& wstate)
{
   return m_game.state().status() != stratego::Status::ONGOING;
}
double NORStrategoGame::reward(Player player)
{
   return _status_to_reward(m_game.state().status(), player);
}
std::vector< stratego::Action > NORStrategoGame::actions(Player player)
{
   return m_game.state().logic()->valid_actions(m_game.state(), to_team(player));
}

void NORStrategoGame::transition(
   const stratego::Action& action,
   world_state_type& worldstate)
{
   m_game.state().logic()->apply_action(worldstate, action);
}

}  // namespace nor::games