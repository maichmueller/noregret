
#include "nor/wrappers/stratego_env.hpp"

#include "stratego/Logic.hpp"

namespace nor::games {

double NORStrategoEnv::_status_to_reward(stratego::Status status, Player player)
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
double NORStrategoEnv::reward(Player player, world_state_type& wstate)
{
   return _status_to_reward(wstate.logic()->check_terminal(wstate), player);
}
bool NORStrategoEnv::is_terminal(NORStrategoEnv::world_state_type& wstate)
{
   return m_state.status() != stratego::Status::ONGOING;
}
double NORStrategoEnv::reward(Player player)
{
   return _status_to_reward(m_state.status(), player);
}
std::vector< stratego::Action > NORStrategoEnv::actions(Player player)
{
   return m_state.logic()->valid_actions(m_state, to_team(player));
}

void NORStrategoEnv::transition(const stratego::Action& action, world_state_type& worldstate)
{
   m_state.logic()->apply_action(worldstate, action);
}
auto NORStrategoEnv::reset()
{
   m_state.logic()->reset(m_state);
}

}  // namespace nor::games