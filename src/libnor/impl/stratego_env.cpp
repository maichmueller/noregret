
#include "nor/wrappers/stratego_env.hpp"

#include "stratego/stratego.hpp"

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
bool NORStrategoEnv::is_terminal(world_state_type& wstate)
{
   return wstate.status() != stratego::Status::ONGOING;
}
std::vector< stratego::Action > NORStrategoEnv::actions(Player player,
   const world_state_type& wstate)
{
   return m_logic->valid_actions(wstate, to_team(player));
}

void NORStrategoEnv::transition(const stratego::Action& action, world_state_type& worldstate)
{
   m_logic->apply_action(worldstate, action);
}
auto NORStrategoEnv::reset(world_state_type& wstate)
{
   m_logic->reset(wstate);
}
NORStrategoEnv::observation_type NORStrategoEnv::private_observation(
   Player player,
   const NORStrategoEnv::world_state_type& wstate)
{
   // TODO: implement string representation of state
}
NORStrategoEnv::observation_type NORStrategoEnv::private_observation(
   Player player,
   const NORStrategoEnv::action_type& action)
{
   // TODO: implement string representation of action
}
NORStrategoEnv::observation_type NORStrategoEnv::public_observation(
   Player player,
   const NORStrategoEnv::world_state_type& wstate)
{
   // TODO: implement string representation of state
}
NORStrategoEnv::observation_type NORStrategoEnv::public_observation(
   Player player,
   const NORStrategoEnv::action_type& action)
{
   // TODO: implement string representation of action
}

}  // namespace nor::games