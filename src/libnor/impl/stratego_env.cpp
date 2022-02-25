
#include "nor/wrappers/stratego_env.hpp"

#include "stratego/stratego.hpp"

namespace nor::games::stratego {

double Environment::_status_to_reward(::stratego::Status status, Player player)
{
   switch(status) {
      case ::stratego::Status::ONGOING:
      case ::stratego::Status::TIE: {
         return 0.;
      }
      case ::stratego::Status::WIN_BLUE: {
         return player == Player::alex ? 1. : -1.;
      }
      case ::stratego::Status::WIN_RED: {
         return player == Player::bob ? 1. : -1.;
      }
      default: {
         throw std::logic_error("Unknown Status enum returned.");
      }
   }
}
double Environment::reward(Player player, world_state_type& wstate)
{
   return _status_to_reward(wstate.logic()->check_terminal(wstate), player);
}
bool Environment::is_terminal(world_state_type& wstate)
{
   return wstate.status() != ::stratego::Status::ONGOING;
}
std::vector< Environment::action_type > Environment::actions(
   Player player,
   const world_state_type& wstate)
{
   return m_logic->valid_actions(wstate, to_team(player));
}

void Environment::transition(const action_type & action, world_state_type& worldstate)
{
   m_logic->apply_action(worldstate, action);
}
auto Environment::reset(world_state_type& wstate)
{
   m_logic->reset(wstate);
}
Environment::observation_type Environment::private_observation(
   Player player,
   const Environment::world_state_type& wstate)
{
   // TODO: implement string representation of state
}
Environment::observation_type Environment::private_observation(
   Player player,
   const Environment::action_type& action)
{
   // TODO: implement string representation of action
}
Environment::observation_type Environment::public_observation(
   Player player,
   const Environment::world_state_type& wstate)
{
   // TODO: implement string representation of state
}
Environment::observation_type Environment::public_observation(
   Player player,
   const Environment::action_type& action)
{
   // TODO: implement string representation of action
}
Environment::Environment(uptr< ::stratego::Logic >&& logic) : m_logic(std::move(logic)) {}

}  // namespace nor::games