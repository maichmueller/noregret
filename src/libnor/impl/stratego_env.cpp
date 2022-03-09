
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
   const world_state_type& wstate) const
{
   return m_logic->valid_actions(wstate, to_team(player));
}

void Environment::transition(world_state_type& worldstate, const action_type& action) const
{
   m_logic->apply_action(worldstate, action);
}
void Environment::reset(world_state_type& wstate) const
{
   m_logic->reset(wstate);
}
Environment::observation_type Environment::private_observation(
   Player player,
   const Environment::world_state_type& wstate) const
{
   // TODO: implement string representation of state
}
Environment::observation_type Environment::private_observation(
   Player player,
   const Environment::action_type& action) const
{
   // TODO: implement string representation of action
}
Environment::observation_type Environment::public_observation(
   Player player,
   const Environment::world_state_type& wstate) const
{
   // TODO: implement string representation of state
}
Environment::observation_type Environment::public_observation(
   Player player,
   const Environment::action_type& action) const
{
   // TODO: implement string representation of action
}

Environment::Environment(uptr< ::stratego::Logic >&& logic) : m_logic(std::move(logic)) {}

Player Environment::active_player(const Environment::world_state_type& wstate) const
{
   return to_player(wstate.active_team());
}

}  // namespace nor::games::stratego