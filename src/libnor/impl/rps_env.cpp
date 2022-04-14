
#include "nor/wrappers/rps_env.hpp"

nor::Player nor::games::rps::Environment::active_player(
   const nor::games::rps::Environment::world_state_type& wstate) const
{
   return to_player(wstate.active_team());
}
bool nor::games::rps::Environment::is_terminal(
   nor::games::rps::Environment::world_state_type& wstate)
{
   return wstate.terminal();
}
double nor::games::rps::Environment::reward(
   nor::Player player,
   nor::games::rps::Environment::world_state_type& wstate)
{
   return wstate.payoff(to_team(player));
}
void nor::games::rps::Environment::transition(
   nor::games::rps::Environment::world_state_type& worldstate,
   const nor::games::rps::Environment::action_type& action) const
{
   worldstate.apply_action(action);
}
nor::games::rps::Environment::observation_type nor::games::rps::Environment::private_observation(
   nor::Player player,
   const nor::games::rps::Environment::world_state_type& wstate) const
{
   std::stringstream ss;
   if(std::none_of(wstate.picks().begin(), wstate.picks().end(), [&](const auto& pick) {
         return pick.has_value();
      })) {
      return "";
   }
   if(player == Player::alex) {
      ss
         << (wstate.picks()[0].has_value() ? common::to_string(wstate.picks()[0].value().hand)
                                           : "");
      ss << "-?";
   } else if(player == Player::bob) {
      ss << "?-";
      ss
         << (wstate.picks()[1].has_value() ? common::to_string(wstate.picks()[1].value().hand)
                                           : "");
   } else {
      throw std::invalid_argument("Passed player is not 'alex' or 'bob'.");
   }
   return ss.str();
}
nor::games::rps::Environment::observation_type nor::games::rps::Environment::private_observation(
   nor::Player player,
   const nor::games::rps::Environment::action_type& action) const
{
   if(action.team == to_team(player)) {
      return std::string(common::to_string(action.hand));
   }
   return "?";
}
nor::games::rps::Environment::observation_type nor::games::rps::Environment::tiny_repr(
   const nor::games::rps::Environment::world_state_type& wstate) const
{
   std::stringstream ss;
   ss
      << (wstate.picks()[0].has_value()
             ? std::string(common::to_string(wstate.picks()[0].value().hand))
             : "");
   ss << "-";
   ss
      << (wstate.picks()[1].has_value()
             ? std::string(common::to_string(wstate.picks()[1].value().hand))
             : "");
   return ss.str();
}
