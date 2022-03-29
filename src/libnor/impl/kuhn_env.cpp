
#include "nor/wrappers/kuhn_env.hpp"

#include <unordered_set>

nor::Player nor::games::kuhn::Environment::active_player(
   const nor::games::kuhn::Environment::world_state_type& wstate) const
{
   return to_nor_player(wstate.active_player());
}

bool nor::games::kuhn::Environment::is_terminal(
   nor::games::kuhn::Environment::world_state_type& wstate)
{
   return wstate.is_terminal();
}

double nor::games::kuhn::Environment::reward(
   nor::Player player,
   nor::games::kuhn::Environment::world_state_type& wstate)
{
   return wstate.payoff(to_kuhn_player(player));
}
void nor::games::kuhn::Environment::transition(
   nor::games::kuhn::Environment::world_state_type& worldstate,
   const nor::games::kuhn::Environment::action_type& action) const
{
   worldstate.apply_action(action);
}
std::string nor::games::kuhn::Environment::private_observation(
   nor::Player player,
   const nor::games::kuhn::Environment::world_state_type& wstate) const
{
   if(not std::unordered_set{nor::Player::alex, nor::Player::bob}.contains(player)) {
      throw std::invalid_argument("Player parameter has to be either 'alex' (0) or 'bob' (1)");
   }
   auto pub_obs = public_observation(wstate);
   // replace the opponents card with a question mark since it cannot be observed.
   pub_obs.replace((1 - static_cast< uint8_t >(player)), 1, "?");
   return pub_obs;
}
std::string nor::games::kuhn::Environment::public_observation(
   const nor::games::kuhn::Environment::world_state_type& wstate) const
{
   std::stringstream ss;
   for(uint8_t i = 0; i < 2; i++) {
      ss << *std::begin(
         wstate.cards()[i].has_value() ? common::enum_name(wstate.cards()[i].value()) : "-");
   }
   for(auto card : wstate.history().sequence) {
      ss << *std::begin(common::enum_name(card));
   }
   return ss.str();
}

std::string nor::games::kuhn::Environment::private_observation(
   nor::Player,
   const nor::games::kuhn::Environment::action_type& action) const
{
   return public_observation(action);
}
std::string nor::games::kuhn::Environment::public_observation(
   const nor::games::kuhn::Environment::action_type& action) const
{
   return std::string(common::enum_name(action));
}
void nor::games::kuhn::Environment::transition(
   nor::games::kuhn::Environment::world_state_type& worldstate,
   const nor::games::kuhn::Environment::chance_outcome_type& action) const
{
   worldstate.apply_action(action);
}
nor::games::kuhn::Environment::observation_type nor::games::kuhn::Environment::private_observation(
   nor::Player,
   const nor::games::kuhn::Environment::chance_outcome_type& action) const
{
   return std::string(common::enum_name(action));
}
nor::games::kuhn::Environment::observation_type nor::games::kuhn::Environment::public_observation(
   const nor::games::kuhn::Environment::chance_outcome_type& action) const
{
   return std::string(common::enum_name(action));
}
