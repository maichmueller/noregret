
#include "nor/env/rps.hpp"

#include "nor/utils/player_informed_type.hpp"

using namespace nor;
using namespace games::rps;

Player Environment::active_player(const Environment::world_state_type& wstate) const
{
   return to_player(wstate.active_team());
}
bool Environment::is_terminal(const Environment::world_state_type& wstate)
{
   return wstate.terminal();
}
double Environment::reward(Player player, const world_state_type& wstate)
{
   return wstate.payoff(to_team(player));
}
void Environment::transition(
   Environment::world_state_type& worldstate,
   const Environment::action_type& action
) const
{
   worldstate.apply_action(action);
}

Environment::observation_type Environment::tiny_repr(const Environment::world_state_type& wstate
) const
{
   std::stringstream ss;
   ss << (wstate.picks()[0].has_value() ? common::to_string(wstate.picks()[0].value()) : "");
   ss << "-";
   ss << (wstate.picks()[1].has_value() ? common::to_string(wstate.picks()[1].value()) : "");
   return ss.str();
}

std::vector< PlayerInformedType< std::optional< Environment::action_variant_type > > >
Environment::public_history(const Environment::world_state_type& wstate) const
{
   using value_type = PlayerInformedType< std::optional< Environment::action_variant_type > >;
   if(wstate.picks()[0].has_value()) {
      if(wstate.picks()[1].has_value()) {
         return {value_type{std::nullopt, Player::alex}, value_type{std::nullopt, Player::bob}};
      }
      return {value_type{std::nullopt, Player::alex}};
   }
   return {};
}

std::vector< PlayerInformedType< Environment::action_variant_type > > Environment::open_history(
   const Environment::world_state_type& wstate
) const
{
   std::vector< PlayerInformedType< action_variant_type > > out;
   for(auto&& [i, outcome_opt] : ranges::views::enumerate(wstate.picks())) {
      if(outcome_opt.has_value()) {
         out.emplace_back(*outcome_opt, Player(i));
      }
   }
   return out;
}

std::vector< PlayerInformedType< std::optional< Environment::action_variant_type > > >
Environment::private_history(Player, const Environment::world_state_type& wstate) const
{
   std::vector< PlayerInformedType< std::optional< action_variant_type > > > out;
   for(auto&& [i, outcome_opt] : ranges::views::enumerate(wstate.picks())) {
      if(outcome_opt.has_value()) {
         out.emplace_back(action_type{*outcome_opt}, Player(i));
      } else {
         out.emplace_back(std::nullopt, Player(i));
      }
   }
   return out;
}
Environment::observation_type Environment::private_observation(
   Player observer,
   const world_state_type& wstate,
   const action_type& action,
   const world_state_type& /*next_wstate*/
) const
{
   if(active_player(wstate) == observer) {
      return common::to_string(action);
   }
   return "";
}

Environment::observation_type Environment::public_observation(
   const world_state_type& wstate,
   const action_type& /*action*/,
   const world_state_type& /*next_wstate*/
) const
{
   return common::to_string(active_player(wstate)) + ":?";
}
