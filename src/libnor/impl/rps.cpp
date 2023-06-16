
#include "nor/env/rps.hpp"

using namespace nor;

Player games::rps::Environment::active_player(
   const games::rps::Environment::world_state_type& wstate
) const
{
   return to_player(wstate.active_team());
}
bool games::rps::Environment::is_terminal(games::rps::Environment::world_state_type& wstate)
{
   return wstate.terminal();
}
double
games::rps::Environment::reward(Player player, games::rps::Environment::world_state_type& wstate)
{
   return wstate.payoff(to_team(player));
}
void games::rps::Environment::transition(
   games::rps::Environment::world_state_type& worldstate,
   const games::rps::Environment::action_type& action
) const
{
   worldstate.apply_action(action);
}

games::rps::Environment::observation_type games::rps::Environment::tiny_repr(
   const games::rps::Environment::world_state_type& wstate
) const
{
   std::stringstream ss;
   ss
      << (wstate.picks()[0].has_value() ? std::string(common::to_string(wstate.picks()[0].value()))
                                        : "");
   ss << "-";
   ss
      << (wstate.picks()[1].has_value() ? std::string(common::to_string(wstate.picks()[1].value()))
                                        : "");
   return ss.str();
}
std::vector< PlayerInformedType<
   std::optional< std::variant< std::monostate, games::rps::Environment::action_type > > > >
games::rps::Environment::public_history(const games::rps::Environment::world_state_type& wstate
) const
{
   if(wstate.picks()[0].has_value()) {
      if(wstate.picks()[1].has_value()) {
         return {{std::nullopt, Player::alex}, {std::nullopt, Player::bob}};
      }
      return {{std::nullopt, Player::alex}};
   }
   return {};
}

std::vector<
   PlayerInformedType< std::variant< std::monostate, games::rps::Environment::action_type > > >
games::rps::Environment::open_history(const games::rps::Environment::world_state_type& wstate) const
{
   std::vector< PlayerInformedType< std::variant< std::monostate, action_type > > > out;
   for(auto&& [i, outcome_opt] : ranges::views::enumerate(wstate.picks())) {
      if(outcome_opt.has_value()) {
         out.emplace_back(outcome_opt.value(), Player(i));
      }
   }
   return out;
}

std::vector< PlayerInformedType<
   std::optional< std::variant< std::monostate, games::rps::Environment::action_type > > > >
games::rps::Environment::private_history(
   Player,
   const games::rps::Environment::world_state_type& wstate
) const
{
   std::vector< PlayerInformedType< std::optional< std::variant< std::monostate, action_type > > > >
      out;
   for(auto&& [i, outcome_opt] : ranges::views::enumerate(wstate.picks())) {
      if(outcome_opt.has_value()) {
         out.emplace_back(outcome_opt.value(), Player(i));
      } else {
         out.emplace_back(std::nullopt, Player(i));
      }
   }
   return out;
}
games::rps::Environment::observation_type games::rps::Environment::private_observation(
   Player observer,
   const world_state_type& wstate,
   const action_type& action,
   const world_state_type& next_wstate
) const
{
   if(active_player(wstate) == observer) {
      return std::string(common::to_string(action));
   }
   return "";
}

games::rps::Environment::observation_type games::rps::Environment::public_observation(
   const world_state_type& wstate,
   const action_type& action,
   const world_state_type& next_wstate
) const
{
   return common::to_string(active_player(wstate)) + ":?";
}
