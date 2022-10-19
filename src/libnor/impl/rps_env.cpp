
#include "nor/env/rps_env.hpp"

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
         << (wstate.picks()[0].has_value() ? common::to_string(wstate.picks()[0].value())
                                           : "");
      ss << "-?";
   } else if(player == Player::bob) {
      ss << "?-";
      ss
         << (wstate.picks()[1].has_value() ? common::to_string(wstate.picks()[1].value())
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
      return std::string(common::to_string(action));
}
nor::games::rps::Environment::observation_type nor::games::rps::Environment::tiny_repr(
   const nor::games::rps::Environment::world_state_type& wstate) const
{
   std::stringstream ss;
   ss
      << (wstate.picks()[0].has_value()
             ? std::string(common::to_string(wstate.picks()[0].value()))
             : "");
   ss << "-";
   ss
      << (wstate.picks()[1].has_value()
             ? std::string(common::to_string(wstate.picks()[1].value()))
             : "");
   return ss.str();
}
std::vector< nor::PlayerInformedType<
   std::optional< std::variant< std::monostate, nor::games::rps::Environment::action_type > > > >
nor::games::rps::Environment::public_history(
   const nor::games::rps::Environment::world_state_type& wstate) const
{
   if(wstate.picks()[0].has_value()) {
      if(wstate.picks()[1].has_value()) {
         return {{std::nullopt, Player::alex}, {std::nullopt, Player::bob}};
      }
      return {{std::nullopt, Player::alex}};
   }
   return {};
}

std::vector< nor::PlayerInformedType<
   std::variant< std::monostate, nor::games::rps::Environment::action_type > > >
nor::games::rps::Environment::open_history(
   const nor::games::rps::Environment::world_state_type& wstate) const
{
   std::vector< PlayerInformedType< std::variant< std::monostate, action_type > > > out;
   for(auto&& [i, outcome_opt] : ranges::views::enumerate(wstate.picks())) {
      if(outcome_opt.has_value()) {
         out.emplace_back(outcome_opt.value(), nor::Player(i));
      }
   }
   return out;
}

std::vector< nor::PlayerInformedType<
   std::optional< std::variant< std::monostate, nor::games::rps::Environment::action_type > > > >
nor::games::rps::Environment::private_history(
   nor::Player,
   const nor::games::rps::Environment::world_state_type& wstate) const
{
   std::vector< PlayerInformedType< std::optional< std::variant< std::monostate, action_type > > > >
      out;
   for(auto&& [i, outcome_opt] : ranges::views::enumerate(wstate.picks())) {
      if(outcome_opt.has_value()) {
         out.emplace_back(outcome_opt.value(), nor::Player(i));
      } else {
         out.emplace_back(std::nullopt, nor::Player(i));
      }
   }
   return out;
}
