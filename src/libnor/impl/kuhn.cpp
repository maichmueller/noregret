
#include "nor/env/kuhn.hpp"

#include <unordered_set>

using namespace nor;

Player games::kuhn::Environment::active_player(const world_state_type& wstate) const
{
   return to_nor_player(wstate.active_player());
}

bool games::kuhn::Environment::is_terminal(world_state_type& wstate)
{
   return wstate.is_terminal();
}

double games::kuhn::Environment::reward(Player player, world_state_type& wstate)
{
   return wstate.payoff(to_kuhn_player(player));
}

std::string games::kuhn::Environment::
   private_observation(Player, const world_state_type&, const action_type&, const world_state_type&)
      const
{
   return "-";
}
std::string games::kuhn::Environment::
   public_observation(const world_state_type&, const action_type& action, const world_state_type&)
      const
{
   return common::to_string(action);
}

games::kuhn::Environment::observation_type games::kuhn::Environment::
   private_observation(Player observer, const world_state_type&, const chance_outcome_type& outcome, const world_state_type&)
      const
{
   if(outcome.player == to_kuhn_player(observer)) {
      return std::string(common::to_string(outcome));
   }
   return "-";
}

games::kuhn::Environment::observation_type games::kuhn::Environment::public_observation(
   const world_state_type& wstate,
   const chance_outcome_type& action,
   const world_state_type& next_wstate
) const
{
   return std::to_string(static_cast< int >(action.player)) + ":?";
}

games::kuhn::Environment::observation_type games::kuhn::Environment::tiny_repr(
   const world_state_type& wstate
) const
{
   std::stringstream ss;
   for(auto [idx, card] : ranges::views::enumerate(wstate.cards())) {
      if(card.has_value()) {
         ss << card.value();
         if((idx == 0 and wstate.cards()[1].has_value()) or not wstate.history().empty()) {
            ss << "-";
         }
      }
   }
   for(auto [idx, action] : ranges::views::enumerate(wstate.history())) {
      ss << action;
      if(idx != wstate.history().size() - 1) {
         ss << "-";
      }
   }
   return ss.str();
}

std::vector< PlayerInformedType< std::optional< std::variant<
   games::kuhn::Environment::chance_outcome_type,
   games::kuhn::Environment::action_type > > > >
games::kuhn::Environment::private_history(Player player, const world_state_type& wstate) const
{
   std::vector<
      PlayerInformedType< std::optional< std::variant< chance_outcome_type, action_type > > > >
      out;
   auto action_history = wstate.history();
   out.reserve(action_history.size() + 2);
   for(auto&& [i, outcome_opt] : ranges::views::enumerate(wstate.cards())) {
      if(not outcome_opt.has_value()) {
         // the card has not been set yet, so we just return, as there is no further history
         break;
      }

      if(i == static_cast< size_t >(player)) {
         out.emplace_back(
            chance_outcome_type{.player = kuhn::Player(i), .card = outcome_opt.value()},
            Player::chance
         );
      } else {
         out.emplace_back(std::nullopt, Player::chance);
      }
   }
   for(auto&& [i, action_or_outcome] : ranges::views::enumerate(action_history)) {
      if(i != static_cast< size_t >(player)) {
         out.emplace_back(action_or_outcome, to_nor_player(kuhn::Player(i)));
      } else {
         out.emplace_back(std::nullopt, to_nor_player(kuhn::Player(i)));
      }
   }
   out.shrink_to_fit();
   return out;
}

std::vector< PlayerInformedType< std::variant<
   games::kuhn::Environment::chance_outcome_type,
   games::kuhn::Environment::action_type > > >
games::kuhn::Environment::open_history(const world_state_type& wstate) const
{
   using action_chance_variant = std::variant< chance_outcome_type, action_type >;
   std::vector< PlayerInformedType< action_chance_variant > > out;
   auto action_history = wstate.history();
   out.reserve(action_history.size() + 2);
   for(auto&& [i, action] : ranges::views::enumerate(action_history)) {
      out.emplace_back(action, to_nor_player(kuhn::Player(i)));
   }
   out.shrink_to_fit();
   return out;
}
std::vector< PlayerInformedType< std::optional< std::variant<
   games::kuhn::Environment::chance_outcome_type,
   games::kuhn::Environment::action_type > > > >
games::kuhn::Environment::public_history(const world_state_type& wstate) const
{
   auto hist = private_history(Player::alex, wstate);
   // hide the first entry too, since this is private information to Alex
   if(not hist.empty()) {
      hist[0].value().reset();
   }
   return hist;
}