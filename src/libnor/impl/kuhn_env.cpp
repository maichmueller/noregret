
#include "nor/env/kuhn_env.hpp"

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
   auto obs = public_observation(wstate);
   // replace the opponents card with a question mark since it cannot be observed.
   if(uint8_t p_card_idx = static_cast< uint8_t >(player); wstate.cards()[p_card_idx].has_value()) {
      obs.replace(
         p_card_idx, 1, common::to_string(wstate.cards()[p_card_idx].value()).substr(0, 1));
   }
   return obs;
}
std::string nor::games::kuhn::Environment::public_observation(
   const nor::games::kuhn::Environment::world_state_type& wstate) const
{
   std::stringstream ss;
   for(uint8_t i = 0; i < 2; i++) {
      ss << (wstate.cards()[i].has_value() ? "?" : "-");
   }
   if(not wstate.history().empty()) {
      ss << "|";
   }
   for(auto card : wstate.history()) {
      ss << common::to_string(card)[0];
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
   return std::string(common::to_string(action));
}
void nor::games::kuhn::Environment::transition(
   nor::games::kuhn::Environment::world_state_type& worldstate,
   const nor::games::kuhn::Environment::chance_outcome_type& action) const
{
   worldstate.apply_action(action);
}
nor::games::kuhn::Environment::observation_type nor::games::kuhn::Environment::private_observation(
   nor::Player observer,
   const nor::games::kuhn::Environment::chance_outcome_type& action) const
{
   if(action.player == to_kuhn_player(observer)) {
      return std::string(common::to_string(action));
   } else {
      return "?";
   }
}
nor::games::kuhn::Environment::observation_type nor::games::kuhn::Environment::public_observation(
   const nor::games::kuhn::Environment::chance_outcome_type& action) const
{
   return "?";
}
nor::games::kuhn::Environment::observation_type nor::games::kuhn::Environment::tiny_repr(
   const nor::games::kuhn::Environment::world_state_type& wstate) const
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

std::vector< nor::PlayerInformedType< std::optional< std::variant<
   nor::games::kuhn::Environment::chance_outcome_type,
   nor::games::kuhn::Environment::action_type > > > >
nor::games::kuhn::Environment::private_history(
   nor::Player player,
   const nor::games::kuhn::Environment::world_state_type& wstate) const
{
   std::vector<
      nor::PlayerInformedType< std::optional< std::variant< chance_outcome_type, action_type > > > >
      out;
   auto action_history = wstate.history();
   out.reserve(action_history.size() + 2);
   for(auto&& [i, outcome_opt] : ranges::views::enumerate(wstate.cards())) {
      if(not outcome_opt.has_value()) {
         // the card has not been set yet so we just return, as there is no further history
         break;
      }

      if(i == static_cast< size_t >(player)) {
         out.emplace_back(
            chance_outcome_type{.player = kuhn::Player(i), .card = outcome_opt.value()},
            Player::chance);
      } else {
         out.emplace_back(std::nullopt, Player::chance);
      }
   }
   for(auto&& [i, action_or_outcome] : ranges::views::enumerate(action_history)) {
      if(i != static_cast< size_t >(player)) {
         out.emplace_back(std::move(action_or_outcome), to_nor_player(kuhn::Player(i)));
      } else {
         out.emplace_back(std::nullopt, to_nor_player(kuhn::Player(i)));
      }
   }
   out.shrink_to_fit();
   return out;
}

std::vector< nor::PlayerInformedType< std::variant<
   nor::games::kuhn::Environment::chance_outcome_type,
   nor::games::kuhn::Environment::action_type > > >
nor::games::kuhn::Environment::open_history(
   const nor::games::kuhn::Environment::world_state_type& wstate) const
{
   using action_chance_variant = std::variant< chance_outcome_type, action_type >;
   std::vector< nor::PlayerInformedType< action_chance_variant > > out;
   auto action_history = wstate.history();
   out.reserve(action_history.size() + 2);
   for(auto&& [i, action] : ranges::views::enumerate(action_history)) {
      out.emplace_back(std::move(action), to_nor_player(kuhn::Player(i)));
   }
   out.shrink_to_fit();
   return out;
}
std::vector< nor::PlayerInformedType< std::optional< std::variant<
   nor::games::kuhn::Environment::chance_outcome_type,
   nor::games::kuhn::Environment::action_type > > > >
nor::games::kuhn::Environment::public_history(
   const nor::games::kuhn::Environment::world_state_type& wstate) const
{
   auto hist = private_history(Player::alex, wstate);
   // hide the first entry too, since this is private information to Alex
   if(hist.size() > 0) {
      hist[0].value().reset();
   }
   return hist;
}
