
#include "nor/env/leduc.hpp"

#include <unordered_set>

#include "nor/utils/player_informed_type.hpp"

using namespace nor;
using namespace games::leduc;

nor::Player Environment::active_player(const world_state_type& wstate) const
{
   return to_nor_player(wstate.active_player());
}

bool Environment::is_terminal(const world_state_type& wstate)
{
   return wstate.is_terminal();
}

double Environment::reward(Player player, world_state_type& wstate)
{
   return wstate.payoff(to_leduc_player(player));
}

Environment::observation_type Environment::
   private_observation(nor::Player, const world_state_type&, const action_type&, const world_state_type&)
      const
{
   return "-";
}
Environment::observation_type Environment::
   public_observation(const world_state_type& wstate, const action_type& action, const world_state_type&)
      const
{
   return fmt::format("{}:{}", to_nor_player(wstate.active_player()), common::to_string(action));
}

Environment::observation_type Environment::private_observation(
   Player observer,
   const world_state_type& /*wstate*/,
   const chance_outcome_type& outcome,
   const world_state_type& next_wstate
) const
{
   // `next_wstate` is the worldstate resulting from applying `outcome` onto `wstate`
   if(next_wstate.public_card().has_value()) {
      // if the next worldstate has a public card then this means that this chance outcome was the
      // selection of the flop --> the outcome is fully public
      return "-";
   }
   auto owner_of_card = Player(next_wstate.cards().size() - 1);
   if(owner_of_card == observer) {
      // we only use the rank as observation of a card! (otherwise the number of suits multiplies
      // the number of infostates to no strategic benefit (in leduc))
      return common::to_string(outcome.rank);
   }
   return "-";
}

Environment::observation_type Environment::public_observation(
   const world_state_type& /*wstate*/,
   const chance_outcome_type& outcome,
   const world_state_type& next_wstate
) const
{
   if(next_wstate.public_card().has_value()) {
      // if the next worldstate has a public card then this means that this chance outcome was the
      // selection of the flop
      return common::to_string(outcome.rank);
   }
   return common::to_string(nor::Player(next_wstate.cards().size() - 1)) + ":?";
}

// Environment::observation_type Environment::tiny_repr(const world_state_type& wstate) const
//{
//    std::stringstream ss;
//    for(auto [idx, card] : ranges::views::enumerate(wstate.cards())) {
//       if(card.has_value()) {
//          ss << card.value();
//          if((idx == 0 and wstate.cards()[1].has_value()) or not wstate.history().empty()) {
//             ss << "-";
//          }
//       }
//    }
//    for(auto [idx, action] : ranges::views::enumerate(wstate.history())) {
//       ss << action;
//       if(idx != wstate.history().size() - 1) {
//          ss << "-";
//       }
//    }
//    return ss.str();
// }

std::vector< PlayerInformedType< std::optional< Environment::action_variant_type > > >
Environment::private_history(Player player, const world_state_type& wstate) const
{
   std::vector< PlayerInformedType< std::optional< action_variant_type > > > out;
   auto action_history = wstate.history();
   const auto n_players_total = wstate.config().n_players;
   out.reserve(action_history.size() + n_players_total);
   for(auto&& [i, card] : ranges::views::enumerate(wstate.cards())) {
      if(i == static_cast< size_t >(player)) {
         out.emplace_back(card, Player::chance);
      } else {
         out.emplace_back(std::nullopt, Player::chance);
      }
   }
   for(auto&& [i, action] : ranges::views::enumerate(action_history)) {
      out.emplace_back(action, nor::Player(i));
   }
   out.shrink_to_fit();
   return out;
}

std::vector< PlayerInformedType< Environment::action_variant_type > > Environment::open_history(
   const world_state_type& wstate
) const
{
   std::vector< PlayerInformedType< action_variant_type > > out;
   auto action_history = wstate.history();
   const auto n_players_total = wstate.config().n_players;
   out.reserve(action_history.size() + n_players_total);
   for(auto&& [i, card] : ranges::views::enumerate(wstate.cards())) {
      out.emplace_back(card, Player::chance);
   }
   for(auto&& [i, action] : ranges::views::enumerate(action_history)) {
      out.emplace_back(action, nor::Player(i));
   }
   out.shrink_to_fit();
   return out;
}
std::vector< PlayerInformedType< std::optional< Environment::action_variant_type > > >
Environment::public_history(const world_state_type& wstate) const
{
   std::vector< PlayerInformedType< std::optional< action_variant_type > > > out;
   auto action_history = wstate.history();
   const auto n_players_total = wstate.config().n_players;
   out.reserve(action_history.size() + n_players_total);
   for([[maybe_unused]] auto&& _ : ranges::views::enumerate(wstate.cards())) {
      out.emplace_back(std::nullopt, Player::chance);
   }
   for(auto&& [i, action] : ranges::views::enumerate(action_history)) {
      out.emplace_back(action, nor::Player(i));
   }
   out.shrink_to_fit();
   return out;
}
