
#include "kuhn_poker/state.hpp"
#include <iostream>

namespace kuhn {

void State::apply_action(Action action)
{
   m_history.sequence.emplace_back(action);
   m_active_player = static_cast< Player >(not static_cast< bool >(m_active_player));
}
void State::apply_action(ChanceAction action)
{
   if(not m_player_cards[1].has_value()) {
      if(not m_player_cards[0].has_value()) {
         m_player_cards[0] = action;
      } else {
         m_player_cards[1] = action;
         m_active_player = Player::one;
      }
   } else {
      throw std::logic_error("All cards have already been assigned.");
   }
}
bool State::is_terminal() const
{
   const auto& terminal_seqs = _all_terminal_histories();
   return std::find(terminal_seqs.begin(), terminal_seqs.end(), m_history) != terminal_seqs.end();
}

int16_t State::payoff(Player player) const
{
   if(player == Player::chance) {
      throw std::invalid_argument("Can't provide payoff for chance player.");
   }
   if(not is_terminal()) {
      return 0;
   }
   bool higher_card_p1 = static_cast< int8_t >(m_player_cards[0].value())
                         > static_cast< int8_t >(m_player_cards[1].value());
   auto n_bets = static_cast< uint8_t >(
      std::count(m_history.sequence.begin(), m_history.sequence.end(), Action::bet));
   uint8_t factor = n_bets > 0 ? n_bets : 1;
   // 2 * x - 1 is a faster computation than std::pow((-1), x)
   return factor * (2 * higher_card_p1 - 1) * (2 * (player == Player::one) - 1);
}
const std::vector< History >& State::_all_terminal_histories()
{
   static const std::vector< History > terminal_sequences = {
      {{Action::check, Action::check}},
      {{Action::check, Action::bet, Action::check}},
      {{Action::check, Action::bet, Action::bet}},
      {{Action::bet, Action::bet}},
      {{Action::bet, Action::check}}};
   return terminal_sequences;
}
bool State::is_valid(Action) const
{
   return _all_cards_engaged();
}
bool State::is_valid(ChanceAction action) const
{
   if(_all_cards_engaged()) {
      return false;
   }
   auto all_actions = chance_actions();
   return std::find(all_actions.begin(), all_actions.end(), action) != all_actions.end();
}
bool State::_all_cards_engaged() const
{
   return std::all_of(m_player_cards.begin(), m_player_cards.end(), [](const auto& opt_card) {
            return opt_card.has_value();
         });
}

std::vector< ChanceAction > State::chance_actions() const
{
   if(not m_history.sequence.empty() or _all_cards_engaged()) {
      return {};
   }
   static const std::vector< Card > all_chance_actions = {Card::jack, Card::queen, Card::king};
   if(m_player_cards[0].has_value()) {
      return ranges::to< std::vector< ChanceAction > >(
         all_chance_actions
         | ranges::views::filter(
            [p1_card = m_player_cards[0].value()](auto card) { return card != p1_card; }));
   }
   return all_chance_actions;
}
std::vector< Action > State::actions() const
{
   if(not is_valid(Action::check)) {
      return {};
   }
   return std::vector< Action >{Action::check, Action::bet};
}
double State::chance_probability(ChanceAction) const
{
   if(m_player_cards[0].has_value()) {
      if(m_player_cards[1].has_value()) {
         return 0.;
      } else {
         return 0.5;
      }
   }
   return 1./3.;
}

}  // namespace kuhn