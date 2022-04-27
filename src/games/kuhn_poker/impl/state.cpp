
#include "kuhn_poker/state.hpp"

#include <iostream>

namespace kuhn {

inline int16_t sign(bool x)
{
   return 2 * x - 1;
}

bool State::_has_higher_card(Player player) const
{
   return static_cast< int8_t >(m_player_cards[static_cast< size_t >(player)].value())
          > static_cast< int8_t >(m_player_cards[1 - static_cast< size_t >(player)].value());
}

void State::apply_action(Action action)
{
   m_history.sequence.emplace_back(action);
   m_active_player = static_cast< Player >(not static_cast< bool >(m_active_player));
}
void State::apply_action(ChanceOutcome action)
{
   if(m_player_cards[static_cast< unsigned int >(action.player)].has_value()) {
      throw std::logic_error("Card has already been assigned.");
   }
   m_player_cards[static_cast< unsigned int >(action.player)] = action.card;
   if(_all_cards_engaged()) {
      m_active_player = static_cast< Player >(1 - static_cast< int >(action.player));
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
   bool p1_has_bet = m_history.sequence.size() < 3 ? m_history.sequence[0] == Action::bet
                                                   : m_history.sequence[2] == Action::bet;
   bool p2_has_bet = m_history.sequence[1] == Action::bet;

   // 2 * x - 1 is a faster computation than std::pow((-1), x)
   for(auto [this_player, has_bet, other_has_bet] : std::array{
          std::tuple{Player::one, p1_has_bet, p2_has_bet},
          std::tuple{Player::two, p2_has_bet, p1_has_bet}}) {
      if(has_bet) {
         if(not other_has_bet) {
            auto value = sign(player == this_player);
            return sign(player == this_player);
         } else {
            auto value = sign(_has_higher_card(this_player)) * sign(player == this_player) * 2;
            return sign(_has_higher_card(this_player)) * sign(player == this_player) * 2;
         }
      }
   }
   return sign(_has_higher_card(Player::one)) * sign(player == Player::one);
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
bool State::is_valid(ChanceOutcome outcome) const
{
   if(m_player_cards[static_cast< unsigned int >(outcome.player)].has_value()) {
      return false;
   }
   auto all_outcomes = chance_actions();
   return std::find_if(
             all_outcomes.begin(),
             all_outcomes.end(),
             [&](const auto& this_outcome) { return this_outcome == outcome; })
          != all_outcomes.end();
}
bool State::_all_cards_engaged() const
{
   return std::all_of(m_player_cards.begin(), m_player_cards.end(), [](const auto& opt_card) {
      return opt_card.has_value();
   });
}

std::vector< ChanceOutcome > State::chance_actions() const
{
   if(not m_history.sequence.empty() or _all_cards_engaged()) {
      return {};
   }
   Player player = m_player_cards[0].has_value() ? (Player::two) : (Player::one);
   auto card_pool = m_card_pool;
   if(m_player_cards[0].has_value()) {
      card_pool.erase(
         std::remove(card_pool.begin(), card_pool.end(), m_player_cards[0].value()),
         card_pool.end());
   }
   auto to_chance_action = [](const auto& player_card) {
      return ChanceOutcome{std::get< 0 >(player_card), std::get< 1 >(player_card)};
   };
   return ranges::to< std::vector< ChanceOutcome > >(
      ranges::views::zip(ranges::views::repeat(player), card_pool)
      | ranges::views::transform(to_chance_action));
}
std::vector< Action > State::actions() const
{
   if(not is_valid(Action::check)) {
      return {};
   }
   return std::vector< Action >{Action::check, Action::bet};
}
double State::chance_probability(ChanceOutcome) const
{
   if(m_player_cards[0].has_value()) {
      if(m_player_cards[1].has_value()) {
         return 0.;
      } else {
         return 0.5;
      }
   }
   return 1. / 3.;
}

}  // namespace kuhn