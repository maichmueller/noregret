
#include "kuhn_poker/state.hpp"

#include <iostream>
#include <stdexcept>

namespace kuhn {

State::State(std::vector< Card > card_pool, size_t player_count)
    : m_player_count(player_count),
      m_player_cards(player_count, std::nullopt),
      m_folded(player_count, 0),
      m_open_responses(static_cast< int >(player_count)),
      m_card_pool(std::move(card_pool))
{
   if(m_player_count < 2 or m_player_count > max_player_count) {
      throw std::invalid_argument(
         "kuhn poker supports between 2 and 13 players, got " + std::to_string(m_player_count)
      );
   }
   if(m_card_pool.size() < m_player_count) {
      throw std::invalid_argument("the card pool must hold at least one unique card per player");
   }
}

void State::apply_action(Action action)
{
   const Player actor = m_active_player;
   m_history.emplace_back(action);
   m_actors.emplace_back(actor);
   if(not m_bet_outstanding) {
      if(action == Action::bet) {
         // opening bet: every other active player now owes a response
         m_bet_outstanding = true;
         m_open_responses = _active_count() - 1;
      } else {
         m_open_responses -= 1;
      }
   } else if(action == Action::check) {
      // folding against an outstanding bet
      m_folded.at(static_cast< size_t >(actor)) = 1;
      m_open_responses -= 1;
   } else {
      // calling (matching) the outstanding bet
      m_open_responses -= 1;
   }
   if(not is_terminal()) {
      m_active_player = _next_active_seat(actor);
   }
}
void State::apply_action(ChanceOutcome action)
{
   auto seat = static_cast< unsigned int >(action.player);
   if(seat >= m_player_count) {
      throw std::logic_error("Dealing to a seat beyond the configured player count.");
   }
   if(m_player_cards[seat].has_value()) {
      throw std::logic_error("Card has already been assigned.");
   }
   m_player_cards[seat] = action.card;
   if(_all_cards_engaged()) {
      m_active_player = static_cast< Player >(0);
      m_open_responses = static_cast< int >(m_player_count);
      m_bet_outstanding = false;
   }
}
bool State::is_terminal() const
{
   if(not _all_cards_engaged()) {
      return false;
   }
   // either everyone except a single player folded (fold-out short-circuit) or the betting
   // has closed (every active player passed without a bet or matched the outstanding one)
   return _active_count() <= 1 or m_open_responses == 0;
}

int State::payoff(Player player) const
{
   if(player == Player::chance) {
      throw std::invalid_argument("Can't provide payoff for chance player.");
   }
   if(not is_terminal()) {
      return 0;
   }

   int pot = 0;
   for(size_t seat = 0; seat < m_player_count; ++seat) {
      pot += _contribution(static_cast< Player >(seat));
   }

   std::vector< size_t > survivors;
   survivors.reserve(m_player_count);
   for(size_t seat = 0; seat < m_player_count; ++seat) {
      if(m_folded[seat] == 0) {
         survivors.emplace_back(seat);
      }
   }

   const size_t this_seat = static_cast< size_t >(player);
   int share = 0;
   if(survivors.size() == 1) {
      // fold-out: the last remaining player takes the whole pot
      if(survivors.front() == this_seat) {
         share = pot;
      }
   } else {
      // showdown: highest card among the survivors wins; true ties split evenly with any
      // remainder chips awarded to the tied players in seat order
      Card best_card = m_player_cards[survivors.front()].value();
      for(auto seat : survivors | ranges::views::drop(1)) {
         best_card = std::max(best_card, m_player_cards[seat].value());
      }
      std::vector< size_t > winners;
      for(auto seat : survivors) {
         if(m_player_cards[seat].value() == best_card) {
            winners.emplace_back(seat);
         }
      }
      auto winner_pos = ranges::find(winners, this_seat);
      if(winner_pos != winners.end()) {
         share = pot / static_cast< int >(winners.size());
         if(ranges::distance(winners.begin(), winner_pos)
            < static_cast< size_t >(pot % static_cast< int >(winners.size()))) {
            share += 1;
         }
      }
   }
   return share - _contribution(player);
}

bool State::is_valid(Action) const
{
   return _all_cards_engaged();
}
bool State::is_valid(ChanceOutcome outcome) const
{
   auto seat = static_cast< unsigned int >(outcome.player);
   if(seat >= m_player_count or m_player_cards[seat].has_value()) {
      return false;
   }
   auto all_outcomes = chance_actions();
   return std::find_if(
             all_outcomes.begin(),
             all_outcomes.end(),
             [&](const auto& this_outcome) { return this_outcome == outcome; }
          )
          != all_outcomes.end();
}
bool State::_all_cards_engaged() const
{
   return std::all_of(m_player_cards.begin(), m_player_cards.end(), [](const auto& opt_card) {
      return opt_card.has_value();
   });
}

size_t State::_dealt_count() const
{
   return static_cast< size_t >(ranges::count_if(m_player_cards, [](const auto& opt_card) {
      return opt_card.has_value();
   }));
}

int State::_active_count() const
{
   return static_cast< int >(m_player_count) - static_cast< int >(ranges::count(m_folded, char(1)));
}

kuhn::Player State::_next_active_seat(Player after) const
{
   const size_t current = static_cast< size_t >(after);
   for(size_t step = 1; step <= m_player_count; ++step) {
      const size_t candidate = (current + step) % m_player_count;
      if(m_folded[candidate] == 0) {
         return static_cast< Player >(candidate);
      }
   }
   return after;
}

int State::_contribution(Player player) const
{
   // every player posts an ante of 1 and adds 1 chip per bet they placed themselves. Since
   // raising is not part of the game each player can wager at most once per deal.
   int contribution = 1;
   for(auto&& [actor, action] : ranges::views::zip(m_actors, m_history)) {
      if(actor == player and action == Action::bet) {
         contribution += 1;
      }
   }
   return contribution;
}

std::vector< ChanceOutcome > State::chance_actions() const
{
   if(not m_history.empty() or _all_cards_engaged()) {
      return {};
   }
   Player next_receiver = static_cast< Player >(_dealt_count());
   std::vector< ChanceOutcome > outcomes;
   outcomes.reserve(m_card_pool.size());
   for(auto card : m_card_pool) {
      const bool already_dealt = ranges::any_of(m_player_cards, [&](const auto& opt_card) {
         return opt_card.has_value() and opt_card.value() == card;
      });
      if(not already_dealt) {
         outcomes.emplace_back(next_receiver, card);
      }
   }
   return outcomes;
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
   if(_all_cards_engaged() or not m_history.empty()) {
      return 0.;
   }
   return 1. / static_cast< double >(m_card_pool.size() - _dealt_count());
}

}  // namespace kuhn
