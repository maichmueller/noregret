
#include "leduc_poker/state.hpp"

#include <unordered_set>

namespace leduc {

inline Player State::_cycle_active_player()
{
   // left rotate all entries (pop front and append to the back)
   ranges::rotate(m_remaining_players, m_remaining_players.begin() + 1);
   return m_remaining_players.front();
}

void State::apply_action(Action action)
{
   switch(action.action_type) {
      case ActionType::bet: {
         m_bets_this_round += 1;
         m_stakes[as_int(m_active_player)] += (m_active_bettor.has_value()
                                                  ? m_history_since_last_bet[*m_active_bettor]->bet
                                                  : 0.)
                                              + action.bet;
         m_active_bettor = m_active_player;
         // with a fresh bet we need to go around and ask every player anew for their response
         m_history_since_last_bet.reset();
         break;
      }
      case ActionType::check: {
         if(m_active_bettor.has_value()) {
            // if there is a bettor then check is actually a call --> add to the player's stakes
            m_stakes[as_int(m_active_player)] = m_stakes[as_int(*m_active_bettor)];
         }
         break;
      }
      case ActionType::fold: {
         // take the folding player out of the competing player range
         m_remaining_players.erase(ranges::remove(m_remaining_players, m_active_player));
         break;
      }
   }
   // append the active player's action to the last action history
   m_history.emplace_back(action);
   m_history_since_last_bet[m_active_player] = std::move(action);

   if(ranges::all_of(
         ranges::views::enumerate(m_history_since_last_bet),
         [&](const auto& player_opt) {
            const auto& [player, opt] = player_opt;
            return opt.has_value() or not ranges::contains(m_remaining_players, Player(player));
         }
      )) [[unlikely]] {
      // everyone left in the game acted in this round and the round is over
      // --> on to the public card or the game is over anyway
      m_active_player = Player::chance;
      m_history_since_last_bet.reset();
      _cycle_active_player();
   } else [[likely]] {
      m_active_player = _cycle_active_player();
   }
   // since the state has changed we need to recompute whether the game has terminated
   m_terminal_checked = false;
}

void State::apply_action(Card action)
{
   if(_all_player_cards_assigned()) [[unlikely]] {
      m_public_card = action;
      m_active_player = m_remaining_players[0];
      // reset the active bettor log. Noone is currently raising the stakes anymore
      m_active_bettor.reset();
      m_history_since_last_bet.reset();
   } else [[likely]] {
      m_player_cards.emplace_back(action);
      // we have to recheck here since we added a card to the player cards
      if(_all_player_cards_assigned()) {
         m_active_player = m_remaining_players[0];
      }
   }
   m_terminal_checked = false;
}

bool State::is_terminal()
{
   if(m_terminal_checked) {
      return m_is_terminal;
   }
   m_is_terminal = std::as_const(*this).is_terminal();
   m_terminal_checked = true;
   return m_is_terminal;
}

bool State::is_terminal() const
{
   if(not m_public_card.has_value()) {
      // if the public card has not yet been set then the game cannot be over with more than 1
      // player remaining
      return false;
   }

   if(m_active_player == Player::chance) {
      // there is a public card and we are back to having the chance player be the next active
      // player, then this means everyone has bet and the betting has concluded
      return true;
   }
   return false;
}

void State::_single_pot_winner(std::vector< double >& payoffs, Player player) const
{
   // the chosen player is the winnign player so all of the pot goes to him, the payoff is the
   // surpluss with respect to his own bet
   payoffs[as_int(player)] = std::accumulate(m_stakes.begin(), m_stakes.end(), 0.)
                             - static_cast< double >(m_stakes[as_int(player)]);
}

std::vector< double > State::payoff()
{
   if(not is_terminal()) {
      return std::vector(m_remaining_players.size(), 0.);
   }
   // initiate payoffs first as negative stakes for each player
   std::vector< double > payoffs;
   payoffs.reserve(m_stakes.size());
   ranges::insert(
      payoffs, payoffs.begin(), m_stakes | ranges::views::transform([](const auto& value) {
                                   return -value;
                                })
   );

   if(auto n_remaining = m_remaining_players.size(); n_remaining == 1) {
      _single_pot_winner(payoffs, m_remaining_players[0]);
   } else {
      assert(m_public_card.has_value());
      // in the following we can also assume that the public card is set, since there is a showdown
      // (aka it is a terminal state)!
      Card pub_card = *m_public_card;
      // there is a showdown between 2 or more players. We thus need to see who has the pair, if
      // any, or otherwise who has the highest card
      std::vector< Player > winners;
      for(auto p : m_remaining_players) {
         if(m_player_cards[as_int(p)].rank == pub_card.rank) {
            winners.emplace_back(p);
         }
      }
      // now we check for pairs with the public card first, and then for highest card
      if(winners.size() == 1) {
         _single_pot_winner(payoffs, winners[0]);
      } else if(winners.size() > 1) {
         // more than one winner --> split pot
         _split_pot(payoffs, winners);
      } else {
         // no player has a pair --> next critera: who has the highest card?
         auto rem_player_begin = m_remaining_players.begin();
         winners.emplace_back(*rem_player_begin);
         Rank highest_rank = m_player_cards[as_int(*rem_player_begin)].rank;
         for(auto rem_player : ranges::subrange{rem_player_begin + 1, m_remaining_players.end()}) {
            auto curr_rank = m_player_cards[as_int(rem_player)].rank;
            if(curr_rank > highest_rank) {
               winners.clear();
               winners.emplace_back(rem_player);
               highest_rank = curr_rank;
            } else if(curr_rank == highest_rank) {
               winners.emplace_back(rem_player);
            }
         }
         if(winners.size() == 1) {
            _single_pot_winner(payoffs, *winners.begin());
         } else {
            _split_pot(payoffs, winners);
         }
      }
   }
   payoffs.shrink_to_fit();
   return payoffs;
}

bool State::is_valid(Action action) const
{
   if(m_active_player == Player::chance) {
      return false;
   }
   if(action.action_type == ActionType::bet) {
      const auto& config = *m_config;
      return (m_bets_this_round < config.n_raises_allowed)
             and ranges::contains(
                m_public_card.has_value() ? config.bet_sizes_round_two : config.bet_sizes_round_one,
                action.bet
             );
   }
   return true;
}

bool State::is_valid(Card outcome) const
{
   auto all_outcomes = chance_actions();
   return std::find_if(
             all_outcomes.begin(),
             all_outcomes.end(),
             [&](const auto& this_outcome) { return this_outcome == outcome; }
          )
          != all_outcomes.end();
}
bool State::_all_player_cards_assigned() const
{
   return m_player_cards.size() == m_config->n_players;
}

std::vector< Card > State::chance_actions() const
{
   if(_all_player_cards_assigned() and m_public_card.has_value()) {
      return {};
   }
   return ranges::to_vector(ranges::views::filter(m_config->available_cards, [&](const auto& card) {
      return not ranges::contains(m_player_cards, card);
   }));
}

std::vector< Action > State::actions() const
{
   if(m_active_player == Player::chance) {
      return {};
   }
   std::vector< Action > all_actions{{ActionType::check}, {ActionType::fold}};
   if(m_config->n_raises_allowed > m_bets_this_round) {
      const auto& all_bets = m_public_card.has_value() ? m_config->bet_sizes_round_two
                                                       : m_config->bet_sizes_round_one;
      all_actions.reserve(2 + all_bets.size());
      for(auto bet_amount : all_bets) {
         all_actions.emplace_back(Action{ActionType::bet, bet_amount});
      }
   }
   return all_actions;
}

double State::chance_probability(Card) const
{
   return 1. / double(chance_actions().size());
}

State::State(sptr< const LeducConfig > config)
    : m_remaining_players(),
      m_player_cards(),
      m_stakes(config->n_players, config->small_blind),  // everyone places at least the small blind
      m_history_since_last_bet(config->n_players),
      m_config(std::move(config))
{
   size_t nr_players = m_config->n_players;
   m_player_cards.reserve(nr_players);
   for(size_t p = 0; p < nr_players; p++) {
      m_remaining_players.emplace_back(Player(p));
   }
}
State::State(LeducConfig config) : State(std::make_shared< const LeducConfig >(std::move(config)))
{
}

}  // namespace leduc
