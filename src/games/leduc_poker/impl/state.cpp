
#include "leduc_poker/state.hpp"

#include <unordered_set>

namespace leduc {

inline Player State::_cycle_active_player(bool folded, size_t shift_amount)
{
   if(folded) {
      // merely boot the front player from the game
      m_remaining_players.pop_front();
   } else {
      // left rotate all entries (pop front and append to the back)
      ranges::rotate(
         m_remaining_players, std::next(m_remaining_players.begin(), long(shift_amount))
      );
   }
   return m_remaining_players.front();
}

void State::apply_action(Action action)
{
   bool folded = false;
   switch(action.action_type) {
      case ActionType::bet: {
         m_bets_this_round += 1;
         _stake(m_active_player) += action.bet  // increment by the current betting amount
                                    + (m_active_bettor.has_value()
                                          ? m_history_since_last_bet[*m_active_bettor]->bet
                                          : 0.);  // + preceding bet amount (if re-raise!)
         m_active_bettor = m_active_player;
         // with a fresh bet we need to go around and ask every player anew for their response
         m_history_since_last_bet.reset();
         break;
      }
      case ActionType::check: {
         if(m_active_bettor.has_value()) {
            // if there is a bettor then check is actually a call --> add to the player's stakes
            _stake(m_active_player) = _stake(*m_active_bettor);
         }
         break;
      }
      case ActionType::fold: {
         // take the folding player out of the competing player range
         folded = true;
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
      m_active_bettor.reset();
      if(not m_public_card.has_value()) {
         // we have just concluded round 1 (no public card yet) --> reset counters and table
         m_bets_this_round = 0;
         // remove the folded player (current active player) from the roster
         _cycle_active_player(folded);
         // since we are moving on to the public card reveal (flop), we need to reset the order of
         // play back to the initial one (if possible)
         _reset_order_of_play();
      }
   } else [[likely]] {
      m_active_player = _cycle_active_player(folded);
   }
   // since the state has changed we need to recompute whether the game has terminated
   m_terminal_checked = false;
}

void State::_reset_order_of_play()
{
   int starting_player = static_cast< int >(config().starting_player);

   auto min_pos_pair = std::pair{
      std::numeric_limits< int >::max(), std::optional< Player >{std::nullopt}
   };
   auto min_neg_pair = std::pair{
      std::numeric_limits< int >::max(), std::optional< Player >{std::nullopt}
   };
   for(Player player : m_remaining_players) {
      int dist = static_cast< int >(player) - starting_player;
      auto& pair_to_update = (dist >= 0) ? min_pos_pair : min_neg_pair;
      if(dist < pair_to_update.first) {
         pair_to_update = std::pair{dist, player};
      }
   }
   Player player_to_go_next =
      ((min_pos_pair.second.has_value()) ? *min_pos_pair.second : *min_neg_pair.second);
   auto cycle_table_by = std::distance(
      m_remaining_players.begin(), ranges::find(m_remaining_players, player_to_go_next)
   );
   _cycle_active_player(false, size_t(cycle_table_by));
}

void State::apply_action(Card action)
{
   if(_all_player_cards_assigned()) [[unlikely]] {
      m_public_card = action;
      m_active_player = m_remaining_players.front();
   } else [[likely]] {
      m_player_cards.emplace_back(action);
      // we have to recheck here since we added a card to the player cards
      if(_all_player_cards_assigned()) {
         m_active_player = m_remaining_players.front();
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
   if(m_remaining_players.size() == 1) {
      return true;
   }
   if(not (round_nr() == 1)) {
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
   // the chosen player is the winnign player so all the pot goes to him, the payoff is everyone
   // else's bet in the game minus one's own contributions
   payoffs[as_int(player)] = ranges::accumulate(m_stakes, 0.) - stake(player);
}

std::vector< double > State::payoff()
{
   if(not is_terminal()) {
      return std::vector(config().n_players, 0.);
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
         if(card(p).rank == pub_card.rank) {
            winners.emplace_back(p);
         }
      }
      if(winners.empty()) {
         // no player has a pair --> next critera: who has the highest card?
         _determine_highest_card_winner(winners);
      }

      if(winners.size() == 1) {
         // a single winnner takes home the whole pot
         _single_pot_winner(payoffs, winners[0]);
      } else {
         // more than one winner --> split pot
         _split_pot(payoffs, winners);
      }
   }
   return payoffs;
}
void State::_determine_highest_card_winner(std::vector< Player >& winners) const
{
   auto rem_player_begin = m_remaining_players.begin();
   winners.emplace_back(*rem_player_begin);
   Rank highest_rank = card(*rem_player_begin).rank;
   for(auto rem_player : ranges::subrange{rem_player_begin + 1, m_remaining_players.end()}) {
      auto curr_rank = card(rem_player).rank;
      if(curr_rank > highest_rank) {
         winners.clear();
         winners.emplace_back(rem_player);
         highest_rank = curr_rank;
      } else if(curr_rank == highest_rank) {
         winners.emplace_back(rem_player);
      }
   }
}

bool State::is_valid(Action action) const
{
   if(m_active_player == Player::chance) {
      return false;
   }
   if(action.action_type == ActionType::bet) {
      return (m_bets_this_round < m_config.n_raises_allowed)
             and ranges::contains(bet_sizes(round_nr()), action.bet);
   }
   return true;
}

bool State::is_valid(Card outcome) const
{
   if(m_active_player != Player::chance) {
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
bool State::_all_player_cards_assigned() const
{
   return m_player_cards.size() == m_config.n_players;
}

std::vector< Card > State::chance_actions() const
{
   if(_all_player_cards_assigned() and m_public_card.has_value()) {
      return {};
   }
   return ranges::to_vector(ranges::views::filter(m_config.available_cards, [&](const auto& card) {
      return not ranges::contains(m_player_cards, card);
   }));
}

std::vector< Action > State::actions() const
{
   if(m_active_player == Player::chance) {
      return {};
   }
   std::vector< Action > all_actions{{ActionType::check}, {ActionType::fold}};
   if(m_config.n_raises_allowed > m_bets_this_round) {
      auto all_bets = bet_sizes(round_nr());
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

State::State(LeducConfig config)
    : m_remaining_players(),
      m_player_cards(),
      m_stakes(config.n_players, config.blind),  // everyone places at least the small blind
      m_history_since_last_bet(config.n_players),
      m_config(std::move(config))
{
   size_t nr_players = m_config.n_players;
   m_player_cards.reserve(nr_players);
   size_t start = static_cast< size_t >(m_config.starting_player);
   // emplace the players into the order queue from the starting player onwards, i.e.
   // (starter, starter + 1, starter + 2, ..., nr_players, 0, 1, ..., starter - 1)
   for(size_t p : ranges::views::concat(
          ranges::views::iota(start, nr_players), ranges::views::iota(size_t(0), start)
       )) {
      m_remaining_players.emplace_back(Player(p));
   }
}

}  // namespace leduc